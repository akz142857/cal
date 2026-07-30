"""Item-3 forward-model benchmark on the randomized-occlusion world.

Prediction target (per the item-3 design decision): the *any-object hidden
occupancy*.  Every predictor outputs an occupancy map; every currently-hidden
object cell is a positive; the field is all hidden non-static arena cells
(positives + empty decoys).  All predictors are scored on the same joint
occupancy, so belief/geometric merge one filter per hidden object to match the
joint maps the neural baselines and the entity graph already produce:

    positives = every occluded object's true cell (1 or 2)
    field     = every hidden, non-static arena cell

Metrics: hidden-field-conditioned top-1 accuracy and mean reciprocal rank of the
best positive; hidden-field-conditioned argmax position error; balanced binary
occupancy NLL, field-normalized categorical NLL, and a proper field-average
binary Brier score; empty-field rate; the exact any-positive random top-1
chance; and single-hidden-object top-1 by occlusion length.  Every predictor map
is projected onto the same hidden field before localization.  Mass outside that
field neither helps nor competes; if a map has no mass inside the field, it
receives an explicit miss.

The benchmark implements four baselines plus the optional deployed entity
graph:

    belief     -- an online occupancy filter over (position, velocity) that
                  models the world's stochastic hidden-maneuver kernel exactly.
                  This is the action-conditioned forward model over object
                  state -- it maintains and propagates a calibrated belief.
    geometric  -- constant-velocity extrapolation that knows this episode's
                  occluder geometry but ignores hidden maneuvers (a point mass).
    gru        -- a small non-object-centric recurrent predictor trained on the
                  sensed-patch + action sequence (the "no object structure"
                  neural control).
    slot       -- a small SlotSSM-style object-centric recurrent predictor with
                  slot attention and spatial-broadcast decoding.
    entity_graph (optional) -- the deployed I1 entity belief graph, updated
                  online from the same sensed-patch + action sequence.

The whole module is NON-GATED analysis: no frozen protocol, no source lock, no
one-shot evidence, and it runs only on fresh unreserved seeds.

Run:
    uv run python -m cal.evaluation.permanence_forward_benchmark
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cal.evaluation.randomized_occlusion_world import (
    _UNIT_VELOCITIES,
    _bounce_advance,
    RandomizedOcclusionWorld,
)
from cal.evaluation.v2_i1_integration import (
    ARENA_HIGH,
    ARENA_LOW,
    CAMERA,
    WARMUP,
    _global_visibility,
)

_EPS = 1e-6
_DEFAULT_TURN_PROBABILITY = 0.35


def _unblocked_others(
    position: tuple[int, int],
    velocity: tuple[int, int],
    static: frozenset[tuple[int, int]],
) -> list[tuple[int, int]]:
    others = []
    for candidate in _UNIT_VELOCITIES:
        if candidate == velocity:
            continue
        moved, _ = _bounce_advance(position, candidate, static)
        if moved != position:
            others.append(candidate)
    return others


def _successors(
    position: tuple[int, int],
    velocity: tuple[int, int],
    static: frozenset[tuple[int, int]],
    turn_probability: float,
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """Exact one-step transition kernel of ``RandomizedOcclusionWorld`` while
    the object is occluded.

    A hidden object turns with probability ``turn_probability`` to a uniformly
    chosen *unblocked* alternative direction (the first unblocked entry of a
    random permutation is uniform over the unblocked set); otherwise it keeps
    its velocity and advances with reflection.
    """

    if not 0.0 <= turn_probability <= 1.0:
        raise ValueError("turn_probability must be in [0, 1]")
    keep_pos, keep_vel = _bounce_advance(position, velocity, static)
    unblocked = _unblocked_others(position, velocity, static)
    if not unblocked or turn_probability <= 0.0:
        return [(keep_pos, keep_vel, 1.0)]
    successors = [(keep_pos, keep_vel, 1.0 - turn_probability)]
    share = turn_probability / len(unblocked)
    for candidate in unblocked:
        moved, new_vel = _bounce_advance(position, candidate, static)
        successors.append((moved, new_vel, share))
    return successors


def _belief_occupancy(
    last_seen: tuple[int, int],
    observed_velocity: tuple[int, int],
    hidden_steps: int,
    static: frozenset[tuple[int, int]],
    visible: np.ndarray,
    turn_probability: float,
) -> dict[tuple[int, int], float]:
    """Propagate and condition a belief through a continuous hidden interval.

    The first step starts at the last *visible* position, so the world cannot
    take a hidden maneuver on that transition.  Every later transition starts
    from an observed-hidden cell and follows the stochastic hidden kernel.
    After each transition, discard hypotheses inconsistent with the continuing
    empty-visible observation.
    """

    belief: dict[tuple[tuple[int, int], tuple[int, int]], float] = {
        (last_seen, observed_velocity): 1.0
    }
    for hidden_index in range(hidden_steps):
        nxt: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        for (pos, vel), mass in belief.items():
            successors = (
                [(*_bounce_advance(pos, vel, static), 1.0)]
                if hidden_index == 0
                else _successors(pos, vel, static, turn_probability)
            )
            for new_pos, new_vel, prob in successors:
                if visible[new_pos[1], new_pos[0]]:
                    continue
                key = (new_pos, new_vel)
                nxt[key] = nxt.get(key, 0.0) + mass * prob
        total = sum(nxt.values())
        if total <= 0.0:
            return {}
        belief = {state: mass / total for state, mass in nxt.items()}
    occupancy: dict[tuple[int, int], float] = {}
    for (pos, _vel), mass in belief.items():
        occupancy[pos] = occupancy.get(pos, 0.0) + mass
    return occupancy


@dataclass
class _Sample:
    seed: int
    step: int
    positive: tuple[int, int]
    negative: tuple[int, int]
    last_seen: tuple[int, int]
    observed_velocity: tuple[int, int]
    hidden_steps: int
    focus_object: str
    focus_occlusion_id: int
    hidden_object_count: int
    static: frozenset[tuple[int, int]]
    visible: np.ndarray
    # Any-object occupancy task: every currently-hidden object cell is a
    # positive; the field (candidate_cells) is all hidden non-static cells
    # (positives + empty decoys). Ranking the positives against the whole field
    # removes decoy-selection bias. hidden_tracks carries one (last_seen,
    # velocity, hidden_steps) per hidden object with a known velocity, so belief
    # and geometric can merge one filter per object. positive/negative/last_seen
    # /observed_velocity/hidden_steps remain the focus object, used for the GRU
    # training query term and occlusion-length binning only.
    positives: tuple[tuple[int, int], ...] = ()
    hidden_tracks: tuple[
        tuple[tuple[int, int], tuple[int, int], int], ...
    ] = ()
    candidate_cells: tuple[tuple[int, int], ...] = ()
    # Sequence context and target for the neural baseline.
    sensed_history: list[np.ndarray] = field(default_factory=list)
    action_history: list[int] = field(default_factory=list)
    hidden_occupancy: np.ndarray | None = None
    # Online occupancy posterior of the real I1 entity-graph agent, snapshotted
    # at the query step (only populated when attach_entity_graph=True).
    agent_occupancy: np.ndarray | None = None


def _collect(
    seed: int,
    *,
    steps: int,
    warmup: int,
    turn_probability: float,
    attach_entity_graph: bool = False,
    audit: Counter[str] | None = None,
    agent_seed: int = 74_000,
) -> list[_Sample]:
    world = RandomizedOcclusionWorld(seed, hidden_turn_probability=turn_probability)
    action_rng = np.random.default_rng(int(seed) + 50_000)
    agent = None
    if attach_entity_graph:
        from cal.model.entity_belief_graph import IntegratedBeliefAgentV2

        # The candidate's RNG must not be derived from the environment seed.
        agent = IntegratedBeliefAgentV2(seed=int(agent_seed))
    samples: list[_Sample] = []

    visible_trail: dict[str, list[tuple[int, int]]] = {"a": [], "b": []}
    last_seen: dict[str, tuple[int, int] | None] = {"a": None, "b": None}
    observed_velocity: dict[str, tuple[int, int]] = {"a": (1, 0), "b": (0, 1)}
    velocity_known = {"a": False, "b": False}
    hidden_steps = {"a": 0, "b": 0}
    occlusion_ids = {"a": 0, "b": 0}
    audit = audit if audit is not None else Counter()

    sensed, local_visibility = world.observe()
    if agent is not None:
        agent.update(sensed, 0)
    visible = _global_visibility(local_visibility, world.grid_size)
    cells = {
        "a": (int(world.distractor_a[0]), int(world.distractor_a[1])),
        "b": (int(world.distractor_b[0]), int(world.distractor_b[1])),
    }
    previous_visible = {n: bool(visible[cells[n][1], cells[n][0]]) for n in ("a", "b")}
    for n in ("a", "b"):
        if previous_visible[n]:
            visible_trail[n].append(cells[n])
            last_seen[n] = cells[n]

    sensed_history = [sensed.astype(np.float32)]
    action_history = [0]

    for step in range(1, steps + 1):
        action = int(action_rng.integers(0, 5))
        sensed, local_visibility = world.step(action)
        if agent is not None:
            agent.update(sensed, action)
        visible = _global_visibility(local_visibility, world.grid_size)
        sensed_history.append(sensed.astype(np.float32))
        action_history.append(action)
        cells = {
            "a": (int(world.distractor_a[0]), int(world.distractor_a[1])),
            "b": (int(world.distractor_b[0]), int(world.distractor_b[1])),
        }
        current_visible = {
            n: bool(visible[cells[n][1], cells[n][0]]) for n in ("a", "b")
        }
        for n in ("a", "b"):
            was_visible = previous_visible[n]
            if was_visible and not current_visible[n]:
                hidden_steps[n] = 1
                occlusion_ids[n] += 1
            elif not current_visible[n]:
                hidden_steps[n] += 1
            else:
                hidden_steps[n] = 0
            if current_visible[n]:
                last_seen[n] = cells[n]
                if not was_visible:
                    # A displacement across an occlusion interval is not a
                    # one-step velocity observation.
                    visible_trail[n] = [cells[n]]
                    velocity_known[n] = False
                else:
                    trail = visible_trail[n]
                    prior = trail[-1]
                    trail.append(cells[n])
                    if len(trail) > 2:
                        trail.pop(0)
                    dx = cells[n][0] - prior[0]
                    dy = cells[n][1] - prior[1]
                    if abs(dx) + abs(dy) == 1:
                        observed_velocity[n] = (dx, dy)
                        velocity_known[n] = True
                    elif velocity_known[n]:
                        expected, new_velocity = _bounce_advance(
                            prior, observed_velocity[n], world.static
                        )
                        if expected == cells[n]:
                            observed_velocity[n] = new_velocity
                        else:
                            velocity_known[n] = False
                    else:
                        velocity_known[n] = False
            previous_visible[n] = current_visible[n]

        if step < warmup:
            continue
        hidden_names = [n for n in ("a", "b") if hidden_steps[n] >= 2]
        if not hidden_names:
            continue
        audit["eligible_query_steps"] += 1
        selected = hidden_names[(int(seed) + step) % len(hidden_names)]
        if last_seen[selected] is None or not velocity_known[selected]:
            audit["skipped_unknown_focus_track"] += 1
            continue
        positive = cells[selected]
        truth = world.truth()
        decoys = [
            (x, y)
            for y in range(ARENA_LOW, ARENA_HIGH + 1)
            for x in range(ARENA_LOW, ARENA_HIGH + 1)
            if not visible[y, x]
            and not bool(truth[y, x])
            and (x, y) not in world.static
        ]
        if not decoys:
            continue
        negative = min(
            decoys,
            key=lambda cell: (
                abs(cell[0] - positive[0]) + abs(cell[1] - positive[1]),
                (cell[0] * 17 + cell[1] * 31 + int(seed) + step) % 97,
            ),
        )
        # Any-object occupancy: all currently-hidden objects are positives; the
        # field is positives + empty hidden decoys; belief/geometric merge one
        # filter per hidden object that has a known velocity.
        hidden_objects = [n for n in ("a", "b") if not current_visible[n]]
        raw_positives = tuple(cells[n] for n in hidden_objects)
        positives = tuple(dict.fromkeys(raw_positives))
        actor_cells = (
            tuple(int(v) for v in world.self_position),
            cells["a"],
            cells["b"],
        )
        if len(positives) != len(raw_positives):
            audit["skipped_duplicate_hidden_positive"] += 1
            continue
        if len(set(actor_cells)) != len(actor_cells):
            audit["skipped_actor_collision"] += 1
            continue
        hidden_tracks = tuple(
            (last_seen[n], observed_velocity[n], hidden_steps[n])
            for n in hidden_objects
            if last_seen[n] is not None and velocity_known[n]
        )
        if len(hidden_tracks) != len(hidden_objects):
            # The analytic belief is an oracle only when every hidden positive
            # has a corresponding tracked state.  Unknown-velocity events need
            # a separately preregistered mixture prior and are excluded here.
            audit["skipped_incomplete_hidden_tracks"] += 1
            continue
        candidate_cells = (*positives, *decoys)
        side = ARENA_HIGH - ARENA_LOW + 1
        hidden_occupancy = np.zeros((side, side), dtype=np.float32)
        for name in ("a", "b"):
            if not current_visible[name]:
                cx, cy = cells[name]
                if ARENA_LOW <= cx <= ARENA_HIGH and ARENA_LOW <= cy <= ARENA_HIGH:
                    hidden_occupancy[cy - ARENA_LOW, cx - ARENA_LOW] = 1.0
        agent_occupancy = None
        if agent is not None:
            agent_occupancy = (
                agent.probability()[
                    ARENA_LOW : ARENA_HIGH + 1, ARENA_LOW : ARENA_HIGH + 1
                ]
                .ravel()
                .astype(np.float64)
            )
        samples.append(
            _Sample(
                seed=seed,
                step=step,
                positive=positive,
                negative=negative,
                last_seen=last_seen[selected],
                observed_velocity=observed_velocity[selected],
                hidden_steps=hidden_steps[selected],
                focus_object=selected,
                focus_occlusion_id=occlusion_ids[selected],
                hidden_object_count=len(hidden_objects),
                static=world.static,
                visible=visible.copy(),
                positives=positives,
                hidden_tracks=hidden_tracks,
                candidate_cells=candidate_cells,
                sensed_history=list(sensed_history),
                action_history=list(action_history),
                hidden_occupancy=hidden_occupancy,
                agent_occupancy=agent_occupancy,
            )
        )
        audit["accepted_samples"] += 1
        audit[f"accepted_seed_{int(seed)}"] += 1
        if len(hidden_objects) > 1:
            audit["accepted_multi_hidden_object_samples"] += 1
        audit["maximum_hidden_steps"] = max(
            audit["maximum_hidden_steps"], hidden_steps[selected]
        )
    return samples


_SIDE = ARENA_HIGH - ARENA_LOW + 1  # 11
_MAX_POSITION_ERROR = 2 * (_SIDE - 1)
_OCCLUSION_BIN_ORDER = ("2-3", "4-5", "6+")


def _occlusion_length_bin(hidden_steps: int) -> str:
    if hidden_steps < 2:
        raise ValueError("permanence samples require at least two hidden steps")
    if hidden_steps <= 3:
        return "2-3"
    if hidden_steps <= 5:
        return "4-5"
    return "6+"


def _cell_index(cell: tuple[int, int]) -> int:
    return (cell[1] - ARENA_LOW) * _SIDE + (cell[0] - ARENA_LOW)


def _index_cell(index: int) -> tuple[int, int]:
    return (index % _SIDE + ARENA_LOW, index // _SIDE + ARENA_LOW)


def _belief_map(sample: _Sample, turn_probability: float) -> np.ndarray:
    """Union of one occupancy belief per hidden object with a known velocity."""

    vector = np.zeros(_SIDE * _SIDE, dtype=np.float64)
    for last_seen, velocity, hidden_steps in sample.hidden_tracks:
        occupancy = _belief_occupancy(
            last_seen,
            velocity,
            hidden_steps,
            sample.static,
            sample.visible,
            turn_probability,
        )
        for cell, mass in occupancy.items():
            index = _cell_index(cell)
            vector[index] = 1.0 - (1.0 - vector[index]) * (1.0 - mass)
    return vector


def _geometric_map(sample: _Sample) -> np.ndarray:
    """Constant-velocity extrapolation point mass, one per hidden object."""

    vector = np.zeros(_SIDE * _SIDE, dtype=np.float64)
    for last_seen, velocity, hidden_steps in sample.hidden_tracks:
        pos, vel = last_seen, velocity
        for _ in range(hidden_steps):
            pos, vel = _bounce_advance(pos, vel, sample.static)
        vector[_cell_index(pos)] = 1.0
    return vector


def _field_geometry_key(
    sample: _Sample, cell: tuple[int, int]
) -> tuple[int, int, int, int]:
    field = set(sample.candidate_cells)
    xs = [candidate[0] for candidate in field]
    ys = [candidate[1] for candidate in field]
    width = max(max(xs) - min(xs), 1)
    height = max(max(ys) - min(ys), 1)
    relative_x = min(3, int(4 * (cell[0] - min(xs)) / (width + 1)))
    relative_y = min(3, int(4 * (cell[1] - min(ys)) / (height + 1)))
    non_field = [
        (x, y)
        for y in range(ARENA_LOW, ARENA_HIGH + 1)
        for x in range(ARENA_LOW, ARENA_HIGH + 1)
        if (x, y) not in field
    ]
    boundary_depth = min(
        3,
        min(
            abs(cell[0] - other[0]) + abs(cell[1] - other[1])
            for other in non_field
        ),
    )
    field_size_bucket = min(5, len(field) // 20)
    return relative_x, relative_y, boundary_depth, field_size_bucket


def _prior_key(
    mode: str, sample: _Sample, cell: tuple[int, int]
) -> tuple[object, ...]:
    if mode == "center_distance":
        return (abs(cell[0] - CAMERA[0]) + abs(cell[1] - CAMERA[1]),)
    if mode == "global_coordinate":
        return cell
    if mode == "field_geometry":
        return _field_geometry_key(sample, cell)
    if mode == "duration_conditioned_coordinate":
        return (_occlusion_length_bin(sample.hidden_steps), *cell)
    raise ValueError(f"unknown prior mode: {mode}")


def _fit_prior_maps(
    train_samples: list[_Sample],
    evaluation_samples: list[_Sample],
    *,
    mode: str,
) -> np.ndarray:
    """Fit a smoothed no-trajectory prior for leakage decomposition."""

    positive_counts: Counter[tuple[object, ...]] = Counter()
    opportunity_counts: Counter[tuple[object, ...]] = Counter()
    total_positives = 0
    total_opportunities = 0
    for sample in train_samples:
        positive_set = set(sample.positives)
        for cell in sample.candidate_cells:
            key = _prior_key(mode, sample, cell)
            opportunity_counts[key] += 1
            total_opportunities += 1
            if cell in positive_set:
                positive_counts[key] += 1
                total_positives += 1
    if total_opportunities <= 0:
        raise ValueError("cannot fit a prior without candidate opportunities")
    base_rate = total_positives / total_opportunities
    alpha = 2.0
    maps = np.zeros((len(evaluation_samples), _SIDE * _SIDE), dtype=np.float64)
    for row, sample in enumerate(evaluation_samples):
        for cell in sample.candidate_cells:
            key = _prior_key(mode, sample, cell)
            count = opportunity_counts[key]
            maps[row, _cell_index(cell)] = (
                positive_counts[key] + alpha * base_rate
            ) / (count + alpha)
    return maps


def _uniform_field_maps(samples: list[_Sample]) -> np.ndarray:
    """Calibrated exchangeable occupancy over each evaluation field."""

    maps = np.zeros((len(samples), _SIDE * _SIDE), dtype=np.float64)
    for row, sample in enumerate(samples):
        probability = len(sample.positives) / len(sample.candidate_cells)
        for cell in sample.candidate_cells:
            maps[row, _cell_index(cell)] = probability
    return maps


def _rank(occupancy: np.ndarray, sample: _Sample) -> dict[str, float]:
    """Score a map after projection onto the shared hidden candidate field.

    Positives are all currently-hidden object cells; the field is all hidden
    non-static cells (positives + empty decoys).  Visible and otherwise
    out-of-field mass is discarded before top-1, distance, MRR, and categorical
    NLL are computed.  A predictor with no field mass gets the worst rank,
    maximum distance, and an explicit empty-field miss.
    """

    if not np.isfinite(occupancy).all():
        raise ValueError("predictor map contains non-finite values")
    positives = set(sample.positives)
    field = sample.candidate_cells
    field_scores = np.clip(occupancy[[_cell_index(c) for c in field]], 0.0, 1.0)
    pos_scores = np.clip(
        np.array([occupancy[_cell_index(c)] for c in sample.positives]), 0.0, 1.0
    )
    decoy_scores = np.clip(
        np.array([occupancy[_cell_index(c)] for c in field if c not in positives]),
        0.0,
        1.0,
    )
    # The balanced NLL is retained as a class-balanced diagnostic.  Brier is a
    # field-average binary score, without outcome-dependent class weighting, so
    # it remains a proper score for each cell's hidden-occupancy probability.
    balanced_binary_nll = float(np.mean(-np.log(np.maximum(pos_scores, _EPS))))
    if decoy_scores.size:
        balanced_binary_nll = 0.5 * (
            balanced_binary_nll
            + float(np.mean(-np.log(np.maximum(1.0 - decoy_scores, _EPS))))
        )
    field_targets = np.asarray(
        [float(cell in positives) for cell in field], dtype=np.float64
    )
    brier = float(np.mean((field_scores - field_targets) ** 2))
    # Field-normalized categorical NLL: -log P(mass on a positive | field).
    field_total = float(field_scores.sum())
    if field_total <= 0.0:
        categorical_nll = float(-np.log(_EPS))
    else:
        categorical_nll = float(
            -np.log(max(float(pos_scores.sum()) / field_total, _EPS))
        )
    # Hidden-field-conditioned localization: visible and non-field cells are
    # dropped, so every predictor competes only on its hidden-object mass.
    field_max = float(field_scores.max()) if field_scores.size else 0.0
    if field_max <= 0.0:
        return {
            "top1": 0.0,
            "rr": 1.0 / len(field),
            "distance": float(_MAX_POSITION_ERROR),
            "balanced_binary_nll": balanced_binary_nll,
            "categorical_nll": categorical_nll,
            "brier": brier,
            "empty": 1.0,
        }
    max_cells = [field[i] for i in np.flatnonzero(field_scores == field_max)]
    top1 = sum(1 for c in max_cells if c in positives) / len(max_cells)
    # Average over all maximum-score ties rather than taking the first field
    # entry.  candidate_cells intentionally lists positives first, so np.argmax
    # would otherwise give a uniform predictor a spurious zero distance.
    distance = float(
        np.mean(
            [
                min(
                    abs(cell[0] - positive[0]) + abs(cell[1] - positive[1])
                    for positive in positives
                )
                for cell in max_cells
            ]
        )
    )
    # Reciprocal rank of the best-scored positive within the field.
    p_best = float(pos_scores.max()) if pos_scores.size else 0.0
    if p_best <= 0.0:
        reciprocal_rank = 1.0 / len(field)
    else:
        greater = int(np.sum(field_scores > p_best))
        tied = int(np.sum(field_scores == p_best))
        reciprocal_rank = 1.0 / (greater + (tied + 1) / 2.0)
    return {
        "top1": top1,
        "rr": reciprocal_rank,
        "distance": distance,
        "balanced_binary_nll": balanced_binary_nll,
        "categorical_nll": categorical_nll,
        "brier": brier,
        "empty": 0.0,
    }


def _score_maps(
    samples: list[_Sample], maps: np.ndarray
) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot score an empty sample set")
    top1 = []
    mrr = []
    binary_nll = []
    categorical_nll = []
    brier = []
    distances = []
    empty_maps = 0
    candidate_counts = []
    random_top1_chances = []
    for sample, occupancy in zip(samples, maps, strict=True):
        candidate_counts.append(len(sample.candidate_cells))
        random_top1_chances.append(
            len(sample.positives) / len(sample.candidate_cells)
        )
        scored = _rank(occupancy, sample)
        empty_maps += int(scored["empty"])
        top1.append(scored["top1"])
        mrr.append(scored["rr"])
        binary_nll.append(scored["balanced_binary_nll"])
        categorical_nll.append(scored["categorical_nll"])
        brier.append(scored["brier"])
        distances.append(scored["distance"])
    return {
        "top1_accuracy": float(np.mean(top1)),
        "mrr": float(np.mean(mrr)),
        "balanced_binary_nll": float(np.mean(binary_nll)),
        "categorical_nll": float(np.mean(categorical_nll)),
        "brier": float(np.mean(brier)),
        "argmax_position_error": float(np.mean(distances)),
        "empty_map_rate": float(empty_maps / len(samples)),
        "mean_candidate_count": float(np.mean(candidate_counts)),
        "mean_any_positive_top1_chance": float(np.mean(random_top1_chances)),
        "sample_count": len(samples),
    }


_AGGREGATE_METRIC_KEYS = (
    "top1_accuracy",
    "mrr",
    "balanced_binary_nll",
    "categorical_nll",
    "brier",
    "argmax_position_error",
    "empty_map_rate",
    "mean_candidate_count",
    "mean_any_positive_top1_chance",
)


def _mean_metric_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        raise ValueError("cannot aggregate empty metric records")
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in _AGGREGATE_METRIC_KEYS
    }


def _sample_metric_record(occupancy: np.ndarray, sample: _Sample) -> dict[str, float]:
    scored = _rank(occupancy, sample)
    return {
        "top1_accuracy": scored["top1"],
        "mrr": scored["rr"],
        "balanced_binary_nll": scored["balanced_binary_nll"],
        "categorical_nll": scored["categorical_nll"],
        "brier": scored["brier"],
        "argmax_position_error": scored["distance"],
        "empty_map_rate": scored["empty"],
        "mean_candidate_count": float(len(sample.candidate_cells)),
        "mean_any_positive_top1_chance": len(sample.positives)
        / len(sample.candidate_cells),
    }


def _episode_binned_score(
    samples: list[_Sample], maps: np.ndarray
) -> dict[str, object]:
    """Equal-weight seed, occlusion-length bin, and focus episode.

    Only single-hidden-object samples have an unambiguous duration label.  The
    reduction order is fixed: time points -> focus episode within a length bin
    -> episodes within seed/bin -> the three bins within a seed -> seeds.  A
    129-step episode therefore has the same weight as a short episode in the
    same seed/bin, and every qualifying seed has equal final weight.
    """

    if len(samples) != len(maps):
        raise ValueError("sample/map length mismatch")
    episode_bins: dict[
        tuple[int, str, int, str], list[dict[str, float]]
    ] = {}
    eligible_sample_count = 0
    for sample, occupancy in zip(samples, maps, strict=True):
        if sample.hidden_object_count != 1:
            continue
        eligible_sample_count += 1
        bin_name = _occlusion_length_bin(sample.hidden_steps)
        key = (
            sample.seed,
            sample.focus_object,
            sample.focus_occlusion_id,
            bin_name,
        )
        episode_bins.setdefault(key, []).append(
            _sample_metric_record(occupancy, sample)
        )

    episode_bin_scores = {
        key: _mean_metric_records(records) for key, records in episode_bins.items()
    }
    seed_bins: dict[tuple[int, str], list[dict[str, float]]] = {}
    for (seed, _object_name, _episode_id, bin_name), score in episode_bin_scores.items():
        seed_bins.setdefault((seed, bin_name), []).append(score)
    seed_bin_scores = {
        key: _mean_metric_records(records) for key, records in seed_bins.items()
    }

    seeds = sorted({sample.seed for sample in samples})
    seed_scores: dict[str, dict[str, object]] = {}
    for seed in seeds:
        available = [
            seed_bin_scores[(seed, bin_name)]
            for bin_name in _OCCLUSION_BIN_ORDER
            if (seed, bin_name) in seed_bin_scores
        ]
        if not available:
            continue
        present_bins = [
            bin_name
            for bin_name in _OCCLUSION_BIN_ORDER
            if (seed, bin_name) in seed_bin_scores
        ]
        seed_scores[str(seed)] = {
            **_mean_metric_records(available),
            "present_bins": present_bins,
            "complete_bins": len(present_bins) == len(_OCCLUSION_BIN_ORDER),
            "episode_bin_count": sum(
                len(seed_bins[(seed, bin_name)]) for bin_name in present_bins
            ),
        }

    complete_seed_records = [
        {key: float(score[key]) for key in _AGGREGATE_METRIC_KEYS}
        for score in seed_scores.values()
        if bool(score["complete_bins"])
    ]
    overall = (
        _mean_metric_records(complete_seed_records)
        if complete_seed_records
        else None
    )
    complete_seed_ids = {
        int(seed) for seed, score in seed_scores.items() if score["complete_bins"]
    }
    by_bin: dict[str, dict[str, float | int]] = {}
    for bin_name in _OCCLUSION_BIN_ORDER:
        records = [
            score
            for (seed, current_bin), score in seed_bin_scores.items()
            if current_bin == bin_name and seed in complete_seed_ids
        ]
        if records:
            by_bin[bin_name] = {
                **_mean_metric_records(records),
                "seed_count": len(records),
                "episode_bin_count": sum(
                    len(seed_bins[(seed, bin_name)])
                    for seed, current_bin in seed_bin_scores
                    if current_bin == bin_name and seed in complete_seed_ids
                ),
            }
    return {
        "policy": (
            "single-hidden samples; time->episode/bin->seed/bin->equal bins->equal "
            "complete seeds; length curves use the same complete-seed population"
        ),
        "eligible_sample_count": eligible_sample_count,
        "excluded_multi_hidden_sample_count": len(samples) - eligible_sample_count,
        "episode_bin_group_count": len(episode_bin_scores),
        "complete_seed_count": len(complete_seed_records),
        "complete_seed_ids": sorted(complete_seed_ids),
        "overall_complete_seeds_only": overall,
        "by_occlusion_length": by_bin,
        "seed_scores": seed_scores,
        "seed_bin_scores": {
            str(seed): {
                bin_name: {
                    key: float(score[key])
                    for key in _AGGREGATE_METRIC_KEYS
                }
                for bin_name in _OCCLUSION_BIN_ORDER
                if (score := seed_bin_scores.get((seed, bin_name)))
                is not None
            }
            for seed in sorted(complete_seed_ids)
        },
    }


def _paired_seed_bootstrap(
    first: dict[str, object],
    second: dict[str, object],
    *,
    first_name: str,
    second_name: str,
    bootstrap_samples: int,
    rng_seed: int,
) -> dict[str, object]:
    """Paired percentile bootstrap over complete seed-level primary scores."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    first_seed_scores = first["seed_scores"]
    second_seed_scores = second["seed_scores"]
    seed_ids = sorted(
        seed
        for seed in set(first_seed_scores) & set(second_seed_scores)
        if first_seed_scores[seed]["complete_bins"]
        and second_seed_scores[seed]["complete_bins"]
    )
    if len(seed_ids) < 2:
        raise ValueError("paired bootstrap requires at least two complete seeds")
    metric_directions = {
        "top1_accuracy": "higher",
        "categorical_nll": "lower",
        "brier": "lower",
        "argmax_position_error": "lower",
    }
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(seed_ids), size=(bootstrap_samples, len(seed_ids)))
    metrics: dict[str, object] = {}
    for metric, direction in metric_directions.items():
        first_values = np.asarray(
            [first_seed_scores[seed][metric] for seed in seed_ids], dtype=np.float64
        )
        second_values = np.asarray(
            [second_seed_scores[seed][metric] for seed in seed_ids], dtype=np.float64
        )
        advantage = (
            first_values - second_values
            if direction == "higher"
            else second_values - first_values
        )
        bootstrap_means = advantage[indices].mean(axis=1)
        metrics[metric] = {
            "direction": direction,
            "advantage_positive_is_better": float(advantage.mean()),
            "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
            "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
            "bootstrap_probability_advantage_positive": float(
                np.mean(bootstrap_means > 0.0)
            ),
        }
    return {
        "first": first_name,
        "second": second_name,
        "paired_seed_count": len(seed_ids),
        "paired_seed_ids": [int(seed) for seed in seed_ids],
        "bootstrap_samples": bootstrap_samples,
        "rng_seed": rng_seed,
        "metrics": metrics,
    }


def _collect_many(
    seeds: list[int],
    *,
    steps: int,
    warmup: int,
    turn_probability: float,
    attach_entity_graph: bool = False,
    audit: Counter[str] | None = None,
    agent_seed_base: int = 74_000,
) -> list[_Sample]:
    samples: list[_Sample] = []
    for episode_index, seed in enumerate(seeds):
        samples.extend(
            _collect(
                seed,
                steps=steps,
                warmup=warmup,
                turn_probability=turn_probability,
                attach_entity_graph=attach_entity_graph,
                audit=audit,
                agent_seed=agent_seed_base + episode_index,
            )
        )
    return samples


def run_benchmark(
    train_seeds: list[int],
    evaluation_seeds: list[int],
    *,
    steps: int,
    warmup: int,
    turn_probability: float,
    include_gru: bool = False,
    include_slot: bool = False,
    include_entity_graph: bool = False,
    paired_bootstrap_samples: int = 0,
) -> dict[str, object]:
    if not 0.0 <= turn_probability <= 1.0:
        raise ValueError("turn_probability must be in [0, 1]")
    evaluation_audit: Counter[str] = Counter()
    eval_samples = _collect_many(
        evaluation_seeds,
        steps=steps,
        warmup=warmup,
        turn_probability=turn_probability,
        attach_entity_graph=include_entity_graph,
        audit=evaluation_audit,
    )
    if not eval_samples:
        raise ValueError(
            "evaluation configuration produced no valid hidden-object samples"
        )
    training_audit: Counter[str] = Counter()
    train_samples = _collect_many(
        train_seeds,
        steps=steps,
        warmup=warmup,
        turn_probability=turn_probability,
        audit=training_audit,
    )
    if not train_samples:
        raise ValueError("training configuration produced no valid hidden-object samples")

    maps: dict[str, np.ndarray] = {
        "belief": np.stack([_belief_map(s, turn_probability) for s in eval_samples]),
        "geometric": np.stack([_geometric_map(s) for s in eval_samples]),
    }
    if include_entity_graph:
        maps["entity_graph"] = np.stack(
            [s.agent_occupancy for s in eval_samples]
        )

    if include_gru or include_slot:
        if include_gru:
            from cal.evaluation._permanence_gru_baseline import gru_predictor_maps

            maps["gru"] = gru_predictor_maps(train_samples, eval_samples)
        if include_slot:
            from cal.evaluation._permanence_slot_baseline import slot_predictor_maps

            maps["slot"] = slot_predictor_maps(train_samples, eval_samples)

    predictors = {
        name: _score_maps(eval_samples, matrix) for name, matrix in maps.items()
    }
    episode_binned_predictors = {
        name: _episode_binned_score(eval_samples, matrix)
        for name, matrix in maps.items()
    }
    paired_comparisons: dict[str, object] = {}
    if paired_bootstrap_samples:
        comparison_pairs = [("belief", "geometric")]
        if "entity_graph" in episode_binned_predictors:
            comparison_pairs.extend(
                ("entity_graph", baseline)
                for baseline in ("geometric", "gru", "slot")
                if baseline in episode_binned_predictors
            )
        for comparison_index, (first_name, second_name) in enumerate(
            comparison_pairs
        ):
            key = f"{first_name}_vs_{second_name}"
            paired_comparisons[key] = _paired_seed_bootstrap(
                episode_binned_predictors[first_name],
                episode_binned_predictors[second_name],
                first_name=first_name,
                second_name=second_name,
                bootstrap_samples=paired_bootstrap_samples,
                rng_seed=73_191 + comparison_index,
            )
    # Keep the public length-curve key, but make it an alias of the primary
    # episode/bin/seed reduction.  Exposing the former time-point-weighted curve
    # beside the primary score would let long events dominate downstream tables.
    ranking_by_occlusion_length = {
        name: score["by_occlusion_length"]
        for name, score in episode_binned_predictors.items()
    }
    single_hidden_count = sum(
        1 for sample in eval_samples if sample.hidden_object_count == 1
    )
    primary_train_samples = [
        sample for sample in train_samples if sample.hidden_object_count == 1
    ]
    primary_eval_samples = [
        sample for sample in eval_samples if sample.hidden_object_count == 1
    ]
    prior_maps = {"uniform_field": _uniform_field_maps(primary_eval_samples)}
    for prior_mode in (
        "center_distance",
        "global_coordinate",
        "field_geometry",
        "duration_conditioned_coordinate",
    ):
        prior_maps[prior_mode] = _fit_prior_maps(
            primary_train_samples,
            primary_eval_samples,
            mode=prior_mode,
        )
    leakage_audit = {
        "population": "single-hidden-object accepted events",
        "baselines": {
            name: {
                "sample_level": _score_maps(primary_eval_samples, matrix),
                "episode_binned_primary": _episode_binned_score(
                    primary_eval_samples, matrix
                ),
                "model_inputs": {
                    "uniform_field": ["shared_hidden_field_mask"],
                    "center_distance": [
                        "shared_hidden_field_mask",
                        "camera_center_distance",
                    ],
                    "global_coordinate": [
                        "shared_hidden_field_mask",
                        "absolute_cell_coordinate",
                    ],
                    "field_geometry": [
                        "shared_hidden_field_mask",
                        "field_relative_geometry",
                    ],
                    "duration_conditioned_coordinate": [
                        "shared_hidden_field_mask",
                        "absolute_cell_coordinate",
                        "focus_hidden_duration_bin",
                    ],
                }[name],
            }
            for name, matrix in prior_maps.items()
        },
        "evaluation_seed_in_model_inputs": False,
        "evaluation_positive_labels_in_model_inputs": False,
        "shared_hidden_field_mask_in_model_inputs": True,
        "note": (
            "Uniform chance, radial center bias, absolute coordinate, field-relative "
            "geometry, and duration-conditioned coordinate priors. The last baseline "
            "isolates duration conditioning but does not identify every possible "
            "acceptance-filter selection effect; no pass/fail gate is defined yet."
        ),
    }
    focus_occlusion_counts = Counter(
        (sample.seed, sample.focus_object, sample.focus_occlusion_id)
        for sample in eval_samples
    )
    edge_case_audit = {
        "accepted_unique_focus_occlusions": len(focus_occlusion_counts),
        "maximum_samples_per_focus_occlusion": max(
            focus_occlusion_counts.values(), default=0
        ),
        "accepted_multi_hidden_object_samples": sum(
            sample.hidden_object_count > 1 for sample in eval_samples
        ),
        "skipped_unknown_focus_track": evaluation_audit.get(
            "skipped_unknown_focus_track", 0
        ),
        "skipped_incomplete_hidden_tracks": evaluation_audit.get(
            "skipped_incomplete_hidden_tracks", 0
        ),
        "skipped_duplicate_hidden_positive": evaluation_audit.get(
            "skipped_duplicate_hidden_positive", 0
        ),
        "skipped_actor_collision": evaluation_audit.get(
            "skipped_actor_collision", 0
        ),
        "maximum_hidden_steps": evaluation_audit.get("maximum_hidden_steps", 0),
        "note": (
            "Collision, duplicate-positive, and incomplete-oracle events are "
            "reported and excluded; long-run concentration is diagnostic only."
        ),
    }
    evaluation_seed_counts = Counter(sample.seed for sample in eval_samples)
    evaluation_seed_coverage = {
        "accepted_samples_by_seed": {
            str(seed): evaluation_seed_counts[seed] for seed in evaluation_seeds
        },
        "zero_sample_seeds": [
            seed for seed in evaluation_seeds if evaluation_seed_counts[seed] == 0
        ],
        "minimum_accepted_samples": min(
            (evaluation_seed_counts[seed] for seed in evaluation_seeds), default=0
        ),
        "maximum_accepted_samples": max(
            (evaluation_seed_counts[seed] for seed in evaluation_seeds), default=0
        ),
        "all_seeds_have_samples": all(
            evaluation_seed_counts[seed] > 0 for seed in evaluation_seeds
        ),
        "note": "Diagnostic only; a minimum event-coverage gate is not frozen.",
    }
    return {
        "world": "randomized",
        "turn_probability": turn_probability,
        "train_seeds": train_seeds,
        "evaluation_seeds": evaluation_seeds,
        "evaluation_sample_count": len(eval_samples),
        "evaluation_collection_audit": dict(sorted(evaluation_audit.items())),
        "training_collection_audit": dict(sorted(training_audit.items())),
        "leakage_audit": leakage_audit,
        "edge_case_audit": edge_case_audit,
        "evaluation_seed_coverage": evaluation_seed_coverage,
        "model_rng_policy": "agent_seed_base_plus_episode_index_not_world_seed",
        "occlusion_length_binning": {
            "policy": "single_hidden_object_events_only",
            "included_sample_count": single_hidden_count,
            "excluded_multi_hidden_sample_count": len(eval_samples)
            - single_hidden_count,
        },
        "predictors": predictors,
        "episode_binned_predictors": episode_binned_predictors,
        "paired_seed_bootstrap": paired_comparisons,
        "ranking_by_occlusion_length": ranking_by_occlusion_length,
    }


def gru_capacity_sweep(
    train_seeds: list[int],
    evaluation_seeds: list[int],
    *,
    steps: int,
    warmup: int,
    turn_probability: float,
    hidden_sizes: tuple[int, ...] = (16, 64, 128, 256),
    epochs_grid: tuple[int, ...] = (25, 80),
) -> dict[str, object]:
    """Sweep GRU capacity/epochs so a near-chance result cannot be dismissed as
    a single-configuration training artifact."""

    from cal.evaluation._permanence_gru_baseline import (
        gru_predictor_maps,
        parameter_count,
    )

    if not 0.0 <= turn_probability <= 1.0:
        raise ValueError("turn_probability must be in [0, 1]")
    train_samples = _collect_many(
        train_seeds, steps=steps, warmup=warmup, turn_probability=turn_probability
    )
    eval_samples = _collect_many(
        evaluation_seeds, steps=steps, warmup=warmup, turn_probability=turn_probability
    )
    if not train_samples:
        raise ValueError(
            "training configuration produced no valid hidden-object samples"
        )
    if not eval_samples:
        raise ValueError(
            "evaluation configuration produced no valid hidden-object samples"
        )
    runs = []
    for hidden in hidden_sizes:
        for epochs in epochs_grid:
            maps = gru_predictor_maps(
                train_samples, eval_samples, epochs=epochs, hidden=hidden
            )
            score = _score_maps(eval_samples, maps)
            runs.append(
                {
                    "hidden": hidden,
                    "epochs": epochs,
                    "parameter_count": parameter_count(hidden),
                    "top1_accuracy": score["top1_accuracy"],
                    "mrr": score["mrr"],
                    "categorical_nll": score["categorical_nll"],
                }
            )
    return {
        "evaluation_sample_count": len(eval_samples),
        "turn_probability": turn_probability,
        "runs": runs,
    }


def _development_audit_summary(report: dict[str, object]) -> dict[str, object]:
    """Compact the non-gated report without changing any computation."""

    predictor_summary = {}
    for name, sample_score in report["predictors"].items():
        primary = report["episode_binned_predictors"][name]
        predictor_summary[name] = {
            "sample_level_top1": sample_score["top1_accuracy"],
            "sample_level_chance": sample_score[
                "mean_any_positive_top1_chance"
            ],
            "episode_binned_overall": primary["overall_complete_seeds_only"],
            "complete_seed_count": primary["complete_seed_count"],
            "episode_bin_group_count": primary["episode_bin_group_count"],
        }
    prior_summary = {}
    for name, baseline in report["leakage_audit"]["baselines"].items():
        primary = baseline["episode_binned_primary"]
        prior_summary[name] = {
            "sample_level_top1": baseline["sample_level"]["top1_accuracy"],
            "sample_level_chance": baseline["sample_level"][
                "mean_any_positive_top1_chance"
            ],
            "episode_binned_overall": primary["overall_complete_seeds_only"],
            "complete_seed_count": primary["complete_seed_count"],
        }
    return {
        "world": report["world"],
        "turn_probability": report["turn_probability"],
        "seed_registry": report.get("seed_registry"),
        "evaluation_sample_count": report["evaluation_sample_count"],
        "evaluation_seed_coverage": report["evaluation_seed_coverage"],
        "edge_case_audit": report["edge_case_audit"],
        "occlusion_length_binning": report["occlusion_length_binning"],
        "predictors": predictor_summary,
        "position_prior_decomposition": prior_summary,
        "paired_seed_bootstrap": report["paired_seed_bootstrap"],
        "stability_checks": {
            "all_evaluation_seeds_have_samples": report[
                "evaluation_seed_coverage"
            ]["all_seeds_have_samples"],
            "all_evaluation_seeds_have_all_primary_bins": all(
                predictor["complete_seed_count"] == len(report["evaluation_seeds"])
                for predictor in report["episode_binned_predictors"].values()
            ),
            "turn_probability_scan_run": False,
            "paired_ci_run": bool(report["paired_seed_bootstrap"]),
            "neural_or_entity_graph_baselines_run": any(
                name in report["predictors"]
                for name in ("gru", "slot", "entity_graph")
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--turn-probability", type=float)
    parser.add_argument("--train-seeds", type=int, default=40)
    parser.add_argument("--eval-seeds", type=int, default=16)
    parser.add_argument("--train-base", type=int, default=61000)
    parser.add_argument("--eval-base", type=int, default=61100)
    parser.add_argument(
        "--seed-registry",
        type=Path,
        help="development registry JSON; overrides base/count seed arguments",
    )
    parser.add_argument("--gru", action="store_true")
    parser.add_argument("--slot", action="store_true")
    parser.add_argument(
        "--entity-graph",
        action="store_true",
        help="run the real I1 IntegratedBeliefAgentV2 as a learned-belief predictor",
    )
    parser.add_argument("--sweep", action="store_true", help="run GRU capacity sweep")
    parser.add_argument(
        "--audit-summary",
        action="store_true",
        help="print a compact development-only audit report",
    )
    parser.add_argument(
        "--paired-ci",
        action="store_true",
        help="run deterministic 10,000-resample paired seed bootstrap CIs",
    )
    args = parser.parse_args()

    if args.seed_registry is not None:
        registry = json.loads(args.seed_registry.read_text(encoding="utf-8"))
        if registry.get("status") != "development_only_non_gated":
            raise ValueError("only a development-only non-gated registry is accepted")
        if registry.get("model_metrics_read") is not False:
            raise ValueError("seed registry must attest model_metrics_read=false")
        train = [int(seed) for seed in registry["train_seeds"]]
        evaluation = [int(seed) for seed in registry["evaluation_seeds"]]
        turn_probability = float(
            registry.get("selected_turn_probability", _DEFAULT_TURN_PROBABILITY)
        )
    else:
        train = [args.train_base + i for i in range(args.train_seeds)]
        evaluation = [args.eval_base + i for i in range(args.eval_seeds)]
        turn_probability = _DEFAULT_TURN_PROBABILITY
    if args.turn_probability is not None:
        turn_probability = args.turn_probability
    common = dict(
        steps=args.steps, warmup=args.warmup, turn_probability=turn_probability
    )
    if args.sweep:
        report = gru_capacity_sweep(train, evaluation, **common)
    else:
        report = run_benchmark(
            train,
            evaluation,
            include_gru=args.gru,
            include_slot=args.slot,
            include_entity_graph=args.entity_graph,
            paired_bootstrap_samples=10_000 if args.paired_ci else 0,
            **common,
        )
    if args.seed_registry is not None:
        report["seed_registry"] = {
            "path": str(args.seed_registry),
            "selection_digest_sha256": registry.get("selection_digest_sha256"),
        }
    if args.audit_summary and not args.sweep:
        report = _development_audit_summary(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
