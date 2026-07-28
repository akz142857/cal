"""Item-3 forward-model benchmark on the randomized-occlusion world.

Prediction target (per the item-3 design decision): the *any-object hidden
occupancy*.  Every predictor outputs an occupancy map; every currently-hidden
object cell is a positive; the field is all hidden non-static arena cells
(positives + empty decoys).  All predictors are scored on the same joint
occupancy, so belief/geometric merge one filter per hidden object to match the
joint maps the neural baselines and the entity graph already produce:

    positives = every occluded object's true cell (1 or 2)
    field     = every hidden, non-static arena cell

Metrics: top-1 accuracy (is the full-map argmax an occluded-object cell) and
mean reciprocal rank of the best positive over the field; argmax position error;
balanced binary occupancy NLL and field-normalized categorical NLL; Brier score;
empty-map rate; and top-1 by occlusion length.  Localization uses the full-map
argmax, so mass placed outside the hidden field is a real miss, not credited.

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
import json
from dataclasses import dataclass, field

import numpy as np

from cal.evaluation.randomized_occlusion_world import (
    _UNIT_VELOCITIES,
    _bounce_advance,
    RandomizedOcclusionWorld,
)
from cal.evaluation.v2_i1_integration import (
    ARENA_HIGH,
    ARENA_LOW,
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
) -> list[_Sample]:
    world = RandomizedOcclusionWorld(seed, hidden_turn_probability=turn_probability)
    action_rng = np.random.default_rng(int(seed) + 50_000)
    agent = None
    if attach_entity_graph:
        from cal.model.entity_belief_graph import IntegratedBeliefAgentV2

        agent = IntegratedBeliefAgentV2(seed=int(seed) + 40_000)
    samples: list[_Sample] = []

    visible_trail: dict[str, list[tuple[int, int]]] = {"a": [], "b": []}
    last_seen: dict[str, tuple[int, int] | None] = {"a": None, "b": None}
    observed_velocity: dict[str, tuple[int, int]] = {"a": (1, 0), "b": (0, 1)}
    velocity_known = {"a": False, "b": False}
    hidden_steps = {"a": 0, "b": 0}

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
        selected = hidden_names[(int(seed) + step) % len(hidden_names)]
        if last_seen[selected] is None or not velocity_known[selected]:
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
        positives = tuple(cells[n] for n in hidden_objects)
        hidden_tracks = tuple(
            (last_seen[n], observed_velocity[n], hidden_steps[n])
            for n in hidden_objects
            if last_seen[n] is not None and velocity_known[n]
        )
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
    return samples


_SIDE = ARENA_HIGH - ARENA_LOW + 1  # 11
_MAX_POSITION_ERROR = 2 * (_SIDE - 1)


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


def _rank(occupancy: np.ndarray, sample: _Sample) -> dict[str, float]:
    """Score an occupancy map for the any-object hidden-localization task.

    Positives are all currently-hidden object cells; the field is all hidden
    non-static cells (positives + empty decoys).  Localization (top-1, argmax
    distance) uses the *full-map* argmax, so mass placed outside the field
    (e.g. an extrapolation that lands on a now-visible cell) is a genuine miss,
    not credited to the true cell.  Ranking (MRR, categorical NLL) is taken over
    the field; a predictor with no field mass gets the worst rank / max penalty.
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
    # Balanced binary occupancy NLL / Brier (per-cell, positive vs decoy).
    balanced_binary_nll = float(np.mean(-np.log(np.maximum(pos_scores, _EPS))))
    brier = float(np.mean((1.0 - pos_scores) ** 2))
    if decoy_scores.size:
        balanced_binary_nll = 0.5 * (
            balanced_binary_nll
            + float(np.mean(-np.log(np.maximum(1.0 - decoy_scores, _EPS))))
        )
        brier = 0.5 * (brier + float(np.mean(decoy_scores**2)))
    # Field-normalized categorical NLL: -log P(mass on a positive | field).
    field_total = float(field_scores.sum())
    if field_total <= 0.0:
        categorical_nll = float(-np.log(_EPS))
    else:
        categorical_nll = float(
            -np.log(max(float(pos_scores.sum()) / field_total, _EPS))
        )
    # Localization is over the map *projected onto the hidden field*: visible
    # and non-field cells are dropped, so every predictor competes only on its
    # hidden-object mass (the entity graph's visible-object occupancy does not
    # count) and mass placed outside the field cannot be credited to a positive.
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
    argmax_cell = field[int(np.argmax(field_scores))]
    distance = float(
        min(abs(argmax_cell[0] - p[0]) + abs(argmax_cell[1] - p[1]) for p in positives)
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
    for sample, occupancy in zip(samples, maps, strict=True):
        candidate_counts.append(len(sample.candidate_cells))
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
        "sample_count": len(samples),
    }


def _binned_ranking(
    samples: list[_Sample], maps: np.ndarray
) -> dict[str, dict[str, float]]:
    bins: dict[str, list[float]] = {}
    for sample, occupancy in zip(samples, maps, strict=True):
        key = "6+" if sample.hidden_steps >= 6 else str(sample.hidden_steps)
        bins.setdefault(key, []).append(_rank(occupancy, sample)["top1"])
    return {
        key: {
            "top1_accuracy": float(np.mean(values)),
            "sample_count": len(values),
        }
        for key, values in sorted(bins.items())
    }


def _collect_many(
    seeds: list[int],
    *,
    steps: int,
    warmup: int,
    turn_probability: float,
    attach_entity_graph: bool = False,
) -> list[_Sample]:
    samples: list[_Sample] = []
    for seed in seeds:
        samples.extend(
            _collect(
                seed,
                steps=steps,
                warmup=warmup,
                turn_probability=turn_probability,
                attach_entity_graph=attach_entity_graph,
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
) -> dict[str, object]:
    if not 0.0 <= turn_probability <= 1.0:
        raise ValueError("turn_probability must be in [0, 1]")
    eval_samples = _collect_many(
        evaluation_seeds,
        steps=steps,
        warmup=warmup,
        turn_probability=turn_probability,
        attach_entity_graph=include_entity_graph,
    )
    if not eval_samples:
        raise ValueError(
            "evaluation configuration produced no valid hidden-object samples"
        )
    maps: dict[str, np.ndarray] = {
        "belief": np.stack([_belief_map(s, turn_probability) for s in eval_samples]),
        "geometric": np.stack([_geometric_map(s) for s in eval_samples]),
    }
    if include_entity_graph:
        maps["entity_graph"] = np.stack(
            [s.agent_occupancy for s in eval_samples]
        )

    if include_gru or include_slot:
        train_samples = _collect_many(
            train_seeds, steps=steps, warmup=warmup, turn_probability=turn_probability
        )
        if not train_samples:
            raise ValueError(
                "training configuration produced no valid hidden-object samples"
            )
        if include_gru:
            from cal.evaluation._permanence_gru_baseline import gru_predictor_maps

            maps["gru"] = gru_predictor_maps(train_samples, eval_samples)
        if include_slot:
            from cal.evaluation._permanence_slot_baseline import slot_predictor_maps

            maps["slot"] = slot_predictor_maps(train_samples, eval_samples)

    predictors = {
        name: _score_maps(eval_samples, matrix) for name, matrix in maps.items()
    }
    ranking_by_occlusion_length = {
        name: _binned_ranking(eval_samples, matrix)
        for name, matrix in maps.items()
    }
    return {
        "world": "randomized",
        "turn_probability": turn_probability,
        "train_seeds": train_seeds,
        "evaluation_seeds": evaluation_seeds,
        "evaluation_sample_count": len(eval_samples),
        "predictors": predictors,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--turn-probability", type=float, default=_DEFAULT_TURN_PROBABILITY)
    parser.add_argument("--train-seeds", type=int, default=40)
    parser.add_argument("--eval-seeds", type=int, default=16)
    parser.add_argument("--train-base", type=int, default=61000)
    parser.add_argument("--eval-base", type=int, default=61100)
    parser.add_argument("--gru", action="store_true")
    parser.add_argument("--slot", action="store_true")
    parser.add_argument(
        "--entity-graph",
        action="store_true",
        help="run the real I1 IntegratedBeliefAgentV2 as a learned-belief predictor",
    )
    parser.add_argument("--sweep", action="store_true", help="run GRU capacity sweep")
    args = parser.parse_args()

    train = [args.train_base + i for i in range(args.train_seeds)]
    evaluation = [args.eval_base + i for i in range(args.eval_seeds)]
    common = dict(
        steps=args.steps, warmup=args.warmup, turn_probability=args.turn_probability
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
            **common,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
