"""Confirmatory diagnostic for the L0 V8 permanence-control failure.

This is a NON-GATED analysis script (no frozen protocol, no source lock, no
one-shot evidence). It exists to empirically confirm the root-cause claim in
``docs/experiments/V2_L0_PERMANENCE_DIAGNOSIS_AND_PREREGISTRATION_DRAFT.md``:

    The current fixed-camera / fixed-occluder environment makes the hidden
    object's position a deterministic function of ``(seed, step)`` that is
    independent of the agent's actions.  The permanence probe therefore does
    not test object permanence -- a pure geometric/kinematic extrapolator with
    no belief state recovers the "hidden cell contains the entity" label at a
    level comparable to (or better than) the formal entity graph.

It re-simulates the exact same ``_IntegratedWorld`` trajectory, the exact same
``hidden_steps`` bookkeeping, and the exact same permanence positive/negative
construction as ``cal.evaluation.v2_l0_language_readout``, then compares three
predictors on the SAME held-out permanence samples:

    1. ``analytic``   -- forward-simulate the deterministic distractor dynamics
                         from the last visible state (no training, no belief).
    2. ``geometry``   -- a ridge linear probe over geometry-only features
                         (query one-hot + last-seen position + observed
                         velocity + hidden_steps).  No sensed grid, no belief.
    3. ``raw_sensor`` -- a ridge linear probe over the sensed occupancy grid +
                         query one-hot, reproducing the readout's raw baseline.

It runs ONLY on fresh, unreserved diagnostic seeds (60000+).  It never touches
any consumed holdout (I1 31000-31015 / 32000-32007, L0 33xxx) and produces no
gated evidence.

Run:
    uv run python -m cal.evaluation.permanence_geometry_diagnostic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from cal.evaluation.randomized_occlusion_world import (
    _bounce_advance,
    RandomizedOcclusionWorld,
)
from cal.evaluation.v2_i1_integration import (
    ARENA_HIGH,
    ARENA_LOW,
    WARMUP,
    _IntegratedWorld,
    _global_visibility,
)

WorldFactory = Callable[[int], object]

WORLD_FACTORIES: dict[str, WorldFactory] = {
    "fixed": lambda seed: _IntegratedWorld(seed),
    "randomized": lambda seed: RandomizedOcclusionWorld(seed),
}

ARENA_SIDE = ARENA_HIGH - ARENA_LOW + 1  # 11


def _local_index(x: int, y: int) -> int:
    return (y - ARENA_LOW) * ARENA_SIDE + (x - ARENA_LOW)


def _query_one_hot(cell: tuple[int, int]) -> np.ndarray:
    vector = np.zeros(ARENA_SIDE * ARENA_SIDE, dtype=np.float32)
    x, y = cell
    if ARENA_LOW <= x <= ARENA_HIGH and ARENA_LOW <= y <= ARENA_HIGH:
        vector[_local_index(x, y)] = 1.0
    return vector


def _forward_extrapolate(
    position: tuple[int, int],
    velocity: tuple[int, int],
    static: set[tuple[int, int]],
    n_steps: int,
) -> tuple[int, int]:
    """Replicate the deterministic distractor update in ``_IntegratedWorld.step``.

    Distractor motion is action-independent, so an observer that knows the last
    visible position, the observed velocity and the static layout can compute
    the occluded position exactly.
    """

    px, py = position
    vx, vy = velocity
    axis = 0 if vx != 0 else 1
    for _ in range(n_steps):
        if axis == 0:
            nxt = px + vx
            blocked = (
                nxt < ARENA_LOW or nxt > ARENA_HIGH or (nxt, py) in static
            )
            if blocked:
                vx = -vx
                nxt = px + vx
            px = nxt
        else:
            nxt = py + vy
            blocked = (
                nxt < ARENA_LOW or nxt > ARENA_HIGH or (px, nxt) in static
            )
            if blocked:
                vy = -vy
                nxt = py + vy
            py = nxt
    return int(px), int(py)


@dataclass
class _Sample:
    seed: int
    step: int
    query: tuple[int, int]
    label: float
    predicted_cell: tuple[int, int]  # analytic extrapolation of the entity
    geometry_features: np.ndarray
    raw_features: np.ndarray


@dataclass
class _Collector:
    samples: list[_Sample] = field(default_factory=list)
    reconstruction_hits: int = 0
    reconstruction_total: int = 0


def _collect_seed(
    seed: int, *, steps: int, warmup: int, world_factory: WorldFactory
) -> _Collector:
    world = world_factory(seed)
    action_rng = np.random.default_rng(int(seed) + 50_000)
    collector = _Collector()

    # Two most recent VISIBLE true positions per distractor -> observed velocity.
    visible_trail: dict[str, list[tuple[int, int]]] = {"a": [], "b": []}
    last_seen: dict[str, tuple[int, int] | None] = {"a": None, "b": None}
    observed_velocity: dict[str, tuple[int, int]] = {
        "a": (1, 0),
        "b": (0, 1),
    }
    velocity_known = {"a": False, "b": False}

    sensed, local_visibility = world.observe()
    visible = _global_visibility(local_visibility, world.grid_size)

    def _actor_cells() -> dict[str, tuple[int, int]]:
        return {
            "self": (int(world.self_position[0]), int(world.self_position[1])),
            "a": (int(world.distractor_a[0]), int(world.distractor_a[1])),
            "b": (int(world.distractor_b[0]), int(world.distractor_b[1])),
        }

    cells = _actor_cells()
    counts = {c: tuple(cells.values()).count(c) for c in cells.values()}
    previous_sensor_visible = {
        name: bool(visible[cells[name][1], cells[name][0]])
        for name in ("a", "b")
    }
    for name in ("a", "b"):
        if previous_sensor_visible[name]:
            visible_trail[name].append(cells[name])
            last_seen[name] = cells[name]
    hidden_steps = {"a": 0, "b": 0}

    for step in range(1, steps + 1):
        action = int(action_rng.integers(0, 5))
        sensed, local_visibility = world.step(action)
        visible = _global_visibility(local_visibility, world.grid_size)
        cells = _actor_cells()
        counts = {c: tuple(cells.values()).count(c) for c in cells.values()}
        current_sensor_visible = {
            name: bool(visible[cells[name][1], cells[name][0]])
            for name in ("a", "b")
        }

        # hidden_steps bookkeeping, identical to the readout's rule.
        for name in ("a", "b"):
            was_visible = previous_sensor_visible[name]
            if was_visible and not current_sensor_visible[name]:
                hidden_steps[name] = 1
            elif not current_sensor_visible[name]:
                hidden_steps[name] += 1
            else:
                hidden_steps[name] = 0
            if current_sensor_visible[name]:
                last_seen[name] = cells[name]
                if not was_visible:
                    visible_trail[name] = [cells[name]]
                    velocity_known[name] = False
                else:
                    trail = visible_trail[name]
                    prior = trail[-1]
                    trail.append(cells[name])
                    if len(trail) > 2:
                        trail.pop(0)
                    dx = cells[name][0] - prior[0]
                    dy = cells[name][1] - prior[1]
                    if abs(dx) + abs(dy) == 1:
                        observed_velocity[name] = (dx, dy)
                        velocity_known[name] = True
                    elif velocity_known[name]:
                        expected, new_velocity = _bounce_advance(
                            prior, observed_velocity[name], world.static
                        )
                        if expected == cells[name]:
                            observed_velocity[name] = new_velocity
                        else:
                            velocity_known[name] = False
                    else:
                        velocity_known[name] = False
            previous_sensor_visible[name] = current_sensor_visible[name]

        if step < warmup:
            continue

        hidden_names = [n for n in ("a", "b") if hidden_steps[n] >= 2]
        if not hidden_names:
            continue
        selected = hidden_names[(int(seed) + step) % len(hidden_names)]
        positive = cells[selected]
        truth = world.truth()
        negative_candidates = [
            (x, y)
            for y in range(ARENA_LOW, ARENA_HIGH + 1)
            for x in range(ARENA_LOW, ARENA_HIGH + 1)
            if not visible[y, x]
            and not bool(truth[y, x])
            and (x, y) not in world.static
        ]
        if not negative_candidates:
            continue
        negative = min(
            negative_candidates,
            key=lambda cell: (
                abs(cell[0] - positive[0]) + abs(cell[1] - positive[1]),
                (cell[0] * 17 + cell[1] * 31 + int(seed) + step) % 97,
            ),
        )

        seen = last_seen[selected]
        if seen is None or not velocity_known[selected]:
            continue
        vel = observed_velocity[selected]
        predicted_cell = _forward_extrapolate(
            seen, vel, world.static, hidden_steps[selected]
        )
        collector.reconstruction_total += 1
        collector.reconstruction_hits += int(predicted_cell == positive)

        seen_arr = np.asarray(
            [
                (seen[0] - ARENA_LOW) / ARENA_SIDE,
                (seen[1] - ARENA_LOW) / ARENA_SIDE,
                vel[0],
                vel[1],
                hidden_steps[selected] / 10.0,
            ],
            dtype=np.float32,
        )
        sensed_grid = sensed.astype(np.float32).ravel()
        for cell, label in ((positive, 1.0), (negative, 0.0)):
            one_hot = _query_one_hot(cell)
            geometry_features = np.concatenate((one_hot, seen_arr))
            raw_features = np.concatenate((sensed_grid, one_hot))
            collector.samples.append(
                _Sample(
                    seed=seed,
                    step=step,
                    query=cell,
                    label=label,
                    predicted_cell=predicted_cell,
                    geometry_features=geometry_features,
                    raw_features=raw_features,
                )
            )
    return collector


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = labels.astype(bool)
    predictions = predictions.astype(bool)
    accuracies = []
    for value in (True, False):
        mask = labels == value
        if not mask.any():
            continue
        accuracies.append(float((predictions[mask] == value).mean()))
    return float(np.mean(accuracies)) if accuracies else float("nan")


def _ridge_fit(features: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    design = np.concatenate(
        (features, np.ones((features.shape[0], 1), dtype=np.float32)), axis=1
    )
    gram = design.T @ design
    gram += alpha * np.eye(gram.shape[0], dtype=gram.dtype)
    return np.linalg.solve(gram, design.T @ targets)


def _ridge_predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate(
        (features, np.ones((features.shape[0], 1), dtype=np.float32)), axis=1
    )
    return design @ weights


def _probe_balanced_accuracy(
    train: list[_Sample],
    evaluation: list[_Sample],
    attribute: str,
    *,
    alpha: float = 1.0,
) -> float:
    train_x = np.stack([getattr(s, attribute) for s in train]).astype(np.float32)
    train_y = np.asarray([s.label * 2.0 - 1.0 for s in train], dtype=np.float32)
    eval_x = np.stack([getattr(s, attribute) for s in evaluation]).astype(np.float32)
    eval_y = np.asarray([s.label for s in evaluation], dtype=np.float32)
    weights = _ridge_fit(train_x, train_y, alpha)
    scores = _ridge_predict(eval_x, weights)
    return _balanced_accuracy(eval_y, scores > 0.0)


def run_diagnostic(
    train_seeds: list[int],
    evaluation_seeds: list[int],
    *,
    steps: int,
    warmup: int,
    world: str = "fixed",
) -> dict[str, object]:
    world_factory = WORLD_FACTORIES[world]
    train_samples: list[_Sample] = []
    eval_samples: list[_Sample] = []
    recon_hits = recon_total = 0
    for seed in train_seeds:
        collector = _collect_seed(
            seed, steps=steps, warmup=warmup, world_factory=world_factory
        )
        train_samples.extend(collector.samples)
        recon_hits += collector.reconstruction_hits
        recon_total += collector.reconstruction_total
    for seed in evaluation_seeds:
        collector = _collect_seed(
            seed, steps=steps, warmup=warmup, world_factory=world_factory
        )
        eval_samples.extend(collector.samples)
        recon_hits += collector.reconstruction_hits
        recon_total += collector.reconstruction_total

    if not train_samples:
        raise ValueError(
            "training configuration produced no valid hidden-object samples"
        )
    if not eval_samples:
        raise ValueError(
            "evaluation configuration produced no valid hidden-object samples"
        )
    eval_labels = np.asarray([s.label for s in eval_samples], dtype=np.float32)
    analytic_predictions = np.asarray(
        [float(s.query == s.predicted_cell) for s in eval_samples],
        dtype=np.float32,
    )
    return {
        "world": world,
        "train_seeds": train_seeds,
        "evaluation_seeds": evaluation_seeds,
        "train_sample_count": len(train_samples),
        "evaluation_sample_count": len(eval_samples),
        "hidden_position_reconstruction_accuracy": (
            recon_hits / recon_total if recon_total else float("nan")
        ),
        "analytic_extrapolation_permanence_balanced_accuracy": _balanced_accuracy(
            eval_labels, analytic_predictions
        ),
        "geometry_probe_permanence_balanced_accuracy": _probe_balanced_accuracy(
            train_samples, eval_samples, "geometry_features"
        ),
        "raw_sensor_probe_permanence_balanced_accuracy": _probe_balanced_accuracy(
            train_samples, eval_samples, "raw_features"
        ),
        "v8_holdout_reference": {
            "formal_entity_graph_permanence": 0.8021,
            "raw_sensor_permanence": 0.8750,
            "no_action_permanence": 0.8854,
            "assume_all_visible_permanence": 0.8542,
            "note": "measured on the consumed V8 holdout seeds 33600-33603; "
            "shown only for scale.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument(
        "--world", choices=sorted(WORLD_FACTORIES), default="fixed"
    )
    parser.add_argument("--train-seeds", type=int, default=40)
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--train-base", type=int, default=60000)
    parser.add_argument("--eval-base", type=int, default=60100)
    args = parser.parse_args()

    train_seeds = [args.train_base + i for i in range(args.train_seeds)]
    evaluation_seeds = [args.eval_base + i for i in range(args.eval_seeds)]
    report = run_diagnostic(
        train_seeds,
        evaluation_seeds,
        steps=args.steps,
        warmup=args.warmup,
        world=args.world,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
