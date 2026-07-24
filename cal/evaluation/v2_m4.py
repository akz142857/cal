"""V2-M4: visual occupancy, self-motion fusion, and object permanence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from cal.infra.provenance import capture_provenance
from cal.model.occupancy import MOTION_DELTAS, VIEW_RADIUS, OccupancyMemory
from cal.evaluation.v2_artifacts import require_authorization


class _OccupancyWorld:
    def __init__(self, seed: int, grid_size: int = 25) -> None:
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        self.camera = np.asarray((grid_size // 2, grid_size // 2), dtype=np.int64)
        jitter = int(self.rng.integers(-1, 2))
        self.static = {
            (4 + jitter, 5),
            (7, 18 - jitter),
            (18 - jitter, 5),
            (20, 19 + jitter),
            *((grid_size // 2, y) for y in range(8, 17)),
        }
        self.moving = np.asarray(
            (5 + int(self.rng.integers(0, 3)), 12), dtype=np.int64
        )
        self.velocity = np.asarray(
            (1 if seed % 2 == 0 else -1, 0), dtype=np.int64
        )

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray]:
        self.camera = np.clip(
            self.camera + MOTION_DELTAS[action],
            VIEW_RADIUS,
            self.grid_size - VIEW_RADIUS - 1,
        )
        candidate = self.moving + self.velocity
        if candidate[0] <= 2 or candidate[0] >= self.grid_size - 3:
            self.velocity[0] *= -1
            candidate = self.moving + self.velocity
        self.moving = candidate
        return self.observe()

    def truth(self) -> np.ndarray:
        result = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for x, y in self.static:
            result[y, x] = 1
        result[self.moving[1], self.moving[0]] = 1
        return result

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        size = 2 * VIEW_RADIUS + 1
        occupancy = np.zeros((size, size), dtype=np.uint8)
        visibility = np.ones((size, size), dtype=np.uint8)
        x0 = int(self.camera[0] - VIEW_RADIUS)
        y0 = int(self.camera[1] - VIEW_RADIUS)
        truth = self.truth()
        for local_y in range(size):
            for local_x in range(size):
                x, y = x0 + local_x, y0 + local_y
                # The central vertical structure occludes its far-side band.
                opposite_sides = (self.camera[0] - 12) * (x - 12) < 0
                behind_screen = opposite_sides and 8 <= y <= 16
                if behind_screen:
                    visibility[local_y, local_x] = 0
                elif truth[y, x]:
                    occupancy[local_y, local_x] = 1
        return occupancy, visibility


def _episode(seed: int, *, steps: int, active: bool) -> dict[str, Any]:
    world = _OccupancyWorld(seed)
    memory = OccupancyMemory(active=active, seed=seed + 20_000)
    initial_occupancy, initial_visibility = world.observe()
    memory.update(initial_occupancy, initial_visibility, 0)
    hidden_probabilities = []
    hidden_targets = []
    moving_hidden_probabilities = []
    confidence_step = None
    for step in range(1, steps + 1):
        action = memory.choose_action()
        occupancy, visibility = world.step(action)
        memory.update(occupancy, visibility, action)
        probability = memory.probability()
        truth = world.truth()
        x0 = int(world.camera[0] - VIEW_RADIUS)
        y0 = int(world.camera[1] - VIEW_RADIUS)
        global_visibility = np.zeros_like(truth, dtype=bool)
        global_visibility[
            y0 : y0 + visibility.shape[0],
            x0 : x0 + visibility.shape[1],
        ] = visibility.astype(bool)
        if step >= 40:
            hidden = ~global_visibility
            hidden_probabilities.extend(probability[hidden].tolist())
            hidden_targets.extend(truth[hidden].tolist())
            if not global_visibility[world.moving[1], world.moving[0]]:
                moving_hidden_probabilities.append(
                    float(probability[world.moving[1], world.moving[0]])
                )
        occupied_probabilities = probability[truth.astype(bool)]
        if (
            confidence_step is None
            and len(occupied_probabilities)
            and float(np.mean(occupied_probabilities >= 0.75)) >= 0.80
        ):
            confidence_step = step
    probability = memory.probability()
    truth = world.truth().astype(bool)
    predicted = probability >= 0.65
    intersection = int(np.logical_and(predicted, truth).sum())
    union = int(np.logical_or(predicted, truth).sum())
    hidden_probability = np.asarray(hidden_probabilities)
    hidden_target = np.asarray(hidden_targets, dtype=bool)
    hidden_prediction = hidden_probability >= 0.65
    hidden_tp = int(np.logical_and(hidden_prediction, hidden_target).sum())
    hidden_fn = int(np.logical_and(~hidden_prediction, hidden_target).sum())
    return {
        "seed": seed,
        "active": active,
        "occupancy_iou": intersection / max(1, union),
        "hidden_occupancy_recall": hidden_tp / max(1, hidden_tp + hidden_fn),
        "hidden_brier": float(
            np.mean((hidden_probability - hidden_target.astype(float)) ** 2)
        ),
        "moving_hidden_probability": (
            mean(moving_hidden_probabilities)
            if moving_hidden_probabilities
            else 0.0
        ),
        "confidence_step": confidence_step if confidence_step is not None else steps,
        "mean_entropy": memory.mean_entropy(),
    }


def run_v2_m4(
    *,
    output_path: str | Path = "results/V2-M4-summary.json",
    prerequisite_path: str | Path = (
        "results/V2-M3-body-graph-holdout-summary.json"
    ),
    seeds: Sequence[int] = tuple(range(500, 516)),
    steps: int = 220,
    enforce_prerequisite: bool = True,
) -> dict[str, Any]:
    prerequisite_passed = True
    try:
        require_authorization(
            prerequisite_path,
            expected_name="V2-M3",
            expected_decision="authorize_v2_m4",
        )
    except RuntimeError:
        prerequisite_passed = False
        if enforce_prerequisite:
            raise
    started = perf_counter()
    active = [_episode(seed, steps=steps, active=True) for seed in seeds]
    random = [_episode(seed, steps=steps, active=False) for seed in seeds]
    aggregate = {
        "occupancy_iou": mean(item["occupancy_iou"] for item in active),
        "hidden_occupancy_recall": mean(
            item["hidden_occupancy_recall"] for item in active
        ),
        "hidden_brier": mean(item["hidden_brier"] for item in active),
        "moving_hidden_probability": mean(
            item["moving_hidden_probability"] for item in active
        ),
        "active_confidence_step": mean(item["confidence_step"] for item in active),
        "random_confidence_step": mean(item["confidence_step"] for item in random),
    }
    aggregate["active_step_reduction"] = (
        1.0
        - aggregate["active_confidence_step"]
        / max(aggregate["random_confidence_step"], 1e-12)
    )
    memory = OccupancyMemory()
    resources = {
        "learnable_parameter_count": memory.learnable_parameter_count,
        "active_state_bytes": memory.active_state_bytes,
        "estimated_mac_per_step": memory.estimated_mac_per_step,
        "steps_per_seed": steps,
        "maximum_replays_per_experience": 0,
        "cpu_wall_seconds": perf_counter() - started,
    }
    gates = {
        "m3_prerequisite_passed": prerequisite_passed,
        "occupancy_iou_ge_0_70": aggregate["occupancy_iou"] >= 0.70,
        "hidden_recall_ge_0_65": aggregate["hidden_occupancy_recall"] >= 0.65,
        "hidden_brier_le_0_12": aggregate["hidden_brier"] <= 0.12,
        "moving_object_probability_ge_0_55": (
            aggregate["moving_hidden_probability"] >= 0.55
        ),
        "active_confirmation_steps_reduced_ge_30pct": (
            aggregate["active_step_reduction"] >= 0.30
        ),
        "resources_pass": (
            memory.learnable_parameter_count <= 100_000
            and memory.active_state_bytes <= 64 * 1024
            and memory.estimated_mac_per_step <= 5_000_000
            and steps <= 100_000
            and resources["cpu_wall_seconds"] <= 7_200
        ),
        "labels_absent_from_learner": True,
        "visual_only_no_privileged_visibility": False,
    }
    summary = {
        "result_schema_version": 1,
        "experiment": "V2-M4",
        "formal_agent_input": "local_visual_occupancy_visibility_and_self_motion",
        "evaluation_labels_used_for_learning": False,
        "diagnostic_only": True,
        "privileged_input": "simulator_generated_visibility_mask",
        "active_episodes": active,
        "random_episodes": random,
        "aggregate": aggregate,
        "resources": resources,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "authorize_reconnection_to_original_m2"
            if all(gates.values())
            else "stop_before_reconnection"
        ),
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/V2-M4-summary.json"))
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument(
        "--prerequisite",
        type=Path,
        default=Path("results/V2-M3-body-graph-holdout-summary.json"),
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="run the diagnostic even when M3 did not authorize M4",
    )
    arguments = parser.parse_args(argv)
    result = run_v2_m4(
        output_path=arguments.output,
        prerequisite_path=arguments.prerequisite,
        steps=arguments.steps,
        enforce_prerequisite=not arguments.exploratory,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
