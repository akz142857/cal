"""V2-M4 without the privileged visibility mask: vision-only occlusion."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from cal.evaluation.v2_artifacts import require_authorization
from cal.evaluation.v2_m4 import _OccupancyWorld
from cal.infra.provenance import capture_provenance
from cal.model.occupancy import (
    MOTION_DELTAS,
    VIEW_RADIUS,
    UnprivilegedOccupancyMemory,
    bresenham_intermediate_cells,
)


DEFAULT_PROTOCOL = Path("experiments/V2_M4_UNPRIVILEGED_PROTOCOL_V3.json")

FROZEN_STATUSES = {
    "frozen_before_unprivileged_visibility_implementation",
    "frozen_after_v1_control_criterion_failure_before_any_holdout_run",
    "frozen_after_v2_control_insensitivity_diagnosis_before_any_holdout_run",
}


class _LineOfSightWorld(_OccupancyWorld):
    """Diagnostic M4 world with physically consistent ray occlusion.

    observe() returns the sensed patch (visibly occupied cells only) for the
    agent, and the true visibility separately for evaluation-side scoring.
    """

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        size = 2 * VIEW_RADIUS + 1
        sensed = np.zeros((size, size), dtype=np.uint8)
        true_visibility = np.ones((size, size), dtype=np.uint8)
        x0 = int(self.camera[0] - VIEW_RADIUS)
        y0 = int(self.camera[1] - VIEW_RADIUS)
        truth = self.truth()
        camera = (int(self.camera[0]), int(self.camera[1]))
        for local_y in range(size):
            for local_x in range(size):
                x, y = x0 + local_x, y0 + local_y
                if (x, y) != camera:
                    for cx, cy in bresenham_intermediate_cells(
                        camera, (x, y)
                    ):
                        if truth[cy, cx]:
                            true_visibility[local_y, local_x] = 0
                            break
                if true_visibility[local_y, local_x] and truth[y, x]:
                    sensed[local_y, local_x] = 1
        return sensed, true_visibility


class _StressedLineOfSightWorld(_LineOfSightWorld):
    """Line-of-sight world with occlusion stress the tracker cannot bridge.

    The moving object pauses for PAUSE_STEPS each time it reaches three
    cells to either side of the central screen column, and two seed-jittered
    static blocks sit inside the screen's habitual shadow bands.
    """

    PAUSE_STEPS = 10
    PAUSE_OFFSET = 3
    WALL_LOW = 2
    WALL_HIGH_MARGIN = 3

    def __init__(self, seed: int, grid_size: int = 25) -> None:
        super().__init__(seed, grid_size)
        jitter = int(self.rng.integers(-1, 2))
        screen_x = grid_size // 2
        self.static |= {
            (screen_x + 3, 10 + jitter),
            (screen_x - 3, 14 - jitter),
        }
        self._pause_remaining = 0
        self._pause_columns = {
            screen_x - self.PAUSE_OFFSET,
            screen_x + self.PAUSE_OFFSET,
        }
        self._paused_at: set[tuple[int, int]] = set()

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray]:
        self.camera = np.clip(
            self.camera + MOTION_DELTAS[action],
            VIEW_RADIUS,
            self.grid_size - VIEW_RADIUS - 1,
        )
        if self._pause_remaining > 0:
            self._pause_remaining -= 1
            return self.observe()
        candidate = self.moving + self.velocity
        if (
            candidate[0] <= self.WALL_LOW
            or candidate[0] >= self.grid_size - self.WALL_HIGH_MARGIN
        ):
            self.velocity[0] *= -1
            candidate = self.moving + self.velocity
            self._paused_at.clear()
        self.moving = candidate
        key = (int(self.moving[0]), int(self.velocity[0]))
        if int(self.moving[0]) in self._pause_columns and key not in self._paused_at:
            self._pause_remaining = self.PAUSE_STEPS
            self._paused_at.add(key)
        return self.observe()


class _DeepShadowLineOfSightWorld(_StressedLineOfSightWorld):
    """V3 stress: pauses in the deepest shadow, travel kept near the window.

    Pause columns sit one cell to either side of the screen, and tighter
    bounce walls keep the object inside the camera's habitual reach, so a
    much larger share of hidden samples is occluded inside the window.
    """

    PAUSE_OFFSET = 1
    WALL_LOW = 5
    WALL_HIGH_MARGIN = 6


_WORLDS = {
    1: _LineOfSightWorld,
    2: _StressedLineOfSightWorld,
    3: _DeepShadowLineOfSightWorld,
}


def _episode(
    seed: int,
    *,
    steps: int,
    active: bool,
    infer_occlusion: bool,
    environment_version: int = 2,
) -> dict[str, Any]:
    world = _WORLDS[environment_version](seed)
    memory = UnprivilegedOccupancyMemory(
        active=active,
        infer_occlusion=infer_occlusion,
        seed=seed + 20_000,
    )
    sensed, _ = world.observe()
    memory.update(sensed, 0)
    hidden_probabilities: list[float] = []
    hidden_targets: list[int] = []
    moving_hidden_probabilities: list[float] = []
    confidence_step = None
    for step in range(1, steps + 1):
        action = memory.choose_action()
        sensed, true_visibility = world.step(action)
        memory.update(sensed, action)
        probability = memory.probability()
        truth = world.truth()
        x0 = int(world.camera[0] - VIEW_RADIUS)
        y0 = int(world.camera[1] - VIEW_RADIUS)
        global_visibility = np.zeros_like(truth, dtype=bool)
        global_visibility[
            y0 : y0 + true_visibility.shape[0],
            x0 : x0 + true_visibility.shape[1],
        ] = true_visibility.astype(bool)
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
        "infer_occlusion": infer_occlusion,
        "occupancy_iou": intersection / max(1, union),
        "hidden_occupancy_recall": hidden_tp / max(1, hidden_tp + hidden_fn),
        "hidden_brier": float(
            np.mean((hidden_probability - hidden_target.astype(float)) ** 2)
        )
        if len(hidden_probability)
        else 1.0,
        "moving_hidden_probability": (
            mean(moving_hidden_probabilities)
            if moving_hidden_probabilities
            else 0.0
        ),
        "confidence_step": (
            confidence_step if confidence_step is not None else steps
        ),
        "mean_entropy": memory.mean_entropy(),
    }


def _load_frozen_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = source.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if digest != expected:
        raise RuntimeError(
            "unprivileged M4 protocol hash does not match its frozen lock"
        )
    protocol = json.loads(source.read_text(encoding="utf-8"))
    if protocol.get("status") not in FROZEN_STATUSES:
        raise RuntimeError("unprivileged M4 protocol is not frozen")
    return protocol, digest


def run_v2_m4_unprivileged(
    *,
    split: str,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    protocol, protocol_digest = _load_frozen_protocol(protocol_path)
    if split not in {"development", "holdout"}:
        raise ValueError("split must be development or holdout")
    expected_output = Path(protocol[split]["result_path"])
    destination = Path(output_path) if output_path else expected_output
    if destination != expected_output:
        raise RuntimeError("output path does not match the frozen protocol")
    if split == "holdout" and destination.exists():
        raise RuntimeError(
            "one-shot unprivileged M4 holdout already exists; reruns forbidden"
        )
    prerequisite = protocol["prerequisite"]
    require_authorization(
        prerequisite["artifact"],
        expected_name=prerequisite["expected_name"],
        expected_decision=prerequisite["expected_decision"],
    )
    if split == "holdout":
        development = json.loads(
            Path(protocol["development"]["result_path"]).read_text(
                encoding="utf-8"
            )
        )
        if development.get("passed") is not True:
            raise RuntimeError(
                "development must pass before the one-shot holdout"
            )
    seeds = tuple(int(seed) for seed in protocol[split]["seeds"])
    steps = int(protocol["fixed_runtime"]["steps_per_seed"])
    fixed = protocol["fixed_gates"]
    limits = protocol["resource_limits"]
    environment_version = int(
        protocol["environment"].get("environment_version", 1)
    )
    started = perf_counter()
    formal = [
        _episode(
            seed,
            steps=steps,
            active=True,
            infer_occlusion=True,
            environment_version=environment_version,
        )
        for seed in seeds
    ]
    random = [
        _episode(
            seed,
            steps=steps,
            active=False,
            infer_occlusion=True,
            environment_version=environment_version,
        )
        for seed in seeds
    ]
    all_visible = [
        _episode(
            seed,
            steps=steps,
            active=True,
            infer_occlusion=False,
            environment_version=environment_version,
        )
        for seed in seeds
    ]
    aggregate = {
        "occupancy_iou": mean(item["occupancy_iou"] for item in formal),
        "hidden_occupancy_recall": mean(
            item["hidden_occupancy_recall"] for item in formal
        ),
        "hidden_brier": mean(item["hidden_brier"] for item in formal),
        "moving_hidden_probability": mean(
            item["moving_hidden_probability"] for item in formal
        ),
        "active_confidence_step": mean(
            item["confidence_step"] for item in formal
        ),
        "random_confidence_step": mean(
            item["confidence_step"] for item in random
        ),
        "assume_all_visible_hidden_recall": mean(
            item["hidden_occupancy_recall"] for item in all_visible
        ),
        "assume_all_visible_moving_hidden_probability": mean(
            item["moving_hidden_probability"] for item in all_visible
        ),
    }
    aggregate["active_step_reduction"] = (
        1.0
        - aggregate["active_confidence_step"]
        / max(aggregate["random_confidence_step"], 1e-12)
    )
    memory = UnprivilegedOccupancyMemory()
    resources = {
        "learnable_parameter_count": memory.learnable_parameter_count,
        "active_state_bytes": memory.active_state_bytes,
        "estimated_mac_per_step": memory.estimated_mac_per_step,
        "steps_per_seed": steps,
        "maximum_replays_per_experience": 0,
        "cpu_wall_seconds": perf_counter() - started,
    }
    update_parameters = set(
        inspect.signature(UnprivilegedOccupancyMemory.update).parameters
    )
    gates = {
        "occupancy_iou_pass": (
            aggregate["occupancy_iou"] >= fixed["occupancy_iou_minimum"]
        ),
        "hidden_recall_pass": (
            aggregate["hidden_occupancy_recall"]
            >= fixed["hidden_occupancy_recall_minimum"]
        ),
        "hidden_brier_pass": (
            aggregate["hidden_brier"] <= fixed["hidden_brier_maximum"]
        ),
        "moving_hidden_probability_pass": (
            aggregate["moving_hidden_probability"]
            >= fixed["moving_hidden_probability_minimum"]
        ),
        "active_step_reduction_pass": (
            aggregate["active_step_reduction"]
            >= fixed["active_step_reduction_minimum"]
        ),
        "assume_all_visible_control_fails": (
            aggregate["assume_all_visible_moving_hidden_probability"]
            <= fixed["assume_all_visible_moving_hidden_probability_maximum"]
            if "assume_all_visible_moving_hidden_probability_maximum" in fixed
            else aggregate["assume_all_visible_hidden_recall"]
            <= fixed["assume_all_visible_hidden_recall_maximum"]
        ),
        "visual_only_no_privileged_visibility": not (
            {"visibility", "local_visibility", "mask"} & update_parameters
        ),
        "labels_absent_from_learner": not any(
            token in name.lower()
            for name in update_parameters
            for token in ("label", "truth", "mask")
        ),
        "resources_pass": (
            resources["learnable_parameter_count"]
            <= limits["learnable_parameters"]
            and resources["active_state_bytes"] <= limits["active_state_bytes"]
            and resources["estimated_mac_per_step"] <= limits["mac_per_step"]
            and steps <= limits["steps_per_seed"]
            and resources["cpu_wall_seconds"] <= limits["wall_seconds"]
        ),
    }
    passed = all(gates.values())
    summary = {
        "result_schema_version": 1,
        "experiment": "V2-M4-unprivileged",
        "review_split": split,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "holdout_run_count": 1 if split == "holdout" else 0,
        "formal_agent_input": "sensed_visible_occupancy_and_action_copy_only",
        "evaluation_labels_used_for_learning": False,
        "privileged_input": None,
        "formal_episodes": formal,
        "random_episodes": random,
        "assume_all_visible_episodes": all_visible,
        "aggregate": aggregate,
        "resources": resources,
        "gates": gates,
        "passed": passed,
        "decision": (
            (
                "authorize_one_shot_unprivileged_holdout"
                if split == "development"
                else "authorize_reconnection_design_review"
            )
            if passed
            else "stop_before_reconnection"
        ),
        "provenance": capture_provenance(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("development", "holdout"),
        required=True,
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    result = run_v2_m4_unprivileged(
        split=arguments.split,
        protocol_path=arguments.protocol,
        output_path=arguments.output,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
