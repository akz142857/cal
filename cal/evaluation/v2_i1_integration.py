"""V2-I1: first system-level closed loop in a single shared world."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from cal.evaluation.v2_artifacts import (
    build_resources,
    constructor_apis_reject_ground_truth,
    load_frozen_protocol,
    require_authorization,
    resources_pass,
)
from cal.infra.provenance import capture_provenance
from cal.model.integrated_agent import (
    ACTION_DELTAS,
    IntegratedSelfWorldAgent,
)
from cal.model.occupancy import VIEW_RADIUS, sense_via_line_of_sight


DEFAULT_PROTOCOL = Path("experiments/V2_I1_INTEGRATION_PROTOCOL.json")
ARENA_LOW, ARENA_HIGH = 7, 17
CAMERA = (12, 12)
WARMUP = 12
PERMANENCE_WARMUP = 40


class _IntegratedWorld:
    """Fixed-camera arena: one controlled point, two distractors, occluders."""

    def __init__(self, seed: int, grid_size: int = 25) -> None:
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        jitter = int(self.rng.integers(-1, 2))
        # The screen has a doorway on row 12 so distractor A can travel
        # through it into the shadow band behind the screen.
        self.static = {
            *((10, y) for y in range(9, 16) if y != 12),
            (15, 9 + jitter),
            (14, 16),
        }
        self.self_position = np.asarray(
            (17 - int(self.rng.integers(0, 2)), 13), dtype=np.int64
        )
        self.distractor_a = np.asarray(
            (16 - int(self.rng.integers(0, 2)), 12), dtype=np.int64
        )
        self.velocity_a = np.asarray(
            (1 if seed % 2 == 0 else -1, 0), dtype=np.int64
        )
        self.distractor_b = np.asarray(
            (16, 8 + int(self.rng.integers(0, 3))), dtype=np.int64
        )
        self.velocity_b = np.asarray((0, 1), dtype=np.int64)

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray]:
        delta = ACTION_DELTAS[int(action)]
        candidate = np.clip(
            self.self_position + delta, ARENA_LOW, ARENA_HIGH
        )
        if tuple(candidate) not in self.static:
            self.self_position = candidate
        for point, velocity in (
            (self.distractor_a, self.velocity_a),
            (self.distractor_b, self.velocity_b),
        ):
            axis = 0 if velocity[0] else 1
            nxt = point[axis] + velocity[axis]
            if nxt < ARENA_LOW or nxt > ARENA_HIGH or (
                axis == 0
                and (int(nxt), int(point[1])) in self.static
            ) or (
                axis == 1
                and (int(point[0]), int(nxt)) in self.static
            ):
                velocity[axis] *= -1
                nxt = point[axis] + velocity[axis]
            point[axis] = nxt
        return self.observe()

    def truth(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for x, y in self.static:
            grid[y, x] = 1
        for point in (
            self.self_position, self.distractor_a, self.distractor_b
        ):
            grid[point[1], point[0]] = 1
        return grid

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        return sense_via_line_of_sight(CAMERA, self.truth())


def _global_visibility(
    visibility: np.ndarray, grid_size: int
) -> np.ndarray:
    result = np.zeros((grid_size, grid_size), dtype=bool)
    x0, y0 = CAMERA[0] - VIEW_RADIUS, CAMERA[1] - VIEW_RADIUS
    result[
        y0 : y0 + visibility.shape[0], x0 : x0 + visibility.shape[1]
    ] = visibility.astype(bool)
    return result


def _episode(
    seed: int,
    *,
    steps: int,
    infer_occlusion: bool = True,
    use_action: bool = True,
    shuffle_lag: int = 0,
) -> dict[str, Any]:
    world = _IntegratedWorld(seed)
    agent = IntegratedSelfWorldAgent(
        infer_occlusion=infer_occlusion,
        use_action=use_action,
        seed=seed + 40_000,
    )
    action_rng = np.random.default_rng(seed + 50_000)
    sensed, _ = world.observe()
    agent.update(sensed, 0)
    executed: list[int] = [0]
    true_positive = false_positive = false_negative = 0
    hidden_probabilities: list[float] = []
    hidden_targets: list[int] = []
    identity_map: dict[str, dict[int, int]] = {"a": {}, "b": {}}
    for step in range(1, steps + 1):
        action = int(action_rng.integers(0, 5))
        sensed, visibility = world.step(action)
        executed.append(action)
        supplied = (
            executed[max(0, len(executed) - 1 - shuffle_lag)]
            if shuffle_lag
            else action
        )
        agent.update(sensed, supplied)
        visible = _global_visibility(visibility, world.grid_size)
        truth = world.truth()
        positions = agent.track_positions()
        if step >= WARMUP:
            self_identity = agent.self_track_identity()
            true_self = (
                int(world.self_position[0]), int(world.self_position[1])
            )
            if visible[true_self[1], true_self[0]]:
                predicted = (
                    positions.get(self_identity)
                    if self_identity is not None
                    else None
                )
                if predicted == true_self:
                    true_positive += 1
                else:
                    false_negative += 1
                    if predicted is not None:
                        false_positive += 1
        for name, point in (
            ("a", world.distractor_a), ("b", world.distractor_b)
        ):
            cell = (int(point[0]), int(point[1]))
            if visible[cell[1], cell[0]]:
                for identity, position in positions.items():
                    if position == cell:
                        identity_map[name][identity] = (
                            identity_map[name].get(identity, 0) + 1
                        )
                        break
        if step >= PERMANENCE_WARMUP:
            probability = agent.probability()
            for point in (world.distractor_a, world.distractor_b):
                cell = (int(point[0]), int(point[1]))
                if not visible[cell[1], cell[0]]:
                    hidden_probabilities.append(
                        float(probability[cell[1], cell[0]])
                    )
                    hidden_targets.append(1)
    f1_denominator = 2 * true_positive + false_positive + false_negative
    consistency = []
    for counts in identity_map.values():
        total = sum(counts.values())
        if total:
            consistency.append(max(counts.values()) / total)
    return {
        "seed": seed,
        "self_f1": (
            2 * true_positive / f1_denominator if f1_denominator else 0.0
        ),
        "identity_consistency": mean(consistency) if consistency else 0.0,
        "distractor_hidden_probability": (
            mean(hidden_probabilities) if hidden_probabilities else 0.0
        ),
        "hidden_sample_count": len(hidden_probabilities),
    }


def _load_frozen_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    return load_frozen_protocol(
        path,
        frozen_statuses="frozen_before_integration_probe_implementation",
    )


def run_v2_i1(
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
            "one-shot integration holdout already exists; reruns forbidden"
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
    started = perf_counter()
    formal = [_episode(seed, steps=steps) for seed in seeds]
    no_action = [
        _episode(seed, steps=steps, use_action=False) for seed in seeds
    ]
    shuffled = [
        _episode(seed, steps=steps, shuffle_lag=5) for seed in seeds
    ]
    all_visible = [
        _episode(seed, steps=steps, infer_occlusion=False)
        for seed in seeds
    ]
    aggregate = {
        "self_f1": mean(e["self_f1"] for e in formal),
        "identity_consistency": mean(
            e["identity_consistency"] for e in formal
        ),
        "distractor_hidden_probability": mean(
            e["distractor_hidden_probability"] for e in formal
        ),
        "no_action_self_f1": mean(e["self_f1"] for e in no_action),
        "shuffled_self_f1": mean(e["self_f1"] for e in shuffled),
        "assume_all_visible_hidden_probability": mean(
            e["distractor_hidden_probability"] for e in all_visible
        ),
        "paired_formal_beats_visible_control": mean(
            1.0 if f["distractor_hidden_probability"]
            > c["distractor_hidden_probability"]
            else 0.0
            for f, c in zip(formal, all_visible)
        ),
    }
    agent = IntegratedSelfWorldAgent()
    resources = build_resources(agent, steps=steps, started=started)
    update_parameters = set(
        inspect.signature(IntegratedSelfWorldAgent.update).parameters
    )
    gates = {
        "self_identification_pass": (
            aggregate["self_f1"]
            >= fixed["self_identification_f1_minimum"]
        ),
        "identity_consistency_pass": (
            aggregate["identity_consistency"]
            >= fixed["identity_consistency_minimum"]
        ),
        "distractor_permanence_pass": (
            aggregate["distractor_hidden_probability"]
            >= fixed["distractor_hidden_probability_minimum"]
        ),
        "no_action_control_fails": (
            aggregate["self_f1"] - aggregate["no_action_self_f1"]
            >= fixed["no_action_self_f1_drop_minimum"]
        ),
        "time_shuffle_control_fails": (
            aggregate["self_f1"] - aggregate["shuffled_self_f1"]
            >= fixed["time_shuffled_self_f1_drop_minimum"]
        ),
        "assume_all_visible_control_fails": (
            aggregate["assume_all_visible_hidden_probability"]
            <= fixed["assume_all_visible_hidden_probability_maximum"]
        ),
        "paired_separation_pass": (
            aggregate["paired_formal_beats_visible_control"]
            >= fixed[
                "paired_formal_beats_visible_control_fraction_minimum"
            ]
        ),
        "single_agent_single_stream": (
            update_parameters == {"self", "sensed_occupancy", "action"}
        ),
        "labels_absent_from_learner": constructor_apis_reject_ground_truth(
            IntegratedSelfWorldAgent.update
        ),
        "resources_pass": resources_pass(resources, limits),
    }
    passed = all(gates.values())
    summary = {
        "result_schema_version": 1,
        "experiment": "V2-I1-integration",
        "review_split": split,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "holdout_run_count": 1 if split == "holdout" else 0,
        "formal_agent_input": "sensed_visible_occupancy_and_action_copy_only",
        "evaluation_labels_used_for_learning": False,
        "privileged_input": None,
        "formal_episodes": formal,
        "no_action_episodes": no_action,
        "shuffled_episodes": shuffled,
        "assume_all_visible_episodes": all_visible,
        "aggregate": aggregate,
        "resources": resources,
        "gates": gates,
        "passed": passed,
        "decision": (
            (
                "authorize_one_shot_integration_holdout"
                if split == "development"
                else "first_system_level_loop_verified"
            )
            if passed
            else "stop_integration_probe"
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
        "--split", choices=("development", "holdout"), required=True
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    result = run_v2_i1(
        split=arguments.split,
        protocol_path=arguments.protocol,
        output_path=arguments.output,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
