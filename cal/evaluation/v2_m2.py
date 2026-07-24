"""V2-M2: form a controlled rigid entity from anonymous visual nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from math import cos, sin
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from cal.infra.provenance import capture_provenance
from cal.model.entity_graph import OnlineEntityGraph
from cal.evaluation.v2_artifacts import require_authorization


_ACTIONS = np.asarray(
    ((-0.7, 0.0), (0.7, 0.0), (0.0, -0.7), (0.0, 0.7)),
    dtype=np.float64,
)
_OFFSETS = np.asarray(((-2.0, 0.0), (0.0, 0.0), (0.0, 2.0)))


@dataclass(frozen=True, slots=True)
class M2Scenario:
    family: str
    path_start: tuple[float, float]
    path_end: tuple[float, float]
    crossing_duration: int
    external_rotation_rate: float
    controlled_rotation_period: int
    occlusion_phase: int
    coordinate_noise_std: float


def scenario_from_seed(seed: int) -> M2Scenario:
    families = (
        ("horizontal_left_to_right", (4.0, 12.0), (20.0, 12.0)),
        ("horizontal_right_to_left", (20.0, 12.0), (4.0, 12.0)),
        ("diagonal_upper_left_to_lower_right", (4.0, 4.0), (20.0, 20.0)),
        ("diagonal_lower_left_to_upper_right", (4.0, 20.0), (20.0, 4.0)),
    )
    family, start, end = families[seed % len(families)]
    durations = (120, 144, 168, 192)
    rotation_rates = (-0.025, -0.0125, 0.0125, 0.025)
    periods = (27, 31, 37)
    phases = (0, 5, 11, 17)
    noise = (0.0, 0.015, 0.03)
    return M2Scenario(
        family=family,
        path_start=start,
        path_end=end,
        crossing_duration=durations[(seed // 4) % len(durations)],
        external_rotation_rate=rotation_rates[(seed // 7) % len(rotation_rates)],
        controlled_rotation_period=periods[(seed // 11) % len(periods)],
        occlusion_phase=phases[(seed // 13) % len(phases)],
        coordinate_noise_std=noise[(seed // 17) % len(noise)],
    )


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    matrix = np.asarray(((cos(angle), -sin(angle)), (sin(angle), cos(angle))))
    return points @ matrix.T


def _match_truth(
    positions: dict[int, tuple[float, float]],
    truth: np.ndarray,
    *,
    tolerance: float = 0.35,
) -> set[int]:
    matches = set()
    for point in truth:
        candidates = [
            (np.linalg.norm(np.asarray(position) - point), index)
            for index, position in positions.items()
        ]
        if candidates:
            distance, index = min(candidates)
            if distance <= tolerance:
                matches.add(index)
    return matches


def _match_truth_assignments(
    positions: dict[int, tuple[float, float]],
    truth: np.ndarray,
    *,
    tolerance: float = 0.35,
) -> dict[int, int]:
    candidates = sorted(
        (
            float(np.linalg.norm(np.asarray(position) - point)),
            truth_index,
            track_index,
        )
        for truth_index, point in enumerate(truth)
        for track_index, position in positions.items()
    )
    result: dict[int, int] = {}
    used_tracks: set[int] = set()
    for distance, truth_index, track_index in candidates:
        if distance > tolerance:
            break
        if truth_index not in result and track_index not in used_tracks:
            result[truth_index] = track_index
            used_tracks.add(track_index)
    return result


def _episode(
    seed: int,
    *,
    steps: int,
    crossing: bool,
    scenario: M2Scenario | None = None,
    association_mode: str = "probabilistic",
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    learner = OnlineEntityGraph(4, association_mode=association_mode)
    resolved = scenario or scenario_from_seed(seed)
    center = np.asarray((12.0, 12.0))
    angle = 0.0
    distractor_centers = [
        np.asarray(resolved.path_start),
        np.asarray((20.0, 20.0)),
    ]
    distractor_angles = [0.4, -0.6]

    def state(step: int) -> tuple[np.ndarray, np.ndarray]:
        controlled = center + _rotate(_OFFSETS, angle)
        external = [
            item + _rotate(_OFFSETS, distractor_angles[index])
            for index, item in enumerate(distractor_centers)
        ]
        visible_controlled = controlled.copy()
        if (step - resolved.occlusion_phase) % 23 in (8, 9):
            visible_controlled = np.delete(visible_controlled, 2, axis=0)
        detections = np.concatenate((visible_controlled, *external), axis=0)
        if resolved.coordinate_noise_std:
            detections = detections + rng.normal(
                0.0, resolved.coordinate_noise_std, size=detections.shape
            )
        rng.shuffle(detections)
        return detections, controlled

    initial, _ = state(0)
    learner.reset(initial)
    tp = fp = fn = 0
    reidentifications = reappearance_count = 0
    previous_occluded = False
    pre_occlusion_ids: dict[int, int] = {}
    last_visible_ids: dict[int, int] = {}
    crossing_reference_ids: dict[int, int] | None = None
    crossing_final_ids: dict[int, int] = {}
    minimum_crossing_distance = 99.0
    association_brier: list[float] = []
    association_entropy: list[float] = []
    crossing_start = 24
    crossing_end = crossing_start + resolved.crossing_duration
    for step in range(1, steps + 1):
        action_index = (step - 1) % 4
        action = np.zeros(4)
        action[action_index] = 1.0
        center += _ACTIONS[action_index]
        if step % resolved.controlled_rotation_period == 0:
            angle += 0.08
        for index in range(len(distractor_centers)):
            distractor_angles[index] += float(rng.normal(0.0, 0.025))
            distractor_centers[index] += rng.normal(0.0, 0.18, size=2)
        if crossing:
            progress = np.clip(
                (step - crossing_start) / resolved.crossing_duration,
                0.0,
                1.0,
            )
            distractor_centers[0] = (
                (1.0 - progress) * np.asarray(resolved.path_start)
                + progress * np.asarray(resolved.path_end)
            )
            distractor_angles[0] += resolved.external_rotation_rate
        detections, controlled = state(step)
        learner.update(detections, action)
        assignment = _match_truth_assignments(learner.positions(), controlled)
        truth_tracks = set(assignment.values())
        predicted = learner.self_tracks()
        if step >= 24:
            tp += len(predicted & truth_tracks)
            fp += len(predicted - truth_tracks)
            fn += len(controlled) - len(predicted & truth_tracks)
        occluded = (step - resolved.occlusion_phase) % 23 in (8, 9)
        if not previous_occluded and occluded:
            pre_occlusion_ids = dict(last_visible_ids)
        if previous_occluded and not occluded and step >= 24:
            reappearance_count += 1
            reidentifications += int(
                len(pre_occlusion_ids) == len(controlled)
                and assignment == pre_occlusion_ids
            )
        if not occluded:
            last_visible_ids = dict(assignment)
        if crossing:
            distance = float(np.linalg.norm(distractor_centers[0] - center))
            minimum_crossing_distance = min(minimum_crossing_distance, distance)
            if step == crossing_start - 1:
                crossing_reference_ids = dict(assignment)
            if step == min(steps, crossing_end + 12):
                crossing_final_ids = dict(assignment)
            if (
                crossing_reference_ids
                and crossing_start <= step <= crossing_end
            ):
                external_points = (
                    distractor_centers[0]
                    + _rotate(_OFFSETS, distractor_angles[0])
                )
                for truth_index, track_index in crossing_reference_ids.items():
                    if occluded and truth_index == 2:
                        continue
                    truth_point = controlled[truth_index]
                    if min(
                        float(np.linalg.norm(truth_point - point))
                        for point in external_points
                    ) <= 0.08:
                        continue
                    distribution = learner.association_distribution(track_index)
                    point_probabilities = {
                        key: value
                        for key, value in distribution.items()
                        if key is not None
                    }
                    if not point_probabilities:
                        continue
                    detections_now = learner.detection_positions()
                    if not detections_now:
                        continue
                    true_key = min(
                        detections_now,
                        key=lambda key: float(
                            np.linalg.norm(np.asarray(key) - truth_point)
                        ),
                    )
                    if float(
                        np.linalg.norm(np.asarray(true_key) - truth_point)
                    ) > 0.20:
                        continue
                    true_probability = point_probabilities.get(true_key, 0.0)
                    association_brier.append(
                        (1.0 - true_probability) ** 2
                        + sum(
                            probability**2
                            for key, probability in point_probabilities.items()
                            if key != true_key
                        )
                    )
                association_entropy.append(learner.association_entropy())
        previous_occluded = occluded
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    truth_tracks = _match_truth(learner.positions(), center + _rotate(_OFFSETS, angle))
    predicted_self = learner.self_tracks()
    predicted_edges = {
        edge
        for edge in learner.rigid_edges()
        if edge[0] in predicted_self and edge[1] in predicted_self
    }
    truth_edges = {
        tuple(sorted((left, right)))
        for left in truth_tracks
        for right in truth_tracks
        if left < right
    }
    edge_tp = len(predicted_edges & truth_edges)
    edge_fp = len(predicted_edges - truth_edges)
    edge_fn = 3 - edge_tp
    edge_f1 = 2 * edge_tp / max(1, 2 * edge_tp + edge_fp + edge_fn)
    return {
        "seed": seed,
        "node_f1": f1,
        "rigid_edge_f1": edge_f1,
        "reidentification_rate": (
            reidentifications / reappearance_count if reappearance_count else 0.0
        ),
        "crossing_stress": crossing,
        "association_mode": association_mode,
        "scenario": {
            "family": resolved.family,
            "crossing_duration": resolved.crossing_duration,
            "external_rotation_rate": resolved.external_rotation_rate,
            "controlled_rotation_period": resolved.controlled_rotation_period,
            "occlusion_phase": resolved.occlusion_phase,
            "coordinate_noise_std": resolved.coordinate_noise_std,
        },
        "minimum_crossing_distance": (
            minimum_crossing_distance if crossing else None
        ),
        "crossing_order_flipped": (
            bool(
                np.dot(
                    distractor_centers[0] - np.asarray(resolved.path_start),
                    np.asarray(resolved.path_end)
                    - np.asarray(resolved.path_start),
                )
                > 0.8
                * float(
                    np.linalg.norm(
                        np.asarray(resolved.path_end)
                        - np.asarray(resolved.path_start)
                    )
                    ** 2
                )
            )
            if crossing
            else None
        ),
        "crossing_identity_retention": (
            sum(
                crossing_final_ids.get(index) == track
                for index, track in (crossing_reference_ids or {}).items()
            )
            / max(1, len(crossing_reference_ids or {}))
            if crossing
            else None
        ),
        "ambiguous_association_brier": (
            mean(association_brier) if association_brier else 1.0
        ),
        "mean_association_entropy": (
            mean(association_entropy) if association_entropy else 0.0
        ),
    }


def run_v2_m2(
    *,
    output_path: str | Path = "results/V2-M2-summary.json",
    prerequisite_path: str | Path = "results/V2-M1-summary.json",
    seeds: Sequence[int] = tuple(range(300, 316)),
    steps: int = 260,
    protocol_path: str | Path | None = None,
    split: str | None = None,
    association_mode: str = "probabilistic",
) -> dict[str, Any]:
    require_authorization(
        prerequisite_path,
        expected_name="V2-M1",
        expected_decision="authorize_v2_m2",
    )
    protocol: dict[str, Any] | None = None
    protocol_sha256: str | None = None
    if protocol_path is not None:
        protocol, protocol_sha256 = _load_frozen_protocol(protocol_path)
        if split not in {"development", "holdout"}:
            raise ValueError("protocol runs require development or holdout split")
        seeds = tuple(int(seed) for seed in protocol[split]["seeds"])
        if split == "holdout":
            expected_output = Path(protocol["holdout"]["result_path"])
            if Path(output_path) != expected_output:
                raise RuntimeError(
                    "holdout output must match the path frozen in the protocol"
                )
            if expected_output.exists():
                raise RuntimeError(
                    "frozen holdout result already exists; the protocol forbids reruns"
                )
    started = perf_counter()
    normal = [
        _episode(
            seed,
            steps=steps,
            crossing=False,
            scenario=scenario_from_seed(seed),
            association_mode=association_mode,
        )
        for seed in seeds
    ]
    crossing = [
        _episode(
            seed,
            steps=steps,
            crossing=True,
            scenario=scenario_from_seed(seed),
            association_mode=association_mode,
        )
        for seed in seeds
    ]
    learner = OnlineEntityGraph(4, association_mode=association_mode)
    aggregate = {
        "node_f1": mean(item["node_f1"] for item in normal),
        "rigid_edge_f1": mean(item["rigid_edge_f1"] for item in normal),
        "reidentification_rate": mean(
            item["reidentification_rate"] for item in normal
        ),
        "crossing_node_f1": mean(item["node_f1"] for item in crossing),
        "crossing_identity_retention": mean(
            item["crossing_identity_retention"] for item in crossing
        ),
        "maximum_crossing_distance": max(
            item["minimum_crossing_distance"] for item in crossing
        ),
        "ambiguous_association_brier": mean(
            item["ambiguous_association_brier"] for item in crossing
        ),
        "crossing_identity_switch_rate": 1.0
        - mean(item["crossing_identity_retention"] for item in crossing),
    }
    family_retention = {
        family: mean(
            item["crossing_identity_retention"]
            for item in crossing
            if item["scenario"]["family"] == family
        )
        for family in sorted(
            {item["scenario"]["family"] for item in crossing}
        )
    }
    aggregate["crossing_identity_retention_by_family"] = family_retention
    aggregate["worst_family_crossing_identity_retention"] = min(
        family_retention.values()
    )
    resources = {
        "learnable_parameter_count": learner.learnable_parameter_count,
        "active_state_bytes": learner.active_state_bytes,
        "estimated_mac_per_step": learner.estimated_mac_per_step,
        "steps_per_seed": steps,
        "maximum_replays_per_experience": 0,
        "cpu_wall_seconds": perf_counter() - started,
    }
    fixed = protocol["fixed_gates"] if protocol is not None else {}
    gates = {
        "m1_prerequisite_passed": True,
        "controlled_node_f1_ge_0_90": aggregate["node_f1"]
        >= fixed.get("normal_node_f1_minimum", 0.90),
        "rigid_edge_f1_ge_0_85": aggregate["rigid_edge_f1"]
        >= fixed.get("normal_rigid_edge_f1_minimum", 0.85),
        "occlusion_reidentification_ge_0_90": (
            aggregate["reidentification_rate"]
            >= fixed.get("occlusion_identity_retention_minimum", 0.90)
        ),
        "crossing_stress_node_f1_ge_0_80": (
            aggregate["crossing_node_f1"]
            >= fixed.get("crossing_node_f1_minimum", 0.80)
        ),
        "crossing_actually_occurred": (
            aggregate["maximum_crossing_distance"] <= 1.0
            and all(item["crossing_order_flipped"] for item in crossing)
        ),
        "crossing_identity_retention_ge_0_90": (
            aggregate["crossing_identity_retention"]
            >= fixed.get("crossing_identity_retention_mean_minimum", 0.90)
        ),
        "worst_family_identity_retention_ge_0_75": (
            aggregate["worst_family_crossing_identity_retention"]
            >= fixed.get(
                "crossing_identity_retention_worst_family_minimum", 0.75
            )
        ),
        "crossing_identity_switch_rate_le_0_10": (
            aggregate["crossing_identity_switch_rate"]
            <= fixed.get("crossing_identity_switch_rate_maximum", 0.10)
        ),
        "ambiguous_association_brier_le_0_20": (
            aggregate["ambiguous_association_brier"]
            <= fixed.get("ambiguous_association_brier_maximum", 0.20)
        ),
        "resources_pass": (
            resources["learnable_parameter_count"] <= 100_000
            and resources["active_state_bytes"] <= 64 * 1024
            and resources["estimated_mac_per_step"] <= 5_000_000
            and steps <= 100_000
            and resources["cpu_wall_seconds"] <= 7_200
        ),
        "labels_absent_from_learner": True,
    }
    summary = {
        "result_schema_version": 1,
        "experiment": "V2-M2",
        "candidate": "probabilistic_multi_trajectory_association",
        "association_mode": association_mode,
        "review_split": split or "legacy_development",
        "protocol_path": str(protocol_path) if protocol_path else None,
        "protocol_sha256": protocol_sha256,
        "holdout_run_count": 1 if split == "holdout" else 0,
        "formal_agent_input": "unordered_sparse_visual_detections_and_action_copy",
        "evaluation_labels_used_for_learning": False,
        "normal_episodes": normal,
        "crossing_episodes": crossing,
        "aggregate": aggregate,
        "resources": resources,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "authorize_v2_m3" if all(gates.values()) else "stop_before_v2_m3",
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_frozen_protocol(
    protocol_path: str | Path,
) -> tuple[dict[str, Any], str]:
    path = Path(protocol_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lock_path = path.with_suffix(".sha256")
    expected = lock_path.read_text(encoding="utf-8").split()[0]
    if digest != expected:
        raise RuntimeError("V2-M2 protocol hash does not match its frozen lock")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != (
        "frozen_before_probabilistic_association_implementation"
    ):
        raise RuntimeError("V2-M2 protocol is not frozen")
    return protocol, digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument(
        "--prerequisite",
        type=Path,
        default=Path("results/V2-M1-summary.json"),
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument(
        "--split",
        choices=("development", "holdout"),
    )
    parser.add_argument(
        "--association-mode",
        choices=("probabilistic", "hard_map", "nearest"),
        default="probabilistic",
    )
    arguments = parser.parse_args(argv)
    output = arguments.output
    if output is None:
        if arguments.protocol and arguments.split == "holdout":
            protocol = json.loads(
                arguments.protocol.read_text(encoding="utf-8")
            )
            output = Path(protocol["holdout"]["result_path"])
        elif arguments.protocol and arguments.split == "development":
            output = Path("results/V2-M2-probabilistic-development-summary.json")
        else:
            output = Path("results/V2-M2-summary.json")
    result = run_v2_m2(
        output_path=output,
        prerequisite_path=arguments.prerequisite,
        steps=arguments.steps,
        protocol_path=arguments.protocol,
        split=arguments.split,
        association_mode=arguments.association_mode,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
