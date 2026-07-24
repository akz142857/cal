"""V2-M3: learn an articulated body graph from control and geometry."""

from __future__ import annotations

import argparse
import json
from math import cos, sin
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from cal.evaluation.v2_m2 import _match_truth, _match_truth_assignments
from cal.infra.provenance import capture_provenance
from cal.model.entity_graph import OnlineEntityGraph
from cal.evaluation.v2_artifacts import require_authorization


def _arm(base: np.ndarray, shoulder: float, elbow: float) -> np.ndarray:
    joint = base + 3.0 * np.asarray((cos(shoulder), sin(shoulder)))
    endpoint = joint + 3.0 * np.asarray(
        (cos(shoulder + elbow), sin(shoulder + elbow))
    )
    return np.stack((base, joint, endpoint))


def _rasterize(points: dict[int, tuple[float, float]], edges: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint8)
    for left, right in edges:
        if left not in points or right not in points:
            continue
        start = np.asarray(points[left])
        end = np.asarray(points[right])
        count = max(2, int(np.linalg.norm(end - start) * 5))
        for alpha in np.linspace(0.0, 1.0, count):
            x, y = np.rint(start * (1.0 - alpha) + end * alpha).astype(int)
            if 0 <= x < 32 and 0 <= y < 32:
                mask[y, x] = 1
    return mask


def _episode(seed: int, *, steps: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    learner = OnlineEntityGraph(4, match_radius=2.5, forgetting=0.90)
    base = np.asarray((13.0, 15.0))
    shoulder = -0.6
    elbow = 1.1
    external_base = np.asarray((24.0, 8.0))
    external_shoulder = 2.1
    external_elbow = -0.8
    truth = _arm(base, shoulder, elbow)
    external = _arm(external_base, external_shoulder, external_elbow)
    initial = np.concatenate((truth, external))
    rng.shuffle(initial)
    learner.reset(initial)
    tp = fp = fn = 0
    prediction_errors: list[float] = []
    identity_hits = identity_count = 0
    previous_truth: dict[int, np.ndarray] = {}
    reference_ids: dict[int, int] | None = None
    pose_ious: list[float] = []
    for step in range(1, steps + 1):
        action_index = (step - 1) % 4
        action = np.zeros(4)
        action[action_index] = 1.0
        predictions = learner.predict_displacements(action)
        if action_index == 0:
            shoulder -= 0.18
        elif action_index == 1:
            shoulder += 0.18
        elif action_index == 2:
            elbow -= 0.20
        else:
            elbow += 0.20
        external_shoulder += float(rng.normal(0.0, 0.025))
        external_elbow += float(rng.normal(0.0, 0.025))
        truth = _arm(base, shoulder, elbow)
        external = _arm(external_base, external_shoulder, external_elbow)
        visible_truth = truth if step % 29 not in (12, 13) else truth[:2]
        detections = np.concatenate((visible_truth, external))
        rng.shuffle(detections)
        learner.update(detections, action)
        assignment = _match_truth_assignments(
            learner.positions(), truth, tolerance=0.45
        )
        truth_tracks = set(assignment.values())
        predicted_self = learner.self_tracks()
        if step >= 32:
            if reference_ids is None and len(assignment) == len(truth):
                reference_ids = dict(assignment)
            tp += len(predicted_self & truth_tracks)
            fp += len(predicted_self - truth_tracks)
            fn += len(truth) - len(predicted_self & truth_tracks)
            if reference_ids is not None:
                identity_hits += sum(
                    assignment.get(index) == track
                    for index, track in reference_ids.items()
                )
                identity_count += len(reference_ids)
            current_edges = {
                edge
                for edge in learner.rigid_edges(
                    maximum_variance=0.12,
                    maximum_length=4.2,
                )
                if edge[0] in predicted_self and edge[1] in predicted_self
            }
            predicted_mask = _rasterize(learner.positions(), current_edges)
            truth_positions_now = {
                index: tuple(point) for index, point in enumerate(truth)
            }
            truth_mask_now = _rasterize(
                truth_positions_now, {(0, 1), (1, 2)}
            )
            intersection_now = int(
                np.logical_and(predicted_mask, truth_mask_now).sum()
            )
            union_now = int(np.logical_or(predicted_mask, truth_mask_now).sum())
            pose_ious.append(intersection_now / max(1, union_now))
        for index in truth_tracks:
            current = np.asarray(learner.positions()[index])
            if index in previous_truth and index in predictions:
                actual = current - previous_truth[index]
                prediction_errors.append(
                    float(np.linalg.norm(actual - np.asarray(predictions[index])))
                )
            previous_truth[index] = current
    node_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    truth_tracks = _match_truth(learner.positions(), truth, tolerance=0.45)
    predicted_self = learner.self_tracks()
    predicted_edges = {
        edge
        for edge in learner.rigid_edges(
            maximum_variance=0.12,
            maximum_length=4.2,
        )
        if edge[0] in predicted_self and edge[1] in predicted_self
    }
    ordered_truth = []
    for point in truth:
        match = _match_truth(learner.positions(), point[None, :], tolerance=0.45)
        ordered_truth.append(next(iter(match)) if match else -1)
    truth_edges = {
        tuple(sorted((ordered_truth[0], ordered_truth[1]))),
        tuple(sorted((ordered_truth[1], ordered_truth[2]))),
    }
    truth_edges = {edge for edge in truth_edges if -1 not in edge}
    edge_tp = len(predicted_edges & truth_edges)
    edge_fp = len(predicted_edges - truth_edges)
    edge_fn = 2 - edge_tp
    edge_f1 = 2 * edge_tp / max(1, 2 * edge_tp + edge_fp + edge_fn)
    predicted_mask = _rasterize(learner.positions(), predicted_edges)
    truth_positions = {index: tuple(point) for index, point in enumerate(truth)}
    truth_mask = _rasterize(truth_positions, {(0, 1), (1, 2)})
    intersection = int(np.logical_and(predicted_mask, truth_mask).sum())
    union = int(np.logical_or(predicted_mask, truth_mask).sum())
    return {
        "seed": seed,
        "node_f1": node_f1,
        "node_precision": tp / max(1, tp + fp),
        "node_recall": tp / max(1, tp + fn),
        "joint_edge_f1": edge_f1,
        "body_projection_iou": mean(pose_ious) if pose_ious else 0.0,
        "worst_pose_projection_iou": min(pose_ious) if pose_ious else 0.0,
        "local_jacobian_prediction_rmse": (
            float(np.sqrt(mean(value * value for value in prediction_errors)))
            if prediction_errors
            else 99.0
        ),
        "pose_identity_rate": identity_hits / max(1, identity_count),
        "final_control_strengths": learner.control_strengths(),
        "final_membership_probabilities": learner.probabilities(),
        "final_truth_tracks": sorted(truth_tracks),
    }


def _mirrored_ambiguity_stress() -> dict[str, Any]:
    """Observable-equivalent arms must remain multiple hypotheses."""

    learner = OnlineEntityGraph(4, forgetting=0.90)
    base = np.asarray((16.0, 16.0))
    angles = [[-0.7, 1.0], [0.7, -1.0]]
    detections = np.concatenate(
        (_arm(base, *angles[0]), _arm(base, *angles[1]))
    )
    learner.reset(detections)
    for step in range(96):
        action_index = step % 4
        action = np.zeros(4)
        action[action_index] = 1.0
        for arm_angles in angles:
            if action_index == 0:
                arm_angles[0] -= 0.15
            elif action_index == 1:
                arm_angles[0] += 0.15
            elif action_index == 2:
                arm_angles[1] -= 0.17
            else:
                arm_angles[1] += 0.17
        detections = np.concatenate(
            (_arm(base, *angles[0]), _arm(base, *angles[1]))
        )
        learner.update(detections, action)
    edges = learner.rigid_edges(maximum_variance=0.12, maximum_length=4.2)
    self_tracks = learner.self_tracks()
    # More than one controlled branch is an explicit multi-hypothesis result,
    # not a hidden-label guess.
    controlled_edges = {
        edge for edge in edges if edge[0] in self_tracks and edge[1] in self_tracks
    }
    return {
        "observable_equivalence_expected": True,
        "controlled_track_count": len(self_tracks),
        "controlled_edge_count": len(controlled_edges),
        "unique_hidden_identity_claimed": len(self_tracks) <= 3,
        "explicit_mutually_exclusive_hypotheses": False,
        "metric_policy": "do_not_score_a_secret_unique_self_label",
    }


def run_v2_m3(
    *,
    output_path: str | Path = "results/V2-M3-summary.json",
    prerequisite_path: str | Path = "results/V2-M2-summary.json",
    seeds: Sequence[int] = tuple(range(400, 416)),
    steps: int = 220,
    enforce_prerequisite: bool = True,
) -> dict[str, Any]:
    prerequisite_passed = True
    try:
        require_authorization(
            prerequisite_path,
            expected_name="V2-M2",
            expected_decision="authorize_v2_m3",
        )
    except RuntimeError:
        prerequisite_passed = False
        if enforce_prerequisite:
            raise
    started = perf_counter()
    episodes = [_episode(seed, steps=steps) for seed in seeds]
    ambiguity = _mirrored_ambiguity_stress()
    aggregate = {
        key: mean(item[key] for item in episodes)
        for key in (
            "node_f1",
            "joint_edge_f1",
            "body_projection_iou",
            "local_jacobian_prediction_rmse",
            "pose_identity_rate",
        )
    }
    learner = OnlineEntityGraph(4)
    resources = {
        "learnable_parameter_count": learner.learnable_parameter_count,
        "active_state_bytes": learner.active_state_bytes,
        "estimated_mac_per_step": learner.estimated_mac_per_step,
        "steps_per_seed": steps,
        "maximum_replays_per_experience": 0,
        "cpu_wall_seconds": perf_counter() - started,
    }
    gates = {
        "m2_prerequisite_passed": prerequisite_passed,
        "controlled_node_f1_ge_0_85": aggregate["node_f1"] >= 0.85,
        "joint_edge_f1_ge_0_85": aggregate["joint_edge_f1"] >= 0.85,
        "body_projection_iou_ge_0_80": aggregate["body_projection_iou"] >= 0.80,
        "pose_identity_rate_ge_0_90": aggregate["pose_identity_rate"] >= 0.90,
        "local_prediction_rmse_le_0_40": (
            aggregate["local_jacobian_prediction_rmse"] <= 0.40
        ),
        "mirrored_case_has_explicit_calibrated_hypotheses": ambiguity[
            "explicit_mutually_exclusive_hypotheses"
        ],
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
        "experiment": "V2-M3",
        "formal_agent_input": "unordered_sparse_visual_nodes_and_joint_action_copy",
        "evaluation_labels_used_for_learning": False,
        "episodes": episodes,
        "aggregate": aggregate,
        "mirrored_shared_base_stress": ambiguity,
        "resources": resources,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "authorize_v2_m4" if all(gates.values()) else "stop_before_v2_m4",
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
    parser.add_argument("--output", type=Path, default=Path("results/V2-M3-summary.json"))
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument(
        "--prerequisite",
        type=Path,
        default=Path("results/V2-M2-summary.json"),
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="run the diagnostic even when M2 did not authorize M3",
    )
    arguments = parser.parse_args(argv)
    result = run_v2_m3(
        output_path=arguments.output,
        prerequisite_path=arguments.prerequisite,
        steps=arguments.steps,
        enforce_prerequisite=not arguments.exploratory,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
