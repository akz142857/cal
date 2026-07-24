"""Frozen V2-M3 review of mutually exclusive complete-body hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from math import log, pi
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from calmodel.evaluation.v2_artifacts import require_authorization
from calmodel.evaluation.v2_m2 import _match_truth_assignments
from calmodel.evaluation.v2_m3 import _arm, _rasterize
from calmodel.infra.provenance import capture_provenance
from calmodel.model.body_hypotheses import (
    BodyGraphCandidate,
    BodyGraphHypothesisFilter,
)
from calmodel.model.entity_graph import OnlineEntityGraph


@dataclass(frozen=True, slots=True)
class M3Scenario:
    symmetric_steps: int
    break_mode: str
    hidden_self_branch: str
    shoulder_start: float
    elbow_start: float
    occlusion_phase: int
    coordinate_noise_std: float


def scenario_from_seed(seed: int) -> M3Scenario:
    return M3Scenario(
        symmetric_steps=(80, 96, 112, 128)[seed % 4],
        break_mode=(
            "external_noop",
            "external_autonomous",
            "external_opposite_elbow",
            "external_quarter_gain",
        )[seed % 4],
        hidden_self_branch=("left", "right")[seed % 2],
        shoulder_start=(-0.9, -0.7, -0.5, -0.3)[(seed // 4) % 4],
        elbow_start=(0.7, 0.9, 1.1, 1.3)[(seed // 7) % 4],
        occlusion_phase=(0, 7, 13, 19)[(seed // 11) % 4],
        coordinate_noise_std=(0.0, 0.01, 0.02)[(seed // 13) % 3],
    )


def _action_delta(action_index: int, gain: float = 1.0) -> tuple[float, float]:
    if action_index == 0:
        return -0.14 * gain, 0.0
    if action_index == 1:
        return 0.14 * gain, 0.0
    if action_index == 2:
        return 0.0, -0.16 * gain
    return 0.0, 0.16 * gain


def _advance_external(
    angles: list[float],
    action_index: int,
    mode: str,
    rng: np.random.Generator,
) -> None:
    if mode == "external_noop":
        return
    if mode == "external_autonomous":
        angles[0] += float(rng.normal(0.0, 0.035))
        angles[1] += float(rng.normal(0.0, 0.035))
        return
    gain = -1.0 if mode == "external_opposite_elbow" else 0.25
    shoulder, elbow = _action_delta(action_index, gain)
    angles[0] += shoulder
    angles[1] += elbow


def _shared_observation(
    base: np.ndarray,
    left_angles: list[float],
    right_angles: list[float],
    *,
    rng: np.random.Generator,
    noise_std: float,
    hide_endpoint: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = _arm(base, left_angles[0], left_angles[1])
    right = _arm(base, right_angles[0], right_angles[1])
    points = [base, left[1], left[2], right[1], right[2]]
    if hide_endpoint == "left":
        points.pop(2)
    elif hide_endpoint == "right":
        points.pop(4)
    detections = np.asarray(points, dtype=np.float64)
    if noise_std:
        detections += rng.normal(0.0, noise_std, size=detections.shape)
    rng.shuffle(detections)
    return detections, left, right


def _candidate_for_branch(
    filter_: BodyGraphHypothesisFilter,
    assignment: dict[int, int],
    branch: str,
) -> BodyGraphCandidate | None:
    indexes = (0, 1, 2) if branch == "left" else (0, 3, 4)
    if not all(index in assignment for index in indexes):
        return None
    expected = BodyGraphCandidate(*(assignment[index] for index in indexes))
    return (
        expected
        if expected in filter_.probability_map()
        else None
    )


def _multiclass_scores(
    probabilities: dict[BodyGraphCandidate, float],
    truth: BodyGraphCandidate,
) -> tuple[float, float]:
    brier = sum(
        (probability - float(candidate == truth)) ** 2
        for candidate, probability in probabilities.items()
    )
    return brier, -log(max(probabilities.get(truth, 1e-12), 1e-12))


def _pose_grid_projection(
    candidate_correct: bool,
    *,
    branch: str,
) -> tuple[float, float]:
    base = np.asarray((16.0, 16.0))
    ious = []
    shoulder_axis = np.linspace(-1.2, 0.0, 7)
    elbow_axis = np.linspace(0.4, 1.6, 7)
    for shoulder in shoulder_axis:
        for elbow in elbow_axis:
            if branch == "left":
                truth = _arm(base, float(shoulder), float(elbow))
                other = _arm(base, pi - float(shoulder), -float(elbow))
            else:
                truth = _arm(base, pi - float(shoulder), -float(elbow))
                other = _arm(base, float(shoulder), float(elbow))
            predicted = truth if candidate_correct else other
            predicted_mask = _rasterize(
                {index: tuple(point) for index, point in enumerate(predicted)},
                {(0, 1), (1, 2)},
            )
            truth_mask = _rasterize(
                {index: tuple(point) for index, point in enumerate(truth)},
                {(0, 1), (1, 2)},
            )
            intersection = int(np.logical_and(predicted_mask, truth_mask).sum())
            union = int(np.logical_or(predicted_mask, truth_mask).sum())
            ious.append(intersection / max(1, union))
    return mean(ious), min(ious)


def _episode(
    seed: int,
    *,
    scenario: M3Scenario,
    broken_steps: int = 72,
    use_prediction_likelihood: bool = True,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    graph = OnlineEntityGraph(
        4,
        match_radius=2.5,
        forgetting=0.90,
        association_commitment=0.0,
        association_miss_cost=30.0,
    )
    filter_ = BodyGraphHypothesisFilter()
    base = np.asarray((16.0, 16.0))
    left_angles = [scenario.shoulder_start, scenario.elbow_start]
    right_angles = [
        pi - scenario.shoulder_start,
        -scenario.elbow_start,
    ]
    initial, _, _ = _shared_observation(
        base,
        left_angles,
        right_angles,
        rng=rng,
        noise_std=scenario.coordinate_noise_std,
        hide_endpoint=None,
    )
    graph.reset(initial)
    symmetric_probabilities: list[tuple[float, float]] = []
    symmetric_brier: list[float] = []
    symmetric_nll: list[float] = []
    broken_true_probabilities: list[float] = []
    broken_brier: list[float] = []
    broken_nll: list[float] = []
    posterior_sum_errors: list[float] = []
    hypothesis_counts: list[int] = []
    symmetric_hypothesis_counts: list[int] = []
    topology_hits = []
    identity_hits = identity_total = 0
    reference_assignment: dict[int, int] | None = None
    convergence_step: int | None = None
    stable_convergence = 0
    total_steps = scenario.symmetric_steps + broken_steps
    for step in range(1, total_steps + 1):
        action_index = (step - 1) % 4
        action = np.zeros(4)
        action[action_index] = 1.0
        shoulder_delta, elbow_delta = _action_delta(action_index)
        if step <= scenario.symmetric_steps:
            for angles in (left_angles, right_angles):
                angles[0] += shoulder_delta
                angles[1] += elbow_delta
        else:
            self_angles = (
                left_angles
                if scenario.hidden_self_branch == "left"
                else right_angles
            )
            external_angles = (
                right_angles
                if scenario.hidden_self_branch == "left"
                else left_angles
            )
            self_angles[0] += shoulder_delta
            self_angles[1] += elbow_delta
            _advance_external(
                external_angles,
                action_index,
                scenario.break_mode,
                rng,
            )
        phase = (step - scenario.occlusion_phase) % 37
        hide = None
        if phase in (17, 18):
            hide = "left" if (step // 37) % 2 == 0 else "right"
        detections, left, right = _shared_observation(
            base,
            left_angles,
            right_angles,
            rng=rng,
            noise_std=scenario.coordinate_noise_std,
            hide_endpoint=hide,
        )
        graph.update(detections, action)
        if use_prediction_likelihood:
            filter_.update_from_entity_graph(graph)
        else:
            filter_.update(filter_.discover_candidates(graph), {})
        truth_points = np.concatenate(
            (base[None, :], left[1:], right[1:]), axis=0
        )
        assignment = _match_truth_assignments(
            graph.positions(), truth_points, tolerance=0.40
        )
        if reference_assignment is None and len(assignment) == 5:
            reference_assignment = dict(assignment)
        if reference_assignment is not None:
            identity_hits += sum(
                assignment.get(index) == track
                for index, track in reference_assignment.items()
            )
            identity_total += len(reference_assignment)
        probabilities = filter_.probability_map()
        if not probabilities:
            continue
        posterior_sum_errors.append(abs(sum(probabilities.values()) - 1.0))
        hypothesis_counts.append(len(probabilities))
        evaluation_assignment = reference_assignment or assignment
        left_candidate = _candidate_for_branch(
            filter_, evaluation_assignment, "left"
        )
        right_candidate = _candidate_for_branch(
            filter_, evaluation_assignment, "right"
        )
        if left_candidate is None or right_candidate is None:
            continue
        topology_hits.append(
            all(len(candidate.nodes) == 3 and len(candidate.edges) == 2)
            for candidate in probabilities
        )
        true_candidate = (
            left_candidate
            if scenario.hidden_self_branch == "left"
            else right_candidate
        )
        brier, nll = _multiclass_scores(probabilities, true_candidate)
        if step <= scenario.symmetric_steps:
            if step >= scenario.symmetric_steps - 23:
                symmetric_hypothesis_counts.append(len(probabilities))
                symmetric_probabilities.append(
                    (
                        probabilities[left_candidate],
                        probabilities[right_candidate],
                    )
                )
                symmetric_brier.append(brier)
                symmetric_nll.append(nll)
        else:
            probability = probabilities[true_candidate]
            broken_true_probabilities.append(probability)
            broken_brier.append(brier)
            broken_nll.append(nll)
            if probability >= 0.90:
                stable_convergence += 1
                if stable_convergence >= 3 and convergence_step is None:
                    convergence_step = step - scenario.symmetric_steps - 2
            else:
                stable_convergence = 0
    probabilities = filter_.probability_map()
    final_assignment = _match_truth_assignments(
        graph.positions(),
        np.concatenate(
            (
                base[None, :],
                _arm(base, *left_angles)[1:],
                _arm(base, *right_angles)[1:],
            ),
            axis=0,
        ),
        tolerance=0.40,
    )
    true_candidate = _candidate_for_branch(
        filter_, final_assignment, scenario.hidden_self_branch
    )
    map_candidate = (
        max(probabilities, key=probabilities.get) if probabilities else None
    )
    projection_mean, projection_worst = _pose_grid_projection(
        true_candidate is not None and map_candidate == true_candidate,
        branch=scenario.hidden_self_branch,
    )
    final_window = broken_true_probabilities[-24:]
    final_brier = broken_brier[-24:]
    final_nll = broken_nll[-24:]
    symmetric_flat = [
        probability
        for pair in symmetric_probabilities
        for probability in pair
    ]
    return {
        "seed": seed,
        "scenario": asdict(scenario),
        "all_symmetric_window_steps_have_two_hypotheses": (
            len(symmetric_hypothesis_counts) == 24
            and all(count == 2 for count in symmetric_hypothesis_counts)
        ),
        "hypothesis_count_mode": (
            max(set(hypothesis_counts), key=hypothesis_counts.count)
            if hypothesis_counts
            else 0
        ),
        "all_hypotheses_complete": all(topology_hits) if topology_hits else False,
        "posterior_sum_error_maximum": max(posterior_sum_errors, default=1.0),
        "symmetric_probability_deviation_maximum": max(
            (abs(value - 0.5) for value in symmetric_flat),
            default=1.0,
        ),
        "symmetric_multiclass_brier": mean(symmetric_brier)
        if symmetric_brier
        else 1.0,
        "symmetric_nll": mean(symmetric_nll) if symmetric_nll else 99.0,
        "broken_true_probability_mean": mean(final_window)
        if final_window
        else 0.0,
        "broken_true_probability_minimum": min(final_window)
        if final_window
        else 0.0,
        "broken_multiclass_brier": mean(final_brier)
        if final_brier
        else 1.0,
        "broken_nll": mean(final_nll) if final_nll else 99.0,
        "broken_convergence_steps": convergence_step
        if convergence_step is not None
        else broken_steps + 1,
        "complete_graph_topology_f1": float(
            true_candidate is not None and map_candidate == true_candidate
        ),
        "pose_grid_projection_iou_mean": projection_mean,
        "pose_grid_projection_iou_worst": projection_worst,
        "pose_identity_retention": identity_hits / max(1, identity_total),
    }


def run_v2_m3_hypothesis_review(
    *,
    output_path: str | Path,
    prerequisite_path: str | Path = (
        "results/V2-M2-probabilistic-holdout-summary.json"
    ),
    protocol_path: str | Path = (
        "experiments/V2_M3_BODY_GRAPH_HYPOTHESIS_PROTOCOL.json"
    ),
    split: str,
    broken_steps: int = 72,
    use_prediction_likelihood: bool = True,
) -> dict[str, Any]:
    require_authorization(
        prerequisite_path,
        expected_name="V2-M2",
        expected_decision="authorize_v2_m3",
    )
    protocol, protocol_digest = _load_protocol(protocol_path)
    if split not in {"development", "holdout"}:
        raise ValueError("split must be development or holdout")
    expected_output = Path(protocol["holdout"]["result_path"])
    if split == "holdout":
        if Path(output_path) != expected_output:
            raise RuntimeError("holdout output does not match frozen protocol")
        if expected_output.exists():
            raise RuntimeError("frozen M3 holdout already exists; reruns forbidden")
    seeds = tuple(int(seed) for seed in protocol[split]["seeds"])
    started = perf_counter()
    episodes = [
        _episode(
            seed,
            scenario=scenario_from_seed(seed),
            broken_steps=broken_steps,
            use_prediction_likelihood=use_prediction_likelihood,
        )
        for seed in seeds
    ]
    aggregate = {
        "all_symmetric_window_steps_have_two_hypotheses": all(
            item["all_symmetric_window_steps_have_two_hypotheses"]
            for item in episodes
        ),
        "hypothesis_count_mode": round(
            mean(item["hypothesis_count_mode"] for item in episodes)
        ),
        "all_hypotheses_complete": all(
            item["all_hypotheses_complete"] for item in episodes
        ),
        "posterior_sum_error_maximum": max(
            item["posterior_sum_error_maximum"] for item in episodes
        ),
        "symmetric_probability_deviation_maximum": max(
            item["symmetric_probability_deviation_maximum"]
            for item in episodes
        ),
        "symmetric_multiclass_brier": mean(
            item["symmetric_multiclass_brier"] for item in episodes
        ),
        "symmetric_nll": mean(item["symmetric_nll"] for item in episodes),
        "broken_true_probability_mean": mean(
            item["broken_true_probability_mean"] for item in episodes
        ),
        "broken_true_probability_minimum": min(
            item["broken_true_probability_minimum"] for item in episodes
        ),
        "broken_multiclass_brier": mean(
            item["broken_multiclass_brier"] for item in episodes
        ),
        "broken_nll": mean(item["broken_nll"] for item in episodes),
        "broken_convergence_steps_maximum": max(
            item["broken_convergence_steps"] for item in episodes
        ),
        "complete_graph_topology_f1": mean(
            item["complete_graph_topology_f1"] for item in episodes
        ),
        "pose_grid_projection_iou_mean": mean(
            item["pose_grid_projection_iou_mean"] for item in episodes
        ),
        "pose_grid_projection_iou_worst": min(
            item["pose_grid_projection_iou_worst"] for item in episodes
        ),
        "pose_identity_retention": mean(
            item["pose_identity_retention"] for item in episodes
        ),
    }
    graph = OnlineEntityGraph(4)
    filter_ = BodyGraphHypothesisFilter()
    resources = {
        "learnable_parameter_count": (
            graph.learnable_parameter_count + filter_.learnable_parameter_count
        ),
        "active_state_bytes": (
            graph.active_state_bytes + filter_.active_state_bytes
        ),
        "estimated_mac_per_step": (
            graph.estimated_mac_per_step + filter_.estimated_mac_per_step
        ),
        "steps_per_seed": max(
            scenario_from_seed(seed).symmetric_steps + broken_steps
            for seed in seeds
        ),
        "maximum_replays_per_experience": 0,
        "cpu_wall_seconds": perf_counter() - started,
    }
    fixed = protocol["fixed_gates"]
    gates = {
        "m2_prerequisite_passed": True,
        "exactly_two_complete_symmetric_hypotheses": (
            aggregate["all_symmetric_window_steps_have_two_hypotheses"]
            and aggregate["all_hypotheses_complete"]
        ),
        "posterior_normalized": aggregate["posterior_sum_error_maximum"]
        <= fixed["posterior_sum_error_maximum"],
        "symmetric_half_calibrated": (
            aggregate["symmetric_probability_deviation_maximum"]
            <= fixed[
                "symmetric_each_probability_deviation_from_half_maximum"
            ]
        ),
        "symmetric_brier_pass": aggregate["symmetric_multiclass_brier"]
        <= fixed["symmetric_multiclass_brier_maximum"],
        "symmetric_nll_pass": (
            fixed["symmetric_nll_minimum"]
            <= aggregate["symmetric_nll"]
            <= fixed["symmetric_nll_maximum"]
        ),
        "broken_probability_mean_pass": (
            aggregate["broken_true_probability_mean"]
            >= fixed["broken_true_hypothesis_probability_mean_minimum"]
        ),
        "broken_probability_minimum_pass": (
            aggregate["broken_true_probability_minimum"]
            >= fixed["broken_true_hypothesis_probability_minimum"]
        ),
        "broken_brier_pass": aggregate["broken_multiclass_brier"]
        <= fixed["broken_multiclass_brier_maximum"],
        "broken_nll_pass": aggregate["broken_nll"]
        <= fixed["broken_nll_maximum"],
        "broken_convergence_pass": (
            aggregate["broken_convergence_steps_maximum"]
            <= fixed["broken_convergence_steps_maximum"]
        ),
        "topology_f1_pass": aggregate["complete_graph_topology_f1"]
        >= fixed["complete_graph_topology_f1_minimum"],
        "pose_projection_mean_pass": aggregate[
            "pose_grid_projection_iou_mean"
        ]
        >= fixed["pose_grid_projection_iou_mean_minimum"],
        "pose_projection_worst_pass": aggregate[
            "pose_grid_projection_iou_worst"
        ]
        >= fixed["pose_grid_projection_iou_worst_minimum"],
        "pose_identity_pass": aggregate["pose_identity_retention"]
        >= fixed["pose_identity_retention_minimum"],
        "resources_pass": (
            resources["learnable_parameter_count"] <= 100_000
            and resources["active_state_bytes"] <= 64 * 1024
            and resources["estimated_mac_per_step"] <= 5_000_000
            and resources["steps_per_seed"] <= 100_000
            and resources["cpu_wall_seconds"] <= 7_200
        ),
        "labels_absent_from_learner": True,
    }
    summary = {
        "result_schema_version": 1,
        "experiment": "V2-M3",
        "candidate": "mutually_exclusive_complete_body_graph_hypotheses",
        "formal_agent_input": (
            "unordered_sparse_visual_nodes_joint_action_and_learned_graph_only"
        ),
        "evaluation_labels_used_for_learning": False,
        "review_split": split,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "holdout_run_count": 1 if split == "holdout" else 0,
        "use_prediction_likelihood": use_prediction_likelihood,
        "episodes": episodes,
        "aggregate": aggregate,
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


def _load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = source.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if digest != expected:
        raise RuntimeError("M3 protocol hash does not match frozen lock")
    protocol = json.loads(source.read_text(encoding="utf-8"))
    if protocol.get("status") != (
        "frozen_before_body_graph_hypothesis_implementation"
    ):
        raise RuntimeError("M3 protocol is not frozen")
    return protocol, digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "experiments/V2_M3_BODY_GRAPH_HYPOTHESIS_PROTOCOL.json"
        ),
    )
    parser.add_argument(
        "--prerequisite",
        type=Path,
        default=Path("results/V2-M2-probabilistic-holdout-summary.json"),
    )
    parser.add_argument(
        "--split",
        choices=("development", "holdout"),
        required=True,
    )
    parser.add_argument(
        "--no-prediction-likelihood",
        action="store_true",
    )
    arguments = parser.parse_args(argv)
    output = arguments.output
    if output is None:
        if arguments.split == "holdout":
            protocol = json.loads(
                arguments.protocol.read_text(encoding="utf-8")
            )
            output = Path(protocol["holdout"]["result_path"])
        else:
            output = Path("results/V2-M3-body-graph-development-summary.json")
    result = run_v2_m3_hypothesis_review(
        output_path=output,
        prerequisite_path=arguments.prerequisite,
        protocol_path=arguments.protocol,
        split=arguments.split,
        use_prediction_likelihood=not arguments.no_prediction_likelihood,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
