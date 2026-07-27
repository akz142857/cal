"""Fresh-data behavioral confirmation of the V2-M1 through V2-M3 chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

from cal.evaluation.v2_m1 import (
    _aggregate as _m1_aggregate,
    _run_episode as _m1_episode,
)
from cal.evaluation.v2_m2 import (
    _episode as _m2_episode,
    scenario_from_seed as m2_scenario_from_seed,
)
from cal.evaluation.v2_m3_hypotheses import (
    _episode as _m3_episode,
    scenario_from_seed as m3_scenario_from_seed,
)
from cal.evaluation.v2_artifacts import constructor_apis_reject_ground_truth
from cal.infra.provenance import capture_provenance
from cal.model.body_hypotheses import BodyGraphHypothesisFilter
from cal.model.entity_graph import OnlineEntityGraph
from cal.model.online_control import (
    ControlAgentConfig,
    OnlineControlAgent,
)


DEFAULT_PROTOCOL = Path(
    "experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL_V7.json"
)
DEFAULT_NAMESPACE_MIGRATION = Path(
    "experiments/CAL_NAMESPACE_MIGRATION.json"
)


def _load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = source.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if digest != expected:
        raise RuntimeError("integrated confirmation protocol hash mismatch")
    protocol = json.loads(source.read_text(encoding="utf-8"))
    if protocol.get("status") not in {
        "frozen_before_integrated_confirmation_evaluator_implementation",
        "frozen_after_v1_development_metric_audit_before_any_confirmation_run",
        "frozen_after_v2_confirmation_gate_computation_bugfix",
        "frozen_after_v3_entity_graph_reacquisition_window_extension",
        "frozen_after_v4_entity_graph_identity_switch_penalty_extension",
        "frozen_after_v5_entity_graph_drift_reset_extension",
        "frozen_after_v6_entity_graph_confidence_adaptive_gating_extension",
    }:
        raise RuntimeError("integrated confirmation protocol is not frozen")
    return protocol, digest


def _load_namespace_migration() -> dict[str, Any]:
    source = DEFAULT_NAMESPACE_MIGRATION
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = source.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if digest != expected:
        raise RuntimeError("namespace migration manifest hash mismatch")
    migration = json.loads(source.read_text(encoding="utf-8"))
    if (
        migration.get("migration_kind") != "python_namespace_only"
        or migration.get("from_namespace") != "calmodel"
        or migration.get("to_namespace") != "cal"
        or migration.get("historical_protocols_immutable") is not True
        or migration.get("historical_results_immutable") is not True
    ):
        raise RuntimeError("invalid namespace migration manifest")
    return migration


def _verify_locked_sources(protocol: dict[str, Any]) -> dict[str, str]:
    locked = protocol["locked_source_sha256"]
    if all(Path(path).exists() for path in locked):
        observed = {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in locked
        }
    else:
        migration = _load_namespace_migration()
        source_map = migration["source_map"]
        observed = {}
        for legacy_path, legacy_digest in locked.items():
            entry = source_map.get(legacy_path)
            if (
                not isinstance(entry, dict)
                or entry.get("legacy_sha256") != legacy_digest
            ):
                raise RuntimeError(
                    f"missing namespace migration for {legacy_path}"
                )
            current_path = Path(entry["current_path"])
            current_digest = hashlib.sha256(
                current_path.read_bytes()
            ).hexdigest()
            if current_digest != entry.get("current_sha256"):
                raise RuntimeError(
                    f"migrated source changed: {current_path}"
                )
            observed[legacy_path] = legacy_digest
    mismatches = {
        path: {
            "expected": locked[path],
            "observed": digest,
        }
        for path, digest in observed.items()
        if digest != locked[path]
    }
    if mismatches:
        raise RuntimeError(
            "locked M1-M3 source changed after protocol freeze: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return observed


def _m1_suite(
    seeds: tuple[int, ...],
    *,
    steps: int,
) -> dict[str, Any]:
    variants = {
        "formal_active": (ControlAgentConfig(active=True), False),
        "random_action": (ControlAgentConfig(active=False), False),
        "no_action_input": (
            ControlAgentConfig(active=True, use_action=False),
            False,
        ),
        "no_failure_update": (
            ControlAgentConfig(active=True, use_failure_update=False),
            False,
        ),
        "time_shuffled_action": (ControlAgentConfig(active=True), True),
    }
    results: dict[str, Any] = {}
    for name, (configuration, shuffled) in variants.items():
        episodes = [
            _m1_episode(
                seed,
                steps=steps,
                agent_config=configuration,
                time_shuffle=shuffled,
            )
            for seed in seeds
        ]
        results[name] = {
            "configuration": asdict(configuration),
            "episodes": episodes,
            "aggregate": _m1_aggregate(episodes),
        }
    formal = results["formal_active"]["aggregate"]
    random = results["random_action"]["aggregate"]
    comparisons = {
        "active_confidence_step_reduction": (
            1.0
            - formal["mean_confidence_step"]
            / max(random["mean_confidence_step"], 1e-12)
        ),
        "no_action_f1_drop": (
            formal["mean_f1"]
            - results["no_action_input"]["aggregate"]["mean_f1"]
        ),
        "no_failure_update_f1_drop": (
            formal["mean_f1"]
            - results["no_failure_update"]["aggregate"]["mean_f1"]
        ),
        "time_shuffled_action_f1_drop": (
            formal["mean_f1"]
            - results["time_shuffled_action"]["aggregate"]["mean_f1"]
        ),
    }
    return {
        "variants": results,
        "comparisons": comparisons,
    }


def _aggregate_m2(
    normal: list[dict[str, Any]],
    crossing: list[dict[str, Any]],
) -> dict[str, Any]:
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
    crossing_retention = mean(
        item["crossing_identity_retention"] for item in crossing
    )
    return {
        "node_f1": mean(item["node_f1"] for item in normal),
        "rigid_edge_f1": mean(item["rigid_edge_f1"] for item in normal),
        "reidentification_rate": mean(
            item["reidentification_rate"] for item in normal
        ),
        "crossing_node_f1": mean(item["node_f1"] for item in crossing),
        "crossing_identity_retention": crossing_retention,
        "crossing_identity_switch_rate": 1.0 - crossing_retention,
        "worst_family_crossing_identity_retention": min(
            family_retention.values()
        ),
        "crossing_identity_retention_by_family": family_retention,
        "ambiguous_association_brier": mean(
            item["ambiguous_association_brier"] for item in crossing
        ),
        "maximum_crossing_distance": max(
            item["minimum_crossing_distance"] for item in crossing
        ),
        "all_crossings_flip_order": all(
            item["crossing_order_flipped"] for item in crossing
        ),
    }


def _m2_suite(
    seeds: tuple[int, ...],
    *,
    steps: int,
) -> dict[str, Any]:
    formal_normal = [
        _m2_episode(
            seed,
            steps=steps,
            crossing=False,
            scenario=m2_scenario_from_seed(seed),
            association_mode="probabilistic",
        )
        for seed in seeds
    ]
    formal_crossing = [
        _m2_episode(
            seed,
            steps=steps,
            crossing=True,
            scenario=m2_scenario_from_seed(seed),
            association_mode="probabilistic",
        )
        for seed in seeds
    ]
    nearest_crossing = [
        _m2_episode(
            seed,
            steps=steps,
            crossing=True,
            scenario=m2_scenario_from_seed(seed),
            association_mode="nearest",
        )
        for seed in seeds
    ]
    return {
        "formal": {
            "normal_episodes": formal_normal,
            "crossing_episodes": formal_crossing,
            "aggregate": _aggregate_m2(formal_normal, formal_crossing),
        },
        "nearest_association": {
            "crossing_episodes": nearest_crossing,
            "aggregate": {
                "crossing_node_f1": mean(
                    item["node_f1"] for item in nearest_crossing
                ),
                "crossing_identity_retention": mean(
                    item["crossing_identity_retention"]
                    for item in nearest_crossing
                ),
                "crossing_identity_switch_rate": (
                    1.0
                    - mean(
                        item["crossing_identity_retention"]
                        for item in nearest_crossing
                    )
                ),
            },
        },
    }


def _aggregate_m3(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_symmetric_window_steps_have_two_hypotheses": all(
            item["all_symmetric_window_steps_have_two_hypotheses"]
            for item in episodes
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


def _m3_suite(
    seeds: tuple[int, ...],
    *,
    broken_steps: int,
) -> dict[str, Any]:
    formal = [
        _m3_episode(
            seed,
            scenario=m3_scenario_from_seed(seed),
            broken_steps=broken_steps,
            use_prediction_likelihood=True,
        )
        for seed in seeds
    ]
    no_likelihood = [
        _m3_episode(
            seed,
            scenario=m3_scenario_from_seed(seed),
            broken_steps=broken_steps,
            use_prediction_likelihood=False,
        )
        for seed in seeds
    ]
    return {
        "formal": {
            "episodes": formal,
            "aggregate": _aggregate_m3(formal),
        },
        "no_causal_prediction_likelihood": {
            "episodes": no_likelihood,
            "aggregate": _aggregate_m3(no_likelihood),
        },
    }


def _resources(cpu_wall_seconds: float, steps: dict[str, int]) -> dict[str, Any]:
    m1 = OnlineControlAgent(ControlAgentConfig())
    graph = OnlineEntityGraph(4)
    body_filter = BodyGraphHypothesisFilter()
    stage_resources = {
        "m1": {
            "learnable_parameter_count": m1.learnable_parameter_count,
            "active_state_bytes": m1.active_state_bytes,
            "estimated_mac_per_step": m1.estimated_mac_per_step,
            "steps_per_seed": steps["m1_steps"],
        },
        "m2": {
            "learnable_parameter_count": graph.learnable_parameter_count,
            "active_state_bytes": graph.active_state_bytes,
            "estimated_mac_per_step": graph.estimated_mac_per_step,
            "steps_per_seed": steps["m2_steps"],
        },
        "m3_composed": {
            "learnable_parameter_count": (
                graph.learnable_parameter_count
                + body_filter.learnable_parameter_count
            ),
            "active_state_bytes": (
                graph.active_state_bytes + body_filter.active_state_bytes
            ),
            "estimated_mac_per_step": (
                graph.estimated_mac_per_step
                + body_filter.estimated_mac_per_step
            ),
            "steps_per_seed": 128 + steps["m3_broken_steps"],
        },
    }
    return {
        "stage_resources": stage_resources,
        "simultaneous_runtime_note": (
            "M1 is a stage-specific confirmation agent. The composed endpoint "
            "is M2 entity graph plus M3 body filter, so M1 state is not "
            "simultaneously resident."
        ),
        "maximum_replays_per_experience": 0,
        "cpu_wall_seconds": cpu_wall_seconds,
    }


def _evaluate_gates(
    protocol: dict[str, Any],
    m1: dict[str, Any],
    m2: dict[str, Any],
    m3: dict[str, Any],
    resources: dict[str, Any],
    *,
    source_digests: dict[str, str],
    old_holdouts_intersected: bool,
) -> dict[str, bool]:
    fixed = protocol["fixed_gates"]
    limits = protocol["resource_limits"]
    m1_formal = m1["variants"]["formal_active"]["aggregate"]
    m1_comparison = m1["comparisons"]
    m2_formal = m2["formal"]["aggregate"]
    m2_nearest = m2["nearest_association"]["aggregate"]
    m3_formal = m3["formal"]["aggregate"]
    m3_no_likelihood = m3[
        "no_causal_prediction_likelihood"
    ]["aggregate"]
    resource_pass = all(
        item["learnable_parameter_count"] <= limits["learnable_parameters"]
        and item["active_state_bytes"] <= limits["active_state_bytes"]
        and item["estimated_mac_per_step"] <= limits["mac_per_step"]
        and item["steps_per_seed"] <= limits["steps_per_seed"]
        for item in resources["stage_resources"].values()
    ) and resources["cpu_wall_seconds"] <= limits["wall_seconds"]
    gates = {
        "locked_sources_unchanged": (
            source_digests == protocol["locked_source_sha256"]
        ),
        "old_holdouts_not_read": not old_holdouts_intersected,
        "m1_formal_f1_pass": (
            m1_formal["mean_f1"] >= fixed["m1_formal_mean_f1_minimum"]
        ),
        "m1_active_information_pass": (
            m1_comparison["active_confidence_step_reduction"]
            >= fixed["m1_active_confidence_step_reduction_minimum"]
        ),
        "m1_no_action_control_pass": (
            m1_comparison["no_action_f1_drop"]
            >= fixed["m1_no_action_f1_drop_minimum"]
        ),
        "m1_no_failure_update_control_pass": (
            m1_comparison["no_failure_update_f1_drop"]
            >= fixed["m1_no_failure_update_f1_drop_minimum"]
        ),
        "m1_time_shuffle_control_pass": (
            m1_comparison["time_shuffled_action_f1_drop"]
            >= fixed["m1_time_shuffled_action_f1_drop_minimum"]
        ),
        "m1_occlusion_calibration_pass": (
            m1_formal["mean_occlusion_brier"]
            <= fixed["m1_occlusion_brier_maximum"]
            and m1_formal["mean_occlusion_nll"]
            <= fixed["m1_occlusion_nll_maximum"]
        ),
        "m1_failure_memory_bounded": m1_formal[
            "all_failure_memories_bounded"
        ],
        "m2_normal_node_pass": (
            m2_formal["node_f1"] >= fixed["m2_normal_node_f1_minimum"]
        ),
        "m2_rigid_edge_pass": (
            m2_formal["rigid_edge_f1"]
            >= fixed["m2_normal_rigid_edge_f1_minimum"]
        ),
        "m2_occlusion_identity_pass": (
            m2_formal["reidentification_rate"]
            >= fixed["m2_occlusion_identity_retention_minimum"]
        ),
        "m2_crossing_node_pass": (
            m2_formal["crossing_node_f1"]
            >= fixed["m2_crossing_node_f1_minimum"]
        ),
        "m2_crossing_occurred": (
            m2_formal["maximum_crossing_distance"] <= 1.0
            and m2_formal["all_crossings_flip_order"]
        ),
        "m2_crossing_identity_pass": (
            m2_formal["crossing_identity_retention"]
            >= fixed["m2_crossing_identity_retention_mean_minimum"]
        ),
        "m2_worst_family_identity_pass": (
            m2_formal["worst_family_crossing_identity_retention"]
            >= fixed[
                "m2_crossing_identity_retention_worst_family_minimum"
            ]
        ),
        "m2_switch_rate_pass": (
            m2_formal["crossing_identity_switch_rate"]
            <= fixed["m2_crossing_identity_switch_rate_maximum"]
        ),
        "m2_association_calibration_pass": (
            m2_formal["ambiguous_association_brier"]
            <= fixed["m2_ambiguous_association_brier_maximum"]
        ),
        "m2_nearest_control_pass": (
            m2_nearest["crossing_identity_retention"]
            <= fixed["m2_nearest_crossing_identity_retention_maximum"]
        ),
        "m3_complete_hypotheses_pass": (
            m3_formal["all_symmetric_window_steps_have_two_hypotheses"]
            and m3_formal["all_hypotheses_complete"]
        ),
        "m3_posterior_normalized": (
            m3_formal["posterior_sum_error_maximum"]
            <= fixed["m3_posterior_sum_error_maximum"]
        ),
        "m3_symmetric_calibration_pass": (
            m3_formal["symmetric_probability_deviation_maximum"]
            <= fixed[
                "m3_symmetric_each_probability_deviation_from_half_maximum"
            ]
            and m3_formal["symmetric_multiclass_brier"]
            <= fixed["m3_symmetric_multiclass_brier_maximum"]
            and fixed["m3_symmetric_nll_minimum"]
            <= m3_formal["symmetric_nll"]
            <= fixed["m3_symmetric_nll_maximum"]
        ),
        "m3_broken_probability_pass": (
            m3_formal["broken_true_probability_mean"]
            >= fixed[
                "m3_broken_true_hypothesis_probability_mean_minimum"
            ]
            and m3_formal["broken_true_probability_minimum"]
            >= fixed["m3_broken_true_hypothesis_probability_minimum"]
        ),
        "m3_broken_calibration_pass": (
            m3_formal["broken_multiclass_brier"]
            <= fixed["m3_broken_multiclass_brier_maximum"]
            and m3_formal["broken_nll"]
            <= fixed["m3_broken_nll_maximum"]
        ),
        "m3_convergence_pass": (
            m3_formal["broken_convergence_steps_maximum"]
            <= fixed["m3_broken_convergence_steps_maximum"]
        ),
        "m3_topology_pass": (
            m3_formal["complete_graph_topology_f1"]
            >= fixed["m3_complete_graph_topology_f1_minimum"]
        ),
        "m3_identity_pass": (
            m3_formal["pose_identity_retention"]
            >= fixed["m3_pose_identity_retention_minimum"]
        ),
        "resources_pass": resource_pass,
        "labels_absent_from_all_formal_learners": (
            constructor_apis_reject_ground_truth(
                ControlAgentConfig,
                OnlineControlAgent,
                OnlineEntityGraph,
                BodyGraphHypothesisFilter,
            )
        ),
    }
    probability_stays_half = (
        abs(m3_no_likelihood["broken_true_probability_mean"] - 0.5)
        <= fixed[
            "m3_no_likelihood_true_probability_deviation_from_half_maximum"
        ]
    )
    if "m3_no_likelihood_topology_f1_maximum" in fixed:
        # Protocol v1 retained for auditability. Its arbitrary MAP tie-break
        # gate failed on development and is not used for confirmation.
        gates["m3_no_likelihood_control_pass"] = (
            probability_stays_half
            and m3_no_likelihood["complete_graph_topology_f1"]
            <= fixed["m3_no_likelihood_topology_f1_maximum"]
        )
    else:
        gates["m3_no_likelihood_control_pass"] = (
            probability_stays_half
            and m3_no_likelihood["broken_multiclass_brier"]
            >= fixed["m3_no_likelihood_brier_minimum"]
            and m3_no_likelihood["broken_nll"]
            >= fixed["m3_no_likelihood_nll_minimum"]
            and m3_no_likelihood["broken_convergence_steps_maximum"]
            >= fixed["m3_no_likelihood_convergence_steps_minimum"]
        )
    return gates


def run_v2_m1_m3_confirmation(
    *,
    split: str,
    output_path: str | Path,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    protocol, protocol_digest = _load_protocol(protocol_path)
    source_digests = _verify_locked_sources(protocol)
    if split not in {"development", "confirmation"}:
        raise ValueError("split must be development or confirmation")
    expected_output = Path(protocol[split]["result_path"])
    destination = Path(output_path)
    if destination != expected_output:
        raise RuntimeError("output path does not match the frozen protocol")
    if split == "confirmation" and destination.exists():
        raise RuntimeError(
            "one-shot integrated confirmation already exists; reruns forbidden"
        )
    old_seed_set = set(
        protocol["historical_holdout_policy"]["old_m2_holdout_seeds"]
        + protocol["historical_holdout_policy"]["old_m3_holdout_seeds"]
    )
    seeds = {
        name: tuple(int(seed) for seed in protocol[split][f"{name}_seeds"])
        for name in ("m1", "m2", "m3")
    }
    old_holdouts_intersected = any(
        old_seed_set.intersection(stage_seeds) for stage_seeds in seeds.values()
    )
    if old_holdouts_intersected:
        raise RuntimeError("integrated confirmation reuses an old holdout seed")
    runtime = protocol["fixed_runtime"]
    started = perf_counter()
    m1 = _m1_suite(seeds["m1"], steps=int(runtime["m1_steps"]))
    m2 = _m2_suite(seeds["m2"], steps=int(runtime["m2_steps"]))
    m3 = _m3_suite(
        seeds["m3"],
        broken_steps=int(runtime["m3_broken_steps"]),
    )
    elapsed = perf_counter() - started
    resources = _resources(elapsed, runtime)
    gates = _evaluate_gates(
        protocol,
        m1,
        m2,
        m3,
        resources,
        source_digests=source_digests,
        old_holdouts_intersected=old_holdouts_intersected,
    )
    passed = all(gates.values())
    summary = {
        "result_schema_version": 1,
        "experiment": "V2-M1-M3-integrated-confirmation",
        "review_split": split,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "locked_source_sha256": source_digests,
        "confirmation_run_count": 1 if split == "confirmation" else 0,
        "historical_holdout_artifacts_read": [],
        "historical_holdout_seeds_used": [],
        "composition_contract": protocol["scope_statement"],
        "formal_agent_inputs": {
            "m1": "binary_visual_frame_and_action_copy_only",
            "m2": "unordered_sparse_visual_detections_and_action_copy",
            "m3": (
                "unordered_sparse_visual_nodes_joint_action_and_learned_graph_only"
            ),
        },
        "evaluation_labels_used_for_learning": False,
        "seeds": {key: list(value) for key, value in seeds.items()},
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "resources": resources,
        "gates": gates,
        "passed": passed,
        "decision": (
            "authorize_design_of_unprivileged_v2_m4"
            if passed
            else "retain_stop_at_m1_m3_confirmation"
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
        choices=("development", "confirmation"),
        required=True,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    output = arguments.output or Path(
        protocol[arguments.split]["result_path"]
    )
    result = run_v2_m1_m3_confirmation(
        split=arguments.split,
        output_path=output,
        protocol_path=arguments.protocol,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
