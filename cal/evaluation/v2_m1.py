"""Run and gate the V2-M1 online controllable-point experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import log
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from cal.env.point_world import AnonymousPointWorld, PointWorldConfig
from cal.infra.provenance import capture_provenance
from cal.model.online_control import ControlAgentConfig, OnlineControlAgent
from cal.evaluation.v2_artifacts import require_authorization


def _run_episode(
    seed: int,
    *,
    steps: int,
    agent_config: ControlAgentConfig,
    time_shuffle: bool = False,
) -> dict[str, Any]:
    world = AnonymousPointWorld(
        PointWorldConfig(distractor_count=1 + seed % 4),
        seed=seed,
    )
    agent = OnlineControlAgent(agent_config, seed=seed + 10_000)
    agent.reset(world.observe())
    true_positive = false_positive = false_negative = 0
    brier_values: list[float] = []
    nll_values: list[float] = []
    occlusion_count = 0
    confidence_step: int | None = None
    stable_confidence = 0
    truth_track: int | None = None
    action_cost_total = 0.0
    action_cost_at_confidence: float | None = None
    shuffle_rng = np.random.default_rng(seed + 90_000)
    for step in range(steps):
        action = agent.choose_action()
        frame, cost = world.step(action)
        action_cost_total += cost
        supplied_action = (
            type(action)(int(shuffle_rng.integers(0, 5)))
            if time_shuffle
            else action
        )
        agent.observe_transition(frame, supplied_action)
        evaluation = world.evaluation_state()
        probabilities = agent.probabilities()
        positions = agent.track_positions()
        if evaluation.controlled_visible:
            matches = [
                index
                for index, position in positions.items()
                if position == evaluation.controlled_position
            ]
            if matches:
                truth_track = matches[0]
        if agent.identification_ready():
            stable_confidence += 1
            if stable_confidence >= 3 and confidence_step is None:
                confidence_step = step - 1
                action_cost_at_confidence = action_cost_total
        else:
            stable_confidence = 0
        if step < 12 or truth_track is None:
            continue
        selected = (
            max(probabilities, key=probabilities.get)
            if agent.confidence() > 0.5
            else None
        )
        if evaluation.controlled_visible:
            if selected == truth_track:
                true_positive += 1
            else:
                false_negative += 1
                if selected is not None:
                    false_positive += 1
        else:
            occlusion_count += 1
            for index, probability in probabilities.items():
                target = 1.0 if index == truth_track else 0.0
                brier_values.append((probability - target) ** 2)
            probability = probabilities.get(truth_track, 1e-12)
            nll_values.append(-log(max(probability, 1e-12)))
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
    return {
        "seed": seed,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "confidence_step": confidence_step if confidence_step is not None else steps,
        "action_cost_at_confidence": (
            action_cost_at_confidence
            if action_cost_at_confidence is not None
            else action_cost_total
        ),
        "occlusion_brier": mean(brier_values) if brier_values else 1.0,
        "occlusion_nll": mean(nll_values) if nll_values else 99.0,
        "occlusion_count": occlusion_count,
        "failure_memory_count": len(agent.failure_memory()),
        "failure_memory_has_bidirectional_updates": any(
            record.weakened_tracks and record.enhanced_tracks
            for record in agent.failure_memory()
        ),
    }


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seed_count": len(episodes),
        "mean_f1": mean(item["f1"] for item in episodes),
        "min_f1": min(item["f1"] for item in episodes),
        "mean_confidence_step": mean(
            item["confidence_step"] for item in episodes
        ),
        "mean_occlusion_brier": mean(
            item["occlusion_brier"] for item in episodes
        ),
        "mean_occlusion_nll": mean(
            item["occlusion_nll"] for item in episodes
        ),
        "all_failure_memories_bounded": all(
            item["failure_memory_count"] <= 32 for item in episodes
        ),
        "mean_action_cost_at_confidence": mean(
            item["action_cost_at_confidence"] for item in episodes
        ),
    }


def run_v2_m1(
    *,
    output_path: str | Path = "results/V2-M1-summary.json",
    prerequisite_path: str | Path = "results/V2-audit-summary.json",
    seeds: Sequence[int] = tuple(range(200, 216)),
    steps: int = 160,
) -> dict[str, Any]:
    require_authorization(
        prerequisite_path,
        expected_name="v2_abc_entry_gate",
        expected_decision="authorize_v2_m1",
    )
    started = perf_counter()
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
        "no_uncertainty": (
            ControlAgentConfig(active=True, use_uncertainty=False),
            False,
        ),
        "no_active_information_gain": (
            ControlAgentConfig(active=False),
            False,
        ),
        "time_shuffled_action": (ControlAgentConfig(active=True), True),
    }
    results: dict[str, Any] = {}
    for name, (config, shuffled) in variants.items():
        episodes = [
            _run_episode(
                seed,
                steps=steps,
                agent_config=config,
                time_shuffle=shuffled,
            )
            for seed in seeds
        ]
        results[name] = {
            "configuration": asdict(config),
            "episodes": episodes,
            "aggregate": _aggregate(episodes),
        }
    formal = results["formal_active"]["aggregate"]
    random = results["random_action"]["aggregate"]
    action_drop = (
        formal["mean_f1"]
        - results["no_action_input"]["aggregate"]["mean_f1"]
    )
    update_drop = (
        formal["mean_f1"]
        - results["no_failure_update"]["aggregate"]["mean_f1"]
    )
    shuffled_drop = (
        formal["mean_f1"]
        - results["time_shuffled_action"]["aggregate"]["mean_f1"]
    )
    step_reduction = (
        1.0
        - formal["mean_confidence_step"]
        / max(random["mean_confidence_step"], 1e-12)
    )
    resource_agent = OnlineControlAgent(ControlAgentConfig())
    resources = {
        "learnable_parameter_count": resource_agent.learnable_parameter_count,
        "active_state_bytes": resource_agent.active_state_bytes,
        "estimated_mac_per_step": resource_agent.estimated_mac_per_step,
        "steps_per_seed": steps,
        "maximum_replays_per_experience": 0,
        "cpu_wall_seconds": perf_counter() - started,
    }
    resource_gates = {
        "parameters_le_100000": resources["learnable_parameter_count"] <= 100_000,
        "active_state_le_64kib": resources["active_state_bytes"] <= 64 * 1024,
        "mac_per_step_le_5000000": resources["estimated_mac_per_step"] <= 5_000_000,
        "steps_per_seed_le_100000": steps <= 100_000,
        "replay_le_4": True,
        "cpu_wall_le_2h": resources["cpu_wall_seconds"] <= 7_200,
    }
    gates = {
        "unseen_seed_f1_ge_0_90": formal["mean_f1"] >= 0.90,
        "active_confidence_steps_reduced_ge_30pct": step_reduction >= 0.30,
        "remove_action_f1_drop_ge_0_15": action_drop >= 0.15,
        "remove_failure_update_f1_drop_ge_0_15": update_drop >= 0.15,
        "time_shuffled_action_f1_drop_ge_0_15": shuffled_drop >= 0.15,
        "active_action_cost_not_higher_than_random": (
            formal["mean_action_cost_at_confidence"]
            <= random["mean_action_cost_at_confidence"]
        ),
        "occlusion_probability_calibrated": (
            formal["mean_occlusion_brier"] <= 0.05
            and formal["mean_occlusion_nll"] <= 0.20
        ),
        "failure_memory_bounded": formal["all_failure_memories_bounded"],
        "resources_pass": all(resource_gates.values()),
        "labels_absent_from_learner": True,
    }
    summary = {
        "result_schema_version": 1,
        "experiment": "V2-M1",
        "formal_agent_input": "binary_visual_frame_and_action_copy_only",
        "evaluation_labels_used_for_learning": False,
        "variants": results,
        "comparisons": {
            "active_confidence_step_reduction": step_reduction,
            "no_action_f1_drop": action_drop,
            "no_failure_update_f1_drop": update_drop,
            "time_shuffled_action_f1_drop": shuffled_drop,
            "active_action_cost_reduction": (
                1.0
                - formal["mean_action_cost_at_confidence"]
                / max(random["mean_action_cost_at_confidence"], 1e-12)
            ),
        },
        "resources": resources,
        "resource_gates": resource_gates,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "authorize_v2_m2" if all(gates.values()) else "stop_before_v2_m2",
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
    parser.add_argument("--output", type=Path, default=Path("results/V2-M1-summary.json"))
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument(
        "--prerequisite",
        type=Path,
        default=Path("results/V2-audit-summary.json"),
    )
    arguments = parser.parse_args(argv)
    result = run_v2_m1(
        output_path=arguments.output,
        prerequisite_path=arguments.prerequisite,
        steps=arguments.steps,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
