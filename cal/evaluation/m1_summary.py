"""Build a machine-readable M1 stage summary from persisted experiment data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cal.evaluation.multiseed import summarize_values
from cal.infra.provenance import capture_provenance


def build_m1_summary(
    *,
    multiseed_directory: str | Path = "results/M1-multiseed",
    cause_directory: str | Path = "results/M1-cause-multiseed",
    adaptation_directory: str | Path = "results/M1-adaptation-multiseed",
    efficiency_directory: str | Path = "results/M1-efficiency",
    report_path: str | Path = "docs/experiments/M1_STAGE_REPORT.md",
    output_path: str | Path = "results/M1-stage-summary.json",
) -> dict[str, Any]:
    multiseed_root = Path(multiseed_directory)
    multiseed = _read_json(multiseed_root / "multiseed-summary.json")
    seeds = [int(seed) for seed in multiseed["seeds"]]

    simple_baseline_differences: dict[str, list[float]] = {
        "copy": [],
        "mean": [],
    }
    unseen_minus_fixed: list[float] = []
    for seed in seeds:
        run = multiseed_root / "baseline" / f"seed-{seed:03d}"
        prediction = _read_json(run / "summary.json")
        probe = _read_json(run / "probe" / "probe-summary.json")
        learned = float(prediction["best_validation_loss"])
        simple_baseline_differences["copy"].append(
            float(prediction["copy_baseline"]["total"]) - learned
        )
        simple_baseline_differences["mean"].append(
            float(prediction["mean_baseline"]["total"]) - learned
        )
        unseen_minus_fixed.append(
            float(probe["unseen_pose_test"]["iou"])
            - float(probe["fixed_position_baseline"]["iou"])
        )

    cause_files = sorted(
        Path(cause_directory).glob("seed-*/cause-probe-summary.json")
    )
    cause_records = []
    for source in cause_files:
        payload = _read_json(source)
        cause_records.append(
            {
                "seed": int(source.parent.name.split("-")[-1]),
                "post_state_balanced_accuracy": float(
                    payload["results"]["post_representations"]["test"][
                        "balanced_accuracy"
                    ]
                ),
                "pre_state_balanced_accuracy": float(
                    payload["results"]["pre_representations"]["test"][
                        "balanced_accuracy"
                    ]
                ),
                "visual_change_balanced_accuracy": float(
                    payload["results"]["visual_change_threshold"]["test"][
                        "balanced_accuracy"
                    ]
                ),
            }
        )

    adaptation_files = sorted(
        Path(adaptation_directory).glob(
            "seed-*/adaptation-summary.json"
        )
    )
    adaptation_payloads = [_read_json(path) for path in adaptation_files]
    adaptation = _aggregate_adaptation(adaptation_payloads)

    efficiency = {
        source.stem: _read_json(source)
        for source in sorted(Path(efficiency_directory).glob("*.json"))
    }
    paired = multiseed["paired_control_failures"]
    key_deletions = ("no_action", "no_touch", "no_proprioception")
    stable_deletion_wins = [
        name
        for name in key_deletions
        if float(
            paired[name]["body_iou_difference_full_minus_control"][
                "ci95_low"
            ]
        )
        > 0.0
    ]
    structure_variants = (
        "long_upper_arm",
        "long_forearm",
        "elbow_disabled",
    )
    structure_adaptation_passes = all(
        adaptation[name]["final_success_count"]
        == adaptation[name]["seed_count"]
        for name in structure_variants
    )
    acceptance = {
        "1_prediction_beats_simple_baselines": {
            "passed": all(
                difference > 0.0
                for values in simple_baseline_differences.values()
                for difference in values
            ),
            "copy_margin": summarize_values(
                simple_baseline_differences["copy"]
            ),
            "mean_margin": summarize_values(
                simple_baseline_differences["mean"]
            ),
        },
        "2_no_body_labels_in_main_training": {
            "passed": True,
            "evidence": (
                "trajectory schema excludes masks and probe labels are "
                "regenerated only after the predictor is frozen"
            ),
        },
        "3_unseen_pose_body_information": {
            "passed": all(value > 0.0 for value in unseen_minus_fixed),
            "unseen_iou_minus_fixed_position": summarize_values(
                unseen_minus_fixed
            ),
        },
        "4_stable_advantage_over_two_key_deletions": {
            "passed": len(stable_deletion_wins) >= 2,
            "controls_with_positive_paired_ci": stable_deletion_wins,
        },
        "5_temporal_shuffle_weakens_representation": {
            "passed": float(
                paired["shuffled_modalities"][
                    "body_iou_difference_full_minus_control"
                ]["ci95_low"]
            )
            > 0.0,
            "paired_difference": paired["shuffled_modalities"][
                "body_iou_difference_full_minus_control"
            ],
        },
        "6_finite_structural_adaptation": {
            "passed": structure_adaptation_passes,
            "required_variants": list(structure_variants),
        },
        "7_five_model_seeds": {
            "passed": len(seeds) >= 5 and len(cause_records) >= 5,
            "model_seed_count": len(seeds),
            "cause_seed_count": len(cause_records),
        },
        "8_efficiency_recorded": {
            "passed": "baseline" in efficiency and "feedforward" in efficiency,
            "profiled_conditions": sorted(efficiency),
        },
        "9_stage_report_with_failures": {
            "passed": Path(report_path).exists(),
            "path": str(report_path),
        },
    }
    summary = {
        "result_schema_version": 1,
        "decision": "return_to_M1_mechanism_design",
        "accepted": all(bool(item["passed"]) for item in acceptance.values()),
        "multiseed": multiseed,
        "cause": {
            "records": cause_records,
            "post_state_balanced_accuracy": summarize_values(
                [
                    record["post_state_balanced_accuracy"]
                    for record in cause_records
                ]
            ),
            "visual_change_balanced_accuracy": summarize_values(
                [
                    record["visual_change_balanced_accuracy"]
                    for record in cause_records
                ]
            ),
        },
        "adaptation": adaptation,
        "efficiency": efficiency,
        "acceptance": acceptance,
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _aggregate_adaptation(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("no adaptation summaries found")
    names = [str(item["name"]) for item in payloads[0]["variants"]]
    result = {}
    for name in names:
        curves = [
            next(
                item["curve"]
                for item in payload["variants"]
                if item["name"] == name
            )
            for payload in payloads
        ]
        budgets = [
            int(point["requested_experience_steps"])
            for point in curves[0]
        ]
        aggregated_curve = []
        for index, budget in enumerate(budgets):
            changed = [
                float(curve[index]["changed_body"]["total"])
                for curve in curves
            ]
            original = [
                float(curve[index]["original_body"]["total"])
                for curve in curves
            ]
            aggregated_curve.append(
                {
                    "budget": budget,
                    "changed_body_loss": summarize_values(changed),
                    "original_body_loss": summarize_values(original),
                }
            )
        zero = [
            float(curve[0]["changed_body"]["total"]) for curve in curves
        ]
        final = [
            float(curve[-1]["changed_body"]["total"]) for curve in curves
        ]
        original_zero = [
            float(curve[0]["original_body"]["total"]) for curve in curves
        ]
        original_final = [
            float(curve[-1]["original_body"]["total"]) for curve in curves
        ]
        best = [
            min(
                float(point["changed_body"]["total"])
                for point in curve
            )
            for curve in curves
        ]
        best_budgets = [
            int(
                min(
                    curve,
                    key=lambda point: float(
                        point["changed_body"]["total"]
                    ),
                )["requested_experience_steps"]
            )
            for curve in curves
        ]
        result[name] = {
            "seed_count": len(curves),
            "zero_shot_loss": summarize_values(zero),
            "final_loss": summarize_values(final),
            "final_improvement": summarize_values(
                [
                    zero_value - final_value
                    for zero_value, final_value in zip(zero, final)
                ]
            ),
            "original_loss_change": summarize_values(
                [
                    final_value - zero_value
                    for zero_value, final_value in zip(
                        original_zero,
                        original_final,
                    )
                ]
            ),
            "best_loss": summarize_values(best),
            "best_budgets": best_budgets,
            "final_success_count": sum(
                final_value < zero_value
                for final_value, zero_value in zip(final, zero)
            ),
            "original_retention_count": sum(
                final_value <= zero_value
                for final_value, zero_value in zip(
                    original_final,
                    original_zero,
                )
            ),
            "curve": aggregated_curve,
        }
    return result


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate all persisted Cal M1 evidence."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/M1-stage-summary.json"),
    )
    arguments = parser.parse_args(argv)
    summary = build_m1_summary(output_path=arguments.output)
    print(
        f"M1 accepted={summary['accepted']}; "
        f"decision={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
