"""Machine-readable pre-registered decision for M1f."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from calmodel.infra.provenance import capture_provenance


def build_m1f_screen_summary(
    root: str | Path = "results/M1f-action-envelope-screen",
    output: str | Path = "results/M1f-screen-summary.json",
) -> dict[str, Any]:
    source = Path(root)
    records = {}
    for condition in ("gru", "no_action", "feedforward", "no_persistence"):
        run = source / f"m1f_action_envelope_{condition}" / "seed-000"
        training = json.loads((run / "summary.json").read_text())
        probe = json.loads((run / "probe/probe-summary.json").read_text())
        body = float(probe["test"]["iou"])
        raw = float(probe["raw_sensor_probe"]["iou"])
        records[condition] = {
            "body_iou": body,
            "raw_sensor_iou": raw,
            "body_gain_over_raw_sensor": body - raw,
            "unseen_pose_iou": float(probe["unseen_pose_test"]["iou"]),
            "prediction_loss": float(training["best_validation_loss"]),
            "mean_baseline_loss": float(training["mean_baseline"]["total"]),
            "parameter_count": int(training["parameter_count"]),
        }
    full = records["gru"]
    gates = {
        "raw_sensor_gain_at_least_0_05":
            full["body_gain_over_raw_sensor"] >= 0.05,
        "over_no_action_at_least_0_04":
            full["body_iou"] - records["no_action"]["body_iou"] >= 0.04,
        "over_no_persistence_at_least_0_03":
            full["body_iou"] - records["no_persistence"]["body_iou"] >= 0.03,
        "over_feedforward_at_least_0_04":
            full["body_iou"] - records["feedforward"]["body_iou"] >= 0.04,
        "unseen_pose_differences_same_direction": all(
            full["unseen_pose_iou"] > records[name]["unseen_pose_iou"]
            for name in ("no_action", "no_persistence", "feedforward")
        ),
        "prediction_and_parameter_budget": (
            full["prediction_loss"] < full["mean_baseline_loss"]
            and full["parameter_count"] <= int(146_195 * 1.1)
        ),
    }
    passed = all(gates.values())
    summary = {
        "result_schema_version": 1,
        "candidate": "M1f-all-action-causal-envelope-part-slots",
        "model_seeds": [0],
        "envelope_quality_screen": "results/M1f-envelope-screen.json",
        "records": records,
        "gates": gates,
        "temporal_gate": {
            "evaluated": False,
            "reason": "ordinary_body_gates_failed",
        },
        "passed": passed,
        "decision": (
            "run_temporal_screen"
            if passed
            else "stop_M1f_before_temporal_and_five_seed_confirmation"
        ),
        "preregistration": "docs/experiments/M1F_PREREGISTRATION.md",
        "provenance": capture_provenance(),
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    result = build_m1f_screen_summary()
    print(f"passed={result['passed']}; decision={result['decision']}")
