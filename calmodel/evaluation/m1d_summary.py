"""Machine-readable pre-registered decision for the M1d ownership screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from calmodel.infra.provenance import capture_provenance


def build_m1d_screen_summary(
    *,
    prediction_root: str | Path = "results/M1d-ownership-screen",
    output_path: str | Path = "results/M1d-screen-summary.json",
) -> dict[str, Any]:
    root = Path(prediction_root)
    records = {}
    for condition in (
        "gru",
        "no_action",
        "feedforward",
        "no_accumulation",
    ):
        run = root / f"m1d_ownership_{condition}" / "seed-000"
        training = _read_json(run / "summary.json")
        probe = _read_json(run / "probe" / "probe-summary.json")
        body_iou = float(_mapping(probe, "test")["iou"])
        raw_iou = float(_mapping(probe, "raw_sensor_probe")["iou"])
        records[condition] = {
            "body_iou": body_iou,
            "raw_sensor_iou": raw_iou,
            "body_gain_over_raw_sensor": body_iou - raw_iou,
            "unseen_pose_iou": float(
                _mapping(probe, "unseen_pose_test")["iou"]
            ),
            "prediction_loss": float(training["best_validation_loss"]),
            "mean_baseline_loss": float(
                _mapping(training, "mean_baseline")["total"]
            ),
            "parameter_count": int(training["parameter_count"]),
        }
    full = records["gru"]
    gates = {
        "raw_sensor_gain_at_least_0_05": (
            full["body_gain_over_raw_sensor"] >= 0.05
        ),
        "over_no_action_at_least_0_04": (
            full["body_iou"] - records["no_action"]["body_iou"] >= 0.04
        ),
        "over_no_accumulation_at_least_0_03": (
            full["body_iou"]
            - records["no_accumulation"]["body_iou"]
            >= 0.03
        ),
        "over_feedforward_at_least_0_04": (
            full["body_iou"] - records["feedforward"]["body_iou"] >= 0.04
        ),
        "prediction_and_parameter_budget": (
            full["prediction_loss"] < full["mean_baseline_loss"]
            and full["parameter_count"] <= int(146_195 * 1.1)
        ),
    }
    passed = all(gates.values())
    summary = {
        "result_schema_version": 1,
        "candidate": "M1d-persistent-controllable-pixel-ownership",
        "model_seeds": [0],
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
            else "stop_M1d_before_temporal_and_five_seed_confirmation"
        ),
        "preregistration": "docs/experiments/M1D_PREREGISTRATION.md",
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the pre-registered M1d screen decision."
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("results/M1d-ownership-screen"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/M1d-screen-summary.json"),
    )
    arguments = parser.parse_args(argv)
    summary = build_m1d_screen_summary(
        prediction_root=arguments.prediction_root,
        output_path=arguments.output,
    )
    print(f"passed={summary['passed']}; decision={summary['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
