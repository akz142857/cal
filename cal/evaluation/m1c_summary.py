"""Build machine-readable screening decisions for M1c candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cal.infra.provenance import capture_provenance


def build_m1c_a_screen_summary(
    *,
    prediction_root: str | Path = "results/M1c-no-residual-screen",
    temporal_root: str | Path = "results/M1c-no-residual-temporal-screen",
    output_path: str | Path = "results/M1c-A-screen-summary.json",
) -> dict[str, Any]:
    """Evaluate the fixed M1c-A pre-registered single-seed gates."""

    prediction = Path(prediction_root)
    temporal = Path(temporal_root)
    names = {
        "gru": "m1c_no_residual_gru",
        "no_action": "m1c_no_residual_no_action",
        "feedforward": "m1c_no_residual_feedforward",
    }
    records = {}
    for condition, name in names.items():
        run = prediction / name / "seed-000"
        training = _read_json(run / "summary.json")
        probe = _read_json(run / "probe" / "probe-summary.json")
        temporal_probe = _read_json(
            temporal / condition / "seed-000" / "temporal-probe-summary.json"
        )
        body_iou = float(_mapping(probe, "test")["iou"])
        raw_iou = float(_mapping(probe, "raw_sensor_probe")["iou"])
        records[condition] = {
            "prediction_loss": float(training["best_validation_loss"]),
            "mean_baseline_loss": float(
                _mapping(training, "mean_baseline")["total"]
            ),
            "body_iou": body_iou,
            "raw_sensor_iou": raw_iou,
            "body_gain_over_raw_sensor": body_iou - raw_iou,
            "unseen_pose_iou": float(
                _mapping(probe, "unseen_pose_test")["iou"]
            ),
            "temporal_body_iou": float(
                _mapping(temporal_probe, "test")["iou"]
            ),
        }
    full = records["gru"]
    no_action = records["no_action"]
    feedforward = records["feedforward"]
    gates = {
        "raw_sensor_gain_at_least_0_03": (
            full["body_gain_over_raw_sensor"] >= 0.03
        ),
        "body_iou_over_no_action_at_least_0_02": (
            full["body_iou"] - no_action["body_iou"] >= 0.02
        ),
        "body_iou_over_feedforward_at_least_0_02": (
            full["body_iou"] - feedforward["body_iou"] >= 0.02
        ),
        "temporal_iou_over_no_action_at_least_0_015": (
            full["temporal_body_iou"]
            - no_action["temporal_body_iou"]
            >= 0.015
        ),
        "prediction_beats_mean_baseline": (
            full["prediction_loss"] < full["mean_baseline_loss"]
        ),
    }
    passed = all(gates.values())
    summary = {
        "result_schema_version": 1,
        "candidate": "M1c-A-no-residual-bypass",
        "model_seeds": [0],
        "records": records,
        "gates": gates,
        "passed": passed,
        "decision": (
            "run_five_seed_confirmation"
            if passed
            else "stop_M1c_A_do_not_run_five_seed_confirmation"
        ),
        "preregistration": "docs/experiments/M1C_PREREGISTRATION.md",
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_m1c_b_screen_summary(
    *,
    prediction_root: str | Path = "results/M1c-control-delta-screen",
    temporal_root: str | Path = "results/M1c-control-delta-temporal-screen",
    output_path: str | Path = "results/M1c-B-screen-summary.json",
) -> dict[str, Any]:
    """Evaluate the fixed M1c-B control-state screen gates."""

    prediction = Path(prediction_root)
    temporal = Path(temporal_root)
    names = {
        "gru": "m1c_control_delta_gru",
        "no_action": "m1c_control_delta_no_action",
        "feedforward": "m1c_control_delta_feedforward",
        "no_aux": "m1c_control_delta_no_aux",
    }
    records = {}
    for condition, name in names.items():
        run = prediction / name / "seed-000"
        training = _read_json(run / "summary.json")
        probe = _read_json(run / "probe" / "probe-summary.json")
        temporal_probe = _read_json(
            temporal / condition / "seed-000" / "temporal-probe-summary.json"
        )
        body_iou = float(_mapping(probe, "test")["iou"])
        raw_iou = float(_mapping(probe, "raw_sensor_probe")["iou"])
        records[condition] = {
            "prediction_loss": float(training["best_validation_loss"]),
            "mean_baseline_loss": float(
                _mapping(training, "mean_baseline")["total"]
            ),
            "parameter_count": int(training["parameter_count"]),
            "body_iou": body_iou,
            "raw_sensor_iou": raw_iou,
            "body_gain_over_raw_sensor": body_iou - raw_iou,
            "unseen_pose_iou": float(
                _mapping(probe, "unseen_pose_test")["iou"]
            ),
            "temporal_body_iou": float(
                _mapping(temporal_probe, "test")["iou"]
            ),
        }
    full = records["gru"]
    no_action = records["no_action"]
    feedforward = records["feedforward"]
    no_aux = records["no_aux"]
    gates = {
        "raw_sensor_gain_at_least_0_03": (
            full["body_gain_over_raw_sensor"] >= 0.03
        ),
        "body_iou_over_no_action_at_least_0_02": (
            full["body_iou"] - no_action["body_iou"] >= 0.02
        ),
        "body_iou_over_feedforward_at_least_0_02": (
            full["body_iou"] - feedforward["body_iou"] >= 0.02
        ),
        "body_iou_over_no_aux_at_least_0_03": (
            full["body_iou"] - no_aux["body_iou"] >= 0.03
        ),
        "temporal_iou_over_no_action_at_least_0_015": (
            full["temporal_body_iou"]
            - no_action["temporal_body_iou"]
            >= 0.015
        ),
        "prediction_and_parameter_budget": (
            full["prediction_loss"] < full["mean_baseline_loss"]
            and full["parameter_count"] <= int(146_195 * 1.1)
        ),
    }
    passed = all(gates.values())
    summary = {
        "result_schema_version": 1,
        "candidate": "M1c-B-control-state-vision-delta",
        "model_seeds": [0],
        "records": records,
        "gates": gates,
        "passed": passed,
        "decision": (
            "run_five_seed_confirmation"
            if passed
            else "stop_M1c_B_do_not_run_five_seed_confirmation"
        ),
        "preregistration": "docs/experiments/M1C_B_PREREGISTRATION.md",
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_m1c_c_screen_summary(
    *,
    prediction_root: str | Path = "results/M1c-paired-control-screen",
    temporal_root: str | Path = (
        "results/M1c-paired-control-temporal-screen"
    ),
    output_path: str | Path = "results/M1c-C-screen-summary.json",
) -> dict[str, Any]:
    """Evaluate the fixed M1c-C cross-scene pairing gates."""

    prediction = Path(prediction_root)
    temporal = Path(temporal_root)
    names = {
        "gru": "m1c_paired_control_gru",
        "no_action": "m1c_paired_control_no_action",
        "feedforward": "m1c_paired_control_feedforward",
        "no_pair": "m1c_paired_control_no_pair",
    }
    records = {}
    for condition, name in names.items():
        run = prediction / name / "seed-000"
        training = _read_json(run / "summary.json")
        probe = _read_json(run / "probe" / "probe-summary.json")
        temporal_probe = _read_json(
            temporal / condition / "seed-000" / "temporal-probe-summary.json"
        )
        body_iou = float(_mapping(probe, "test")["iou"])
        raw_iou = float(_mapping(probe, "raw_sensor_probe")["iou"])
        records[condition] = {
            "prediction_loss": float(training["best_validation_loss"]),
            "mean_baseline_loss": float(
                _mapping(training, "mean_baseline")["total"]
            ),
            "parameter_count": int(training["parameter_count"]),
            "body_iou": body_iou,
            "raw_sensor_iou": raw_iou,
            "body_gain_over_raw_sensor": body_iou - raw_iou,
            "unseen_pose_iou": float(
                _mapping(probe, "unseen_pose_test")["iou"]
            ),
            "temporal_body_iou": float(
                _mapping(temporal_probe, "test")["iou"]
            ),
        }
    full = records["gru"]
    no_action = records["no_action"]
    feedforward = records["feedforward"]
    no_pair = records["no_pair"]
    gates = {
        "raw_sensor_gain_at_least_0_03": (
            full["body_gain_over_raw_sensor"] >= 0.03
        ),
        "body_iou_over_no_action_at_least_0_02": (
            full["body_iou"] - no_action["body_iou"] >= 0.02
        ),
        "body_iou_over_feedforward_at_least_0_02": (
            full["body_iou"] - feedforward["body_iou"] >= 0.02
        ),
        "body_iou_over_no_pair_at_least_0_03": (
            full["body_iou"] - no_pair["body_iou"] >= 0.03
        ),
        "temporal_iou_over_no_action_at_least_0_015": (
            full["temporal_body_iou"]
            - no_action["temporal_body_iou"]
            >= 0.015
        ),
        "prediction_and_parameter_budget": (
            full["prediction_loss"] < full["mean_baseline_loss"]
            and full["parameter_count"] <= int(146_195 * 1.1)
        ),
    }
    passed = all(gates.values())
    summary = {
        "result_schema_version": 1,
        "candidate": "M1c-C-cross-scene-paired-control-state",
        "model_seeds": [0],
        "records": records,
        "gates": gates,
        "passed": passed,
        "decision": (
            "run_five_seed_confirmation"
            if passed
            else "stop_M1c_C_do_not_run_five_seed_confirmation"
        ),
        "preregistration": "docs/experiments/M1C_C_PREREGISTRATION.md",
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_m1c_f_screen_summary(
    *,
    prediction_root: str | Path = (
        "results/M1c-counterfactual-effect-screen"
    ),
    output_path: str | Path = "results/M1c-F-screen-summary.json",
) -> dict[str, Any]:
    """Evaluate F's ordinary-body gates and stop before temporal if failed."""

    prediction = Path(prediction_root)
    records = {}
    for condition in ("gru", "no_action", "feedforward"):
        name = f"m1c_counterfactual_effect_{condition}"
        run = prediction / name / "seed-000"
        training = _read_json(run / "summary.json")
        probe = _read_json(run / "probe" / "probe-summary.json")
        body_iou = float(_mapping(probe, "test")["iou"])
        raw_iou = float(_mapping(probe, "raw_sensor_probe")["iou"])
        records[condition] = {
            "prediction_loss": float(training["best_validation_loss"]),
            "mean_baseline_loss": float(
                _mapping(training, "mean_baseline")["total"]
            ),
            "parameter_count": int(training["parameter_count"]),
            "body_iou": body_iou,
            "raw_sensor_iou": raw_iou,
            "body_gain_over_raw_sensor": body_iou - raw_iou,
            "unseen_pose_iou": float(
                _mapping(probe, "unseen_pose_test")["iou"]
            ),
        }
    full = records["gru"]
    no_action = records["no_action"]
    feedforward = records["feedforward"]
    gates = {
        "raw_sensor_gain_at_least_0_05": (
            full["body_iou"] - 0.3451043338683788 >= 0.05
        ),
        "body_iou_over_no_action_at_least_0_03": (
            full["body_iou"] - no_action["body_iou"] >= 0.03
        ),
        "body_iou_over_feedforward_at_least_0_03": (
            full["body_iou"] - feedforward["body_iou"] >= 0.03
        ),
        "prediction_beats_mean_baseline": (
            full["prediction_loss"] < full["mean_baseline_loss"]
        ),
        "parameter_count_within_1_1x_m1b": (
            full["parameter_count"] <= int(146_195 * 1.1)
        ),
    }
    ordinary_passed = all(gates.values())
    summary = {
        "result_schema_version": 1,
        "candidate": "M1c-F-counterfactual-action-effect",
        "model_seeds": [0],
        "records": records,
        "gates": gates,
        "temporal_gate": {
            "evaluated": False,
            "reason": (
                "ordinary_body_gates_failed"
                if not ordinary_passed
                else "pending"
            ),
        },
        "passed": False,
        "decision": (
            "stop_M1c_F_before_temporal_and_five_seed_confirmation"
            if not ordinary_passed
            else "run_temporal_screen"
        ),
        "preregistration": "docs/experiments/M1C_F_PREREGISTRATION.md",
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
        description="Build the pre-registered M1c-A screen decision."
    )
    parser.add_argument(
        "--candidate",
        choices=("A", "B", "C", "F"),
        default="A",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    arguments = parser.parse_args(argv)
    if arguments.candidate == "A":
        summary = build_m1c_a_screen_summary(
            prediction_root=(
                arguments.prediction_root
                or Path("results/M1c-no-residual-screen")
            ),
            temporal_root=(
                arguments.temporal_root
                or Path("results/M1c-no-residual-temporal-screen")
            ),
            output_path=(
                arguments.output
                or Path("results/M1c-A-screen-summary.json")
            ),
        )
    elif arguments.candidate == "B":
        summary = build_m1c_b_screen_summary(
            prediction_root=(
                arguments.prediction_root
                or Path("results/M1c-control-delta-screen")
            ),
            temporal_root=(
                arguments.temporal_root
                or Path("results/M1c-control-delta-temporal-screen")
            ),
            output_path=(
                arguments.output
                or Path("results/M1c-B-screen-summary.json")
            ),
        )
    elif arguments.candidate == "C":
        summary = build_m1c_c_screen_summary(
            prediction_root=(
                arguments.prediction_root
                or Path("results/M1c-paired-control-screen")
            ),
            temporal_root=(
                arguments.temporal_root
                or Path("results/M1c-paired-control-temporal-screen")
            ),
            output_path=(
                arguments.output
                or Path("results/M1c-C-screen-summary.json")
            ),
        )
    else:
        summary = build_m1c_f_screen_summary(
            prediction_root=(
                arguments.prediction_root
                or Path("results/M1c-counterfactual-effect-screen")
            ),
            output_path=(
                arguments.output
                or Path("results/M1c-F-screen-summary.json")
            ),
        )
    print(f"passed={summary['passed']}; decision={summary['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
