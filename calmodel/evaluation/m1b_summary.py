"""Aggregate the persisted M1b mechanism screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from calmodel.evaluation.multiseed import summarize_values
from calmodel.infra.provenance import capture_provenance


def build_m1b_summary(
    *,
    output_path: str | Path = "results/M1b-stage-summary.json",
    temporal_path: str | Path = (
        "results/M1b-temporal-persistent-probes/"
        "temporal-multiseed-summary.json"
    ),
    action_confirmation_path: str | Path = (
        "results/M1b-temporal-intervention-confirmatory/"
        "temporal-multiseed-summary.json"
    ),
    action_transfer_root: str | Path = (
        "results/M1b-temporal-intervention-transfer"
    ),
    matched_body_probe_path: str | Path = (
        "results/M1b-matched-body-probes/"
        "body-probe-multiseed-summary.json"
    ),
    adaptation_path: str | Path = (
        "results/M1b-temporal-adaptation-confirmatory/"
        "adaptation-multiseed-summary.json"
    ),
) -> dict[str, Any]:
    conditions = {
        "full": (
            Path("results/M1b-passthrough-multiseed") / "m1b_motion_contrastive",
            Path("results/M1b-passthrough-cause"),
        ),
        "no_motion": (
            Path("results/M1b-multiseed") / "m1b_no_motion",
            Path("results/M1b-cause") / "m1b_no_motion",
        ),
        "no_auxiliary": (
            Path("results/M1b-passthrough-noaux") / "m1b_no_auxiliary",
            Path("results/M1b-passthrough-noaux-cause"),
        ),
        "balanced": (
            Path("results/M1b-balanced-multiseed") / "m1b_balanced",
            Path("results/M1b-balanced-cause"),
        ),
        "balanced_feedforward": (
            Path("results/M1b-balanced-ff-multiseed")
            / "m1b_balanced_feedforward",
            Path("results/M1b-balanced-ff-cause"),
        ),
    }
    aggregate: dict[str, Any] = {}
    raw_cause: dict[str, list[float]] = {}
    for name, (prediction_root, cause_root) in conditions.items():
        prediction_files = sorted(prediction_root.glob("seed-*/summary.json"))
        cause_files = sorted(cause_root.glob("seed-*/cause-probe-summary.json"))
        predictions = [_read_json(path) for path in prediction_files]
        causes = [_read_json(path) for path in cause_files]
        losses = [float(item["best_validation_loss"]) for item in predictions]
        ious = [
            float(
                _read_json(
                    path.parent / "probe" / "probe-summary.json"
                )["test"]["iou"]
            )
            for path in prediction_files
        ]
        post = [
            float(
                item["results"]["post_representations"]["test"][
                    "balanced_accuracy"
                ]
            )
            for item in causes
        ]
        pre = [
            float(
                item["results"]["pre_representations"]["test"][
                    "balanced_accuracy"
                ]
            )
            for item in causes
        ]
        raw_cause[name] = post
        aggregate[name] = {
            "seed_count": len(predictions),
            "prediction_loss": summarize_values(losses),
            "body_iou": summarize_values(ious),
            "cause_post": summarize_values(post),
            "cause_pre": summarize_values(pre),
        }

    difference = [
        full - no_motion
        for full, no_motion in zip(
            raw_cause["full"],
            raw_cause["no_motion"],
        )
    ]
    pre = aggregate["full"]["cause_pre"]
    temporal_source = Path(temporal_path)
    temporal = _read_json(temporal_source) if temporal_source.exists() else None
    temporal_paired = (
        temporal.get("paired_differences", {})
        if isinstance(temporal, dict)
        else {}
    )
    recurrent_difference = temporal_paired.get("gru_minus_feedforward", {})
    action_difference = temporal_paired.get("gru_minus_no_action", {})
    recurrent_iou = recurrent_difference.get("body_iou", {})
    action_confirmation_source = Path(action_confirmation_path)
    action_confirmation = (
        _read_json(action_confirmation_source)
        if action_confirmation_source.exists()
        else None
    )
    confirmation_paired = (
        action_confirmation.get("paired_differences", {})
        if isinstance(action_confirmation, dict)
        else {}
    )
    confirmation_recurrent_iou = confirmation_paired.get(
        "gru_minus_feedforward",
        {},
    ).get("body_iou", recurrent_iou)
    confirmation_action_iou = confirmation_paired.get(
        "gru_minus_no_action",
        {},
    ).get("body_iou", action_difference.get("body_iou", {}))
    transfer_root = Path(action_transfer_root)
    action_transfers = {}
    action_transfer_gates = {}
    for name in ("long_first", "large_step"):
        source = transfer_root / name / "temporal-multiseed-summary.json"
        transfer = _read_json(source) if source.exists() else None
        action_transfers[name] = transfer
        paired = (
            transfer.get("paired_differences", {})
            if isinstance(transfer, dict)
            else {}
        )
        action_transfer_gates[name] = (
            float(
                paired.get("gru_minus_no_action", {})
                .get("body_iou", {})
                .get("ci95_low", float("-inf"))
            )
            > 0.0
        )
    matched_body_source = Path(matched_body_probe_path)
    matched_body = (
        _read_json(matched_body_source)
        if matched_body_source.exists()
        else None
    )
    matched_body_paired = (
        matched_body.get("paired_differences", {})
        if isinstance(matched_body, dict)
        else {}
    )
    body_no_action_iou = matched_body_paired.get(
        "gru_minus_no_action",
        {},
    ).get("body_iou", {})
    body_feedforward_iou = matched_body_paired.get(
        "gru_minus_feedforward",
        {},
    ).get("body_iou", {})
    body_representation_gate = (
        float(body_no_action_iou.get("ci95_low", float("-inf"))) > 0.0
        and float(body_feedforward_iou.get("ci95_low", float("-inf"))) > 0.0
    )
    adaptation_source = Path(adaptation_path)
    adaptation = (
        _read_json(adaptation_source)
        if adaptation_source.exists()
        else None
    )
    adaptation_gate = (
        bool(adaptation.get("adaptation_gate_passed"))
        if isinstance(adaptation, dict)
        else False
    )
    causal_gate = aggregate["full"]["cause_post"]["ci95_low"] > 0.55
    pre_event_gate = pre["ci95_low"] <= 0.5 <= pre["ci95_high"]
    persistent_state_gate = (
        float(
            confirmation_recurrent_iou.get(
                "ci95_low",
                float("-inf"),
            )
        )
        > 0.0
    )
    action_integration_gate = (
        float(
            confirmation_action_iou.get(
                "ci95_low",
                float("-inf"),
            )
        )
        > 0.0
    )
    accepted = all(
        (
            causal_gate,
            pre_event_gate,
            persistent_state_gate,
            action_integration_gate,
            body_representation_gate,
            adaptation_gate,
        )
    )
    summary = {
        "result_schema_version": 1,
        "accepted": accepted,
        "decision": (
            "enter_M2"
            if accepted
            else "stop_current_M1b_mechanism_do_not_enter_M2"
        ),
        "causal_gate_passed": causal_gate,
        "pre_event_gate_passed": pre_event_gate,
        "persistent_state_gate_passed": persistent_state_gate,
        "action_integration_gate_passed": action_integration_gate,
        "body_representation_gate_passed": body_representation_gate,
        "adaptation_gate_passed": adaptation_gate,
        "action_transfer_gate_passed": (
            bool(action_transfer_gates)
            and all(action_transfer_gates.values())
        ),
        "action_transfer_gates": action_transfer_gates,
        "conditions": aggregate,
        "full_minus_no_motion_cause_difference": summarize_values(difference),
        "temporal_probe": temporal,
        "action_integration_confirmation": action_confirmation,
        "action_integration_transfers": action_transfers,
        "matched_dimension_body_probe": matched_body,
        "adaptation_confirmation": adaptation,
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate the M1b mechanism screen."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/M1b-stage-summary.json"),
    )
    parser.add_argument(
        "--temporal",
        type=Path,
        default=Path(
            "results/M1b-temporal-persistent-probes/"
            "temporal-multiseed-summary.json"
        ),
    )
    parser.add_argument(
        "--action-confirmation",
        type=Path,
        default=Path(
            "results/M1b-temporal-intervention-confirmatory/"
            "temporal-multiseed-summary.json"
        ),
    )
    parser.add_argument(
        "--action-transfer-root",
        type=Path,
        default=Path("results/M1b-temporal-intervention-transfer"),
    )
    parser.add_argument(
        "--matched-body-probe",
        type=Path,
        default=Path(
            "results/M1b-matched-body-probes/"
            "body-probe-multiseed-summary.json"
        ),
    )
    parser.add_argument(
        "--adaptation",
        type=Path,
        default=Path(
            "results/M1b-temporal-adaptation-confirmatory/"
            "adaptation-multiseed-summary.json"
        ),
    )
    arguments = parser.parse_args(argv)
    summary = build_m1b_summary(
        output_path=arguments.output,
        temporal_path=arguments.temporal,
        action_confirmation_path=arguments.action_confirmation,
        action_transfer_root=arguments.action_transfer_root,
        matched_body_probe_path=arguments.matched_body_probe,
        adaptation_path=arguments.adaptation,
    )
    print(
        f"causal_gate={summary['causal_gate_passed']}; "
        f"pre_event_gate={summary['pre_event_gate_passed']}; "
        f"persistent_state_gate={summary['persistent_state_gate_passed']}; "
        f"action_integration_gate={summary['action_integration_gate_passed']}; "
        f"action_transfer_gate={summary['action_transfer_gate_passed']}; "
        f"body_representation_gate="
        f"{summary['body_representation_gate_passed']}; "
        f"adaptation_gate={summary['adaptation_gate_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
