"""Tests for multi-seed temporal probe aggregation."""

import json

import pytest

from calmodel.evaluation.temporal_summary import (
    aggregate_temporal_length_curve,
    aggregate_temporal_probe_results,
)


def test_temporal_summary_uses_paired_model_seed_differences(
    tmp_path: object,
) -> None:
    root = tmp_path / "temporal"  # type: ignore[operator]
    for condition, values in {
        "gru": (0.5, 0.7),
        "feedforward": (0.2, 0.4),
    }.items():
        for seed, value in enumerate(values):
            directory = root / condition / f"seed-{seed:03d}"
            directory.mkdir(parents=True)
            (directory / "temporal-probe-summary.json").write_text(
                json.dumps(
                    {
                        "blackout": {"period": 8, "start": 2, "length": 4},
                        "action_policy": "persistent",
                        "test": {"iou": value, "f1": value + 0.1},
                        "fixed_position_baseline": {
                            "iou": 0.1,
                            "f1": 0.2,
                        },
                    }
                ),
                encoding="utf-8",
            )

    summary = aggregate_temporal_probe_results(
        root,
        conditions=("gru", "feedforward"),
        seeds=(0, 1),
    )

    paired = summary["paired_differences"]["gru_minus_feedforward"]
    assert paired["body_iou"]["mean"] == pytest.approx(0.3)
    assert summary["action_policy"] == "persistent"
    assert (root / "temporal-multiseed-summary.json").exists()


def test_temporal_length_curve_detects_passing_action_difference(
    tmp_path: object,
) -> None:
    root = tmp_path / "lengths"  # type: ignore[operator]
    for length in (2, 4):
        for condition, values in {
            "gru": (0.5, 0.6),
            "no_action": (
                (0.51, 0.59) if length == 2 else (0.3, 0.4)
            ),
        }.items():
            for seed, value in enumerate(values):
                directory = (
                    root
                    / f"length-{length:02d}"
                    / condition
                    / f"seed-{seed:03d}"
                )
                directory.mkdir(parents=True)
                (directory / "temporal-probe-summary.json").write_text(
                    json.dumps(
                        {
                            "test": {"iou": value, "f1": value},
                            "fixed_position_baseline": {
                                "iou": 0.1,
                                "f1": 0.2,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

    summary = aggregate_temporal_length_curve(
        root,
        lengths=(2, 4),
        conditions=("gru", "no_action"),
        seeds=(0, 1),
    )

    assert summary["action_integration_passing_lengths"] == [4]
    assert (
        summary["action_integration_length_trend"][
            "body_iou_slope_per_step"
        ]["ci95_low"]
        > 0.0
    )
