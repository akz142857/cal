"""Tests for matched-dimension body-probe aggregation."""

import json

import pytest

from cal.evaluation.body_probe_summary import (
    aggregate_body_probe_results,
)


def test_body_probe_summary_uses_paired_seed_differences(
    tmp_path: object,
) -> None:
    root = tmp_path / "probes"  # type: ignore[operator]
    for condition, values in {
        "gru": (0.5, 0.7),
        "feedforward": (0.2, 0.4),
    }.items():
        for seed, value in enumerate(values):
            directory = root / condition / f"seed-{seed:03d}"
            directory.mkdir(parents=True)
            (directory / "probe-summary.json").write_text(
                json.dumps(
                    {
                        "test": {"iou": value, "f1": value + 0.1},
                        "unseen_pose_test": {"iou": value - 0.1},
                        "raw_sensor_probe": {"iou": 0.1},
                        "original_representation_size": 16,
                        "probe_representation_size": 8,
                    }
                ),
                encoding="utf-8",
            )

    summary = aggregate_body_probe_results(
        root,
        conditions=("gru", "feedforward"),
        seeds=(0, 1),
    )

    paired = summary["paired_differences"]["gru_minus_feedforward"]
    assert paired["body_iou"]["mean"] == pytest.approx(0.3)
    assert (root / "body-probe-multiseed-summary.json").exists()
