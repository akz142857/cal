"""Tests for multi-seed adaptation gate aggregation."""

import json

from cal.evaluation.adaptation_summary import (
    aggregate_adaptation_results,
)


def test_adaptation_summary_requires_improvement_and_retention(
    tmp_path: object,
) -> None:
    root = tmp_path / "adaptation"  # type: ignore[operator]
    for seed in (0, 1):
        directory = root / f"seed-{seed:03d}"
        directory.mkdir(parents=True)
        (directory / "adaptation-summary.json").write_text(
            json.dumps(
                {
                    "variants": [
                        {
                            "name": "variant",
                            "curve": [
                                {
                                    "requested_experience_steps": 0,
                                    "changed_body": {"total": 1.0},
                                    "original_body": {"total": 0.5},
                                },
                                {
                                    "requested_experience_steps": 32,
                                    "changed_body": {"total": 0.8},
                                    "original_body": {"total": 0.4},
                                },
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    summary = aggregate_adaptation_results(root, seeds=(0, 1))

    assert summary["adaptation_gate_passed"]
