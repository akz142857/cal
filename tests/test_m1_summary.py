"""Tests for M1 adaptation aggregation."""

import pytest

from cal.evaluation.m1_summary import _aggregate_adaptation


def test_aggregate_adaptation_counts_final_success() -> None:
    payloads = [
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
                            "original_body": {"total": 0.6},
                        },
                    ],
                }
            ]
        },
        {
            "variants": [
                {
                    "name": "variant",
                    "curve": [
                        {
                            "requested_experience_steps": 0,
                            "changed_body": {"total": 1.1},
                            "original_body": {"total": 0.5},
                        },
                        {
                            "requested_experience_steps": 32,
                            "changed_body": {"total": 1.2},
                            "original_body": {"total": 0.6},
                        },
                    ],
                }
            ]
        },
    ]

    summary = _aggregate_adaptation(payloads)["variant"]

    assert summary["seed_count"] == 2
    assert summary["final_success_count"] == 1
    assert summary["best_budgets"] == [32, 0]
    assert summary["final_improvement"]["mean"] == pytest.approx(0.05)
    assert summary["original_loss_change"]["mean"] == pytest.approx(0.1)
