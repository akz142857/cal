"""Tests for multi-seed aggregate statistics."""

import pytest

from cal.evaluation.multiseed import summarize_values


def test_summarize_values_uses_sample_statistics() -> None:
    summary = summarize_values((1.0, 2.0, 3.0, 4.0, 5.0))

    assert summary["count"] == 5
    assert summary["mean"] == 3.0
    assert summary["sample_std"] == pytest.approx(1.58113883)
    assert summary["ci95_low"] < 3.0 < summary["ci95_high"]


def test_summarize_single_value_has_zero_width_interval() -> None:
    summary = summarize_values((2.5,))

    assert summary["sample_std"] == 0.0
    assert summary["ci95_low"] == 2.5
    assert summary["ci95_high"] == 2.5
