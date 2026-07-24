"""Tests for pre-registered M1c screen decisions."""

from calmodel.evaluation.m1c_summary import _mapping


def test_m1c_summary_requires_mapping_values() -> None:
    assert _mapping({"metrics": {"iou": 0.5}}, "metrics")["iou"] == 0.5
