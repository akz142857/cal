"""Tests for the M1d machine-readable screen decision."""

from calmodel.evaluation.m1d_summary import _mapping


def test_m1d_summary_requires_mapping_values() -> None:
    assert _mapping({"metrics": {"iou": 0.5}}, "metrics")["iou"] == 0.5
