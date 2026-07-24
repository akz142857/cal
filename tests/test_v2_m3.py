"""Tests for the articulated V2-M3 stage."""

import json

from calmodel.evaluation.v2_m3 import _arm, run_v2_m3

import numpy as np


def test_arm_preserves_two_rigid_link_lengths() -> None:
    points = _arm(np.asarray((10.0, 10.0)), 0.4, -0.8)

    assert np.isclose(np.linalg.norm(points[1] - points[0]), 3.0)
    assert np.isclose(np.linalg.norm(points[2] - points[1]), 3.0)


def test_m3_enforces_m2_prerequisite(tmp_path: object) -> None:
    prerequisite = tmp_path / "m2.json"  # type: ignore[operator]
    prerequisite.write_text(json.dumps({"passed": False}), encoding="utf-8")
    try:
        run_v2_m3(
            output_path=tmp_path / "m3.json",  # type: ignore[operator]
            prerequisite_path=prerequisite,
            seeds=(1,),
            steps=20,
        )
    except RuntimeError as error:
        assert "M2" in str(error)
    else:
        raise AssertionError("M3 must enforce its prerequisite")
