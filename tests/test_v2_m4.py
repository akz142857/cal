"""Tests for V2-M4 occupancy and permanence."""

import json

import numpy as np

from calmodel.evaluation.v2_m4 import run_v2_m4
from calmodel.model.occupancy import OccupancyMemory


def test_occupancy_memory_updates_only_visible_local_cells() -> None:
    memory = OccupancyMemory()
    occupancy = np.zeros((11, 11), dtype=np.uint8)
    visibility = np.ones((11, 11), dtype=np.uint8)
    occupancy[5, 5] = 1
    visibility[5, 6] = 0
    memory.update(occupancy, visibility, 0)
    probability = memory.probability()

    assert probability[12, 12] > 0.5
    assert probability[12, 13] == 0.5


def test_m4_enforces_m3_prerequisite(tmp_path: object) -> None:
    prerequisite = tmp_path / "m3.json"  # type: ignore[operator]
    prerequisite.write_text(json.dumps({"passed": False}), encoding="utf-8")
    try:
        run_v2_m4(
            output_path=tmp_path / "m4.json",  # type: ignore[operator]
            prerequisite_path=prerequisite,
            seeds=(1,),
            steps=20,
        )
    except RuntimeError as error:
        assert "M3" in str(error)
    else:
        raise AssertionError("M4 must enforce its prerequisite")
