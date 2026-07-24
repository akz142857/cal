"""Tests for V2-M2 rigid visual entity formation."""

import json

import numpy as np

from calmodel.evaluation.v2_m2 import (
    _load_frozen_protocol,
    scenario_from_seed,
    run_v2_m2,
)
from calmodel.model.entity_graph import OnlineEntityGraph


def test_entity_graph_consumes_unordered_detections() -> None:
    learner = OnlineEntityGraph(4)
    points = np.asarray(((4.0, 4.0), (2.0, 2.0), (2.0, 4.0)))
    learner.reset(points)
    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    learner.update(points + np.asarray((-0.7, 0.0)), action)

    assert len(learner.positions()) == 3
    assert learner.learnable_parameter_count < 100_000


def test_probabilistic_association_keeps_multiple_close_hypotheses() -> None:
    learner = OnlineEntityGraph(4)
    learner.reset(np.asarray(((4.0, 5.0), (6.0, 5.0))))
    action = np.zeros(4)
    learner.update(np.asarray(((4.9, 5.0), (5.1, 5.0))), action)

    assert learner.association_hypothesis_count() > 1
    assert learner.association_entropy() > 0.0
    for track_index in learner.positions():
        distribution = learner.association_distribution(track_index)
        assert np.isclose(sum(distribution.values()), 1.0)


def test_frozen_protocol_hash_and_scenarios_are_deterministic() -> None:
    protocol, digest = _load_frozen_protocol(
        "experiments/V2_M2_PROBABILISTIC_ASSOCIATION_PROTOCOL.json"
    )

    assert digest.startswith("8dd8206f")
    assert protocol["holdout"]["seeds"][0] == 9101
    assert scenario_from_seed(9101) == scenario_from_seed(9101)
    assert scenario_from_seed(9101).family != scenario_from_seed(9102).family


def test_m2_refuses_to_run_without_m1_gate(tmp_path: object) -> None:
    prerequisite = tmp_path / "m1.json"  # type: ignore[operator]
    prerequisite.write_text(json.dumps({"passed": False}), encoding="utf-8")

    try:
        run_v2_m2(
            output_path=tmp_path / "m2.json",  # type: ignore[operator]
            prerequisite_path=prerequisite,
            seeds=(1,),
            steps=20,
        )
    except RuntimeError as error:
        assert "M1" in str(error)
    else:
        raise AssertionError("M2 must enforce its prerequisite")
