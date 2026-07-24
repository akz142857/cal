"""Tests for the minimal failure-driven control-identification agent."""

import numpy as np

from cal.env.point_world import AnonymousPointWorld, PointAction
from cal.evaluation.v2_m1 import run_v2_m1
from cal.model.online_control import OnlineControlAgent


def test_point_world_observation_contains_no_identity_channel() -> None:
    world = AnonymousPointWorld(seed=3)
    frame = world.observe()

    assert frame.dtype == np.uint8
    assert frame.ndim == 2
    assert set(np.unique(frame)) <= {0, 1}
    assert not hasattr(frame, "controlled_position")


def test_agent_records_bounded_label_free_failure_updates() -> None:
    world = AnonymousPointWorld(seed=5)
    agent = OnlineControlAgent(seed=5)
    agent.reset(world.observe())
    for _ in range(80):
        action = agent.choose_action()
        frame, _ = world.step(action)
        agent.observe_transition(frame, action)

    assert len(agent.failure_memory()) == 32
    assert agent.learnable_parameter_count < 100_000
    assert agent.active_state_bytes < 64 * 1024
    assert all(record.action in range(len(PointAction)) for record in agent.failure_memory())


def test_tiny_v2_m1_runs_all_required_ablations(tmp_path: object) -> None:
    output = tmp_path / "m1.json"  # type: ignore[operator]
    result = run_v2_m1(output_path=output, seeds=(200, 201), steps=60)

    assert output.exists()
    assert set(result["variants"]) == {
        "formal_active",
        "random_action",
        "no_action_input",
        "no_failure_update",
        "no_uncertainty",
        "no_active_information_gain",
        "time_shuffled_action",
    }
    assert result["evaluation_labels_used_for_learning"] is False
    assert result["resource_gates"]["replay_le_4"]
