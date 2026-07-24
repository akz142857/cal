"""Tests for contiguous multimodal blackout probes."""

import torch

from cal.env.body import BodyConfig
from cal.env.world import WorldConfig
from cal.evaluation.temporal_probe import (
    BlackoutConfig,
    _blackout_mask,
    _collect_probe_trajectories,
    extract_blackout_probe_data,
)
from cal.learning.dataset import collect_trajectories
from cal.model.predictors import PredictorConfig, SensorimotorPredictor


def test_blackout_mask_selects_contiguous_periodic_steps() -> None:
    selected = _blackout_mask(
        12,
        BlackoutConfig(period=6, start=2, length=2),
    )

    assert torch.nonzero(selected, as_tuple=False).flatten().tolist() == [
        2,
        3,
        8,
        9,
    ]


def test_blackout_probe_extracts_only_zeroed_observations() -> None:
    trajectories = collect_trajectories(
        WorldConfig(object_count=1),
        BodyConfig(),
        (23,),
        steps_per_seed=12,
    )
    data = extract_blackout_probe_data(
        SensorimotorPredictor(PredictorConfig(hidden_size=16)),
        trajectories,
        blackout=BlackoutConfig(period=6, start=2, length=2),
    )

    assert len(data) == 4
    assert data.representations.shape == (4, 16)
    assert not bool(data.visions.any())


def test_persistent_policy_holds_each_action_for_a_full_block() -> None:
    trajectory = _collect_probe_trajectories(
        WorldConfig(object_count=1),
        (23,),
        steps=12,
        policy="persistent",
        block_length=4,
    )[0]
    actions = tuple(item.action for item in trajectory.experiences)

    assert len(set(actions[:4])) == 1
    assert len(set(actions[4:8])) == 1
    assert actions[0] != actions[4]


def test_intervention_policy_switches_action_at_blackout_start() -> None:
    trajectory = _collect_probe_trajectories(
        WorldConfig(object_count=1),
        (23,),
        steps=16,
        policy="intervention",
        block_length=8,
        intervention_start=3,
    )[0]
    actions = tuple(item.action for item in trajectory.experiences)

    assert len(set(actions[:3])) == 1
    assert len(set(actions[3:8])) == 1
    assert actions[2] != actions[3]
    assert len(set(actions[8:11])) == 1
    assert actions[7] != actions[8]


def test_probe_trajectory_accepts_unseen_body_dynamics() -> None:
    body = BodyConfig(link_lengths=(0.28, 0.18), angle_step=0.24)

    trajectory = _collect_probe_trajectories(
        WorldConfig(object_count=1),
        (23,),
        steps=8,
        policy="intervention",
        block_length=8,
        intervention_start=3,
        body_config=body,
    )[0]

    assert trajectory.body_config == body
