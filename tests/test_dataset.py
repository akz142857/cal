"""Tests for trajectory windows and reproducible seed splits."""

import pytest
import torch

from cal.env.body import BodyConfig
from cal.env.world import WorldConfig
from cal.learning.dataset import (
    CounterfactualTrajectorySequenceDataset,
    CompressedTrajectorySequenceIterableDataset,
    PairedTrajectorySequenceDataset,
    SeedSplit,
    SequenceTransformConfig,
    TrajectorySequenceDataset,
    _sensor_blackout_mask,
    action_from_index,
    collect_trajectories,
    collect_paired_trajectories,
)
from cal.learning.replay import save_trajectory


def test_sequence_dataset_has_expected_tensor_shapes() -> None:
    trajectories = collect_trajectories(
        WorldConfig(image_size=(16, 16)),
        BodyConfig(),
        (0, 1),
        steps_per_seed=12,
    )
    dataset = TrajectorySequenceDataset(
        trajectories,
        sequence_length=4,
        stride=4,
    )

    sample = dataset[0]

    assert sample["vision"].shape == (4, 1, 16, 16)
    assert sample["proprioception"].shape == (4, 4)
    assert sample["touch"].shape == (4, 2)
    assert sample["actions"].shape == (4,)
    assert sample["actions"].dtype == torch.long
    assert sample["next_vision"].shape == (4, 1, 16, 16)


def test_seed_split_rejects_leakage_between_groups() -> None:
    with pytest.raises(ValueError, match="overlap"):
        SeedSplit(train=(1,), validation=(1,), test=(2,))


def test_action_indexes_round_trip() -> None:
    trajectories = collect_trajectories(
        WorldConfig(),
        BodyConfig(),
        (4,),
        steps_per_seed=4,
    )
    dataset = TrajectorySequenceDataset(trajectories, sequence_length=4)
    sample = dataset[0]

    decoded = tuple(action_from_index(int(value)) for value in sample["actions"])

    assert decoded == tuple(
        experience.action for experience in trajectories[0].experiences
    )


def test_negative_control_transforms_are_deterministic() -> None:
    trajectories = collect_trajectories(
        WorldConfig(),
        BodyConfig(),
        (5,),
        steps_per_seed=8,
    )
    transform = SequenceTransformConfig(
        temporal_shuffle=True,
        randomize_actions=True,
        seed=99,
    )
    dataset = TrajectorySequenceDataset(
        trajectories,
        sequence_length=8,
        transform=transform,
    )

    first = dataset[0]
    second = dataset[0]

    assert all(torch.equal(first[key], second[key]) for key in first)
    assert not torch.equal(
        first["proprioception"],
        torch.tensor(
            [
                item.observation.proprioception
                for item in trajectories[0].experiences
            ]
        ),
    )


def test_full_sensor_blackout_preserves_clean_targets() -> None:
    trajectories = collect_trajectories(
        WorldConfig(),
        BodyConfig(),
        (8,),
        steps_per_seed=4,
    )
    dataset = TrajectorySequenceDataset(
        trajectories,
        sequence_length=4,
        transform=SequenceTransformConfig(
            sensor_blackout_probability=1.0,
        ),
    )

    sample = dataset[0]

    assert not bool(sample["vision"].any())
    assert not bool(sample["proprioception"].any())
    assert not bool(sample["touch"].any())
    assert bool(sample["next_vision"].any())


def test_contiguous_sensor_blackout_uses_requested_span() -> None:
    generator = torch.Generator().manual_seed(19)

    blackout = _sensor_blackout_mask(
        16,
        probability=0.25,
        span=4,
        generator=generator,
    )
    selected = torch.nonzero(blackout, as_tuple=False).flatten()

    assert len(selected) == 4
    assert selected.tolist() == list(
        range(int(selected[0]), int(selected[0]) + 4)
    )


def test_compressed_streaming_windows_match_in_memory_dataset(
    tmp_path: object,
) -> None:
    trajectory = collect_trajectories(
        WorldConfig(),
        BodyConfig(),
        (27,),
        steps_per_seed=12,
    )[0]
    path = tmp_path / "trajectory.jsonl.gz"  # type: ignore[operator]
    save_trajectory(trajectory, path)
    expected = TrajectorySequenceDataset(
        (trajectory,),
        sequence_length=4,
        stride=2,
    )
    streamed = CompressedTrajectorySequenceIterableDataset(
        (path,),
        sequence_length=4,
        stride=2,
    )

    actual_samples = list(streamed)

    assert len(actual_samples) == len(expected)
    for index, actual in enumerate(actual_samples):
        assert all(
            torch.equal(actual[name], expected[index][name])
            for name in actual
        )


def test_paired_dataset_aligns_actions_across_independent_scenes() -> None:
    pairs = collect_paired_trajectories(
        WorldConfig(),
        BodyConfig(),
        (30, 31),
        steps_per_seed=12,
        action_repeat_probability=0.75,
    )
    dataset = PairedTrajectorySequenceDataset(
        pairs,
        sequence_length=4,
        stride=4,
    )

    sample = dataset[0]

    assert len(dataset) == 6
    assert torch.equal(sample["actions"], sample["pair_actions"])
    assert not torch.equal(sample["vision"], sample["pair_vision"])


def test_counterfactual_dataset_cancels_exogenous_distractor_motion() -> None:
    trajectories = collect_trajectories(
        WorldConfig(
            object_count=0,
            distractor_body_count=2,
            distractor_body_motion_probability=1.0,
        ),
        BodyConfig(),
        (33,),
        steps_per_seed=12,
    )
    dataset = CounterfactualTrajectorySequenceDataset(
        trajectories,
        sequence_length=6,
        stride=6,
    )

    sample = dataset[0]

    assert sample["action_effect"].shape == (6, 1, 16, 16)
    assert sample["ownership_target"].shape == (6, 1, 16, 16)
    assert sample["action_envelope"].shape == (6, 1, 16, 16)
    assert sample["all_action_effects"].shape == (6, 5, 1, 16, 16)
    assert torch.all(sample["ownership_target"] <= sample["next_vision"])
    assert torch.all(sample["action_envelope"] <= sample["vision"])
    noop = sample["actions"] == 0
    assert torch.count_nonzero(sample["action_effect"][noop]) == 0
    assert torch.count_nonzero(sample["ownership_target"][noop]) == 0
    assert torch.all(
        torch.count_nonzero(
            sample["action_envelope"].flatten(start_dim=1),
            dim=1,
        )
        > 0
    )
    assert torch.count_nonzero(sample["action_effect"][~noop]) > 0
    assert torch.count_nonzero(sample["all_action_effects"][:, 0]) == 0
