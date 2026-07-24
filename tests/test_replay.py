"""Tests for unlabeled trajectory recording, storage, and replay."""

import json
from dataclasses import replace

import pytest

from cal.env.body import BodyAction
from cal.env.world import BodyDiscoveryWorld, WorldConfig
from cal.learning.replay import (
    ReplayMismatchError,
    Trajectory,
    iter_compressed_experiences,
    load_trajectory,
    record_action_trajectory,
    record_random_trajectory,
    replay_trajectory,
    save_trajectory,
    verify_compressed_trajectory,
)


def test_recorded_experiences_are_temporally_continuous() -> None:
    trajectory = record_random_trajectory(
        BodyDiscoveryWorld(WorldConfig(seed=17)),
        12,
    )

    assert len(trajectory) == 12
    assert trajectory.experiences[0].observation == trajectory.initial_observation
    assert all(
        current.next_observation == following.observation
        for current, following in zip(
            trajectory.experiences,
            trajectory.experiences[1:],
        )
    )


def test_full_action_repeat_probability_holds_initial_random_action() -> None:
    trajectory = record_random_trajectory(
        BodyDiscoveryWorld(WorldConfig(seed=17)),
        12,
        action_repeat_probability=1.0,
    )

    assert len({item.action for item in trajectory.experiences}) == 1
    replay_trajectory(trajectory)


def test_explicit_action_seed_shares_schedule_across_scenes() -> None:
    first = record_random_trajectory(
        BodyDiscoveryWorld(WorldConfig(seed=17)),
        12,
        action_repeat_probability=0.75,
        action_seed=99,
    )
    second = record_random_trajectory(
        BodyDiscoveryWorld(WorldConfig(seed=18)),
        12,
        action_repeat_probability=0.75,
        action_seed=99,
    )

    assert tuple(item.action for item in first.experiences) == tuple(
        item.action for item in second.experiences
    )
    assert first.initial_observation.vision != second.initial_observation.vision


def test_saved_trajectory_contains_no_privileged_labels(tmp_path: object) -> None:
    path = tmp_path / "trajectory.json"  # type: ignore[operator]
    trajectory = record_random_trajectory(
        BodyDiscoveryWorld(WorldConfig(seed=3)),
        5,
    )

    save_trajectory(trajectory, path)
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert "body_mask" not in text
    assert "object_mask" not in text
    assert set(payload["initial_observation"]) == {
        "vision",
        "proprioception",
        "touch",
    }


def test_json_round_trip_preserves_trajectory(tmp_path: object) -> None:
    path = tmp_path / "trajectory.json"  # type: ignore[operator]
    original = record_random_trajectory(
        BodyDiscoveryWorld(WorldConfig(seed=9)),
        8,
    )

    save_trajectory(original, path)
    restored = load_trajectory(path)

    assert restored == original


def test_compressed_stream_round_trip_and_incremental_replay(
    tmp_path: object,
) -> None:
    path = tmp_path / "trajectory.jsonl.gz"  # type: ignore[operator]
    original = record_random_trajectory(
        BodyDiscoveryWorld(WorldConfig(seed=14)),
        12,
    )

    save_trajectory(original, path)

    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert tuple(iter_compressed_experiences(path)) == original.experiences
    assert verify_compressed_trajectory(path) == 12
    assert load_trajectory(path) == original


def test_strict_replay_matches_every_observation() -> None:
    trajectory = record_random_trajectory(
        BodyDiscoveryWorld(WorldConfig(seed=23)),
        20,
    )

    result = replay_trajectory(trajectory)

    assert len(result.observations) == len(trajectory) + 1
    assert result.observations[-1] == trajectory.experiences[-1].next_observation


def test_explicit_action_schedule_is_recorded_and_replays_strictly() -> None:
    actions = (
        BodyAction.SHOULDER_INCREASE,
        BodyAction.SHOULDER_INCREASE,
        BodyAction.ELBOW_DECREASE,
        BodyAction.NOOP,
    )
    trajectory = record_action_trajectory(
        BodyDiscoveryWorld(WorldConfig(seed=31)),
        actions,
    )

    assert tuple(item.action for item in trajectory.experiences) == actions
    assert replay_trajectory(trajectory).observations[-1] == (
        trajectory.experiences[-1].next_observation
    )


def test_strict_replay_matches_external_object_motion() -> None:
    trajectory = record_random_trajectory(
        BodyDiscoveryWorld(
            WorldConfig(
                seed=29,
                external_object_motion_probability=0.5,
            )
        ),
        20,
    )

    saved_result = replay_trajectory(trajectory)

    assert saved_result.observations[-1] == (
        trajectory.experiences[-1].next_observation
    )


def test_strict_replay_matches_isomorphic_distractor_motion() -> None:
    trajectory = record_random_trajectory(
        BodyDiscoveryWorld(
            WorldConfig(
                seed=47,
                object_count=0,
                distractor_body_count=2,
                distractor_body_motion_probability=1.0,
            )
        ),
        12,
    )

    saved_result = replay_trajectory(trajectory)

    assert saved_result.observations[-1] == (
        trajectory.experiences[-1].next_observation
    )


def test_strict_replay_rejects_tampered_initial_observation() -> None:
    original = record_random_trajectory(
        BodyDiscoveryWorld(WorldConfig(seed=4)),
        2,
    )
    altered_proprioception = (0.5, 0.5, 0.5, 0.5)
    altered_initial = replace(
        original.initial_observation,
        proprioception=altered_proprioception,
    )
    altered_first = replace(
        original.experiences[0],
        observation=altered_initial,
    )
    tampered = Trajectory(
        seed=original.seed,
        world_config=original.world_config,
        body_config=original.body_config,
        initial_observation=altered_initial,
        experiences=(altered_first, *original.experiences[1:]),
    )

    with pytest.raises(ReplayMismatchError):
        replay_trajectory(tampered)
