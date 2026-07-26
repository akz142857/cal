"""Tests for the V2-I1 connected-component detection front end and the
integrated agent's persistent self-identification lock."""

import numpy as np

from cal.model.integrated_agent import (
    ACTION_DELTAS,
    IntegratedSelfWorldAgent,
    _SELF_LOCK_STREAK_REQUIRED,
    connected_component_centroids,
)
from cal.model.occupancy import VIEW_RADIUS


def _grid(*rows: str) -> np.ndarray:
    return np.asarray(
        [[1 if char == "#" else 0 for char in row] for row in rows],
        dtype=np.uint8,
    )


def _patch_at(point: np.ndarray) -> np.ndarray:
    size = 2 * VIEW_RADIUS + 1
    patch = np.zeros((size, size), dtype=np.uint8)
    patch[point[1], point[0]] = 1
    return patch


def test_empty_patch_returns_no_detections() -> None:
    patch = _grid("...", "...", "...")

    result = connected_component_centroids(patch, 0, 0)

    assert result.shape == (0, 2)


def test_isolated_cells_are_separate_detections() -> None:
    patch = _grid(
        "#...",
        "....",
        "..#.",
    )

    result = connected_component_centroids(patch, 0, 0)

    assert len(result) == 2
    points = {tuple(row) for row in result}
    assert points == {(0.0, 0.0), (2.0, 2.0)}


def test_touching_cells_merge_into_one_centroid() -> None:
    # A 1x3 horizontal blob: an isolation filter would drop all three
    # cells (each has a neighbor); connected components must report one
    # detection at the blob's centroid instead.
    patch = _grid(
        "....",
        ".###",
        "....",
    )

    result = connected_component_centroids(patch, 0, 0)

    assert len(result) == 1
    assert tuple(result[0]) == (2.0, 1.0)


def test_diagonal_cells_merge_under_eight_connectivity() -> None:
    patch = _grid(
        "#..",
        ".#.",
        "..#",
    )

    merged = connected_component_centroids(patch, 0, 0, connectivity=8)
    separate = connected_component_centroids(patch, 0, 0, connectivity=4)

    assert len(merged) == 1
    assert len(separate) == 3


def test_offset_shifts_absolute_coordinates() -> None:
    patch = _grid("#.", "..")

    result = connected_component_centroids(patch, 5, 9)

    assert tuple(result[0]) == (5.0, 9.0)


def test_dense_wall_is_a_single_stable_blob() -> None:
    # A vertical wall of six adjacent cells (as in the V2-I1 arena) must
    # collapse to one centroid, not six flickering single-cell detections.
    patch = _grid(
        ".#.",
        ".#.",
        ".#.",
        ".#.",
        ".#.",
        ".#.",
    )

    result = connected_component_centroids(patch, 0, 0)

    assert len(result) == 1
    assert tuple(result[0]) == (1.0, 2.5)


def test_self_lock_starts_unset() -> None:
    agent = IntegratedSelfWorldAgent(seed=1)
    point = np.asarray((5, 5))
    agent.update(_patch_at(point), 0)

    assert agent.self_track_identity() is None


def test_self_lock_acquires_and_holds_under_correlated_motion() -> None:
    # A single point moved exactly by the commanded action every step is
    # the cleanest possible controlled-motion signal: the lock must
    # eventually engage and, once engaged, keep reporting the same track
    # index rather than re-deciding every step.
    agent = IntegratedSelfWorldAgent(seed=1)
    point = np.asarray((5, 5))
    agent.update(_patch_at(point), 0)
    rng = np.random.default_rng(0)
    lock_step = None
    identity = None
    for step in range(1, 60):
        action = int(rng.integers(1, 5))
        point = np.clip(point + ACTION_DELTAS[action], 0, 2 * VIEW_RADIUS)
        agent.update(_patch_at(point), action)
        current = agent.self_track_identity()
        if lock_step is None and current is not None:
            lock_step = step
            identity = current
        elif lock_step is not None:
            assert current == identity

    assert lock_step is not None


def test_self_lock_never_engages_under_uncorrelated_action() -> None:
    # The no_action control: the action fed to the agent is unrelated to
    # the point's actual motion, so the RLS control-evidence this lock is
    # built on should never sustain the streak required to engage.
    agent = IntegratedSelfWorldAgent(seed=1)
    point = np.asarray((5, 5))
    agent.update(_patch_at(point), 0)
    motion_rng = np.random.default_rng(0)
    action_rng = np.random.default_rng(99)
    for _ in range(1, 60):
        motion_action = int(motion_rng.integers(1, 5))
        point = np.clip(point + ACTION_DELTAS[motion_action], 0, 2 * VIEW_RADIUS)
        supplied_action = int(action_rng.integers(1, 5))
        agent.update(_patch_at(point), supplied_action)

    assert agent.self_track_identity() is None


def test_streak_bookkeeping_does_not_accumulate_for_a_rival_while_locked() -> None:
    # Regression test: _update_self_lock must not keep updating
    # _leader_track/_leader_streak for some OTHER track while a lock is
    # already held. If it did, a rival that happened to qualify for many
    # steps in the background (while unrelated to the locked track) could
    # instantly re-lock the moment the original lock's track is pruned,
    # with no fresh confirmation streak after the handoff - defeating the
    # whole point of requiring a sustained run before trusting an identity.
    #
    # Point `a` moves under the commanded action and is dropped from the
    # detections once locked, so its track goes stale and is eventually
    # pruned. From the same moment, point `b` (a stand-in for a rival
    # entity) starts moving under the *same* commanded action, i.e. it
    # would itself have qualified for a lock throughout that whole window.
    agent = IntegratedSelfWorldAgent(seed=1)
    a = np.asarray((2, 2))
    b = np.asarray((2 * VIEW_RADIUS - 2, 2))
    agent.update(_patch_at(a), 0)
    rng = np.random.default_rng(0)
    lock_step = None
    freeze_step = None
    for step in range(1, 120):
        action = int(rng.integers(1, 5))
        if freeze_step is None:
            a = np.clip(a + ACTION_DELTAS[action], 0, 2 * VIEW_RADIUS)
            point = a
        else:
            b = np.clip(b + ACTION_DELTAS[action], 0, 2 * VIEW_RADIUS)
            point = b
        previous_lock = agent._self_lock
        agent.update(_patch_at(point), action)
        if lock_step is None and agent.self_track_identity() is not None:
            lock_step = step
            freeze_step = step
        if (
            freeze_step is not None
            and previous_lock is not None
            and agent._self_lock != previous_lock
        ):
            # The instant the original lock is dropped (its track pruned),
            # _leader_streak for the rival must NOT already be at or past
            # the requirement - it must still need a fresh run.
            assert agent._leader_streak < _SELF_LOCK_STREAK_REQUIRED
            break
    else:
        raise AssertionError("original lock was never dropped within the test window")

    assert lock_step is not None
