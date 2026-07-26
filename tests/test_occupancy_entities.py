"""Tests for OccupancyMemory's stale-entity pruning (stale_entity_horizon)."""

import numpy as np

from cal.model.occupancy import OccupancyMemory, _VisualEntity


def _lone_entity(*, motion_confidence: int = 0) -> _VisualEntity:
    return _VisualEntity(
        position=np.asarray((1.0, 1.0)),
        velocity=np.zeros(2),
        last_seen=0,
        motion_confidence=motion_confidence,
    )


def test_default_stale_entity_horizon_is_none() -> None:
    memory = OccupancyMemory()

    assert memory.stale_entity_horizon is None


def test_stale_entity_horizon_rejects_non_positive() -> None:
    try:
        OccupancyMemory(stale_entity_horizon=0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive stale_entity_horizon must raise")


def test_disabled_by_default_never_prunes_stale_entities() -> None:
    # This is what makes M4's own (single-moving-point) worlds behave
    # exactly as before this parameter existed: a never-rematched, never-
    # confident entity would normally look exactly like noise, but with
    # pruning disabled it must survive indefinitely (up to the 40 cap).
    memory = OccupancyMemory()
    memory._entities = [_lone_entity()]

    for _ in range(50):
        memory._step += 1
        memory._update_entities([])

    assert len(memory._entities) == 1


def test_enabled_prunes_a_stale_low_confidence_entity() -> None:
    memory = OccupancyMemory(stale_entity_horizon=5)
    memory._entities = [_lone_entity()]

    for _ in range(10):
        memory._step += 1
        memory._update_entities([])

    assert memory._entities == []


def test_enabled_keeps_a_confident_entity_even_when_stale() -> None:
    # A genuinely-tracked entity that earned real motion confidence must
    # not be evicted just because it went briefly unseen (e.g. occlusion) -
    # only entities that never proved themselves are pruned.
    memory = OccupancyMemory(stale_entity_horizon=5)
    memory._entities = [_lone_entity(motion_confidence=2)]

    for _ in range(10):
        memory._step += 1
        memory._update_entities([])

    assert len(memory._entities) == 1
