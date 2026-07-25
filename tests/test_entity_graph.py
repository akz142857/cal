"""Tests for OnlineEntityGraph, including the reacquisition_window parameter."""

import numpy as np

from cal.model.entity_graph import OnlineEntityGraph


def _train_moving_track(learner: OnlineEntityGraph) -> np.ndarray:
    """Give the sole track a few real, informative steps of motion."""

    position = np.asarray(((5.0, 5.0),))
    learner.reset(position)
    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    for _ in range(4):
        position = position + np.asarray((0.3, 0.0))
        learner.update(position, action)
    return position


def test_default_reacquisition_window_is_six() -> None:
    learner = OnlineEntityGraph(4)

    assert learner.reacquisition_window == 6


def test_reacquisition_window_rejects_non_positive() -> None:
    try:
        OnlineEntityGraph(4, reacquisition_window=0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive reacquisition_window must raise")


def test_default_window_loses_identity_after_long_occlusion() -> None:
    learner = OnlineEntityGraph(4)
    position = _train_moving_track(learner)
    trained_evidence = learner._tracks[0].control_evidence
    assert trained_evidence > 0.0

    # The object keeps moving at the same learned rate while unobserved -
    # only the detections are withheld, not the object's real motion.
    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    empty = np.zeros((0, 2))
    for _ in range(7):  # exceeds the default 6-step window
        position = position + np.asarray((0.3, 0.0))
        learner.update(empty, action)
    position = position + np.asarray((0.3, 0.0))
    learner.update(position, action)

    assert len(learner._tracks) == 2
    fresh = max(learner._tracks, key=lambda track: track.index)
    assert fresh.control_evidence == 0.0


def test_wider_window_preserves_identity_across_the_same_occlusion() -> None:
    learner = OnlineEntityGraph(4, reacquisition_window=20)
    position = _train_moving_track(learner)
    trained_evidence = learner._tracks[0].control_evidence
    assert trained_evidence > 0.0

    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    empty = np.zeros((0, 2))
    for _ in range(7):  # would exceed the default window, not this one
        position = position + np.asarray((0.3, 0.0))
        learner.update(empty, action)
    position = position + np.asarray((0.3, 0.0))
    learner.update(position, action)

    # Identity survived: one track, carrying the theta it already learned
    # rather than a freshly initialized one (control_evidence naturally
    # decays while unmatched, so a strict floor isn't the right check -
    # what matters is that it is still the same track, not a new index 1).
    assert len(learner._tracks) == 1
    assert learner._tracks[0].index == 0
    assert np.linalg.norm(learner._tracks[0].theta[:, 0]) > 0.1


def test_entity_graph_consumes_unordered_detections() -> None:
    learner = OnlineEntityGraph(4)
    points = np.asarray(((4.0, 4.0), (2.0, 2.0), (2.0, 4.0)))
    learner.reset(points)
    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    learner.update(points + np.asarray((-0.7, 0.0)), action)

    assert len(learner.positions()) == 3
    assert learner.learnable_parameter_count < 100_000
