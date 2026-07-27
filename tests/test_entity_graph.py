"""Tests for OnlineEntityGraph, including the reacquisition_window parameter."""

import numpy as np

from cal.model.entity_graph import OnlineEntityGraph, _Edge


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


def test_default_identity_switch_penalty_weight_is_zero() -> None:
    learner = OnlineEntityGraph(4)

    assert learner.identity_switch_penalty_weight == 0.0


def test_identity_switch_penalty_weight_rejects_negative() -> None:
    try:
        OnlineEntityGraph(4, identity_switch_penalty_weight=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative identity_switch_penalty_weight must raise")


def test_identity_switch_penalty_is_zero_when_disabled() -> None:
    # Default weight 0.0 must be a hard no-op, not just numerically small -
    # this is what makes the default-parameter equivalence to every
    # existing caller exact rather than approximate.
    learner = OnlineEntityGraph(4)
    detection = np.asarray((5.0, 0.0))
    predicted_positions = {
        0: np.asarray((4.0, 0.0)),
        1: np.asarray((5.0, 0.0)),  # a rival predicting exactly at detection
    }

    penalty = learner._identity_switch_penalty(
        0, detection, distance=1.0, predicted_positions=predicted_positions
    )

    assert penalty == 0.0


def test_identity_switch_penalty_is_zero_when_own_track_is_closest() -> None:
    learner = OnlineEntityGraph(4, identity_switch_penalty_weight=1.0)
    detection = np.asarray((5.0, 0.0))
    predicted_positions = {
        0: np.asarray((4.0, 0.0)),  # own: distance 1.0
        1: np.asarray((2.0, 0.0)),  # rival: distance 3.0, farther away
    }

    penalty = learner._identity_switch_penalty(
        0, detection, distance=1.0, predicted_positions=predicted_positions
    )

    assert penalty == 0.0


def test_identity_switch_penalty_grows_with_rival_gap() -> None:
    learner = OnlineEntityGraph(4, identity_switch_penalty_weight=1.0)
    detection = np.asarray((5.0, 0.0))
    predicted_positions = {
        0: np.asarray((4.0, 0.0)),  # own: distance 1.0
        1: np.asarray((5.0, 0.0)),  # rival: distance 0.0, a perfect fit
    }

    penalty = learner._identity_switch_penalty(
        0, detection, distance=1.0, predicted_positions=predicted_positions
    )

    assert penalty == 1.0 * (1.0 / 0.32) ** 2
    # A weaker rival advantage produces a smaller (but still positive)
    # penalty - the term scales with the gap, not just a fixed hit for any
    # closer rival.
    weaker_predicted_positions = {0: np.asarray((4.0, 0.0)), 1: np.asarray((4.6, 0.0))}
    weaker_penalty = learner._identity_switch_penalty(
        0, detection, distance=1.0, predicted_positions=weaker_predicted_positions
    )
    assert 0.0 < weaker_penalty < penalty


def test_default_drift_reset_after_is_none() -> None:
    learner = OnlineEntityGraph(4)

    assert learner.drift_reset_after is None


def test_drift_reset_after_rejects_non_positive() -> None:
    try:
        OnlineEntityGraph(4, drift_reset_after=0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive drift_reset_after must raise")


def test_disabled_by_default_never_resets_a_negative_evidence_track() -> None:
    # This is what makes M4's own worlds behave exactly as before this
    # parameter existed: a track whose control_evidence has gone negative
    # (as a genuinely mis-associated one would) must keep coasting on its
    # existing theta/autonomous_velocity indefinitely, not get reset.
    learner = OnlineEntityGraph(4)
    _train_moving_track(learner)
    track = learner._tracks[0]
    track.control_evidence = -1.0
    poisoned_theta = track.theta.copy()

    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    empty = np.zeros((0, 2))
    for _ in range(30):
        learner.update(empty, action)

    assert np.array_equal(learner._tracks[0].theta, poisoned_theta)


def test_enabled_resets_a_negative_evidence_track_after_the_horizon() -> None:
    learner = OnlineEntityGraph(4, reacquisition_window=50, drift_reset_after=5)
    _train_moving_track(learner)
    track = learner._tracks[0]
    track.control_evidence = -1.0
    assert np.linalg.norm(track.theta) > 0.0

    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    empty = np.zeros((0, 2))
    # age = step - last_seen exceeds drift_reset_after=5 for the first
    # time on the 6th unmatched step; stop right there, before further
    # per-step decay (which applies regardless of the reset) would move
    # control_evidence/probability away from the just-reset neutral prior.
    for _ in range(6):
        learner.update(empty, action)

    reset_track = learner._tracks[0]
    assert reset_track.index == track.index
    assert np.array_equal(reset_track.theta, np.zeros_like(reset_track.theta))
    assert np.array_equal(
        reset_track.autonomous_velocity, np.zeros_like(reset_track.autonomous_velocity)
    )
    assert reset_track.control_evidence == 0.0
    assert reset_track.probability == 0.5


def test_enabled_does_not_reset_a_track_with_non_negative_evidence() -> None:
    # A track whose control_evidence hasn't gone negative is treated as a
    # real, ongoing occlusion (what reacquisition_window exists to
    # protect), not a corrupted one - it must not be reset just for going
    # briefly unseen.
    learner = OnlineEntityGraph(4, reacquisition_window=50, drift_reset_after=5)
    _train_moving_track(learner)
    track = learner._tracks[0]
    poisoned_theta = track.theta.copy()
    assert track.control_evidence >= 0.0

    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    empty = np.zeros((0, 2))
    for _ in range(10):
        learner.update(empty, action)

    assert np.array_equal(learner._tracks[0].theta, poisoned_theta)


def test_reset_clears_stale_rigid_edges_so_propagation_cannot_undo_it() -> None:
    # Regression test: _reset_track_state must drop this track's pairwise
    # geometry statistics. Otherwise a stale rigid edge to a confident
    # neighbor lets _propagate_membership (called later in the same
    # update()) immediately raise the just-reset probability back up,
    # defeating the reset before any caller ever observes the neutral
    # prior it was supposed to produce.
    learner = OnlineEntityGraph(4, reacquisition_window=50, drift_reset_after=5)
    corrupted = learner._new_track(np.asarray((0.0, 0.0)))
    corrupted.control_evidence = -1.0
    confident_neighbor = learner._new_track(np.asarray((1.0, 0.0)))
    confident_neighbor.probability = 0.99
    learner._tracks = [corrupted, confident_neighbor]
    # A rigid edge (low variance, enough samples) between the two tracks,
    # as _update_edges would build from many consistent co-observations.
    edge_key = tuple(sorted((corrupted.index, confident_neighbor.index)))
    edge = _Edge()
    for _ in range(12):
        edge.update(1.0)
    learner._edges[edge_key] = edge

    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    empty = np.zeros((0, 2))
    for _ in range(6):
        learner.update(empty, action)

    assert edge_key not in learner._edges
    reset_track = next(
        t for t in learner._tracks if t.index == corrupted.index
    )
    assert reset_track.probability == 0.5


def test_default_confidence_adaptive_gating_weight_is_zero() -> None:
    learner = OnlineEntityGraph(4)

    assert learner.confidence_adaptive_gating_weight == 0.0


def test_confidence_adaptive_gating_weight_rejects_negative() -> None:
    try:
        OnlineEntityGraph(4, confidence_adaptive_gating_weight=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "negative confidence_adaptive_gating_weight must raise"
        )


def test_motion_cost_scale_is_fixed_when_disabled() -> None:
    # Default weight 0.0 must be a hard no-op: the scale is exactly 0.32
    # regardless of covariance or action, matching every existing
    # caller's behavior bit-for-bit, not just numerically.
    learner = OnlineEntityGraph(4)
    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    confident_track = learner._new_track(np.asarray((0.0, 0.0)))
    confident_track.covariance = np.eye(4) * 0.001
    uncertain_track = learner._new_track(np.asarray((0.0, 0.0)))
    uncertain_track.covariance = np.eye(4) * 100.0

    assert learner._motion_cost_scale(confident_track, action) == 0.32
    assert learner._motion_cost_scale(uncertain_track, action) == 0.32


def test_motion_cost_scale_tightens_for_a_less_confident_track() -> None:
    # Direction confirmed by a calibration-set sweep, not the original
    # hypothesis (see the docstring on _motion_cost_scale): a confident
    # (low-covariance) track gets a WIDER tolerance; an uncertain
    # (high-covariance) track gets a TIGHTER one.
    learner = OnlineEntityGraph(4, confidence_adaptive_gating_weight=1.0)
    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    confident_track = learner._new_track(np.asarray((0.0, 0.0)))
    confident_track.covariance = np.eye(4) * 0.1
    # A brand-new track's own default covariance (eye(4) * 6.0) is the
    # reference point: it must behave identically to the disabled scale,
    # so a never-yet-matched track is never harder to match than baseline
    # whether or not this mechanism is enabled.
    reference_track = learner._new_track(np.asarray((0.0, 0.0)))
    assert np.array_equal(reference_track.covariance, np.eye(4) * 6.0)
    uncertain_track = learner._new_track(np.asarray((0.0, 0.0)))
    uncertain_track.covariance = np.eye(4) * 20.0

    confident_scale = learner._motion_cost_scale(confident_track, action)
    reference_scale = learner._motion_cost_scale(reference_track, action)
    uncertain_scale = learner._motion_cost_scale(uncertain_track, action)

    assert reference_scale == 0.32
    assert uncertain_scale < reference_scale < confident_scale


def test_motion_cost_scale_does_not_discriminate_on_a_stay_action() -> None:
    # A "stay" (all-zero) action makes action @ covariance @ action == 0
    # for every track regardless of individual confidence - the formula
    # must degrade to a single uniform (bounded, non-crashing) value on
    # that step rather than dividing by zero or otherwise blowing up.
    learner = OnlineEntityGraph(4, confidence_adaptive_gating_weight=1.0)
    stay_action = np.zeros(4)
    confident_track = learner._new_track(np.asarray((0.0, 0.0)))
    confident_track.covariance = np.eye(4) * 0.1
    uncertain_track = learner._new_track(np.asarray((0.0, 0.0)))
    uncertain_track.covariance = np.eye(4) * 20.0

    confident_scale = learner._motion_cost_scale(confident_track, stay_action)
    uncertain_scale = learner._motion_cost_scale(uncertain_track, stay_action)

    assert confident_scale == uncertain_scale
    assert 0.0 < confident_scale < float("inf")


def test_motion_cost_scale_does_not_crash_on_extreme_covariance() -> None:
    # RLS covariance can "wind up" over hundreds of steps in a rarely
    # excited action dimension (forgetting divides it every step in every
    # direction, but only shrinks it back along the currently-excited
    # one). Without a clamp on the exponent, this drives `gap` far enough
    # negative that exp() underflows to exactly 0.0, which then makes
    # _probabilistic_assign's `distance / scale` raise ZeroDivisionError.
    learner = OnlineEntityGraph(4, confidence_adaptive_gating_weight=1.0)
    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    wound_up_track = learner._new_track(np.asarray((0.0, 0.0)))
    wound_up_track.covariance = np.eye(4) * 1e8

    scale = learner._motion_cost_scale(wound_up_track, action)

    assert 0.0 < scale < float("inf")


def test_motion_cost_scale_does_not_crash_on_extreme_weight() -> None:
    # Symmetrically, a large weight against a near-zero-variance track
    # pushes the exponent far positive; without a clamp exp() raises
    # OverflowError.
    learner = OnlineEntityGraph(4, confidence_adaptive_gating_weight=200.0)
    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    confident_track = learner._new_track(np.asarray((0.0, 0.0)))
    confident_track.covariance = np.eye(4) * 1e-6

    scale = learner._motion_cost_scale(confident_track, action)

    assert 0.0 < scale < float("inf")


def test_entity_graph_consumes_unordered_detections() -> None:
    learner = OnlineEntityGraph(4)
    points = np.asarray(((4.0, 4.0), (2.0, 2.0), (2.0, 4.0)))
    learner.reset(points)
    action = np.asarray((1.0, 0.0, 0.0, 0.0))
    learner.update(points + np.asarray((-0.7, 0.0)), action)

    assert len(learner.positions()) == 3
    assert learner.learnable_parameter_count < 100_000
