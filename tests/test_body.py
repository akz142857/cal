"""Tests for deterministic articulated-body kinematics."""

from calmodel.env.body import ArticulatedBody, BodyAction, BodyConfig, BodyState


def test_shoulder_action_moves_tip_and_preserves_elbow_angle() -> None:
    body = ArticulatedBody()
    initial_tip = body.points()[-1]

    transition = body.apply(BodyAction.SHOULDER_INCREASE)

    assert transition.previous_tip == initial_tip
    assert transition.current_tip != initial_tip
    assert transition.current.shoulder_angle == body.config.angle_step
    assert transition.current.elbow_angle == 0.0


def test_joint_angles_are_clamped_to_body_limits() -> None:
    config = BodyConfig(
        shoulder_limits=(-0.2, 0.2),
        elbow_limits=(-0.1, 0.1),
        angle_step=0.5,
    )
    body = ArticulatedBody(config)

    body.apply(BodyAction.SHOULDER_INCREASE)
    body.apply(BodyAction.ELBOW_DECREASE)

    assert body.state == BodyState(shoulder_angle=0.2, elbow_angle=-0.1)


def test_reset_clamps_supplied_state() -> None:
    body = ArticulatedBody()

    state = body.reset(BodyState(shoulder_angle=99.0, elbow_angle=-99.0))

    assert state.shoulder_angle == body.config.shoulder_limits[1]
    assert state.elbow_angle == body.config.elbow_limits[0]


def test_disabled_joint_ignores_its_motor_commands() -> None:
    body = ArticulatedBody(BodyConfig(elbow_enabled=False))

    transition = body.apply(BodyAction.ELBOW_INCREASE)

    assert transition.current.elbow_angle == 0.0
    assert transition.previous_tip == transition.current_tip
