"""Tests for deterministic world evolution and object interaction."""

from math import cos, sin

from calmodel.env.body import BodyAction
from calmodel.env.world import BodyDiscoveryWorld, CircleObject, WorldConfig


def test_reset_and_seeded_actions_are_reproducible() -> None:
    first = BodyDiscoveryWorld(WorldConfig(seed=7))
    second = BodyDiscoveryWorld(WorldConfig(seed=7))

    first_actions = [first.sample_action() for _ in range(20)]
    second_actions = [second.sample_action() for _ in range(20)]

    assert first.objects == second.objects
    assert first_actions == second_actions


def test_same_action_sequence_produces_same_observations() -> None:
    first = BodyDiscoveryWorld(WorldConfig(seed=11))
    second = BodyDiscoveryWorld(WorldConfig(seed=11))
    actions = [
        BodyAction.SHOULDER_INCREASE,
        BodyAction.ELBOW_DECREASE,
        BodyAction.SHOULDER_DECREASE,
    ]

    first_frames = [first.step(action).observation for action in actions]
    second_frames = [second.step(action).observation for action in actions]

    assert first_frames == second_frames


def test_tip_can_push_an_object() -> None:
    world = BodyDiscoveryWorld(WorldConfig(object_count=0))
    tip_x, tip_y = world.body.points()[-1]
    step = world.body.config.angle_step
    next_tip = (
        world.body.config.base[0]
        + world.body.config.link_lengths[0] * cos(step)
        + world.body.config.link_lengths[1] * cos(step),
        world.body.config.base[1]
        + world.body.config.link_lengths[0] * sin(step)
        + world.body.config.link_lengths[1] * sin(step),
    )
    radius = 0.025
    world.set_objects(
        (
            CircleObject(
                x=(tip_x + next_tip[0]) / 2,
                y=(tip_y + next_tip[1]) / 2,
                radius=radius,
            ),
        )
    )
    before = world.objects[0]

    world.step(BodyAction.SHOULDER_INCREASE)

    assert world.objects[0] != before


def test_ground_truth_is_only_available_through_evaluation_snapshot() -> None:
    world = BodyDiscoveryWorld(WorldConfig(seed=5))

    observation = world.observe()
    snapshot = world.evaluation_snapshot()

    assert not hasattr(observation, "body_mask")
    assert any(value for row in snapshot.masks.body for value in row)


def test_sensor_noise_is_deterministic_per_seed_and_step() -> None:
    config = WorldConfig(
        seed=31,
        vision_noise_probability=0.1,
        proprioception_noise_std=0.05,
        touch_dropout_probability=0.5,
    )
    first = BodyDiscoveryWorld(config)
    second = BodyDiscoveryWorld(config)

    assert first.observe() == second.observe()
    assert (
        first.step(BodyAction.SHOULDER_INCREASE).observation
        == second.step(BodyAction.SHOULDER_INCREASE).observation
    )


def test_full_touch_dropout_removes_all_contacts() -> None:
    world = BodyDiscoveryWorld(
        WorldConfig(object_count=0, touch_dropout_probability=1.0)
    )
    world.set_objects((CircleObject(x=0.61, y=0.53, radius=0.02),))

    assert world.observe().touch == (False, False)


def test_external_object_motion_is_labeled_and_deterministic() -> None:
    config = WorldConfig(
        seed=41,
        object_count=2,
        external_object_motion_probability=1.0,
        external_object_motion_distance=0.08,
    )
    first = BodyDiscoveryWorld(config)
    second = BodyDiscoveryWorld(config)
    before = first.objects

    first_step = first.step(BodyAction.NOOP)
    second_step = second.step(BodyAction.NOOP)

    assert first_step.external_motion
    assert first_step.externally_moved_object in {0, 1}
    assert first.objects != before
    assert first.objects == second.objects
    assert first_step.observation == second_step.observation


def test_isomorphic_distractor_bodies_are_deterministic_and_independent() -> None:
    config = WorldConfig(
        seed=43,
        object_count=0,
        distractor_body_count=2,
        distractor_body_motion_probability=1.0,
    )
    first = BodyDiscoveryWorld(config)
    second = BodyDiscoveryWorld(config)
    initial_states = tuple(body.state for body in first.distractor_bodies)

    first_step = first.step(BodyAction.SHOULDER_INCREASE)
    second_step = second.step(BodyAction.SHOULDER_INCREASE)
    current_states = tuple(body.state for body in first.distractor_bodies)

    assert first.body.config == first.distractor_bodies[0].config
    assert current_states != initial_states
    assert current_states == tuple(
        body.state for body in second.distractor_bodies
    )
    assert first_step.observation == second_step.observation
    assert first.evaluation_snapshot().distractor_body_states == current_states
