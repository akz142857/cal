"""Tests for learner-facing and evaluation-only sensor boundaries."""

from dataclasses import fields

from calmodel.env.body import ArticulatedBody, BodyAction
from calmodel.env.sensors import SensorObservation, SensorSuite, VisionConfig
from calmodel.env.world import CircleObject


def test_observation_does_not_contain_segmentation_labels() -> None:
    names = {field.name for field in fields(SensorObservation)}

    assert names == {"vision", "proprioception", "touch"}


def test_body_and_object_use_same_visual_intensity() -> None:
    body = ArticulatedBody()
    sensors = SensorSuite(VisionConfig(width=32, height=32))
    item = CircleObject(x=0.15, y=0.15, radius=0.06)

    frame = sensors.observe(body, (item,)).vision
    values = {value for row in frame for value in row}

    assert values == {0.0, 1.0}


def test_visual_leak_control_can_distinguish_body_and_objects() -> None:
    body = ArticulatedBody()
    sensors = SensorSuite(
        VisionConfig(
            width=32,
            height=32,
            body_value=1.0,
            object_value=0.25,
        )
    )
    item = CircleObject(x=0.15, y=0.15, radius=0.06)

    frame = sensors.observe(body, (item,)).vision
    values = {value for row in frame for value in row}

    assert values == {0.0, 0.25, 1.0}


def test_touch_reports_link_contact() -> None:
    body = ArticulatedBody()
    sensors = SensorSuite()
    # Neutral first link runs from (0.5, 0.5) to (0.72, 0.5).
    item = CircleObject(x=0.61, y=0.53, radius=0.02)

    touch = sensors.observe(body, (item,)).touch

    assert touch == (True, False)


def test_evaluation_masks_are_separate_and_nonempty() -> None:
    body = ArticulatedBody()
    sensors = SensorSuite(VisionConfig(width=32, height=32))
    item = CircleObject(x=0.15, y=0.15, radius=0.06)

    masks = sensors.evaluation_masks(body, (item,))

    assert any(value for row in masks.body for value in row)
    assert any(value for row in masks.objects for value in row)


def test_isomorphic_distractor_is_visual_external_not_proprioceptive() -> None:
    body = ArticulatedBody()
    distractor = ArticulatedBody()
    distractor.apply(BodyAction.SHOULDER_INCREASE)
    sensors = SensorSuite(VisionConfig(width=32, height=32))

    without = sensors.observe(body, ())
    with_distractor = sensors.observe(body, (), (distractor,))
    masks = sensors.evaluation_masks(body, (), (distractor,))

    assert with_distractor.vision != without.vision
    assert with_distractor.proprioception == without.proprioception
    assert with_distractor.touch == without.touch
    assert any(value for row in masks.objects for value in row)
