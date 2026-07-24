"""Deterministic 2D world for the body-discovery experiment.

The first implementation should own simulation state, apply one body action
per step, update movable objects, and expose observations without semantic
labels. Ground-truth body masks belong only in evaluation output.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, sin, tau
from random import Random
from typing import Iterable

from calmodel.env.body import (
    ArticulatedBody,
    BodyAction,
    BodyConfig,
    BodyState,
    BodyTransition,
)
from calmodel.env.sensors import EvaluationMasks, SensorObservation, SensorSuite


@dataclass(frozen=True, slots=True)
class CircleObject:
    """A visually anonymous movable object in normalized world coordinates."""

    x: float
    y: float
    radius: float = 0.045

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("object radius must be positive")
        if not (
            self.radius <= self.x <= 1.0 - self.radius
            and self.radius <= self.y <= 1.0 - self.radius
        ):
            raise ValueError("object must fit inside the unit square")


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """Configuration for a reproducible body-discovery world."""

    image_size: tuple[int, int] = (16, 16)
    object_count: int = 3
    object_radius: float = 0.045
    body_visual_value: float = 1.0
    object_visual_value: float = 1.0
    vision_noise_probability: float = 0.0
    proprioception_noise_std: float = 0.0
    touch_dropout_probability: float = 0.0
    external_object_motion_probability: float = 0.0
    external_object_motion_distance: float = 0.08
    distractor_body_count: int = 0
    distractor_body_motion_probability: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size dimensions must be positive")
        if self.object_count < 0:
            raise ValueError("object_count cannot be negative")
        if not 0.0 < self.object_radius < 0.5:
            raise ValueError("object_radius must be between zero and 0.5")
        if not all(
            0.0 <= value <= 1.0
            for value in (self.body_visual_value, self.object_visual_value)
        ):
            raise ValueError("visual values must be between zero and one")
        if not 0.0 <= self.vision_noise_probability <= 1.0:
            raise ValueError("vision_noise_probability must be in [0, 1]")
        if self.proprioception_noise_std < 0.0:
            raise ValueError("proprioception_noise_std cannot be negative")
        if not 0.0 <= self.touch_dropout_probability <= 1.0:
            raise ValueError("touch_dropout_probability must be in [0, 1]")
        if not 0.0 <= self.external_object_motion_probability <= 1.0:
            raise ValueError(
                "external_object_motion_probability must be in [0, 1]"
            )
        if self.external_object_motion_distance <= 0.0:
            raise ValueError("external_object_motion_distance must be positive")
        if self.distractor_body_count < 0:
            raise ValueError("distractor_body_count cannot be negative")
        if not 0.0 <= self.distractor_body_motion_probability <= 1.0:
            raise ValueError(
                "distractor_body_motion_probability must be in [0, 1]"
            )


@dataclass(frozen=True, slots=True)
class WorldStep:
    """Result of applying one action to the world."""

    index: int
    transition: BodyTransition
    observation: SensorObservation
    externally_moved_object: int | None = None

    @property
    def external_motion(self) -> bool:
        """Privileged event label for evaluation, not a learner observation."""

        return self.externally_moved_object is not None


@dataclass(frozen=True, slots=True)
class EvaluationSnapshot:
    """Privileged simulator state that must not be passed to the learner."""

    step_index: int
    body_state: BodyState
    objects: tuple[CircleObject, ...]
    distractor_body_states: tuple[BodyState, ...]
    masks: EvaluationMasks


class BodyDiscoveryWorld:
    """Closed-loop environment for discovering an unlabeled body."""

    def __init__(
        self,
        config: WorldConfig | None = None,
        body_config: BodyConfig | None = None,
    ) -> None:
        self.config = config or WorldConfig()
        width, height = self.config.image_size
        from calmodel.env.sensors import VisionConfig

        self.body = ArticulatedBody(body_config)
        self.sensors = SensorSuite(
            VisionConfig(
                width=width,
                height=height,
                body_value=self.config.body_visual_value,
                object_value=self.config.object_visual_value,
            )
        )
        self._rng = Random(self.config.seed)
        self._objects: tuple[CircleObject, ...] = ()
        self._distractor_bodies: tuple[ArticulatedBody, ...] = ()
        self._step_index = 0
        self._episode_seed = self.config.seed
        self.reset()

    @property
    def objects(self) -> tuple[CircleObject, ...]:
        return self._objects

    @property
    def distractor_bodies(self) -> tuple[ArticulatedBody, ...]:
        """External isomorphic bodies, exposed only as simulator state."""

        return self._distractor_bodies

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def actions(self) -> tuple[BodyAction, ...]:
        return tuple(BodyAction)

    def reset(self, seed: int | None = None) -> SensorObservation:
        """Reset body, random generator, and objects deterministically."""

        self._episode_seed = self.config.seed if seed is None else seed
        self._rng = Random(self._episode_seed)
        self.body.reset()
        self._step_index = 0
        self._objects = self._spawn_objects()
        self._distractor_bodies = self._spawn_distractor_bodies()
        return self.observe()

    def observe(self) -> SensorObservation:
        """Return only measurements available to the learning system."""

        raw = self.sensors.observe(
            self.body,
            self._objects,
            self._distractor_bodies,
        )
        return self._apply_sensor_noise(raw)

    def step(self, action: BodyAction | str) -> WorldStep:
        """Apply an action, push touched objects, and observe the new state."""

        transition = self.body.apply(action)
        self._push_objects(transition)
        externally_moved_object = self._move_external_object()
        self._move_distractor_bodies()
        self._step_index += 1
        return WorldStep(
            index=self._step_index,
            transition=transition,
            observation=self.observe(),
            externally_moved_object=externally_moved_object,
        )

    def sample_action(self) -> BodyAction:
        """Sample from the world's seeded generator for reproducible rollouts."""

        return self._rng.choice(self.actions)

    def set_objects(self, objects: Iterable[CircleObject]) -> SensorObservation:
        """Install an explicit scene, primarily for controlled experiments."""

        candidate = tuple(objects)
        self._validate_non_overlapping(candidate)
        self._objects = candidate
        return self.observe()

    def evaluation_snapshot(self) -> EvaluationSnapshot:
        """Return labels and state for metrics, never for representation input."""

        return EvaluationSnapshot(
            step_index=self._step_index,
            body_state=self.body.state,
            objects=self._objects,
            distractor_body_states=tuple(
                body.state for body in self._distractor_bodies
            ),
            masks=self.sensors.evaluation_masks(
                self.body,
                self._objects,
                self._distractor_bodies,
            ),
        )

    def _spawn_distractor_bodies(self) -> tuple[ArticulatedBody, ...]:
        """Create visually isomorphic arms without consuming action RNG."""

        random = Random((self._episode_seed + 1) * 3_000_017)
        bodies: list[ArticulatedBody] = []
        for _ in range(self.config.distractor_body_count):
            body = ArticulatedBody(self.body.config)
            body.reset(
                BodyState(
                    shoulder_angle=random.uniform(
                        *body.config.shoulder_limits
                    ),
                    elbow_angle=random.uniform(*body.config.elbow_limits),
                )
            )
            bodies.append(body)
        return tuple(bodies)

    def _spawn_objects(self) -> tuple[CircleObject, ...]:
        objects: list[CircleObject] = []
        radius = self.config.object_radius
        attempts = 0
        while len(objects) < self.config.object_count and attempts < 10_000:
            attempts += 1
            candidate = CircleObject(
                x=self._rng.uniform(radius, 1.0 - radius),
                y=self._rng.uniform(radius, 1.0 - radius),
                radius=radius,
            )
            if self._object_clear_of_body(candidate) and all(
                _objects_separated(candidate, existing) for existing in objects
            ):
                objects.append(candidate)
        if len(objects) != self.config.object_count:
            raise RuntimeError("could not place requested objects without overlap")
        return tuple(objects)

    def _object_clear_of_body(self, candidate: CircleObject) -> bool:
        return not any(self.sensors.touch.observe(self.body, (candidate,)))

    @staticmethod
    def _validate_non_overlapping(objects: tuple[CircleObject, ...]) -> None:
        for index, item in enumerate(objects):
            if any(
                not _objects_separated(item, other)
                for other in objects[index + 1 :]
            ):
                raise ValueError("objects must not overlap")

    def _push_objects(self, transition: BodyTransition) -> None:
        delta_x = transition.current_tip[0] - transition.previous_tip[0]
        delta_y = transition.current_tip[1] - transition.previous_tip[1]
        if delta_x == 0.0 and delta_y == 0.0:
            return

        updated: list[CircleObject] = []
        for item in self._objects:
            contact_distance = item.radius + self.body.config.tip_radius
            touches_tip = (
                hypot(
                    item.x - transition.current_tip[0],
                    item.y - transition.current_tip[1],
                )
                <= contact_distance
            )
            if touches_tip:
                updated.append(
                    CircleObject(
                        x=_clamp(item.x + delta_x, item.radius, 1.0 - item.radius),
                        y=_clamp(item.y + delta_y, item.radius, 1.0 - item.radius),
                        radius=item.radius,
                    )
                )
            else:
                updated.append(item)
        self._objects = tuple(updated)

    def _move_external_object(self) -> int | None:
        """Apply a step-indexed exogenous motion without consuming action RNG."""

        if (
            not self._objects
            or self.config.external_object_motion_probability == 0.0
        ):
            return None
        event_index = self._step_index + 1
        random = Random(
            (self._episode_seed + 1) * 2_000_003
            + event_index * 193_939
        )
        if random.random() >= self.config.external_object_motion_probability:
            return None

        object_index = random.randrange(len(self._objects))
        item = self._objects[object_index]
        angle = random.random() * tau
        distance = self.config.external_object_motion_distance
        candidate_x = _clamp(
            item.x + distance * cos(angle),
            item.radius,
            1.0 - item.radius,
        )
        candidate_y = _clamp(
            item.y + distance * sin(angle),
            item.radius,
            1.0 - item.radius,
        )
        if candidate_x == item.x and candidate_y == item.y:
            candidate_x = _clamp(
                item.x - distance * cos(angle),
                item.radius,
                1.0 - item.radius,
            )
            candidate_y = _clamp(
                item.y - distance * sin(angle),
                item.radius,
                1.0 - item.radius,
            )
        updated = list(self._objects)
        updated[object_index] = CircleObject(
            x=candidate_x,
            y=candidate_y,
            radius=item.radius,
        )
        self._objects = tuple(updated)
        return object_index

    def _move_distractor_bodies(self) -> None:
        """Move each external arm independently of learner and action RNG."""

        if (
            not self._distractor_bodies
            or self.config.distractor_body_motion_probability == 0.0
        ):
            return
        event_index = self._step_index + 1
        for body_index, body in enumerate(self._distractor_bodies):
            random = Random(
                (self._episode_seed + 1) * 4_000_037
                + event_index * 389_003
                + body_index * 97_409
            )
            if (
                random.random()
                >= self.config.distractor_body_motion_probability
            ):
                continue
            body.apply(random.choice(self.actions))

    def _apply_sensor_noise(
        self,
        observation: SensorObservation,
    ) -> SensorObservation:
        if (
            self.config.vision_noise_probability == 0.0
            and self.config.proprioception_noise_std == 0.0
            and self.config.touch_dropout_probability == 0.0
        ):
            return observation
        random = Random(
            (self._episode_seed + 1) * 1_000_003
            + self._step_index * 97_409
        )
        vision = tuple(
            tuple(
                1.0 - value
                if random.random() < self.config.vision_noise_probability
                else value
                for value in row
            )
            for row in observation.vision
        )
        proprioception = tuple(
            _clamp(
                value + random.gauss(0.0, self.config.proprioception_noise_std),
                -1.0,
                1.0,
            )
            for value in observation.proprioception
        )
        touch = tuple(
            value
            and random.random() >= self.config.touch_dropout_probability
            for value in observation.touch
        )
        return SensorObservation(
            vision=vision,
            proprioception=proprioception,  # type: ignore[arg-type]
            touch=touch,  # type: ignore[arg-type]
        )


def _objects_separated(first: CircleObject, second: CircleObject) -> bool:
    return hypot(first.x - second.x, first.y - second.y) > (
        first.radius + second.radius
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
