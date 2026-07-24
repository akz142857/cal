"""Vision, proprioception, and touch sensors.

Sensors should expose raw measurements. They must not identify body pixels or
external objects in the learner-facing observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, sin
from typing import Callable, Protocol, Sequence

from cal.env.body import ArticulatedBody, Point, Segment

VisionFrame = tuple[tuple[float, ...], ...]
BinaryMask = tuple[tuple[bool, ...], ...]


class CircularObject(Protocol):
    """Structural type required by the sensors."""

    x: float
    y: float
    radius: float


@dataclass(frozen=True, slots=True)
class VisionConfig:
    width: int = 16
    height: int = 16
    background_value: float = 0.0
    body_value: float = 1.0
    object_value: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("vision dimensions must be positive")


@dataclass(frozen=True, slots=True)
class SensorObservation:
    """Learner-facing measurements with no semantic ground-truth labels."""

    vision: VisionFrame
    proprioception: tuple[float, float, float, float]
    touch: tuple[bool, bool]


@dataclass(frozen=True, slots=True)
class EvaluationMasks:
    """Privileged labels used only by evaluation probes."""

    body: BinaryMask
    objects: BinaryMask


class VisionSensor:
    """Rasterize body and objects with identical visual appearance."""

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()

    def observe(
        self,
        body: ArticulatedBody,
        objects: Sequence[CircularObject],
        distractor_bodies: Sequence[ArticulatedBody] = (),
    ) -> VisionFrame:
        """Return occupancy only; body and objects use the same intensity."""

        body_mask = self._body_mask(body)
        object_mask = self._external_mask(objects, distractor_bodies)
        return tuple(
            tuple(
                self.config.body_value
                if body_mask[y][x]
                else (
                    self.config.object_value
                    if object_mask[y][x]
                    else self.config.background_value
                )
                for x in range(self.config.width)
            )
            for y in range(self.config.height)
        )

    def evaluation_masks(
        self,
        body: ArticulatedBody,
        objects: Sequence[CircularObject],
        distractor_bodies: Sequence[ArticulatedBody] = (),
    ) -> EvaluationMasks:
        """Return privileged segmentation labels for evaluation only."""

        return EvaluationMasks(
            body=self._body_mask(body),
            objects=self._external_mask(objects, distractor_bodies),
        )

    def _body_mask(self, body: ArticulatedBody) -> BinaryMask:
        segments = body.segments()
        base, elbow, tip = body.points()
        config = body.config
        pixel_radius = self._pixel_radius

        def occupied(point: Point) -> bool:
            return (
                any(
                    _distance_sq_to_segment(point, segment)
                    <= (config.link_radius + pixel_radius) ** 2
                    for segment in segments
                )
                or _distance_sq(point, base)
                <= (config.joint_radius + pixel_radius) ** 2
                or _distance_sq(point, elbow)
                <= (config.joint_radius + pixel_radius) ** 2
                or _distance_sq(point, tip)
                <= (config.tip_radius + pixel_radius) ** 2
            )

        return self._rasterize(occupied)

    def _object_mask(self, objects: Sequence[CircularObject]) -> BinaryMask:
        pixel_radius = self._pixel_radius

        def occupied(point: Point) -> bool:
            return any(
                _distance_sq(point, (item.x, item.y))
                <= (item.radius + pixel_radius) ** 2
                for item in objects
            )

        return self._rasterize(occupied)

    def _external_mask(
        self,
        objects: Sequence[CircularObject],
        distractor_bodies: Sequence[ArticulatedBody],
    ) -> BinaryMask:
        object_mask = self._object_mask(objects)
        distractor_masks = tuple(
            self._body_mask(body) for body in distractor_bodies
        )
        return tuple(
            tuple(
                object_mask[y][x]
                or any(mask[y][x] for mask in distractor_masks)
                for x in range(self.config.width)
            )
            for y in range(self.config.height)
        )

    @property
    def _pixel_radius(self) -> float:
        """Radius of a circle enclosing half of one rectangular pixel."""

        return 0.5 * hypot(
            1.0 / self.config.width,
            1.0 / self.config.height,
        )

    def _rasterize(self, predicate: Callable[[Point], bool]) -> BinaryMask:
        # Kept as a small pure-Python rasterizer so early experiments remain
        # deterministic and independent of a graphics or array library.
        return tuple(
            tuple(
                predicate(
                    (
                        (x + 0.5) / self.config.width,
                        (y + 0.5) / self.config.height,
                    )
                )
                for x in range(self.config.width)
            )
            for y in range(self.config.height)
        )


class ProprioceptionSensor:
    """Encode joint angles continuously without exposing world coordinates."""

    def observe(
        self, body: ArticulatedBody
    ) -> tuple[float, float, float, float]:
        shoulder = body.state.shoulder_angle
        elbow = body.state.elbow_angle
        return sin(shoulder), cos(shoulder), sin(elbow), cos(elbow)


class TouchSensor:
    """Report contact independently for the two body links."""

    def observe(
        self,
        body: ArticulatedBody,
        objects: Sequence[CircularObject],
    ) -> tuple[bool, bool]:
        radius = body.config.link_radius
        return tuple(
            any(
                _distance_sq_to_segment((item.x, item.y), segment)
                <= (item.radius + radius) ** 2
                for item in objects
            )
            for segment in body.segments()
        )  # type: ignore[return-value]


class SensorSuite:
    """Bundle the three learner-facing sensor modalities."""

    def __init__(self, vision_config: VisionConfig | None = None) -> None:
        self.vision = VisionSensor(vision_config)
        self.proprioception = ProprioceptionSensor()
        self.touch = TouchSensor()

    def observe(
        self,
        body: ArticulatedBody,
        objects: Sequence[CircularObject],
        distractor_bodies: Sequence[ArticulatedBody] = (),
    ) -> SensorObservation:
        return SensorObservation(
            vision=self.vision.observe(body, objects, distractor_bodies),
            proprioception=self.proprioception.observe(body),
            touch=self.touch.observe(body, objects),
        )

    def evaluation_masks(
        self,
        body: ArticulatedBody,
        objects: Sequence[CircularObject],
        distractor_bodies: Sequence[ArticulatedBody] = (),
    ) -> EvaluationMasks:
        return self.vision.evaluation_masks(
            body,
            objects,
            distractor_bodies,
        )


def _distance_sq(first: Point, second: Point) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _distance_sq_to_segment(point: Point, segment: Segment) -> float:
    start, end = segment
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return _distance_sq(point, start)
    projection = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_sq
    projection = max(0.0, min(1.0, projection))
    closest = start[0] + projection * dx, start[1] + projection * dy
    return _distance_sq(point, closest)
