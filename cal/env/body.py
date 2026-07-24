"""Agent body state, kinematics, and discrete action definitions.

Coordinates are normalized to the unit square. The first body is deliberately
small: a fixed base with two rotary joints. It is complex enough to generate
action-dependent visual and proprioceptive changes without introducing a
physics engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import cos, sin

Point = tuple[float, float]
Segment = tuple[Point, Point]


class BodyAction(str, Enum):
    """Discrete motor commands available to the learner."""

    NOOP = "noop"
    SHOULDER_INCREASE = "shoulder_increase"
    SHOULDER_DECREASE = "shoulder_decrease"
    ELBOW_INCREASE = "elbow_increase"
    ELBOW_DECREASE = "elbow_decrease"


@dataclass(frozen=True, slots=True)
class BodyConfig:
    """Geometry, limits, and motor resolution of the articulated body."""

    base: Point = (0.5, 0.5)
    link_lengths: tuple[float, float] = (0.22, 0.18)
    shoulder_limits: tuple[float, float] = (-2.6, 2.6)
    elbow_limits: tuple[float, float] = (-2.4, 2.4)
    angle_step: float = 0.16
    link_radius: float = 0.018
    joint_radius: float = 0.024
    tip_radius: float = 0.025
    shoulder_enabled: bool = True
    elbow_enabled: bool = True

    def __post_init__(self) -> None:
        if not all(0.0 <= coordinate <= 1.0 for coordinate in self.base):
            raise ValueError("base coordinates must be inside the unit square")
        if any(length <= 0.0 for length in self.link_lengths):
            raise ValueError("link lengths must be positive")
        if self.angle_step <= 0.0:
            raise ValueError("angle_step must be positive")
        if any(radius <= 0.0 for radius in self.radii):
            raise ValueError("body radii must be positive")
        for name, limits in (
            ("shoulder_limits", self.shoulder_limits),
            ("elbow_limits", self.elbow_limits),
        ):
            if limits[0] >= limits[1]:
                raise ValueError(f"{name} must be ordered low to high")

    @property
    def radii(self) -> tuple[float, float, float]:
        return self.link_radius, self.joint_radius, self.tip_radius


@dataclass(frozen=True, slots=True)
class BodyState:
    """Minimal dynamic state; elbow angle is relative to the first link."""

    shoulder_angle: float = 0.0
    elbow_angle: float = 0.0


@dataclass(frozen=True, slots=True)
class BodyTransition:
    """State change caused by one motor command."""

    action: BodyAction
    previous: BodyState
    current: BodyState
    previous_tip: Point
    current_tip: Point


class ArticulatedBody:
    """A deterministic two-link body with a fixed base."""

    def __init__(self, config: BodyConfig | None = None) -> None:
        self.config = config or BodyConfig()
        self._state = BodyState()

    @property
    def state(self) -> BodyState:
        return self._state

    def reset(self, state: BodyState | None = None) -> BodyState:
        """Reset to a supplied valid state or the neutral pose."""

        candidate = state or BodyState()
        self._state = BodyState(
            shoulder_angle=_clamp(
                candidate.shoulder_angle, self.config.shoulder_limits
            ),
            elbow_angle=_clamp(candidate.elbow_angle, self.config.elbow_limits),
        )
        return self._state

    def apply(self, action: BodyAction | str) -> BodyTransition:
        """Apply one action and return the complete kinematic transition."""

        resolved = BodyAction(action)
        previous = self._state
        previous_tip = self.points(previous)[-1]
        shoulder = previous.shoulder_angle
        elbow = previous.elbow_angle

        if resolved is BodyAction.SHOULDER_INCREASE and self.config.shoulder_enabled:
            shoulder += self.config.angle_step
        elif (
            resolved is BodyAction.SHOULDER_DECREASE
            and self.config.shoulder_enabled
        ):
            shoulder -= self.config.angle_step
        elif resolved is BodyAction.ELBOW_INCREASE and self.config.elbow_enabled:
            elbow += self.config.angle_step
        elif resolved is BodyAction.ELBOW_DECREASE and self.config.elbow_enabled:
            elbow -= self.config.angle_step

        self._state = BodyState(
            shoulder_angle=_clamp(shoulder, self.config.shoulder_limits),
            elbow_angle=_clamp(elbow, self.config.elbow_limits),
        )
        return BodyTransition(
            action=resolved,
            previous=previous,
            current=self._state,
            previous_tip=previous_tip,
            current_tip=self.points()[-1],
        )

    def points(self, state: BodyState | None = None) -> tuple[Point, Point, Point]:
        """Return base, elbow, and end-effector points for a body state."""

        resolved = state or self._state
        base_x, base_y = self.config.base
        first_length, second_length = self.config.link_lengths
        elbow = (
            base_x + first_length * cos(resolved.shoulder_angle),
            base_y + first_length * sin(resolved.shoulder_angle),
        )
        forearm_angle = resolved.shoulder_angle + resolved.elbow_angle
        tip = (
            elbow[0] + second_length * cos(forearm_angle),
            elbow[1] + second_length * sin(forearm_angle),
        )
        return self.config.base, elbow, tip

    def segments(self, state: BodyState | None = None) -> tuple[Segment, Segment]:
        """Return upper-arm and forearm line segments."""

        base, elbow, tip = self.points(state)
        return (base, elbow), (elbow, tip)


def _clamp(value: float, limits: tuple[float, float]) -> float:
    return max(limits[0], min(limits[1], value))
