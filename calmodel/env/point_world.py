"""Minimal anonymous visual-point world for V2-M1.

The learner receives only a binary image.  Simulator identities and the
controlled-point position are exposed through ``evaluation_state`` and must
remain on the evaluation side of the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from random import Random

import numpy as np


class PointAction(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


ACTION_DELTAS: dict[PointAction, tuple[int, int]] = {
    PointAction.NOOP: (0, 0),
    PointAction.LEFT: (-1, 0),
    PointAction.RIGHT: (1, 0),
    PointAction.UP: (0, -1),
    PointAction.DOWN: (0, 1),
}


@dataclass(frozen=True, slots=True)
class PointWorldConfig:
    grid_size: int = 21
    distractor_count: int = 4
    occlusion_period: int = 19
    occlusion_length: int = 2
    distractor_motion_probability: float = 0.45
    action_cost: float = 0.01

    def __post_init__(self) -> None:
        if self.grid_size < 11:
            raise ValueError("grid_size must be at least 11")
        if not 1 <= self.distractor_count <= 4:
            raise ValueError("distractor_count must be in [1, 4]")
        if self.occlusion_period <= self.occlusion_length + 2:
            raise ValueError("occlusion period must exceed its length")
        if not 0.0 <= self.distractor_motion_probability <= 1.0:
            raise ValueError("invalid distractor motion probability")


@dataclass(frozen=True, slots=True)
class PointEvaluationState:
    controlled_position: tuple[int, int]
    distractor_positions: tuple[tuple[int, int], ...]
    controlled_visible: bool


class AnonymousPointWorld:
    """A controlled point among visually identical autonomous points."""

    def __init__(
        self,
        config: PointWorldConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.config = config or PointWorldConfig()
        self._seed = seed
        self._rng = Random(seed)
        self._step = 0
        self._controlled = (0, 0)
        self._distractors: list[tuple[int, int]] = []
        self._distractor_anchors: tuple[tuple[int, int], ...] = ()
        self.reset(seed)

    @property
    def actions(self) -> tuple[PointAction, ...]:
        return tuple(PointAction)

    @property
    def step_index(self) -> int:
        return self._step

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._seed = self._seed if seed is None else seed
        self._rng = Random(self._seed)
        self._step = 0
        n = self.config.grid_size
        # Stratified starts make nearest-neighbour visual tracking well posed
        # without exposing the strata or identities to the learner.
        self._controlled = (n // 2, n // 2)
        anchors = (
            (2, 2),
            (n - 3, 2),
            (2, n - 3),
            (n - 3, n - 3),
        )
        self._distractor_anchors = anchors[: self.config.distractor_count]
        self._distractors = list(anchors[: self.config.distractor_count])
        return self.observe()

    def observe(self) -> np.ndarray:
        frame = np.zeros(
            (self.config.grid_size, self.config.grid_size),
            dtype=np.uint8,
        )
        if not self._controlled_occluded():
            x, y = self._controlled
            frame[y, x] = 1
        for x, y in self._distractors:
            frame[y, x] = 1
        return frame

    def step(self, action: PointAction) -> tuple[np.ndarray, float]:
        action = PointAction(action)
        dx, dy = ACTION_DELTAS[action]
        self._controlled = self._bounded_move(self._controlled, dx, dy)
        next_distractors = []
        autonomous = tuple(ACTION_DELTAS.values())
        for index, position in enumerate(self._distractors):
            if self._rng.random() < self.config.distractor_motion_probability:
                ddx, ddy = self._rng.choice(autonomous)
                candidate = self._bounded_move(position, ddx, ddy)
                anchor = self._distractor_anchors[index]
                # Autonomous points roam locally.  This avoids turning M1 into
                # the crossing/re-identification problem reserved for M2.
                if (
                    abs(candidate[0] - anchor[0]) <= 3
                    and abs(candidate[1] - anchor[1]) <= 3
                ):
                    position = candidate
            next_distractors.append(position)
        self._distractors = next_distractors
        self._step += 1
        cost = 0.0 if action == PointAction.NOOP else self.config.action_cost
        return self.observe(), cost

    def evaluation_state(self) -> PointEvaluationState:
        return PointEvaluationState(
            controlled_position=self._controlled,
            distractor_positions=tuple(self._distractors),
            controlled_visible=not self._controlled_occluded(),
        )

    def _controlled_occluded(self) -> bool:
        phase = self._step % self.config.occlusion_period
        return 0 < phase <= self.config.occlusion_length

    def _bounded_move(
        self,
        position: tuple[int, int],
        dx: int,
        dy: int,
    ) -> tuple[int, int]:
        high = self.config.grid_size - 2
        return (
            min(high, max(1, position[0] + dx)),
            min(high, max(1, position[1] + dy)),
        )
