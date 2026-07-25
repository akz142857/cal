"""Probabilistic motion hypotheses for occluded moving entities.

When a tracked moving entity becomes unobserved, a small discrete filter
maintains mutually exclusive hypotheses over (position, direction, pause
countdown) along its travel row. Pause regularities are identified online
from visible stationary episodes; hypotheses are pruned only by the agent's
own evidence: cells it currently estimates as visible and empty.
"""

from __future__ import annotations

import numpy as np


MAX_PAUSE = 16
PRUNE_FACTOR = 1e-4
SMOOTHING = 1e-3


class PauseLearner:
    """Online table of where and for how long the object habitually pauses."""

    def __init__(self) -> None:
        self._durations: dict[int, list[int]] = {}

    MIN_DURATION = 4
    MAX_DURATION = 14

    def record(self, position: int, duration: int) -> None:
        # Below the band: tracker jitter. Above it: a moving entity glued
        # to a static detection by the loose match gate. Both poison the
        # transition model far more than a missed real pause does.
        if not (self.MIN_DURATION <= duration <= self.MAX_DURATION):
            return
        self._durations.setdefault(position, []).append(duration)

    def locations(self) -> dict[int, tuple[int, float]]:
        """Map position -> (expected duration, trigger confidence)."""

        result = {}
        for position, durations in self._durations.items():
            # Observed stationary runs are censored from below (the camera
            # may look away mid-pause), so trust the longest run seen.
            expected = min(MAX_PAUSE, max(durations))
            confidence = min(0.9, len(durations) / (len(durations) + 1.0))
            result[position] = (expected, confidence)
        return result

    @property
    def learned_value_count(self) -> int:
        return 2 * len(self._durations)


class MotionHypothesisFilter:
    """Discrete-state filter over one lost entity's travel line."""

    def __init__(
        self,
        grid_size: int,
        *,
        axis: int,
        row: int,
        start: int,
        direction: int,
        learner: PauseLearner,
    ) -> None:
        self.grid_size = grid_size
        self.axis = axis
        self.row = row
        self.learner = learner
        # belief[x, d, k]: position x, direction index d (0:-1, 1:+1),
        # pause countdown k (0 = moving).
        self.belief = np.zeros((grid_size, 2, MAX_PAUSE + 1))
        d = 1 if direction >= 0 else 0
        anchor = int(np.clip(start, 0, grid_size - 1))
        self.belief[anchor, d, 0] = 1.0
        # Never-shrinking reachable interval: with speed at most one cell
        # per step the object provably stays inside it while unobserved.
        self.reachable_low = anchor
        self.reachable_high = anchor
        self.retired = False

    def predict(self) -> None:
        self.reachable_low = max(0, self.reachable_low - 1)
        self.reachable_high = min(self.grid_size - 1, self.reachable_high + 1)
        table = self.learner.locations()
        successor = np.zeros_like(self.belief)
        moving = self.belief[:, :, 0]
        for x in range(self.grid_size):
            for d in (0, 1):
                mass = moving[x, d]
                if mass <= 0.0:
                    continue
                step = 1 if d == 1 else -1
                nx, nd = x + step, d
                if nx < 0 or nx >= self.grid_size:
                    nd = 1 - d
                    nx = x + (1 if nd == 1 else -1)
                if nx in table:
                    duration, confidence = table[nx]
                    successor[nx, nd, duration] += mass * confidence
                    successor[nx, nd, 0] += mass * (1.0 - confidence)
                else:
                    successor[nx, nd, 0] += mass
        # Paused states count down; the final tick resumes motion in place.
        for k in range(1, MAX_PAUSE + 1):
            successor[:, :, k - 1] += self.belief[:, :, k]
        total = successor.sum()
        if total <= 0.0:
            successor[:, :, 0] = 1.0 / (2 * self.grid_size)
            total = successor.sum()
        successor /= total
        uniform = 1.0 / successor.size
        self.belief = (1.0 - SMOOTHING) * successor + SMOOTHING * uniform

    def observe(
        self,
        *,
        visible_empty: set[int],
        detection: int | None,
    ) -> None:
        if detection is not None:
            # The object is visible: re-anchor instead of dying, so the
            # next disappearance resumes with a tight, current belief.
            previous = getattr(self, "_last_detection", None)
            if previous is not None and previous != detection:
                self._direction = 1 if detection > previous else 0
            self._last_detection = detection
            self.belief[:, :, :] = 0.0
            self.belief[detection, getattr(self, "_direction", 1), 0] = 1.0
            self.reachable_low = detection
            self.reachable_high = detection
            return
        for x in visible_empty:
            if 0 <= x < self.grid_size:
                self.belief[x, :, :] *= PRUNE_FACTOR
        total = self.belief.sum()
        if total <= 0.0:
            self.belief[:, :, 0] = 1.0 / (2 * self.grid_size)
            total = self.belief.sum()
        self.belief /= total

    def marginal(self) -> np.ndarray:
        return self.belief.sum(axis=(1, 2))

    @property
    def state_bytes(self) -> int:
        return self.belief.nbytes

    @property
    def estimated_mac_per_step(self) -> int:
        return int(self.belief.size * 4)
