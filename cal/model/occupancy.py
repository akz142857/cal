"""Small allocentric probabilistic occupancy memory for V2-M4."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np


VIEW_RADIUS = 5
MOTION_DELTAS = np.asarray(
    ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
    dtype=np.int64,
)


@dataclass(slots=True)
class _VisualEntity:
    position: np.ndarray
    velocity: np.ndarray
    last_seen: int
    motion_confidence: int = 0


class OccupancyMemory:
    """Fuse local visual occupancy using integrated visual-agent motion."""

    def __init__(self, grid_size: int = 25, *, active: bool = True, seed: int = 0):
        self.grid_size = grid_size
        self.active = active
        self._rng = np.random.default_rng(seed)
        self._log_odds = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._camera = np.asarray((grid_size // 2, grid_size // 2), dtype=np.int64)
        self._entities: list[_VisualEntity] = []
        self._step = 0

    @property
    def learnable_parameter_count(self) -> int:
        return self.grid_size * self.grid_size + 8 * 5

    @property
    def active_state_bytes(self) -> int:
        return self._log_odds.nbytes + 40 * 5 * 8

    @property
    def estimated_mac_per_step(self) -> int:
        return (2 * VIEW_RADIUS + 1) ** 2 * 12

    def choose_action(self) -> int:
        if not self.active:
            return int(self._rng.integers(0, 5))
        probability = self.probability()
        uncertainty = 1.0 - 2.0 * np.abs(probability - 0.5)
        candidates = []
        for action in range(1, 5):
            camera = self._bounded_camera(self._camera + MOTION_DELTAS[action])
            x, y = camera
            patch = uncertainty[
                y - VIEW_RADIUS : y + VIEW_RADIUS + 1,
                x - VIEW_RADIUS : x + VIEW_RADIUS + 1,
            ]
            frontier = float(patch.sum())
            # Tiny deterministic tie-break makes exploration reproducible.
            candidates.append((frontier, -action, action))
        return max(candidates)[2]

    def update(
        self,
        local_occupancy: np.ndarray,
        local_visibility: np.ndarray,
        action: int,
    ) -> None:
        expected = (2 * VIEW_RADIUS + 1, 2 * VIEW_RADIUS + 1)
        if local_occupancy.shape != expected or local_visibility.shape != expected:
            raise ValueError("unexpected local observation shape")
        self._step += 1
        self._camera = self._bounded_camera(
            self._camera + MOTION_DELTAS[int(action)]
        )
        self._log_odds *= 0.999
        x0 = int(self._camera[0] - VIEW_RADIUS)
        y0 = int(self._camera[1] - VIEW_RADIUS)
        visible = local_visibility.astype(bool)
        patch = self._log_odds[y0 : y0 + expected[0], x0 : x0 + expected[1]]
        patch[visible & (local_occupancy > 0)] += 2.2
        patch[visible & (local_occupancy == 0)] -= 2.2
        np.clip(self._log_odds, -8.0, 8.0, out=self._log_odds)
        occupied_visible = visible & (local_occupancy > 0)
        isolated = occupied_visible.copy()
        for y, x in np.argwhere(occupied_visible):
            y0n, y1n = max(0, y - 1), min(expected[0], y + 2)
            x0n, x1n = max(0, x - 1), min(expected[1], x + 2)
            if int(occupied_visible[y0n:y1n, x0n:x1n].sum()) > 1:
                isolated[y, x] = False
        detections = np.argwhere(isolated)
        global_detections = [
            np.asarray((x0 + item[1], y0 + item[0]), dtype=np.float64)
            for item in detections
        ]
        self._update_entities(global_detections)

    def probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self._log_odds))

    def mean_entropy(self) -> float:
        probability = self.probability()
        clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
        entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
        return float(entropy.mean() / log(2.0))

    def _update_entities(self, detections: list[np.ndarray]) -> None:
        used: set[int] = set()
        # Match moving hypotheses first and against their predicted location;
        # otherwise dense static screen cells can steal the moving detection.
        ordered_entities = sorted(
            self._entities,
            key=lambda item: float(np.linalg.norm(item.velocity)),
            reverse=True,
        )
        for entity in ordered_entities:
            predicted = entity.position + entity.velocity
            candidates = [
                (float(np.linalg.norm(point - predicted)), index, point)
                for index, point in enumerate(detections)
                if index not in used
            ]
            if not candidates:
                continue
            distance, index, point = min(candidates)
            if distance <= 2.1:
                displacement = point - entity.position
                speed = float(np.linalg.norm(displacement))
                if 0.5 <= speed <= 1.5:
                    entity.motion_confidence = min(
                        8, entity.motion_confidence + 1
                    )
                else:
                    entity.motion_confidence = max(
                        0, entity.motion_confidence - 1
                    )
                entity.velocity = 0.10 * entity.velocity + 0.90 * displacement
                entity.position = point
                entity.last_seen = self._step
                used.add(index)
        for index, point in enumerate(detections):
            if index not in used and len(self._entities) < 40:
                self._entities.append(
                    _VisualEntity(point, np.zeros(2), self._step)
                )
        for entity in self._entities:
            if (
                entity.last_seen == self._step
                or self._step - entity.last_seen > 30
                or entity.motion_confidence < 2
            ):
                continue
            old_x, old_y = np.rint(entity.position).astype(int)
            if 0 <= old_x < self.grid_size and 0 <= old_y < self.grid_size:
                self._log_odds[old_y, old_x] = min(
                    self._log_odds[old_y, old_x], 0.0
                )
            entity.position = entity.position + entity.velocity
            x, y = np.rint(entity.position).astype(int)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self._log_odds[y, x] = max(self._log_odds[y, x], 1.6)

    def _bounded_camera(self, camera: np.ndarray) -> np.ndarray:
        return np.clip(
            camera,
            VIEW_RADIUS,
            self.grid_size - VIEW_RADIUS - 1,
        )
