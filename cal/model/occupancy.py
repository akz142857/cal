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
            if (
                0 <= old_x < self.grid_size
                and 0 <= old_y < self.grid_size
                and self._may_clear_cell(int(old_x), int(old_y))
            ):
                self._log_odds[old_y, old_x] = min(
                    self._log_odds[old_y, old_x], 0.0
                )
            entity.position = entity.position + entity.velocity
            x, y = np.rint(entity.position).astype(int)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self._log_odds[y, x] = max(self._log_odds[y, x], 1.6)

    def _may_clear_cell(self, x: int, y: int) -> bool:
        """Whether model-driven motion may clear evidence at a grid cell."""

        return True

    def _bounded_camera(self, camera: np.ndarray) -> np.ndarray:
        return np.clip(
            camera,
            VIEW_RADIUS,
            self.grid_size - VIEW_RADIUS - 1,
        )


def bresenham_intermediate_cells(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    """Strictly intermediate cells of the discrete ray from start to end."""

    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    step_x = 1 if x1 > x0 else -1
    step_y = 1 if y1 > y0 else -1
    error = dx - dy
    x, y = x0, y0
    cells: list[tuple[int, int]] = []
    while True:
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x += step_x
        if doubled < dx:
            error += dx
            y += step_y
        if (x, y) == (x1, y1):
            return cells
        cells.append((x, y))


def infer_visibility_from_sensed_occupancy(
    sensed_occupancy: np.ndarray,
) -> np.ndarray:
    """Shadow-cast over a sensed patch to estimate which cells are visible.

    The camera sits at the patch center. A cell is estimated hidden when any
    strictly intermediate cell on its ray is sensed occupied. This uses only
    the agent's own observation and deterministic geometry.
    """

    size = sensed_occupancy.shape[0]
    center = (size // 2, size // 2)
    visibility = np.ones_like(sensed_occupancy, dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if (x, y) == center:
                continue
            for cx, cy in bresenham_intermediate_cells(center, (x, y)):
                if sensed_occupancy[cy, cx] > 0:
                    visibility[y, x] = 0
                    break
    return visibility


class UnprivilegedOccupancyMemory(OccupancyMemory):
    """Occupancy memory whose visibility estimate is self-inferred.

    update() deliberately has no visibility parameter: the only inputs are
    the sensed occupancy patch and the executed action copy.
    """

    def __init__(
        self,
        grid_size: int = 25,
        *,
        active: bool = True,
        infer_occlusion: bool = True,
        seed: int = 0,
    ):
        super().__init__(grid_size, active=active, seed=seed)
        self.infer_occlusion = infer_occlusion

    @property
    def estimated_mac_per_step(self) -> int:
        window = (2 * VIEW_RADIUS + 1) ** 2
        return super().estimated_mac_per_step + window * (2 * VIEW_RADIUS + 2)

    def update(  # type: ignore[override]
        self,
        sensed_occupancy: np.ndarray,
        action: int,
    ) -> None:
        if self.infer_occlusion:
            visibility = infer_visibility_from_sensed_occupancy(
                sensed_occupancy
            )
        else:
            visibility = np.ones_like(sensed_occupancy, dtype=np.uint8)
        self._last_estimated_visibility = visibility
        super().update(sensed_occupancy, visibility, action)
        stamps = getattr(self, "_empty_seen_step", None)
        if stamps is None:
            stamps = np.full(
                (self.grid_size, self.grid_size), -10_000, dtype=np.int64
            )
            self._empty_seen_step = stamps
        x0 = int(self._camera[0] - VIEW_RADIUS)
        y0 = int(self._camera[1] - VIEW_RADIUS)
        size = 2 * VIEW_RADIUS + 1
        empty = (visibility.astype(bool)) & (sensed_occupancy == 0)
        stamps[y0 : y0 + size, x0 : x0 + size][empty] = self._step

    def _may_clear_cell(self, x: int, y: int) -> bool:
        """Only visual evidence, never blind extrapolation, erases memory.

        A cell outside the current window, or estimated occluded inside it,
        keeps its accumulated evidence while the tracked entity is unseen.
        """

        visibility = getattr(self, "_last_estimated_visibility", None)
        if visibility is None:
            return True
        return self._cell_currently_visible(x, y, visibility)

    def _cell_currently_visible(
        self,
        x: int,
        y: int,
        visibility: np.ndarray,
    ) -> bool:
        local_x = x - int(self._camera[0] - VIEW_RADIUS)
        local_y = y - int(self._camera[1] - VIEW_RADIUS)
        size = 2 * VIEW_RADIUS + 1
        if not (0 <= local_x < size and 0 <= local_y < size):
            return False
        return bool(visibility[local_y, local_x])

    REACHABLE_LOG_ODDS = 0.45
    EMPTY_EVIDENCE_FRESH_STEPS = 5

    def _update_entities(self, detections: list[np.ndarray]) -> None:
        super()._update_entities(detections)
        visibility = getattr(self, "_last_estimated_visibility", None)
        if visibility is None:
            return
        corridors = getattr(self, "_corridors", None)
        if corridors is None:
            corridors = {}
            self._corridors = corridors
        # A corridor collapses only when the object is actually re-found:
        # a visible detection lands inside its span. Entity identity churn
        # (timeouts, re-spawns) must not silently kill the reachable set.
        found_keys = set()
        for point in detections:
            x, y = np.rint(point).astype(int)
            for key, corridor in corridors.items():
                along = x if corridor["axis"] == 0 else y
                row = y if corridor["axis"] == 0 else x
                if (
                    row == corridor["row"]
                    and corridor["low"] <= along <= corridor["high"]
                ):
                    found_keys.add(key)
        for key in found_keys:
            corridors.pop(key, None)
        for entity in self._entities:
            if entity.last_seen == self._step or entity.motion_confidence < 2:
                continue
            anchor = np.clip(
                np.rint(entity.position).astype(int),
                0,
                self.grid_size - 1,
            )
            axis = int(np.argmax(np.abs(entity.velocity)))
            key = (axis, int(anchor[1 - axis]))
            if key not in corridors:
                corridors[key] = {
                    "axis": axis,
                    "row": int(anchor[1 - axis]),
                    "low": int(anchor[axis]),
                    "high": int(anchor[axis]),
                }
        for corridor in corridors.values():
            # The unobserved object may keep moving, pause, or bounce, so
            # its reachable set widens symmetrically one cell per step.
            corridor["low"] = max(0, corridor["low"] - 1)
            corridor["high"] = min(
                self.grid_size - 1, corridor["high"] + 1
            )
            self._paint_corridor(corridor, visibility)

    def _paint_corridor(
        self,
        corridor: dict[str, int],
        visibility: np.ndarray,
    ) -> None:
        stamps = getattr(self, "_empty_seen_step", None)
        for position in range(corridor["low"], corridor["high"] + 1):
            if corridor["axis"] == 0:
                x, y = position, corridor["row"]
            else:
                x, y = corridor["row"], position
            if self._cell_currently_visible(x, y, visibility):
                continue
            if (
                stamps is not None
                and self._step - int(stamps[y, x])
                <= self.EMPTY_EVIDENCE_FRESH_STEPS
            ):
                continue
            self._log_odds[y, x] = max(
                self._log_odds[y, x],
                self.REACHABLE_LOG_ODDS,
            )
