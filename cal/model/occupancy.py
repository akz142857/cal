"""Small allocentric probabilistic occupancy memory for V2-M4."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any

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

    def __init__(
        self,
        grid_size: int = 25,
        *,
        active: bool = True,
        seed: int = 0,
        stale_entity_horizon: int | None = None,
    ):
        self.grid_size = grid_size
        self.active = active
        self._rng = np.random.default_rng(seed)
        self._log_odds = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._camera = np.asarray((grid_size // 2, grid_size // 2), dtype=np.int64)
        self._entities: list[_VisualEntity] = []
        self._step = 0
        if stale_entity_horizon is not None and stale_entity_horizon < 1:
            raise ValueError("stale_entity_horizon must be positive")
        # None (default) preserves the original behavior exactly: entities
        # are never pruned by staleness alone, only capped at 40 total.
        # Callers whose world has more simultaneous objects than this
        # tracker's native calibration (one moving point) may opt into
        # pruning entities that never earn real motion confidence, so they
        # stop permanently occupying match priority and MAX_FILTERS slots.
        self.stale_entity_horizon = stale_entity_horizon

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
        # Entities are never expired by default (unlike OnlineEntityGraph's
        # tracks, which gained pruning during the V2-I1 investigation): a
        # one-off detection that's never matched again - noise, or a
        # momentarily isolated cell from an unrelated object - stays in the
        # list forever, up to the 40-entity cap, permanently competing for
        # match priority (sorted by velocity, so a spurious high-velocity
        # entity could out-rank a genuine one for a nearby detection) and
        # for the 4 MAX_FILTERS motion-hypothesis slots via LRU eviction.
        # This is a real cost in a world with several simultaneous objects
        # (V2-I1), but stale_entity_horizon defaults to None (disabled) so
        # M4's own single-moving-point world, whose passing development
        # and one-shot holdout results were produced without this pruning,
        # is completely unaffected unless a caller opts in.
        if self.stale_entity_horizon is not None:
            self._entities = [
                entity
                for entity in self._entities
                if not (
                    self._step - entity.last_seen > self.stale_entity_horizon
                    and entity.motion_confidence < 2
                )
            ]
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


def sense_via_line_of_sight(
    camera: tuple[int, int],
    truth: np.ndarray,
    *,
    view_radius: int = VIEW_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """Shadow-cast ground truth into a local sensed+visibility patch.

    World-simulator side of discrete line-of-sight occlusion: a cell is
    visible iff no strictly intermediate cell on its ray from the camera is
    truly occupied, and sensed marks cells that are both visible and truly
    occupied. Shared by every world that renders this occlusion model
    around a camera cell over ground truth it alone has access to; compare
    `infer_visibility_from_sensed_occupancy` below, which is the agent-side
    analogue over its own sensed (not ground-truth) patch.
    """

    size = 2 * view_radius + 1
    sensed = np.zeros((size, size), dtype=np.uint8)
    visibility = np.ones((size, size), dtype=np.uint8)
    x0, y0 = camera[0] - view_radius, camera[1] - view_radius
    for local_y in range(size):
        for local_x in range(size):
            x, y = x0 + local_x, y0 + local_y
            if (x, y) != camera:
                for cx, cy in bresenham_intermediate_cells(camera, (x, y)):
                    if truth[cy, cx]:
                        visibility[local_y, local_x] = 0
                        break
            if visibility[local_y, local_x] and truth[y, x]:
                sensed[local_y, local_x] = 1
    return sensed, visibility


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
        stale_entity_horizon: int | None = None,
    ):
        super().__init__(
            grid_size,
            active=active,
            seed=seed,
            stale_entity_horizon=stale_entity_horizon,
        )
        self.infer_occlusion = infer_occlusion

    @property
    def learnable_parameter_count(self) -> int:
        # Base grid/tracker values plus the online pause table's worst case
        # (one duration/confidence pair per column).
        return super().learnable_parameter_count + 2 * self.grid_size

    @property
    def active_state_bytes(self) -> int:
        # Worst case: MAX_FILTERS beliefs of grid x 2 x (MAX_PAUSE+1)
        # float64 states, the sensed/visibility caches, and a bounded
        # allowance for the pause table and tracking dicts.
        from cal.model.motion_hypotheses import MAX_PAUSE

        window = (2 * VIEW_RADIUS + 1) ** 2
        filters = (
            self.MAX_FILTERS * self.grid_size * 2 * (MAX_PAUSE + 1) * 8
        )
        return (
            super().active_state_bytes
            + filters
            + 2 * window
            + 2 * self.grid_size * 8
            + 1024
        )

    @property
    def estimated_mac_per_step(self) -> int:
        from cal.model.motion_hypotheses import MAX_PAUSE

        window = (2 * VIEW_RADIUS + 1) ** 2
        ray_casting = window * (2 * VIEW_RADIUS + 2)
        filter_states = self.grid_size * 2 * (MAX_PAUSE + 1)
        filters = self.MAX_FILTERS * filter_states * 4
        painting = self.MAX_FILTERS * self.grid_size * 8
        return super().estimated_mac_per_step + ray_casting + filters + painting

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
        self._last_sensed_occupancy = sensed_occupancy
        super().update(sensed_occupancy, visibility, action)

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

    MAX_FILTERS = 4
    STATIC_LOG_ODDS = 2.0
    PAINT_FLOOR = 0.02
    PAINT_CEILING = 0.85
    SUPPORT_LOG_ODDS = 0.45

    def _update_entities(self, detections: list[np.ndarray]) -> None:
        super()._update_entities(detections)
        visibility = getattr(self, "_last_estimated_visibility", None)
        if visibility is None:
            return
        from cal.model.motion_hypotheses import (
            MotionHypothesisFilter,
            PauseLearner,
        )

        if not hasattr(self, "_filters"):
            self._filters: dict[tuple[int, int], Any] = {}
            self._filter_last_touched: dict[tuple[int, int], int] = {}
            self._pause_learner = PauseLearner()
            self._stationary_runs: dict[int, tuple[int, int, int]] = {}
            self._entity_history: dict[int, int] = {}
        self._learn_pauses()
        for entity in self._entities:
            key = id(entity)
            speed = float(np.linalg.norm(entity.velocity))
            if entity.motion_confidence >= 2 and speed >= 0.5:
                axis = int(np.argmax(np.abs(entity.velocity)))
                direction = 1 if entity.velocity[axis] >= 0 else -1
                self._entity_history[key] = (axis, direction)
            if entity.last_seen == self._step:
                continue
            if key not in self._entity_history:
                continue
            axis, direction = self._entity_history[key]
            anchor = np.clip(
                np.rint(entity.position).astype(int),
                0,
                self.grid_size - 1,
            )
            filter_key = (axis, int(anchor[1 - axis]))
            if filter_key not in self._filters:
                if len(self._filters) >= self.MAX_FILTERS:
                    # Evict the least-recently-touched filter rather than
                    # refusing to track this entity: without this, once
                    # MAX_FILTERS distinct (axis, row) keys have ever been
                    # minted the slots never free up (filters have no other
                    # expiry), silently disabling permanence tracking for
                    # every subsequently occluded entity/location.
                    oldest_key = min(
                        self._filters,
                        key=lambda k: self._filter_last_touched.get(k, -1),
                    )
                    del self._filters[oldest_key]
                    self._filter_last_touched.pop(oldest_key, None)
                self._filters[filter_key] = MotionHypothesisFilter(
                    self.grid_size,
                    axis=axis,
                    row=int(anchor[1 - axis]),
                    start=int(anchor[axis]),
                    direction=direction,
                    learner=self._pause_learner,
                )
            self._filter_last_touched[filter_key] = self._step
        self._advance_filters(detections, visibility)

    def _learn_pauses(self) -> None:
        """Record visible stationary episodes into the pause table."""

        for entity in self._entities:
            key = id(entity)
            if entity.last_seen != self._step:
                continue
            x, y = np.rint(entity.position).astype(int)
            run = self._stationary_runs.get(key)
            if run is None:
                self._stationary_runs[key] = (int(x), int(y), 1)
            else:
                run_x, run_y, count = run
                if (int(x), int(y)) == (run_x, run_y):
                    self._stationary_runs[key] = (run_x, run_y, count + 1)
                else:
                    # Only a clean single-cell resume marks a genuine pause;
                    # multi-cell jumps are rematches, not resumed motion.
                    displacement = abs(int(x) - run_x) + abs(int(y) - run_y)
                    if displacement == 1:
                        axis = 0 if run_y == int(y) else 1
                        along = run_x if axis == 0 else run_y
                        self._pause_learner.record(axis, int(along), count)
                    self._stationary_runs[key] = (int(x), int(y), 1)

    def _advance_filters(
        self,
        detections: list[np.ndarray],
        visibility: np.ndarray,
    ) -> None:
        detection_cells = {
            (int(point[0]), int(point[1]))
            for point in (np.rint(p).astype(int) for p in detections)
        }
        for key in list(self._filters):
            hypothesis = self._filters[key]
            hypothesis.predict()
            axis, row = key
            visible_empty = set()
            detection = None
            for position in range(self.grid_size):
                x, y = (
                    (position, row) if axis == 0 else (row, position)
                )
                if (x, y) in detection_cells:
                    detection = position
                if self._cell_currently_visible(x, y, visibility):
                    local_x = x - int(self._camera[0] - VIEW_RADIUS)
                    local_y = y - int(self._camera[1] - VIEW_RADIUS)
                    if self._last_sensed_occupancy[local_y, local_x] == 0:
                        visible_empty.add(position)
            hypothesis.observe(
                visible_empty=visible_empty,
                detection=detection,
            )
            if detection is not None:
                # Actively reacquired this step: protect it from LRU
                # eviction in _update_entities ahead of filters that have
                # been coasting on prediction alone.
                self._filter_last_touched[key] = self._step
            self._paint_marginal(key, hypothesis, visibility)

    def _paint_marginal(
        self,
        key: tuple[int, int],
        hypothesis: Any,
        visibility: np.ndarray,
    ) -> None:
        axis, row = key
        marginal = hypothesis.marginal()
        for position in range(self.grid_size):
            x, y = (position, row) if axis == 0 else (row, position)
            if self._cell_currently_visible(x, y, visibility):
                continue
            if self._log_odds[y, x] > self.STATIC_LOG_ODDS:
                continue
            probability = float(
                np.clip(marginal[position], self.PAINT_FLOOR, self.PAINT_CEILING)
            )
            log_odds = float(np.log(probability / (1.0 - probability)))
            # The never-shrinking reachable interval provably contains the
            # unobserved object, so it carries a moderate floor even where
            # the calibrated marginal has been diluted by timing mismatch;
            # learned pause regularities raise concentrated cells above it.
            if (
                hypothesis.reachable_low
                <= position
                <= hypothesis.reachable_high
            ):
                log_odds = max(log_odds, self.SUPPORT_LOG_ODDS)
            self._log_odds[y, x] = log_odds
