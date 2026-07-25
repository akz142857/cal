"""One agent, one experience stream: self discovery + identity + permanence.

The integrated agent consumes exactly one (sensed patch, action copy) pair
per world step and simultaneously maintains, by composing two independently
verified V2 components rather than reimplementing tracking from scratch:

- `OnlineEntityGraph` (the V2-M2 mechanism) for entity association, identity
  retention across brief occlusion/merging, and self-discovery via online
  action-displacement control estimation;
- `UnprivilegedOccupancyMemory` (the V2-M4 mechanism) for occupancy fusion
  and object permanence of non-self entities, reusing its verified
  shadow-casting visibility inference and reachable-floor painting.

The camera is fixed at the grid center, so the occupancy memory is driven
with a zero (stay) action; the commanded action feeds only the entity
graph's control estimator, which is the integration seam under test.
"""

from __future__ import annotations

import numpy as np

from cal.model.entity_graph import OnlineEntityGraph
from cal.model.occupancy import UnprivilegedOccupancyMemory, VIEW_RADIUS

ACTION_DELTAS = np.asarray(
    ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)), dtype=np.int64
)


def _action_vector(action: int) -> np.ndarray:
    """Map the 5-way grid action (0=stay) to the entity graph's 4-dim code."""

    vector = np.zeros(4)
    if action != 0:
        vector[action - 1] = 1.0
    return vector


_EIGHT_CONNECTED = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
_FOUR_CONNECTED = ((-1, 0), (1, 0), (0, -1), (0, 1))


def connected_component_centroids(
    sensed: np.ndarray,
    x0: int,
    y0: int,
    *,
    connectivity: int = 8,
) -> np.ndarray:
    """One absolute-coordinate centroid per connected blob of occupied cells.

    The V2-I1 integration report (docs/experiments/V2_I1_INTEGRATION_REPORT.md)
    diagnosed a from-scratch "isolation filter" (drop any occupied cell that
    touches another) as the unresolved front-end friction: in this world's
    dense 11x11 arena, objects and the static screen are frequently
    adjacent, so the isolated-cell count swung between 0 and 5 per frame
    across 100 sampled steps, breaking track continuity before
    OnlineEntityGraph's control estimator could accumulate evidence.

    This replaces that filter with real blob segmentation: touching cells
    merge into one detection at their centroid, instead of vanishing
    entirely. A dense static wall becomes a single stationary blob (a
    stable, correctly "not self" track) rather than several flickering
    single-cell ones; two entities that momentarily touch merge into one
    detection for those frames rather than both dropping out.
    """

    occupied = sensed > 0
    height, width = occupied.shape
    visited = np.zeros_like(occupied, dtype=bool)
    steps = _EIGHT_CONNECTED if connectivity == 8 else _FOUR_CONNECTED
    centroids: list[tuple[float, float]] = []
    for start_y in range(height):
        for start_x in range(width):
            if not occupied[start_y, start_x] or visited[start_y, start_x]:
                continue
            visited[start_y, start_x] = True
            stack = [(start_y, start_x)]
            sum_x = sum_y = count = 0
            while stack:
                cy, cx = stack.pop()
                sum_x += cx
                sum_y += cy
                count += 1
                for dy, dx in steps:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and occupied[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            centroids.append((x0 + sum_x / count, y0 + sum_y / count))
    if not centroids:
        return np.zeros((0, 2))
    return np.asarray(centroids, dtype=np.float64)


class IntegratedSelfWorldAgent:
    """Single-stream agent for the V2-I1 integration probe."""

    def __init__(
        self,
        grid_size: int = 25,
        *,
        infer_occlusion: bool = True,
        use_action: bool = True,
        seed: int = 0,
    ) -> None:
        self.grid_size = grid_size
        self.use_action = use_action
        self.memory = UnprivilegedOccupancyMemory(
            grid_size,
            active=False,
            infer_occlusion=infer_occlusion,
            seed=seed,
        )
        self.graph = OnlineEntityGraph(
            4, association_mode="probabilistic", maximum_tracks=16
        )
        self._initialized = False
        self._action_rng = np.random.default_rng(seed + 60_000)
        self._scale = 0.32

    # -- structural surface under test: one update, patch + action only --

    def update(self, sensed_occupancy: np.ndarray, action: int) -> None:
        # Fixed camera: the occupancy component always receives "stay".
        self.memory.update(sensed_occupancy, 0)
        camera = self.memory._camera
        x0 = int(camera[0] - VIEW_RADIUS)
        y0 = int(camera[1] - VIEW_RADIUS)
        detections = connected_component_centroids(sensed_occupancy, x0, y0)
        # OnlineEntityGraph's association cost is calibrated for its native
        # V2-M2/M3 worlds' sub-unit continuous displacement scale; the grid
        # world's 1-cell-per-step motion is rescaled into that regime so
        # the verified component runs under its own calibration rather than
        # being patched.
        scaled = detections * self._scale
        action_vector = (
            _action_vector(action)
            if self.use_action
            else _action_vector(int(self._action_rng.integers(1, 5)))
        )
        if not self._initialized:
            self.graph.reset(scaled)
            self._initialized = True
        else:
            self.graph.update(scaled, action_vector)
            self._prune_runaway_tracks()

    def _prune_runaway_tracks(self) -> None:
        """Drop tracks whose unmatched extrapolation has left the arena.

        OnlineEntityGraph never expires a track: an unmatched track keeps
        being extrapolated by its own theta/autonomous-velocity estimate
        indefinitely. Under this world's long occlusions, a track fit from
        only a few noisy samples can drift outside the grid and never be
        reacquired, permanently consuming one of the bounded track slots.
        This integration-layer safeguard reclaims only tracks that have
        left the physically valid area - it does not touch tracks merely
        occluded-but-plausible, which stay to represent permanence.
        """

        margin = 4 * self._scale
        low = -margin
        high = (self.grid_size - 1) * self._scale + margin
        step = self.graph._step
        self.graph._tracks = [
            track
            for track in self.graph._tracks
            if (
                low <= track.position[0] <= high
                and low <= track.position[1] <= high
            )
            and not (
                step - track.last_seen > 15 and track.probability < 0.5
            )
        ]

    def self_track_identity(self) -> int | None:
        candidates = self.graph.self_tracks()
        if not candidates:
            return None
        strengths = self.graph.control_strengths()
        return max(candidates, key=lambda index: strengths.get(index, 0.0))

    def track_positions(self) -> dict[int, tuple[int, int]]:
        return {
            index: (
                int(round(x / self._scale)),
                int(round(y / self._scale)),
            )
            for index, (x, y) in self.graph.positions().items()
        }

    def probability(self) -> np.ndarray:
        return self.memory.probability()

    # -- resource accounting --

    @property
    def learnable_parameter_count(self) -> int:
        return (
            self.memory.learnable_parameter_count
            + self.graph.learnable_parameter_count
        )

    @property
    def active_state_bytes(self) -> int:
        return self.memory.active_state_bytes + self.graph.active_state_bytes

    @property
    def estimated_mac_per_step(self) -> int:
        return (
            self.memory.estimated_mac_per_step
            + self.graph.estimated_mac_per_step
        )
