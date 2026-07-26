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

# Self-identification lock calibration (see self_track_identity /
# _update_self_lock below). Chosen from the mechanism's own control_evidence
# distribution on the development seeds, characterized against the two
# negative controls this protocol requires - not fit against the self_f1
# gate itself. Measured on the 16 development seeds with this exact
# configuration (floor/margin/streak below, live-tracks-only candidacy):
# - the true self track engages the lock on all 16/16 seeds, by step 154
#   at the latest (median well under 100);
# - no_action (random action copy) spuriously engages the lock on 3/16
#   seeds and time-shuffled (5-step lag) on 6/16 - a genuinely uncorrelated
#   or stale action does occasionally produce a short accidental streak,
#   so this is not a clean guarantee, only a strong bias. It is sufficient
#   in aggregate: mean self_f1 stays >=0.15 above both controls' mean
#   self_f1 (the protocol's required control-drop margin) because the
#   spurious locks are late, rare, and typically short-lived compared to
#   the correctly-engaged formal lock. Raising the streak requirement
#   further reduces false engagement but delays real engagement by the
#   same mechanism, trading one failure mode for the other; see
#   docs/experiments/V2_I1_INTEGRATION_REPORT.md for why the residual gap
#   to the 0.90 self_f1 gate is a separate, deeper association-layer
#   problem (track identity switching) that this lock cannot paper over.
_SELF_LOCK_CONTROL_EVIDENCE_FLOOR = 0.5
_SELF_LOCK_MARGIN = 0.3
_SELF_LOCK_STREAK_REQUIRED = 5


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
        # This world's fixed-camera occlusion runs far longer than
        # OnlineEntityGraph's native V2-M2/M3 worlds (measured up to 34
        # consecutive steps across 20 seeds, p99 ~28); 40 gives comfortable
        # margin above that tail without being unbounded.
        #
        # identity_switch_penalty_weight is set from a sweep over the 16
        # development seeds (mean self_f1 at weight = 0/1/2/5/10/20/50/100/
        # 300/1000: 0.296/0.264/0.293/0.297/0.298/0.320/0.287/0.273/0.285/
        # 0.262): mean self_f1 peaks around weight=20 and declines on both
        # sides. It is a real, modest improvement (0.296 -> 0.320), not a
        # fix - it reduces how often a bad association happens in the first
        # place, but cannot undo one that already occurred (the corrupted
        # theta/autonomous_velocity from a mis-association keep generating
        # biased predictions afterward), which is why the self_f1 gate
        # (>=0.90) still fails by a wide margin. See "第六处摩擦" and its
        # follow-up in docs/experiments/V2_I1_INTEGRATION_REPORT.md.
        self.graph = OnlineEntityGraph(
            4,
            association_mode="probabilistic",
            maximum_tracks=16,
            reacquisition_window=40,
            identity_switch_penalty_weight=20.0,
        )
        self._initialized = False
        self._action_rng = np.random.default_rng(seed + 60_000)
        self._scale = 0.32
        # Persistent self-identification: see _update_self_lock. A per-step
        # argmax over raw control_evidence is too noisy (stay actions, wall
        # bounces, and brief blob merges all dip it), so identity is locked
        # in once evidence has clearly favored one track for a sustained run
        # and held afterward, rather than re-decided fresh every step.
        self._self_lock: int | None = None
        self._leader_track: int | None = None
        self._leader_streak: int = 0

    # -- structural surface under test: one update, patch + action only --

    def update(self, sensed_occupancy: np.ndarray, action: int) -> None:
        # Fixed camera: the occupancy component always receives "stay".
        self.memory.update(sensed_occupancy, 0)
        camera = self.memory._camera
        x0 = int(camera[0] - VIEW_RADIUS)
        y0 = int(camera[1] - VIEW_RADIUS)
        # 4-connectivity: this world's objects and walls are axis-aligned
        # grid cells, so two entities diagonally touching are still
        # visually distinct rather than one blob (measured: raises the
        # self-position exact-detection rate from 66% to 80% over the
        # development seeds by no longer merging the self point into a
        # diagonally adjacent wall or distractor).
        detections = connected_component_centroids(
            sensed_occupancy, x0, y0, connectivity=4
        )
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
        self._update_self_lock()

    def _update_self_lock(self) -> None:
        if self._self_lock is not None:
            if any(track.index == self._self_lock for track in self.graph._tracks):
                # Already locked and that track still exists: nothing to
                # decide. Deliberately skip the streak bookkeeping below
                # entirely while locked (see the comment on it) instead of
                # letting it keep running for some other track in the
                # background.
                return
            self._self_lock = None
            self._leader_track = None
            self._leader_streak = 0
        # Only a track actually matched to a detection this step is eligible
        # to lead: a stale/extrapolated track's control_evidence is a frozen
        # (decaying) leftover, not fresh evidence, and letting it win the
        # per-step argmax just because live candidates dipped is how a
        # ghost fragment gets mistaken for self. This bookkeeping only runs
        # while unlocked (the early return above): if it kept accumulating
        # for a different track while one was already locked, that stale
        # streak would let the moment the old lock's track is pruned
        # trigger an unverified "instant" re-lock, defeating the point of
        # requiring a fresh sustained run right after losing an identity.
        step = self.graph._step
        live = {
            track.index: track.control_evidence
            for track in self.graph._tracks
            if track.last_seen == step
        }
        if not live:
            self._leader_track = None
            self._leader_streak = 0
            return
        best_index = max(live, key=live.get)
        best_value = live[best_index]
        # -inf when best_index is the only live track this step: with no
        # competitor to out-margin, only the absolute floor gates
        # qualification for that step (a deliberately weaker bar than the
        # floor+margin combination used whenever a rival is live).
        runner_up = max(
            (value for index, value in live.items() if index != best_index),
            default=-float("inf"),
        )
        qualifies = (
            best_value >= _SELF_LOCK_CONTROL_EVIDENCE_FLOOR
            and best_value - runner_up >= _SELF_LOCK_MARGIN
        )
        if qualifies and best_index == self._leader_track:
            self._leader_streak += 1
        elif qualifies:
            self._leader_track = best_index
            self._leader_streak = 1
        else:
            self._leader_track = None
            self._leader_streak = 0
        if self._leader_streak >= _SELF_LOCK_STREAK_REQUIRED:
            self._self_lock = self._leader_track

    def _prune_runaway_tracks(self) -> None:
        """Drop tracks that are both stale and no longer plausible.

        OnlineEntityGraph never expires a track: an unmatched track keeps
        being extrapolated by its own theta/autonomous-velocity estimate
        indefinitely. Under this world's long occlusions, a track fit from
        only a few noisy samples can drift outside the grid and never be
        reacquired, permanently consuming one of the bounded track slots.

        The out-of-bounds check alone must NOT fire before
        reacquisition_window elapses: a noisy track can drift past a fixed
        margin in a handful of steps (a single bad RLS update from one
        mis-associated detection can inflate theta well above real motion),
        which would evict it long before OnlineEntityGraph's own widened
        window ever gives it a chance to be reacquired - silently undercutting
        the wider-window guarantee for exactly the tracks it exists to
        protect. So both conditions require the same staleness gate: this
        reclaims tracks OnlineEntityGraph's own reacquisition_window has
        already given up on AND that are either implausibly positioned or
        low-confidence, never tracks still within their guaranteed window.
        """

        margin = 4 * self._scale
        low = -margin
        high = (self.grid_size - 1) * self._scale + margin
        step = self.graph._step
        stale_after = self.graph.reacquisition_window + 10
        self.graph._tracks = [
            track
            for track in self.graph._tracks
            if not (
                step - track.last_seen > stale_after
                and (
                    not (
                        low <= track.position[0] <= high
                        and low <= track.position[1] <= high
                    )
                    or track.probability < 0.5
                )
            )
        ]

    def self_track_identity(self) -> int | None:
        return self._self_lock

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
        window = (2 * VIEW_RADIUS + 1) ** 2
        # connected_component_centroids visits every cell and, worst case,
        # its 4 neighbors once each (4-connectivity, see update());
        # _prune_runaway_tracks and _update_self_lock each scan every track.
        blob_detection = window * 5
        track_pruning = self.graph.maximum_tracks * 4
        self_lock_bookkeeping = self.graph.maximum_tracks * 4
        return (
            self.memory.estimated_mac_per_step
            + self.graph.estimated_mac_per_step
            + blob_detection
            + track_pruning
            + self_lock_bookkeeping
        )
