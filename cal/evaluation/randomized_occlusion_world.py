"""Randomized-occlusion world for the permanence re-test (item 2 / item 3).

Motivation (see
``docs/experiments/V2_L0_PERMANENCE_DIAGNOSIS_AND_PREREGISTRATION_DRAFT.md``):
the frozen ``_IntegratedWorld`` has a *fixed* occluder screen and *deterministic*
distractor motion, so a hidden object's position is a deterministic function of
``(seed, step)`` that a belief-free constant-velocity extrapolator reconstructs
~99% of the time.  The permanence probe therefore does not test object
permanence at all.

``RandomizedOcclusionWorld`` is a drop-in sibling of ``_IntegratedWorld`` (same
public interface: ``grid_size``, ``static``, ``self_position``,
``distractor_a/b``, ``velocity_a/b``, ``step``, ``observe``, ``truth``) that
breaks that shortcut with two changes:

1.  **Per-episode randomized occluder geometry** -- the screen column, extent,
    doorway row and one or two extra blockers are resampled every episode, so a
    probe cannot memorize a single fixed layout that transfers across seeds.
2.  **Stochastic hidden maneuvers** -- while a distractor is occluded from the
    camera it may change direction, drawn from a *hidden* RNG stream that is not
    reflected in any observation until the object reappears.  A constant-velocity
    extrapolation from the last visible state can therefore no longer determine
    the occluded position; only a belief that represents uncertainty over the
    hidden maneuvers can stay calibrated.

It preserves every V2 perception-fairness constraint: the agent-facing signal is
still only the ``sense_via_line_of_sight`` sensed patch (occluded occupancy is
indistinguishable from empty) plus the action copy; the visibility patch and
``truth()`` are evaluation-only; and the whole trajectory is deterministic given
the episode seed.  This module builds NO frozen protocol and NO gated evidence.
"""

from __future__ import annotations

import hashlib
import hmac

import numpy as np


# Salt for the hidden-maneuver stream.  Development and calibration runs use
# this published value so their episodes stay reproducible; a custodian holding
# a one-shot split MUST pass a secret salt instead, otherwise the split can be
# seed-inverted (see the 2026-08-08 review, finding F9).
DEFAULT_HIDDEN_STREAM_SALT = b"cal/v2-p1/permanence/hidden-stream/development"


def hidden_stream_key(seed: int, *, salt: bytes) -> int:
    """Derive the hidden-maneuver stream key from the episode seed via HMAC.

    The visible layout is a deterministic function of ``seed``, so any key that
    is a published function of ``seed`` alone (such as ``seed + 90_000``) lets
    an observer recover the seed from the layout and replay the hidden
    trajectory.  Keying through HMAC with a salt breaks that link: without the
    salt the hidden stream is not derivable from anything observable.
    """

    if not isinstance(salt, (bytes, bytearray)) or not salt:
        raise ValueError("hidden-stream salt must be non-empty bytes")
    digest = hmac.new(
        bytes(salt), int(seed).to_bytes(16, "big", signed=True), hashlib.sha256
    ).digest()
    return int.from_bytes(digest, "big")

from cal.evaluation.v2_i1_integration import (
    ARENA_HIGH,
    ARENA_LOW,
    CAMERA,
)
from cal.model.integrated_agent import ACTION_DELTAS
from cal.model.occupancy import sense_via_line_of_sight

_UNIT_VELOCITIES: tuple[tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)


def _arena_neighbors(
    cell: tuple[int, int], static: frozenset[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """Return traversable cardinal neighbours inside the evaluation arena."""

    x, y = cell
    return tuple(
        (nx, ny)
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        if ARENA_LOW <= nx <= ARENA_HIGH
        and ARENA_LOW <= ny <= ARENA_HIGH
        and (nx, ny) not in static
    )


def _layout_diagnostics(
    static: frozenset[tuple[int, int]],
    shadow: frozenset[tuple[int, int]],
) -> dict[str, int | bool]:
    """Audit connectivity and visibility structure without consulting a model."""

    free = {
        (x, y)
        for y in range(ARENA_LOW, ARENA_HIGH + 1)
        for x in range(ARENA_LOW, ARENA_HIGH + 1)
        if (x, y) not in static
    }
    visible = free - set(shadow)
    reachable: set[tuple[int, int]] = set()
    if free:
        frontier = [next(iter(free))]
        while frontier:
            cell = frontier.pop()
            if cell in reachable:
                continue
            reachable.add(cell)
            frontier.extend(
                neighbour
                for neighbour in _arena_neighbors(cell, static)
                if neighbour not in reachable
            )
    movable = {cell for cell in free if _arena_neighbors(cell, static)}
    visibility_boundary_edges = sum(
        1
        for cell in shadow
        for neighbour in _arena_neighbors(cell, static)
        if neighbour in visible
    )
    valid = (
        CAMERA in free
        and len(shadow) >= 3
        and len(visible) >= 3
        and reachable == free
        and len(movable) >= 3
        and visibility_boundary_edges >= 1
    )
    return {
        "valid": valid,
        "free_cell_count": len(free),
        "reachable_free_cell_count": len(reachable),
        "hidden_free_cell_count": len(shadow),
        "visible_free_cell_count": len(visible),
        "movable_free_cell_count": len(movable),
        "visibility_boundary_edge_count": visibility_boundary_edges,
    }


def _bounce_advance(
    position: tuple[int, int],
    velocity: tuple[int, int],
    static: frozenset[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Deterministic single-axis move with reflection off bounds and occluders.

    Mirrors the distractor update in ``_IntegratedWorld.step`` but is written
    against an explicit ``(position, velocity)`` pair so it can be reused for
    the hidden-maneuver logic.
    """

    px, py = position
    vx, vy = velocity
    axis = 0 if vx != 0 else 1
    if axis == 0:
        nxt = px + vx
        if nxt < ARENA_LOW or nxt > ARENA_HIGH or (nxt, py) in static:
            vx = -vx
            nxt = px + vx
            if nxt < ARENA_LOW or nxt > ARENA_HIGH or (nxt, py) in static:
                nxt = px  # boxed in on both sides: stay put this step
        px = nxt
    else:
        nxt = py + vy
        if nxt < ARENA_LOW or nxt > ARENA_HIGH or (px, nxt) in static:
            vy = -vy
            nxt = py + vy
            if nxt < ARENA_LOW or nxt > ARENA_HIGH or (px, nxt) in static:
                nxt = py
        py = nxt
    return (int(px), int(py)), (int(vx), int(vy))


class RandomizedOcclusionWorld:
    """Fixed-camera arena with per-episode geometry and stochastic occlusion."""

    def __init__(
        self,
        seed: int,
        grid_size: int = 25,
        *,
        hidden_turn_probability: float = 0.35,
        max_layout_attempts: int = 64,
        hidden_stream_salt: bytes = DEFAULT_HIDDEN_STREAM_SALT,
    ) -> None:
        if grid_size <= ARENA_HIGH:
            raise ValueError(
                f"grid_size must be greater than arena high bound {ARENA_HIGH}"
            )
        if not 0.0 <= hidden_turn_probability <= 1.0:
            raise ValueError("hidden_turn_probability must be in [0, 1]")
        if max_layout_attempts < 1:
            raise ValueError("max_layout_attempts must be positive")
        self.grid_size = grid_size
        self.hidden_turn_probability = float(hidden_turn_probability)
        self._layout_rng = np.random.default_rng(int(seed))
        # A separate stream drives hidden maneuvers so they are not observable
        # from pre-occlusion observation.  It is keyed through HMAC rather than
        # `seed + constant`: the visible layout is a deterministic function of
        # the episode seed, so an offset stream lets an attacker recover the
        # seed by matching layouts and then replay the hidden trajectory
        # exactly, which is a permanence oracle with no belief at all.  With a
        # secret salt, recovering the seed does not reveal the hidden stream.
        self._hidden_rng = np.random.default_rng(
            hidden_stream_key(seed, salt=hidden_stream_salt)
        )

        static, shadow_cells = self._sample_layout(max_layout_attempts)
        self.static = static
        self.shadow_cells = shadow_cells

        movable_cells = [
            (x, y)
            for y in range(ARENA_LOW, ARENA_HIGH + 1)
            for x in range(ARENA_LOW, ARENA_HIGH + 1)
            if (x, y) not in self.static and (x, y) != CAMERA
            and _arena_neighbors((x, y), self.static)
        ]
        placement = self._layout_rng.permutation(len(movable_cells))
        chosen = [movable_cells[int(i)] for i in placement[:3]]
        self.self_position = np.asarray(chosen[0], dtype=np.int64)
        self.distractor_a = np.asarray(chosen[1], dtype=np.int64)
        self.distractor_b = np.asarray(chosen[2], dtype=np.int64)
        self.velocity_a = np.asarray(
            _UNIT_VELOCITIES[int(self._layout_rng.integers(0, 4))],
            dtype=np.int64,
        )
        self.velocity_b = np.asarray(
            _UNIT_VELOCITIES[int(self._layout_rng.integers(0, 4))],
            dtype=np.int64,
        )

    # -- layout ---------------------------------------------------------------

    def _sample_layout(
        self, max_attempts: int
    ) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
        for _ in range(max_attempts):
            screen_x = int(self._layout_rng.integers(ARENA_LOW + 2, ARENA_HIGH - 1))
            row_lo = int(self._layout_rng.integers(ARENA_LOW, CAMERA[1]))
            row_hi = int(self._layout_rng.integers(CAMERA[1] + 1, ARENA_HIGH + 1))
            door_row = int(self._layout_rng.integers(row_lo, row_hi + 1))
            cells = {
                (screen_x, y)
                for y in range(row_lo, row_hi + 1)
                if y != door_row
            }
            extra_count = int(self._layout_rng.integers(1, 3))
            for _ in range(extra_count):
                ex = int(self._layout_rng.integers(ARENA_LOW, ARENA_HIGH + 1))
                ey = int(self._layout_rng.integers(ARENA_LOW, ARENA_HIGH + 1))
                cells.add((ex, ey))
            cells.discard(CAMERA)
            if not cells:
                continue
            static = frozenset(cells)
            shadow = self._shadow_cells(static)
            if bool(_layout_diagnostics(static, shadow)["valid"]):
                return static, shadow
        raise RuntimeError("failed to sample a valid randomized occluder layout")

    def _shadow_cells(
        self, static: frozenset[tuple[int, int]]
    ) -> frozenset[tuple[int, int]]:
        """Arena cells occluded from the camera by the static layout alone."""

        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for x, y in static:
            grid[y, x] = 1
        _, visibility = sense_via_line_of_sight(CAMERA, grid)
        hidden: set[tuple[int, int]] = set()
        x0, y0 = CAMERA[0] - (visibility.shape[1] // 2), CAMERA[1] - (
            visibility.shape[0] // 2
        )
        for local_y in range(visibility.shape[0]):
            for local_x in range(visibility.shape[1]):
                x, y = x0 + local_x, y0 + local_y
                if (
                    ARENA_LOW <= x <= ARENA_HIGH
                    and ARENA_LOW <= y <= ARENA_HIGH
                    and not visibility[local_y, local_x]
                    and (x, y) not in static
                ):
                    hidden.add((x, y))
        return frozenset(hidden)

    def layout_diagnostics(self) -> dict[str, int | bool]:
        """Return the model-independent validity audit for this episode layout."""

        return _layout_diagnostics(self.static, self.shadow_cells)

    # -- dynamics -------------------------------------------------------------

    def _camera_sees(self, cell: tuple[int, int]) -> bool:
        # Visibility is a property of the sampled static occluder layout.
        # Dynamic actors do not become transient "screens"; otherwise the
        # hidden-turn kernel would depend on the unobserved joint state of all
        # three actors and could not be represented by a single-object filter.
        return cell not in self.shadow_cells

    def _maybe_hidden_turn(
        self, position: tuple[int, int], velocity: tuple[int, int]
    ) -> tuple[int, int]:
        """If occluded, possibly pick a new (non-immediately-blocked) direction."""

        if self._camera_sees(position):
            return velocity
        if self._hidden_rng.random() >= self.hidden_turn_probability:
            return velocity
        options = [v for v in _UNIT_VELOCITIES if v != velocity]
        self._hidden_rng.shuffle(options)
        for candidate in options:
            moved, _ = _bounce_advance(position, candidate, self.static)
            if moved != position:
                return candidate
        return velocity

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray]:
        delta = ACTION_DELTAS[int(action)]
        candidate = np.clip(
            self.self_position + delta, ARENA_LOW, ARENA_HIGH
        )
        if tuple(int(v) for v in candidate) not in self.static:
            self.self_position = candidate

        for name in ("a", "b"):
            point = getattr(self, f"distractor_{name}")
            velocity = getattr(self, f"velocity_{name}")
            position = (int(point[0]), int(point[1]))
            velocity_tuple = (int(velocity[0]), int(velocity[1]))
            velocity_tuple = self._maybe_hidden_turn(position, velocity_tuple)
            moved, new_velocity = _bounce_advance(
                position, velocity_tuple, self.static
            )
            setattr(self, f"distractor_{name}", np.asarray(moved, dtype=np.int64))
            setattr(
                self, f"velocity_{name}", np.asarray(new_velocity, dtype=np.int64)
            )
        return self.observe()

    # -- observation ----------------------------------------------------------

    def truth(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for x, y in self.static:
            grid[y, x] = 1
        for point in (self.self_position, self.distractor_a, self.distractor_b):
            grid[int(point[1]), int(point[0])] = 1
        return grid

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        # Compute visibility from static geometry only, then render every
        # currently occupied visible cell.  This keeps the observation process
        # state-independent and makes the documented single-object transition
        # and observation model exact.
        static_truth = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for x, y in self.static:
            static_truth[y, x] = 1
        _, visibility = sense_via_line_of_sight(CAMERA, static_truth)
        radius = visibility.shape[0] // 2
        x0, y0 = CAMERA[0] - radius, CAMERA[1] - radius
        truth_patch = self.truth()[
            y0 : y0 + visibility.shape[0],
            x0 : x0 + visibility.shape[1],
        ]
        sensed = np.where(visibility, truth_patch, 0).astype(np.uint8)
        return sensed, visibility
