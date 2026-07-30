"""Bounded stochastic kinematic filtering for permanence Phase R.

The packed class is candidate-side and imports no evaluator/world code.  The
unpruned numerical reference lives in the evaluator namespace so it cannot be
imported into a formal candidate by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any, Mapping

import numpy as np


UNIT_VELOCITIES: tuple[tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)
VELOCITY_INDEX = {velocity: index for index, velocity in enumerate(UNIT_VELOCITIES)}
PYTHON_CONTAINER_GUARD_BYTES = 8_192
MAX_SUCCESSORS_PER_STATE = 12


class EmptyPosteriorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GridSpec:
    grid_size: int = 25
    arena_low: int = 7
    arena_high: int = 17

    def __post_init__(self) -> None:
        if self.grid_size < 1:
            raise ValueError("grid_size must be positive")
        if not 0 <= self.arena_low <= self.arena_high < self.grid_size:
            raise ValueError("invalid arena bounds")

    def encode(
        self, position: tuple[int, int], velocity: tuple[int, int]
    ) -> int:
        x, y = position
        if not (
            self.arena_low <= x <= self.arena_high
            and self.arena_low <= y <= self.arena_high
        ):
            raise ValueError(f"state outside arena: {position}")
        try:
            velocity_index = VELOCITY_INDEX[velocity]
        except KeyError as error:
            raise ValueError(f"invalid unit velocity: {velocity}") from error
        code = (y * self.grid_size + x) * len(UNIT_VELOCITIES) + velocity_index
        if code > np.iinfo(np.uint16).max:
            raise ValueError("state code does not fit uint16")
        return code

    def decode(self, code: int) -> tuple[tuple[int, int], tuple[int, int]]:
        cell, velocity_index = divmod(int(code), len(UNIT_VELOCITIES))
        y, x = divmod(cell, self.grid_size)
        position = (x, y)
        if not (
            self.arena_low <= x <= self.arena_high
            and self.arena_low <= y <= self.arena_high
        ):
            raise ValueError(f"encoded state outside arena: {code}")
        return position, UNIT_VELOCITIES[velocity_index]

    @property
    def state_capacity(self) -> int:
        width = self.arena_high - self.arena_low + 1
        return width * width * len(UNIT_VELOCITIES)

    def compact_index(self, code: int) -> int:
        (x, y), velocity = self.decode(code)
        width = self.arena_high - self.arena_low + 1
        return (
            ((y - self.arena_low) * width + (x - self.arena_low))
            * len(UNIT_VELOCITIES)
            + VELOCITY_INDEX[velocity]
        )


def _static_probability(
    cell: tuple[int, int],
    static_probability: np.ndarray,
    spec: GridSpec,
) -> float:
    x, y = cell
    if not (
        spec.arena_low <= x <= spec.arena_high
        and spec.arena_low <= y <= spec.arena_high
    ):
        return 1.0
    return float(np.clip(static_probability[y, x], 0.0, 1.0))


def _probability_grid(
    value: np.ndarray,
    *,
    name: str,
    spec: GridSpec,
) -> np.ndarray:
    result = np.asarray(value)
    expected_shape = (spec.grid_size, spec.grid_size)
    if result.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not (
        np.issubdtype(result.dtype, np.number)
        or np.issubdtype(result.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be numeric")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _probabilistic_bounce_distribution_validated(
    position: tuple[int, int],
    velocity: tuple[int, int],
    static_probability: np.ndarray,
    spec: GridSpec,
) -> tuple[tuple[tuple[int, int], tuple[int, int], float], ...]:
    """Bounce marginal after the probability grid has been validated."""

    forward = (position[0] + velocity[0], position[1] + velocity[1])
    reflected_velocity = (-velocity[0], -velocity[1])
    reflected = (
        position[0] + reflected_velocity[0],
        position[1] + reflected_velocity[1],
    )
    p_forward_static = _static_probability(forward, static_probability, spec)
    p_reflected_static = _static_probability(reflected, static_probability, spec)
    raw = (
        (forward, velocity, 1.0 - p_forward_static),
        (
            reflected,
            reflected_velocity,
            p_forward_static * (1.0 - p_reflected_static),
        ),
        (
            position,
            reflected_velocity,
            p_forward_static * p_reflected_static,
        ),
    )
    combined: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for new_position, new_velocity, probability in raw:
        if probability <= 0.0:
            continue
        key = (new_position, new_velocity)
        combined[key] = combined.get(key, 0.0) + probability
    total = sum(combined.values())
    if not np.isfinite(total) or total <= 0.0:
        raise EmptyPosteriorError("bounce transition has no probability mass")
    return tuple(
        (new_position, new_velocity, probability / total)
        for (new_position, new_velocity), probability in sorted(combined.items())
    )


def probabilistic_bounce_distribution(
    position: tuple[int, int],
    velocity: tuple[int, int],
    static_probability: np.ndarray,
    spec: GridSpec,
) -> tuple[tuple[tuple[int, int], tuple[int, int], float], ...]:
    """Marginalize the first forward and reflected cells of a bounce."""

    validated = _probability_grid(
        static_probability, name="static_probability", spec=spec
    )
    return _probabilistic_bounce_distribution_validated(
        position, velocity, validated, spec
    )


def _autonomous_successors_validated(
    position: tuple[int, int],
    velocity: tuple[int, int],
    static_probability: np.ndarray,
    *,
    turn_probability: float,
    allow_turn: bool,
    spec: GridSpec,
) -> tuple[tuple[tuple[int, int], tuple[int, int], float], ...]:
    keep = _probabilistic_bounce_distribution_validated(
        position, velocity, static_probability, spec
    )
    if not allow_turn or turn_probability <= 0.0:
        return keep
    alternatives: list[
        tuple[float, tuple[tuple[tuple[int, int], tuple[int, int], float], ...]]
    ] = []
    for candidate in UNIT_VELOCITIES:
        if candidate == velocity:
            continue
        distribution = _probabilistic_bounce_distribution_validated(
            position, candidate, static_probability, spec
        )
        movable = sum(
            probability
            for new_position, _new_velocity, probability in distribution
            if new_position != position
        )
        if movable > 0.0:
            alternatives.append((movable, distribution))
    if not alternatives:
        return keep
    total_availability = sum(value for value, _distribution in alternatives)
    combined: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for new_position, new_velocity, probability in keep:
        key = (new_position, new_velocity)
        value = (1.0 - turn_probability) * probability
        if value > 0.0:
            combined[key] = combined.get(key, 0.0) + value
    for availability, distribution in alternatives:
        direction_probability = turn_probability * availability / total_availability
        for new_position, new_velocity, probability in distribution:
            if new_position == position:
                continue
            conditional = probability / availability
            key = (new_position, new_velocity)
            combined[key] = combined.get(key, 0.0) + (
                direction_probability * conditional
            )
    total = sum(combined.values())
    if not np.isfinite(total) or total <= 0.0:
        raise EmptyPosteriorError("turn transition has no probability mass")
    return tuple(
        (new_position, new_velocity, probability / total)
        for (new_position, new_velocity), probability in sorted(combined.items())
        if probability > 0.0
    )


def autonomous_successors(
    position: tuple[int, int],
    velocity: tuple[int, int],
    static_probability: np.ndarray,
    *,
    turn_probability: float,
    allow_turn: bool,
    spec: GridSpec,
) -> tuple[tuple[tuple[int, int], tuple[int, int], float], ...]:
    """Return the bounded stochastic transition under uncertain static cells."""

    if not 0.0 <= turn_probability <= 1.0:
        raise ValueError("turn_probability must be in [0, 1]")
    validated = _probability_grid(
        static_probability, name="static_probability", spec=spec
    )
    return _autonomous_successors_validated(
        position,
        velocity,
        validated,
        turn_probability=turn_probability,
        allow_turn=allow_turn,
        spec=spec,
    )


def bayesian_no_detection_update(
    conditional_probability: np.ndarray,
    no_detection_probability: np.ndarray,
    existence: float,
) -> tuple[np.ndarray, float, float, float]:
    """Update ``q(state|exists)`` and Bernoulli existence exactly once."""

    q = np.asarray(conditional_probability, dtype=np.float64)
    likelihood = np.asarray(no_detection_probability, dtype=np.float64)
    if q.shape != likelihood.shape or q.ndim != 1:
        raise ValueError("probability/likelihood vectors must be aligned")
    if not 0.0 <= existence <= 1.0:
        raise ValueError("existence must be in [0, 1]")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(likelihood)):
        raise ValueError("probability/likelihood vectors must be finite")
    if np.any(q < 0.0) or np.any((likelihood < 0.0) | (likelihood > 1.0)):
        raise ValueError("invalid probability")
    q_total = float(q.sum())
    if q_total <= 0.0:
        raise EmptyPosteriorError("empty conditional posterior")
    q = q / q_total
    l_no = float(np.dot(q, likelihood))
    evidence = (1.0 - existence) + existence * l_no
    if l_no <= 0.0 or evidence <= 0.0:
        raise EmptyPosteriorError("no-detection observation has zero evidence")
    q_post = q * likelihood / l_no
    existence_post = existence * l_no / evidence
    return q_post, existence_post, l_no, log(evidence)


class PackedKinematicFilter:
    """Fixed-capacity packed posterior with deterministic tail pruning."""

    _MAX_SUCCESSORS_PER_STATE = MAX_SUCCESSORS_PER_STATE

    def __init__(
        self,
        k_max: int,
        *,
        spec: GridSpec = GridSpec(),
        probability_dtype: np.dtype[np.floating[Any]] = np.dtype(np.float32),
    ) -> None:
        if k_max < 1:
            raise ValueError("k_max must be positive")
        dtype = np.dtype(probability_dtype)
        if not np.issubdtype(dtype, np.floating):
            raise ValueError("probability_dtype must be floating point")
        self.spec = spec
        self.k_max = int(k_max)
        self.codes = np.zeros(self.k_max, dtype=np.uint16)
        self.probability = np.zeros(self.k_max, dtype=dtype)
        workspace = self.k_max * self._MAX_SUCCESSORS_PER_STATE
        self._next_codes = np.zeros(workspace, dtype=np.uint16)
        self._next_probability = np.zeros(workspace, dtype=np.float64)
        index_dtype = np.int16 if workspace <= np.iinfo(np.int16).max else np.int32
        self._next_index = np.full(spec.state_capacity, -1, dtype=index_dtype)
        self.count = 0
        self.branch_log_weight = 0.0
        self.cumulative_retained_probability = 1.0
        self.maximum_step_pruned_mass = 0.0

    def reset(
        self, position: tuple[int, int], velocity: tuple[int, int]
    ) -> None:
        self.codes.fill(0)
        self.probability.fill(0.0)
        self.codes[0] = self.spec.encode(position, velocity)
        self.probability[0] = 1.0
        self.count = 1
        self.branch_log_weight = 0.0
        self.cumulative_retained_probability = 1.0
        self.maximum_step_pruned_mass = 0.0

    def _accumulate(self, code: int, mass: float, next_count: int) -> int:
        compact_index = self.spec.compact_index(code)
        existing = int(self._next_index[compact_index])
        if existing >= 0:
            self._next_probability[existing] += mass
            return next_count
        if next_count >= self._next_codes.size:
            raise RuntimeError("fixed propagation workspace overflow")
        self._next_codes[next_count] = code
        self._next_probability[next_count] = mass
        self._next_index[compact_index] = next_count
        return next_count + 1

    def step(
        self,
        *,
        static_probability: np.ndarray,
        no_detection_probability: np.ndarray,
        turn_probability: float,
        allow_turn: bool,
    ) -> dict[str, float | int]:
        if self.count < 1:
            raise EmptyPosteriorError("filter has not been reset")
        validated_static = _probability_grid(
            static_probability, name="static_probability", spec=self.spec
        )
        validated_no_detection = _probability_grid(
            no_detection_probability,
            name="no_detection_probability",
            spec=self.spec,
        )
        self._next_codes.fill(0)
        self._next_probability.fill(0.0)
        self._next_index.fill(-1)
        next_count = 0
        for index in range(self.count):
            code = int(self.codes[index])
            mass = float(self.probability[index])
            position, velocity = self.spec.decode(code)
            successors = _autonomous_successors_validated(
                position,
                velocity,
                validated_static,
                turn_probability=turn_probability,
                allow_turn=allow_turn,
                spec=self.spec,
            )
            for new_position, new_velocity, transition in successors:
                new_code = self.spec.encode(new_position, new_velocity)
                next_count = self._accumulate(
                    new_code, mass * transition, next_count
                )
        for index in range(next_count):
            position, _velocity = self.spec.decode(int(self._next_codes[index]))
            likelihood = float(
                validated_no_detection[position[1], position[0]]
            )
            self._next_probability[index] *= likelihood
        observation_evidence = float(self._next_probability[:next_count].sum())
        if observation_evidence <= 0.0:
            raise EmptyPosteriorError("observation removed the complete posterior")
        self._next_probability[:next_count] /= observation_evidence
        order = np.lexsort(
            (
                self._next_codes[:next_count].astype(np.int64),
                -self._next_probability[:next_count],
            )
        )
        retain_count = min(self.k_max, next_count)
        retained_indices = order[:retain_count]
        retained_probability = min(
            1.0,
            float(self._next_probability[retained_indices].sum()),
        )
        if retained_probability <= 0.0:
            raise EmptyPosteriorError("pruning removed the complete posterior")
        step_pruned_mass = max(0.0, 1.0 - retained_probability)
        candidate_probability = (
            self._next_probability[retained_indices] / retained_probability
        ).astype(self.probability.dtype)
        normalizer = float(candidate_probability.sum())
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            raise EmptyPosteriorError("retained posterior is not representable")
        candidate_probability /= normalizer
        if not np.all(np.isfinite(candidate_probability)):
            raise EmptyPosteriorError("retained posterior is not finite")

        new_branch_log_weight = self.branch_log_weight + log(
            observation_evidence
        ) + log(retained_probability)
        new_cumulative_retained = (
            self.cumulative_retained_probability * retained_probability
        )
        new_maximum_step_pruned = max(
            self.maximum_step_pruned_mass, step_pruned_mass
        )
        self.codes.fill(0)
        self.probability.fill(0.0)
        self.codes[:retain_count] = self._next_codes[retained_indices]
        self.probability[:retain_count] = candidate_probability
        self.count = retain_count
        self.branch_log_weight = new_branch_log_weight
        self.cumulative_retained_probability = new_cumulative_retained
        self.maximum_step_pruned_mass = new_maximum_step_pruned
        return {
            "pre_pruning_support": next_count,
            "retained_support": retain_count,
            "observation_evidence": observation_evidence,
            "retained_probability": retained_probability,
            "step_pruned_mass": step_pruned_mass,
            "cumulative_pruned_mass": self.cumulative_pruned_mass,
        }

    @property
    def cumulative_pruned_mass(self) -> float:
        return 1.0 - self.cumulative_retained_probability

    def items(self) -> tuple[tuple[int, float], ...]:
        return tuple(
            sorted(
                (
                    (int(self.codes[index]), float(self.probability[index]))
                    for index in range(self.count)
                )
            )
        )

    def position_marginal(self) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for code, mass in self.items():
            position, _velocity = self.spec.decode(code)
            result[position] = result.get(position, 0.0) + mass
        return result

    @property
    def active_state_bytes(self) -> int:
        arrays = (
            self.codes,
            self.probability,
            self._next_codes,
            self._next_probability,
            self._next_index,
        )
        return sum(array.nbytes for array in arrays) + PYTHON_CONTAINER_GUARD_BYTES

    @property
    def learnable_parameter_count(self) -> int:
        return 1

    @property
    def estimated_mac_per_step(self) -> int:
        return self.k_max * self._MAX_SUCCESSORS_PER_STATE * 32


def position_total_variation(
    first: Mapping[tuple[int, int], float],
    second: Mapping[tuple[int, int], float],
) -> float:
    cells = set(first) | set(second)
    return 0.5 * sum(
        abs(float(first.get(cell, 0.0)) - float(second.get(cell, 0.0)))
        for cell in cells
    )


class PackedPosteriorPool:
    """Worst-case fully detached 5x11 bank used for Phase-R capacity proof."""

    __slots__ = (
        "spec",
        "hypotheses",
        "entities",
        "k_max",
        "grid_size",
        "s_max",
        "codes",
        "probability",
        "scratch_codes",
        "scratch_probability",
        "expansion_codes",
        "expansion_probability",
        "expansion_index",
        "counts",
        "existence",
        "self_probability",
        "hypothesis_weight",
        "assignment_workspace",
        "static_probability",
        "front_static_score",
        "last_visibility",
        "last_sensed",
        "rng_state",
        "kernel_parameters",
        "configuration",
    )

    def __init__(
        self,
        *,
        hypotheses: int = 5,
        entities: int = 11,
        k_max: int = 40,
        grid_size: int = 25,
        arena_low: int = 7,
        arena_high: int = 17,
    ) -> None:
        if min(hypotheses, entities, k_max, grid_size) < 1:
            raise ValueError("capacity dimensions must be positive")
        if k_max > np.iinfo(np.uint8).max:
            raise ValueError("k_max does not fit count dtype")
        self.spec = GridSpec(
            grid_size=int(grid_size),
            arena_low=int(arena_low),
            arena_high=int(arena_high),
        )
        state_capacity = self.spec.state_capacity
        if k_max > state_capacity:
            raise ValueError("k_max exceeds distinct valid kinematic states")
        self.hypotheses = int(hypotheses)
        self.entities = int(entities)
        self.k_max = int(k_max)
        self.grid_size = int(grid_size)
        self.s_max = self.hypotheses * self.entities * self.k_max
        self.spec.encode(
            (self.spec.arena_high, self.spec.arena_high), UNIT_VELOCITIES[-1]
        )
        self.codes = np.zeros(self.s_max, dtype=np.uint16)
        self.probability = np.zeros(self.s_max, dtype=np.float32)
        self.scratch_codes = np.zeros(self.s_max, dtype=np.uint16)
        self.scratch_probability = np.zeros(self.s_max, dtype=np.float32)
        self.expansion_codes = np.zeros(
            MAX_SUCCESSORS_PER_STATE * self.k_max, dtype=np.uint16
        )
        self.expansion_probability = np.zeros(
            MAX_SUCCESSORS_PER_STATE * self.k_max, dtype=np.float64
        )
        expansion_size = MAX_SUCCESSORS_PER_STATE * self.k_max
        expansion_index_dtype = (
            np.int16 if expansion_size <= np.iinfo(np.int16).max else np.int32
        )
        self.expansion_index = np.full(
            self.spec.state_capacity, -1, dtype=expansion_index_dtype
        )
        self.counts = np.zeros(
            (self.hypotheses, self.entities), dtype=np.uint8
        )
        self.existence = np.zeros(
            (self.hypotheses, self.entities), dtype=np.float32
        )
        self.self_probability = np.zeros(
            (self.hypotheses, self.entities + 1), dtype=np.float32
        )
        self.hypothesis_weight = np.zeros(self.hypotheses, dtype=np.float32)
        self.assignment_workspace = np.zeros(
            (self.hypotheses, self.entities, self.hypotheses),
            dtype=np.float32,
        )
        self.static_probability = np.zeros(
            (grid_size, grid_size), dtype=np.float16
        )
        self.front_static_score = np.zeros(
            (grid_size, grid_size), dtype=np.int16
        )
        self.last_visibility = np.zeros((grid_size, grid_size), dtype=np.bool_)
        self.last_sensed = np.zeros((grid_size, grid_size), dtype=np.bool_)
        self.rng_state = np.zeros(4, dtype=np.uint64)
        self.kernel_parameters = np.zeros(16, dtype=np.float32)
        self.configuration = np.zeros(16, dtype=np.uint32)

    def fill_fully_detached(self) -> None:
        valid_codes = np.asarray(
            [
                self.spec.encode((x, y), velocity)
                for y in range(self.spec.arena_low, self.spec.arena_high + 1)
                for x in range(self.spec.arena_low, self.spec.arena_high + 1)
                for velocity in UNIT_VELOCITIES
            ][: self.k_max],
            dtype=np.uint16,
        )
        self.codes[:] = np.tile(
            valid_codes, self.hypotheses * self.entities
        )
        self.probability[:] = 1.0 / self.k_max
        self.scratch_codes[:] = self.codes
        self.scratch_probability[:] = self.probability
        self.counts.fill(self.k_max)
        self.existence.fill(1.0)
        self.self_probability.fill(0.0)
        self.self_probability[:, 0] = 1.0
        self.hypothesis_weight.fill(1.0 / self.hypotheses)

    def replace_factor_atomic(
        self,
        hypothesis: int,
        entity: int,
        codes: np.ndarray,
        probability: np.ndarray,
    ) -> None:
        """Replace one factor through the full scratch pool or change nothing."""

        if not 0 <= hypothesis < self.hypotheses:
            raise IndexError("hypothesis outside pool")
        if not 0 <= entity < self.entities:
            raise IndexError("entity outside pool")
        candidate_codes = np.asarray(codes, dtype=np.uint16)
        candidate_probability = np.asarray(probability, dtype=np.float64)
        if candidate_codes.ndim != 1 or candidate_probability.ndim != 1:
            raise ValueError("factor arrays must be one-dimensional")
        if candidate_codes.shape != candidate_probability.shape:
            raise ValueError("factor arrays must be aligned")
        count = int(candidate_codes.size)
        if count < 1:
            raise EmptyPosteriorError("factor posterior is empty")
        if count > self.k_max:
            raise RuntimeError("packed posterior pool overflow")
        for code in candidate_codes:
            self.spec.decode(int(code))
        if not np.all(np.isfinite(candidate_probability)):
            raise ValueError("factor probability must be finite")
        if np.any(candidate_probability < 0.0):
            raise ValueError("factor probability must be non-negative")
        total = float(candidate_probability.sum())
        if total <= 0.0:
            raise EmptyPosteriorError("factor posterior has no probability mass")
        normalized = (candidate_probability / total).astype(np.float32)
        normalized_total = float(normalized.sum())
        if not np.isfinite(normalized_total) or normalized_total <= 0.0:
            raise EmptyPosteriorError("factor posterior is not representable")
        normalized /= normalized_total

        start = (hypothesis * self.entities + entity) * self.k_max
        stop = start + self.k_max
        self.scratch_codes[:] = self.codes
        self.scratch_probability[:] = self.probability
        self.scratch_codes[start:stop] = 0
        self.scratch_probability[start:stop] = 0.0
        self.scratch_codes[start : start + count] = candidate_codes
        self.scratch_probability[start : start + count] = normalized
        self.codes, self.scratch_codes = self.scratch_codes, self.codes
        self.probability, self.scratch_probability = (
            self.scratch_probability,
            self.probability,
        )
        self.counts[hypothesis, entity] = count

    @property
    def persistent_arrays(self) -> dict[str, np.ndarray]:
        return {
            name: value
            for name in self.__slots__
            if isinstance((value := getattr(self, name)), np.ndarray)
        }

    @property
    def persistent_array_bytes(self) -> int:
        return sum(value.nbytes for value in self.persistent_arrays.values())

    @property
    def active_state_bytes(self) -> int:
        return self.persistent_array_bytes + PYTHON_CONTAINER_GUARD_BYTES

    @property
    def learnable_parameter_count(self) -> int:
        return int(self.kernel_parameters.size)

    @property
    def estimated_mac_per_step(self) -> int:
        return (
            self.hypotheses
            * self.entities
            * self.k_max
            * MAX_SUCCESSORS_PER_STATE
            * 32
        )

    def capacity_contract(self) -> dict[str, int | str | bool]:
        return {
            "H_max": self.hypotheses,
            "E_max": self.entities,
            "K_max": self.k_max,
            "grid_size": self.grid_size,
            "S_max": self.s_max,
            "S_max_minimum": self.hypotheses * self.entities * self.k_max,
            "fully_detached_safe": self.s_max
            >= self.hypotheses * self.entities * self.k_max,
            "copy_on_write": False,
            "arena_low": self.spec.arena_low,
            "arena_high": self.spec.arena_high,
            "valid_state_capacity": (
                self.spec.state_capacity
            ),
            "state_code_dtype": str(self.codes.dtype),
            "probability_dtype": str(self.probability.dtype),
            "overflow_policy": "atomic_fail_without_partial_commit",
            "update_schedule": (
                "sequential_factor_expansion_into_full_scratch_then_array_swap"
            ),
            "shared_expansion_workspace_size": int(self.expansion_codes.size),
            "shared_expansion_workspace_required": (
                self.k_max * MAX_SUCCESSORS_PER_STATE
            ),
            "direct_index_accumulator": True,
            "direct_index_capacity": int(self.expansion_index.size),
            "python_container_guard_bytes": PYTHON_CONTAINER_GUARD_BYTES,
            "pruning_tie_break": "probability_desc_then_state_code_asc",
        }
