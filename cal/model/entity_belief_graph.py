"""Bounded multi-hypothesis entity beliefs for the V2-I1 integration.

This module deliberately does not reuse either of the two trackers composed
by the first I1 probe. One shared entity store drives association, online
action-dependence evidence, self attribution, and occluded occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Iterable

import numpy as np

from cal.model.occupancy import (
    VIEW_RADIUS,
    infer_visibility_from_sensed_occupancy,
)


GRID_DELTAS = np.asarray(
    ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
    dtype=np.int16,
)
DELTA_CATEGORIES = 6  # stay, four neighbours, and other
STATIC_THRESHOLD = 5
SELF_POSTERIOR_MINIMUM = 0.55
SELF_NULL_LOGIT = 3.0


def _delta_category(delta: np.ndarray) -> int:
    for index, candidate in enumerate(GRID_DELTAS):
        if np.array_equal(delta, candidate):
            return index
    return 5


def _softmax(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return array
    shifted = array - float(np.max(array))
    weights = np.exp(np.clip(shifted, -60.0, 0.0))
    return weights / max(float(weights.sum()), 1e-12)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-float(np.clip(value, -30.0, 30.0))))


@dataclass(slots=True)
class EntityBelief:
    """One branch-local physical-entity state."""

    index: int
    position: np.ndarray
    velocity: np.ndarray
    last_seen: int
    age: int
    missed: int
    existence: float
    self_logit: float
    action_delta_counts: np.ndarray
    motion_delta_counts: np.ndarray

    def clone(self) -> "EntityBelief":
        return EntityBelief(
            index=self.index,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            last_seen=self.last_seen,
            age=self.age,
            missed=self.missed,
            existence=self.existence,
            self_logit=self.self_logit,
            action_delta_counts=self.action_delta_counts.copy(),
            motion_delta_counts=self.motion_delta_counts.copy(),
        )


@dataclass(slots=True)
class GlobalHypothesis:
    """One globally consistent association history."""

    entities: list[EntityBelief]
    log_weight: float
    weight: float = 1.0


class DynamicCellFrontEnd:
    """Separate persistent static occupancy from entity detections online."""

    def __init__(
        self,
        grid_size: int,
        *,
        infer_occlusion: bool,
    ) -> None:
        self.grid_size = grid_size
        self.infer_occlusion = infer_occlusion
        self.static_score = np.zeros(
            (grid_size, grid_size), dtype=np.int16
        )
        self.last_visibility = np.zeros(
            (grid_size, grid_size), dtype=bool
        )
        self.last_sensed = np.zeros((grid_size, grid_size), dtype=bool)

    def update(
        self,
        sensed_occupancy: np.ndarray,
        *,
        protected_cells: set[tuple[int, int]] | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        expected = (2 * VIEW_RADIUS + 1, 2 * VIEW_RADIUS + 1)
        if sensed_occupancy.shape != expected:
            raise ValueError(f"expected sensed occupancy shape {expected}")
        local_visibility = (
            infer_visibility_from_sensed_occupancy(sensed_occupancy)
            if self.infer_occlusion
            else np.ones_like(sensed_occupancy, dtype=np.uint8)
        )
        visibility = np.zeros(
            (self.grid_size, self.grid_size), dtype=bool
        )
        sensed = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        center = self.grid_size // 2
        x0 = center - VIEW_RADIUS
        y0 = center - VIEW_RADIUS
        ys = slice(y0, y0 + expected[0])
        xs = slice(x0, x0 + expected[1])
        visibility[ys, xs] = local_visibility.astype(bool)
        sensed[ys, xs] = sensed_occupancy.astype(bool)

        occupied_visible = visibility & sensed
        empty_visible = visibility & ~sensed
        self.static_score[occupied_visible] += 1
        self.static_score[empty_visible] -= 3
        np.clip(self.static_score, -12, 12, out=self.static_score)
        for x, y in protected_cells or set():
            if (
                0 <= x < self.grid_size
                and 0 <= y < self.grid_size
                and occupied_visible[y, x]
            ):
                # A previously moving entity is allowed to pause. Do not
                # convert its occupied cell into background merely because
                # it remains still for STATIC_THRESHOLD frames.
                self.static_score[y, x] = min(
                    int(self.static_score[y, x]),
                    STATIC_THRESHOLD - 1,
                )
        static = self.static_score >= STATIC_THRESHOLD
        dynamic_cells = np.argwhere(occupied_visible & ~static)
        detections = [
            np.asarray((int(x), int(y)), dtype=np.int16)
            for y, x in dynamic_cells
        ]
        self.last_visibility = visibility
        self.last_sensed = sensed
        return detections, static

    @property
    def active_state_bytes(self) -> int:
        return (
            self.static_score.nbytes
            + self.last_visibility.nbytes
            + self.last_sensed.nbytes
        )


class EntityBeliefGraph:
    """Online bounded global association and shared entity belief state."""

    def __init__(
        self,
        grid_size: int = 25,
        *,
        maximum_hypotheses: int = 5,
        maximum_entities: int = 11,
        maximum_missed_steps: int = 40,
    ) -> None:
        if maximum_hypotheses < 1 or maximum_entities < 1:
            raise ValueError("hypothesis and entity limits must be positive")
        self.grid_size = grid_size
        self.maximum_hypotheses = maximum_hypotheses
        self.maximum_entities = maximum_entities
        self.maximum_missed_steps = maximum_missed_steps
        self._hypotheses: list[GlobalHypothesis] = [
            GlobalHypothesis([], 0.0, 1.0)
        ]
        self._step = 0
        self._next_entity_index = 0
        self._birth_ids: dict[tuple[int, int, int], int] = {}
        self._last_visibility = np.zeros(
            (grid_size, grid_size), dtype=bool
        )
        self._last_sensed = np.zeros((grid_size, grid_size), dtype=bool)
        self._last_static = np.zeros((grid_size, grid_size), dtype=bool)

    def update(
        self,
        detections: list[np.ndarray],
        visibility: np.ndarray,
        sensed: np.ndarray,
        static: np.ndarray,
        action: int,
    ) -> None:
        if action not in range(5):
            raise ValueError("action must be in [0, 4]")
        self._step += 1
        self._last_visibility = visibility
        self._last_sensed = sensed
        self._last_static = static
        expanded: list[GlobalHypothesis] = []
        for hypothesis in self._hypotheses:
            expanded.extend(
                self._expand_hypothesis(
                    hypothesis,
                    detections=detections,
                    visibility=visibility,
                    static=static,
                    action=action,
                )
            )
        self._hypotheses = self._select_hypotheses(expanded)
        # Birth IDs only need to be shared across branches within this step.
        self._birth_ids.clear()

    def _expand_hypothesis(
        self,
        hypothesis: GlobalHypothesis,
        *,
        detections: list[np.ndarray],
        visibility: np.ndarray,
        static: np.ndarray,
        action: int,
    ) -> list[GlobalHypothesis]:
        viable = [
            entity
            for entity in hypothesis.entities
            if (
                entity.missed <= self.maximum_missed_steps
                and entity.existence >= 0.08
                and not (
                    static[
                        int(entity.position[1]),
                        int(entity.position[0]),
                    ]
                    and entity.missed > 0
                    and entity.self_logit < 0.5
                )
            )
        ]
        viable.sort(
            key=lambda item: (item.existence, item.self_logit, item.age),
            reverse=True,
        )
        partials: list[
            tuple[float, list[EntityBelief], frozenset[int]]
        ] = [(hypothesis.log_weight, [], frozenset())]
        for entity in viable:
            candidates: list[
                tuple[float, list[EntityBelief], frozenset[int]]
            ] = []
            predictions = self._prediction_distribution(
                entity, action=action, static=static
            )
            for score, states, used in partials:
                for position, prior in sorted(
                    predictions.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]:
                    missed, miss_score = self._missed_state(
                        entity,
                        position=position,
                        visibility=visibility,
                    )
                    candidates.append(
                        (
                            score
                            + log(max(prior, 1e-9))
                            + miss_score,
                            states + [missed],
                            used,
                        )
                    )
                for detection_index, detection in enumerate(detections):
                    if detection_index in used:
                        continue
                    likelihood = self._match_likelihood(
                        detection, predictions
                    )
                    if likelihood <= 1e-8:
                        continue
                    matched = self._matched_state(
                        entity,
                        detection=detection,
                        action=action,
                    )
                    candidates.append(
                        (
                            score + log(likelihood) + log(
                                max(entity.existence, 1e-6)
                            ),
                            states + [matched],
                            used | {detection_index},
                        )
                    )
            candidates.sort(key=lambda item: item[0], reverse=True)
            partials = candidates[: self.maximum_hypotheses]

        completed: list[GlobalHypothesis] = []
        for score, states, used in partials:
            remaining = [
                (index, detection)
                for index, detection in enumerate(detections)
                if index not in used
            ]
            capacity = self.maximum_entities - len(states)
            # Births are evidence-explaining alternatives, but carry a
            # penalty so a stable predicted identity wins over fragmentation.
            for _, detection in remaining[: max(0, capacity)]:
                states.append(self._new_entity(detection))
                score -= 1.8
            completed.append(GlobalHypothesis(states, score))
        return completed or [GlobalHypothesis([], hypothesis.log_weight - 4.0)]

    def _prediction_distribution(
        self,
        entity: EntityBelief,
        *,
        action: int,
        static: np.ndarray,
    ) -> dict[tuple[int, int], float]:
        action_counts = entity.action_delta_counts[action]
        action_probability = action_counts[:5] / max(
            float(action_counts.sum()), 1e-12
        )
        motion_counts = entity.motion_delta_counts.astype(np.float64)
        velocity_category = _delta_category(entity.velocity)
        if velocity_category < 5:
            motion_counts[velocity_category] += 5.0
            if velocity_category:
                reverse = velocity_category + (
                    1 if velocity_category % 2 else -1
                )
                motion_counts[reverse] += 0.8
        motion_probability = motion_counts[:5] / max(
            float(motion_counts.sum()), 1e-12
        )
        control_weight = min(0.85, _sigmoid(entity.self_logit - 1.0))
        mixture = (
            control_weight * action_probability
            + (1.0 - control_weight) * motion_probability
        )
        positions: dict[tuple[int, int], float] = {}
        x, y = (int(entity.position[0]), int(entity.position[1]))
        for category, probability in enumerate(mixture):
            candidate = entity.position + GRID_DELTAS[category]
            cx = int(np.clip(candidate[0], 0, self.grid_size - 1))
            cy = int(np.clip(candidate[1], 0, self.grid_size - 1))
            if static[cy, cx]:
                cx, cy = x, y
            positions[(cx, cy)] = positions.get((cx, cy), 0.0) + float(
                probability
            )
        # Deliberately do not renormalize away the "other displacement"
        # category. Its missing probability mass lowers every supported
        # association likelihood after outliers have been learned.
        return positions

    @staticmethod
    def _match_likelihood(
        detection: np.ndarray,
        predictions: dict[tuple[int, int], float],
    ) -> float:
        dx, dy = int(detection[0]), int(detection[1])
        likelihood = 0.0
        for (px, py), probability in predictions.items():
            distance = abs(dx - px) + abs(dy - py)
            if distance <= 2:
                likelihood += probability * exp(-2.2 * distance)
        return likelihood

    def _matched_state(
        self,
        entity: EntityBelief,
        *,
        detection: np.ndarray,
        action: int,
    ) -> EntityBelief:
        result = entity.clone()
        old_position = entity.position.copy()
        result.position = detection.astype(np.int16, copy=True)
        result.age += 1
        result.missed = 0
        result.existence = min(0.995, 0.62 + 0.38 * entity.existence)
        if entity.last_seen == self._step - 1:
            delta = result.position - old_position
            category = _delta_category(delta)
            action_row = entity.action_delta_counts[action]
            action_probability = (
                float(action_row[category] + 0.5)
                / float(action_row.sum() + 0.5 * DELTA_CATEGORIES)
            )
            motion_probability = (
                float(entity.motion_delta_counts[category] + 0.5)
                / float(
                    entity.motion_delta_counts.sum()
                    + 0.5 * DELTA_CATEGORIES
                )
            )
            evidence = float(
                np.clip(
                    log(max(action_probability, 1e-9))
                    - log(max(motion_probability, 1e-9)),
                    -1.5,
                    1.5,
                )
            )
            result.self_logit = 0.985 * entity.self_logit + evidence
            result.action_delta_counts[action, category] += 1.0
            result.motion_delta_counts[category] += 1.0
            if category < 5:
                result.velocity = GRID_DELTAS[category].copy()
        else:
            # Reacquisition is useful identity evidence but not a one-step
            # transition sample; treating the whole gap as one delta would
            # corrupt both control and autonomous models.
            result.self_logit = 0.995 * entity.self_logit
        result.last_seen = self._step
        return result

    def _missed_state(
        self,
        entity: EntityBelief,
        *,
        position: tuple[int, int],
        visibility: np.ndarray,
    ) -> tuple[EntityBelief, float]:
        result = entity.clone()
        result.position = np.asarray(position, dtype=np.int16)
        result.age += 1
        result.missed += 1
        visible = bool(visibility[position[1], position[0]])
        occupied = bool(self._last_sensed[position[1], position[0]])
        if visible and occupied:
            # More physical entities than occupied cells is the expected
            # signature of a point merge. Preserve the unassigned identity
            # as a co-located hypothesis instead of treating the occupied
            # cell as evidence that the entity vanished.
            result.existence *= 0.99
            result.self_logit *= 0.995
            return result, log(0.46)
        center = self.grid_size // 2
        in_sensor_window = (
            center - VIEW_RADIUS <= position[0] <= center + VIEW_RADIUS
            and center - VIEW_RADIUS <= position[1] <= center + VIEW_RADIUS
        )
        if not in_sensor_window:
            # The fixed-camera I1 task exposes its whole arena in this
            # window. A branch extrapolating far outside it is almost always
            # a one-sample velocity ghost. Decay it without confusing that
            # condition with genuine within-window geometric occlusion.
            result.existence *= 0.62
            result.self_logit *= 0.98
            return result, log(0.08)
        if visible:
            result.existence *= 0.18
            result.self_logit -= 0.25
            return result, log(0.035)
        result.existence *= 0.997
        result.self_logit *= 0.997
        return result, log(0.72)

    def _new_entity(self, detection: np.ndarray) -> EntityBelief:
        x, y = int(detection[0]), int(detection[1])
        key = (self._step, x, y)
        index = self._birth_ids.get(key)
        if index is None:
            index = self._next_entity_index
            self._next_entity_index += 1
            self._birth_ids[key] = index
        return EntityBelief(
            index=index,
            position=detection.astype(np.int16, copy=True),
            velocity=np.zeros(2, dtype=np.int16),
            last_seen=self._step,
            age=1,
            missed=0,
            existence=0.82,
            self_logit=0.0,
            action_delta_counts=np.ones(
                (5, DELTA_CATEGORIES), dtype=np.float32
            ),
            motion_delta_counts=np.ones(
                DELTA_CATEGORIES, dtype=np.float32
            ),
        )

    def _select_hypotheses(
        self,
        hypotheses: list[GlobalHypothesis],
    ) -> list[GlobalHypothesis]:
        if not hypotheses:
            return [GlobalHypothesis([], 0.0, 1.0)]
        # Equivalent branches arise frequently from already-dead tracks.
        # Deduplicate before normalization so they do not receive accidental
        # multiplicity weight.
        unique: dict[tuple[object, ...], GlobalHypothesis] = {}
        for hypothesis in hypotheses:
            signature = tuple(
                sorted(
                    (
                        entity.index,
                        int(entity.position[0]),
                        int(entity.position[1]),
                        int(entity.velocity[0]),
                        int(entity.velocity[1]),
                        entity.last_seen,
                        entity.age,
                        entity.missed,
                        np.float64(entity.existence).tobytes(),
                        np.float64(entity.self_logit).tobytes(),
                        entity.action_delta_counts.tobytes(),
                        entity.motion_delta_counts.tobytes(),
                    )
                    for entity in hypothesis.entities
                )
            )
            previous = unique.get(signature)
            if previous is None:
                unique[signature] = hypothesis
            else:
                # These branches are equal in every state variable that can
                # affect future inference. Combine their probability mass;
                # position-only equality is intentionally insufficient.
                previous.log_weight = float(
                    np.logaddexp(
                        previous.log_weight, hypothesis.log_weight
                    )
                )
        selected = sorted(
            unique.values(), key=lambda item: item.log_weight, reverse=True
        )[: self.maximum_hypotheses]
        maximum = selected[0].log_weight
        weights = np.asarray(
            [exp(float(np.clip(item.log_weight - maximum, -60.0, 0.0)))
             for item in selected],
            dtype=np.float64,
        )
        weights /= max(float(weights.sum()), 1e-12)
        for hypothesis, weight in zip(selected, weights):
            hypothesis.weight = float(weight)
            # Keep accumulated evidence bounded without changing ratios.
            hypothesis.log_weight -= maximum
        return selected

    def self_posterior(self) -> dict[int, float]:
        posterior: dict[int, float] = {}
        for hypothesis in self._hypotheses:
            candidates = [
                entity
                for entity in hypothesis.entities
                if entity.existence > 1e-9
            ]
            local = _softmax(
                [
                    *(
                        entity.self_logit
                        + log(max(entity.existence, 1e-9))
                        + log(
                            max(
                                entity.age / (entity.age + 3.0),
                                1e-9,
                            )
                        )
                        for entity in candidates
                    ),
                    SELF_NULL_LOGIT,
                ]
            )
            for entity, probability in zip(candidates, local):
                posterior[entity.index] = posterior.get(
                    entity.index, 0.0
                ) + hypothesis.weight * float(probability)
        return posterior

    def self_identity(self) -> int | None:
        posterior = self.self_posterior()
        if not posterior:
            return None
        identity = max(posterior, key=posterior.get)
        if posterior[identity] < SELF_POSTERIOR_MINIMUM:
            return None
        return identity

    def positions(self) -> dict[int, tuple[int, int]]:
        position_mass: dict[int, dict[tuple[int, int], float]] = {}
        total_mass: dict[int, float] = {}
        for hypothesis in self._hypotheses:
            for entity in hypothesis.entities:
                mass = hypothesis.weight * entity.existence
                if mass <= 0.0:
                    continue
                position = (
                    int(entity.position[0]),
                    int(entity.position[1]),
                )
                cells = position_mass.setdefault(entity.index, {})
                cells[position] = cells.get(position, 0.0) + mass
                total_mass[entity.index] = total_mass.get(
                    entity.index, 0.0
                ) + mass
        self_probability = self.self_posterior()
        cell_candidates: dict[
            tuple[int, int], list[tuple[float, int]]
        ] = {}
        for identity, cells in position_mass.items():
            if total_mass.get(identity, 0.0) < 0.25:
                continue
            position = max(cells, key=cells.get)
            score = (
                cells[position]
                + 2.0 * self_probability.get(identity, 0.0)
            )
            cell_candidates.setdefault(position, []).append(
                (score, identity)
            )
        # A binary occupancy cell cannot expose two identities when physical
        # points merge. Emit one delayed-commitment identity per observed
        # cell, prioritizing a strongly supported self identity and otherwise
        # the largest marginal branch mass. Hidden alternatives remain inside
        # the hypothesis bank and can separate again later.
        result: dict[int, tuple[int, int]] = {}
        for position, candidates in cell_candidates.items():
            _, identity = max(candidates)
            result[identity] = position
        return dict(sorted(result.items()))

    def probability(self) -> np.ndarray:
        dynamic_probability = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float64
        )
        for hypothesis in self._hypotheses:
            branch_probability: dict[tuple[int, int], float] = {}
            for entity in hypothesis.entities:
                x, y = int(entity.position[0]), int(entity.position[1])
                prior = branch_probability.get((x, y), 0.0)
                branch_probability[(x, y)] = (
                    1.0 - (1.0 - prior) * (1.0 - entity.existence)
                )
            for (x, y), branch_value in branch_probability.items():
                dynamic_probability[y, x] += (
                    hypothesis.weight * branch_value
                )
        static_probability = np.zeros_like(dynamic_probability)
        static_probability[self._last_static] = 0.96
        probability = 1.0 - (
            1.0 - static_probability
        ) * (1.0 - dynamic_probability)
        np.clip(probability, 0.0, 0.995, out=probability)
        visible_empty = self._last_visibility & ~self._last_sensed
        probability[visible_empty] = np.minimum(
            probability[visible_empty], 0.02
        )
        probability[self._last_sensed] = np.maximum(
            probability[self._last_sensed], 0.99
        )
        return probability

    def dynamic_positions(self) -> set[tuple[int, int]]:
        """Cells occupied by entities with observed non-stationary history."""

        result: set[tuple[int, int]] = set()
        for hypothesis in self._hypotheses:
            for entity in hypothesis.entities:
                non_stay_updates = float(
                    entity.motion_delta_counts[1:5].sum() - 4.0
                )
                if entity.existence >= 0.35 and non_stay_updates >= 1.0:
                    result.add(
                        (
                            int(entity.position[0]),
                            int(entity.position[1]),
                        )
                    )
        return result

    @property
    def learnable_parameter_count(self) -> int:
        per_entity = 5 * DELTA_CATEGORIES + DELTA_CATEGORIES
        return (
            self.grid_size * self.grid_size
            + self.maximum_hypotheses * self.maximum_entities * per_entity
        )

    @property
    def active_state_bytes(self) -> int:
        # Conservative total-bank accounting established by the V4 review.
        # The nominal per-slot allocation is supplemented by an 8 KiB guard
        # for Python/NumPy headers, containers, RNG state, and allocator
        # variation; the full-agent deep-size regression checks the total.
        per_entity = 768
        return (
            self.maximum_hypotheses * self.maximum_entities * per_entity
            + self.grid_size * self.grid_size * 3
            + self.maximum_hypotheses
            * (256 + self.maximum_entities * 8)
            + 8192
        )

    @property
    def estimated_mac_per_step(self) -> int:
        assignments = (
            self.maximum_hypotheses
            * self.maximum_entities
            * self.maximum_hypotheses
            * (2 * VIEW_RADIUS + 1) ** 2
            * 5
            * 24
        )
        rendering = (
            self.maximum_hypotheses * self.maximum_entities * 8
            + self.grid_size * self.grid_size * 4
        )
        return assignments + rendering


class IntegratedBeliefAgentV2:
    """Single-stream I1-V2 agent backed by one entity belief graph."""

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
        self.front_end = DynamicCellFrontEnd(
            grid_size, infer_occlusion=infer_occlusion
        )
        self.graph = EntityBeliefGraph(grid_size)
        self._action_rng = np.random.default_rng(seed + 80_000)

    def update(self, sensed_occupancy: np.ndarray, action: int) -> None:
        supplied_action = (
            int(action)
            if self.use_action
            else int(self._action_rng.integers(0, 5))
        )
        detections, static = self.front_end.update(
            sensed_occupancy,
            protected_cells=self.graph.dynamic_positions(),
        )
        self.graph.update(
            detections,
            self.front_end.last_visibility,
            self.front_end.last_sensed,
            static,
            supplied_action,
        )

    def self_track_identity(self) -> int | None:
        return self.graph.self_identity()

    def track_positions(self) -> dict[int, tuple[int, int]]:
        return self.graph.positions()

    def probability(self) -> np.ndarray:
        return self.graph.probability()

    @property
    def learnable_parameter_count(self) -> int:
        return self.graph.learnable_parameter_count

    @property
    def active_state_bytes(self) -> int:
        return self.front_end.active_state_bytes + self.graph.active_state_bytes

    @property
    def estimated_mac_per_step(self) -> int:
        ray_casting = (
            (2 * VIEW_RADIUS + 1) ** 2 * (2 * VIEW_RADIUS + 2)
        )
        return self.graph.estimated_mac_per_step + ray_casting
