"""Tests for the unified V2-I1 entity belief graph."""

from __future__ import annotations

import inspect
import sys

import numpy as np

from cal.evaluation.v2_i1_integration_v2 import _episode
from cal.model.entity_belief_graph import (
    DELTA_CATEGORIES,
    STATIC_THRESHOLD,
    DynamicCellFrontEnd,
    EntityBelief,
    EntityBeliefGraph,
    GlobalHypothesis,
    IntegratedBeliefAgentV2,
)


def test_front_end_learns_static_cell_but_keeps_moving_cell() -> None:
    front_end = DynamicCellFrontEnd(25, infer_occlusion=False)
    patch = np.zeros((11, 11), dtype=np.uint8)
    patch[3, 3] = 1
    for _ in range(STATIC_THRESHOLD):
        detections, static = front_end.update(patch)

    assert static[10, 10]
    assert (10, 10) not in {tuple(point) for point in detections}

    moved = patch.copy()
    moved[7, 8] = 1
    detections, _ = front_end.update(moved)

    assert (15, 14) in {tuple(point) for point in detections}


def test_front_end_does_not_absorb_a_known_mover_during_pause() -> None:
    front_end = DynamicCellFrontEnd(25, infer_occlusion=False)
    patch = np.zeros((11, 11), dtype=np.uint8)
    patch[3, 3] = 1
    protected = {(10, 10)}

    for _ in range(STATIC_THRESHOLD + 3):
        detections, static = front_end.update(
            patch, protected_cells=protected
        )

    assert not static[10, 10]
    assert (10, 10) in {tuple(point) for point in detections}


def test_agent_wiring_preserves_identity_through_long_pause() -> None:
    agent = IntegratedBeliefAgentV2(infer_occlusion=False)

    def observation(local_x: int) -> np.ndarray:
        patch = np.zeros((11, 11), dtype=np.uint8)
        patch[5, local_x] = 1
        return patch

    agent.update(observation(3), 2)
    agent.update(observation(4), 2)
    original_identity = next(iter(agent.track_positions()))
    for _ in range(STATIC_THRESHOLD + 3):
        agent.update(observation(4), 0)
    agent.update(observation(5), 2)

    assert agent.track_positions()[original_identity] == (12, 12)


def _entity(
    index: int,
    *,
    position: tuple[int, int] = (6, 6),
    velocity: tuple[int, int] = (0, 0),
    existence: float = 1.0,
    self_logit: float = 0.0,
) -> EntityBelief:
    return EntityBelief(
        index=index,
        position=np.asarray(position, dtype=np.int16),
        velocity=np.asarray(velocity, dtype=np.int16),
        last_seen=4,
        age=4,
        missed=0,
        existence=existence,
        self_logit=self_logit,
        action_delta_counts=np.ones(
            (5, DELTA_CATEGORIES), dtype=np.float32
        ),
        motion_delta_counts=np.ones(
            DELTA_CATEGORIES, dtype=np.float32
        ),
    )


def test_hypothesis_dedup_uses_complete_future_relevant_state() -> None:
    graph = EntityBeliefGraph(maximum_hypotheses=4)
    stationary = _entity(1)
    moving = _entity(1, velocity=(1, 0))

    selected = graph._select_hypotheses(
        [
            GlobalHypothesis([stationary], 0.0),
            GlobalHypothesis([moving], 0.0),
        ]
    )

    assert len(selected) == 2
    assert {tuple(item.entities[0].velocity) for item in selected} == {
        (0, 0),
        (1, 0),
    }


def test_exact_duplicate_hypotheses_combine_probability_mass() -> None:
    graph = EntityBeliefGraph(maximum_hypotheses=4)
    duplicate = _entity(1)
    alternative = _entity(2, position=(7, 6))

    selected = graph._select_hypotheses(
        [
            GlobalHypothesis([duplicate], 0.0),
            GlobalHypothesis([duplicate.clone()], 0.0),
            GlobalHypothesis([alternative], 0.0),
        ]
    )
    weights = {
        item.entities[0].index: item.weight for item in selected
    }

    assert len(selected) == 2
    assert np.isclose(weights[1], 2.0 / 3.0)
    assert np.isclose(weights[2], 1.0 / 3.0)


def test_self_posterior_keeps_competitors_and_explicit_null_mass() -> None:
    graph = EntityBeliefGraph()
    graph._hypotheses = [
        GlobalHypothesis(
            [
                _entity(1, self_logit=3.0),
                _entity(2, self_logit=2.99),
            ],
            0.0,
            1.0,
        )
    ]

    posterior = graph.self_posterior()

    assert sum(posterior.values()) < 1.0
    assert max(posterior.values()) < 0.55
    assert graph.self_identity() is None


def test_self_posterior_has_no_existence_or_age_cutoff_jump() -> None:
    graph = EntityBeliefGraph()
    incumbent = _entity(1, self_logit=4.0)
    competitor = _entity(2, existence=0.349999, self_logit=10.0)
    competitor.age = 3
    graph._hypotheses = [
        GlobalHypothesis([incumbent, competitor], 0.0, 1.0)
    ]
    before = graph.self_posterior()
    competitor.existence = 0.35
    competitor.age = 4
    after = graph.self_posterior()

    assert before[2] > before[1]
    assert after[2] > after[1]
    assert abs(after[2] - before[2]) < 0.01


def test_co_located_occupancy_uses_bernoulli_union_per_branch() -> None:
    graph = EntityBeliefGraph()
    graph._hypotheses = [
        GlobalHypothesis(
            [
                _entity(1, existence=0.6),
                _entity(2, existence=0.6),
            ],
            0.0,
            1.0,
        )
    ]

    assert np.isclose(graph.probability()[6, 6], 0.84)


def test_other_motion_mass_is_not_renormalized_into_local_moves() -> None:
    graph = EntityBeliefGraph()
    entity = _entity(1)
    entity.action_delta_counts[:, 5] = 95.0
    entity.motion_delta_counts[5] = 95.0

    prediction = graph._prediction_distribution(
        entity,
        action=0,
        static=np.zeros((25, 25), dtype=bool),
    )

    assert sum(prediction.values()) < 0.2


def _deep_size(value: object, seen: set[int] | None = None) -> int:
    visited = seen if seen is not None else set()
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            _deep_size(key, visited) + _deep_size(item, visited)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_deep_size(item, visited) for item in value)
    if hasattr(value, "__dict__"):
        size += _deep_size(vars(value), visited)
    for slot in getattr(type(value), "__slots__", ()):
        if hasattr(value, slot):
            size += _deep_size(getattr(value, slot), visited)
    return size


def test_declared_state_bound_covers_a_full_hypothesis_bank() -> None:
    graph = EntityBeliefGraph(
        maximum_hypotheses=5,
        maximum_entities=11,
    )
    graph._hypotheses = [
        GlobalHypothesis(
            [
                _entity(
                    hypothesis * 12 + index,
                    position=(index, hypothesis),
                )
                for index in range(11)
            ],
            0.0,
            0.2,
        )
        for hypothesis in range(5)
    ]

    assert _deep_size(graph) <= graph.active_state_bytes
    assert graph.active_state_bytes < 64 * 1024

    agent = IntegratedBeliefAgentV2()
    agent.graph._hypotheses = graph._hypotheses
    assert _deep_size(agent) <= agent.active_state_bytes


def test_one_shared_store_drives_all_outputs() -> None:
    agent = IntegratedBeliefAgentV2()
    parameters = set(inspect.signature(agent.update).parameters)

    assert parameters == {"sensed_occupancy", "action"}
    assert hasattr(agent, "graph")
    assert not hasattr(agent, "memory")
    assert agent.learnable_parameter_count <= 100_000
    assert agent.active_state_bytes <= 64 * 1024
    assert agent.estimated_mac_per_step <= 5_000_000


def test_action_conditioned_self_requires_real_action_alignment() -> None:
    formal = _episode(30000, steps=200)
    no_action = _episode(30000, steps=200, use_action=False)
    shuffled = _episode(30000, steps=200, shuffle_lag=5)

    assert formal["self_f1"] >= 0.90
    assert no_action["self_f1"] <= formal["self_f1"] - 0.15
    assert shuffled["self_f1"] <= formal["self_f1"] - 0.15


def test_multi_hypothesis_identity_and_entity_conditioned_permanence() -> None:
    formal = _episode(30001, steps=200)
    all_visible = _episode(
        30001, steps=200, infer_occlusion=False
    )

    assert formal["identity_consistency"] >= 0.90
    assert formal["visible_identity_coverage"] >= 0.90
    assert formal["distractor_hidden_probability"] >= 0.55
    assert all_visible["distractor_hidden_probability"] <= 0.55
    assert (
        formal["distractor_hidden_probability"]
        > all_visible["distractor_hidden_probability"]
    )
