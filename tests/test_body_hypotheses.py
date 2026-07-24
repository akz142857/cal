from __future__ import annotations

import numpy as np

from calmodel.model.body_hypotheses import (
    BodyGraphCandidate,
    BodyGraphHypothesisFilter,
)
from calmodel.model.entity_graph import OnlineEntityGraph


def test_complete_hypothesis_has_three_nodes_and_two_edges() -> None:
    candidate = BodyGraphCandidate(base=5, joint=2, endpoint=9)

    assert candidate.nodes == (5, 2, 9)
    assert candidate.edges == ((2, 5), (2, 9))


def test_symmetric_evidence_preserves_normalized_half_posterior() -> None:
    left = BodyGraphCandidate(0, 1, 2)
    right = BodyGraphCandidate(0, 3, 4)
    filter_ = BodyGraphHypothesisFilter()

    for _ in range(20):
        filter_.update(
            (left, right),
            {1: 0.08, 2: 0.06, 3: 0.08, 4: 0.06},
        )

    probabilities = filter_.probability_map()
    assert sum(probabilities.values()) == 1.0
    assert probabilities[left] == 0.5
    assert probabilities[right] == 0.5


def test_asymmetric_prediction_errors_select_complete_hypothesis() -> None:
    left = BodyGraphCandidate(0, 1, 2)
    right = BodyGraphCandidate(0, 3, 4)
    filter_ = BodyGraphHypothesisFilter()
    filter_.update((left, right), {})

    for _ in range(3):
        filter_.update(
            (left, right),
            {1: 0.02, 2: 0.03, 3: 0.55, 4: 0.60},
        )

    probabilities = filter_.probability_map()
    assert probabilities[left] > 0.99
    assert probabilities[right] < 0.01
    assert sum(probabilities.values()) == 1.0


def test_complete_candidates_survive_temporary_rediscovery_gap() -> None:
    left = BodyGraphCandidate(0, 1, 2)
    right = BodyGraphCandidate(0, 3, 4)
    filter_ = BodyGraphHypothesisFilter()
    filter_.update((left, right), {})

    filter_.update((left,), {})

    assert set(filter_.probability_map()) == {left, right}
    assert filter_.probability_map()[left] == 0.5
    assert filter_.probability_map()[right] == 0.5


def test_two_hypothesis_space_rejects_late_spurious_candidate() -> None:
    left = BodyGraphCandidate(0, 1, 2)
    right = BodyGraphCandidate(0, 3, 4)
    spurious = BodyGraphCandidate(0, 1, 4)
    filter_ = BodyGraphHypothesisFilter()
    filter_.update((left, right), {})

    filter_.update((left, right, spurious), {})

    assert set(filter_.probability_map()) == {left, right}


def test_missing_branch_observation_cannot_win_by_absence() -> None:
    left = BodyGraphCandidate(0, 1, 2)
    right = BodyGraphCandidate(0, 3, 4)
    filter_ = BodyGraphHypothesisFilter()
    filter_.update((left, right), {})

    filter_.update(
        (left, right),
        {1: 0.20, 2: 0.20, 3: 0.20},
    )

    assert filter_.probability_map()[left] == 0.5
    assert filter_.probability_map()[right] == 0.5


def test_candidate_discovery_uses_geometry_and_action_roles() -> None:
    class GraphStub:
        def control_matrices(self) -> dict[int, np.ndarray]:
            base = np.zeros((2, 4))
            joint_left = np.zeros((2, 4))
            endpoint_left = np.zeros((2, 4))
            joint_right = np.zeros((2, 4))
            endpoint_right = np.zeros((2, 4))
            joint_left[0, 0] = 0.30
            joint_right[0, 0] = 0.30
            endpoint_left[1, 2] = 0.30
            endpoint_right[1, 2] = 0.30
            return {
                0: base,
                1: joint_left,
                2: endpoint_left,
                3: joint_right,
                4: endpoint_right,
            }

        def rigid_edges(
            self,
            *,
            minimum_samples: int,
            maximum_variance: float,
            maximum_length: float,
        ) -> set[tuple[int, int]]:
            assert minimum_samples == 12
            assert maximum_variance == 0.03
            assert maximum_length == 3.35
            return {(0, 1), (1, 2), (0, 3), (3, 4)}

    candidates = BodyGraphHypothesisFilter().discover_candidates(GraphStub())  # type: ignore[arg-type]

    assert candidates == (
        BodyGraphCandidate(0, 1, 2),
        BodyGraphCandidate(0, 3, 4),
    )


def test_visible_unmatched_track_emits_causal_prediction_error() -> None:
    graph = OnlineEntityGraph(
        1,
        maximum_tracks=1,
        match_radius=0.1,
        association_mode="nearest",
    )
    graph.reset(np.asarray(((0.0, 0.0),)))

    graph.update(
        np.asarray(((1.0, 0.0),)),
        np.asarray((1.0,)),
    )

    assert graph.causal_prediction_errors()[0] == 1.0
