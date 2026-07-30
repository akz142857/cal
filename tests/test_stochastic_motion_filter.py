from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from cal.evaluation.permanence_forward_benchmark import _successors
from cal.evaluation.stochastic_permanence_kernel_diagnostic import (
    PrivilegedUnprunedKinematicReference,
    _deep_size,
    _known_topology_kernel_alignment,
)
from cal.model.stochastic_motion_filter import (
    EmptyPosteriorError,
    GridSpec,
    PackedKinematicFilter,
    PackedPosteriorPool,
    autonomous_successors,
    bayesian_no_detection_update,
    position_total_variation,
    probabilistic_bounce_distribution,
)


def _static_map(cells: set[tuple[int, int]]) -> np.ndarray:
    result = np.zeros((25, 25), dtype=np.float64)
    for x, y in cells:
        result[y, x] = 1.0
    return result


def test_known_topology_successors_match_privileged_world_kernel() -> None:
    static = frozenset({(14, 12), (9, 10)})
    model = autonomous_successors(
        (13, 12),
        (1, 0),
        _static_map(set(static)),
        turn_probability=0.35,
        allow_turn=True,
        spec=GridSpec(),
    )
    world = _successors((13, 12), (1, 0), static, 0.35)

    model_probabilities = {
        (position, velocity): probability
        for position, velocity, probability in model
    }
    world_probabilities: dict[
        tuple[tuple[int, int], tuple[int, int]], float
    ] = {}
    for position, velocity, probability in world:
        key = (position, velocity)
        world_probabilities[key] = world_probabilities.get(key, 0.0) + probability

    assert model_probabilities.keys() == world_probabilities.keys()
    for key, probability in model_probabilities.items():
        assert np.isclose(probability, world_probabilities[key])


def test_probabilistic_bounce_marginalizes_unknown_forward_and_reflected_cells() -> None:
    static = np.zeros((25, 25), dtype=np.float64)
    static[12, 14] = 0.25
    static[12, 12] = 0.50

    distribution = probabilistic_bounce_distribution(
        (13, 12), (1, 0), static, GridSpec()
    )
    probabilities = {
        (position, velocity): probability
        for position, velocity, probability in distribution
    }

    assert probabilities[((14, 12), (1, 0))] == 0.75
    assert probabilities[((12, 12), (-1, 0))] == 0.125
    assert probabilities[((13, 12), (-1, 0))] == 0.125


def test_probabilistic_bounce_matches_explicit_static_map_mixture() -> None:
    static_probability = np.zeros((25, 25), dtype=np.float64)
    static_probability[12, 14] = 0.25
    static_probability[12, 12] = 0.50
    candidate = {
        (position, velocity): probability
        for position, velocity, probability in probabilistic_bounce_distribution(
            (13, 12), (1, 0), static_probability, GridSpec()
        )
    }
    mixture: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for forward_static, reflected_static in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        weight = (0.25 if forward_static else 0.75) * (
            0.50 if reflected_static else 0.50
        )
        static = frozenset(
            cell
            for cell, occupied in (
                ((14, 12), forward_static),
                ((12, 12), reflected_static),
            )
            if occupied
        )
        for position, velocity, probability in _successors(
            (13, 12), (1, 0), static, 0.0
        ):
            key = (position, velocity)
            mixture[key] = mixture.get(key, 0.0) + weight * probability

    assert candidate.keys() == mixture.keys()
    for key in candidate:
        assert np.isclose(candidate[key], mixture[key])


def test_known_topology_kernel_matches_world_exhaustively() -> None:
    alignment = _known_topology_kernel_alignment()

    assert alignment["checked_transition_cases"] == 38_400
    assert alignment["support_mismatch_count"] == 0
    assert alignment["maximum_probability_l1"] <= 1e-12


def test_packed_filter_matches_unpruned_reference_without_pruning() -> None:
    exact = PrivilegedUnprunedKinematicReference()
    packed = PackedKinematicFilter(k_max=512)
    exact.reset((13, 12), (1, 0))
    packed.reset((13, 12), (1, 0))
    static = _static_map({(10, 9), (10, 10), (15, 14)})
    no_detection = np.ones((25, 25), dtype=np.float64)
    no_detection[12, 16] = 0.0

    for hidden_index in range(12):
        exact.step(
            static_probability=static,
            no_detection_probability=no_detection,
            turn_probability=0.35,
            allow_turn=hidden_index > 0,
        )
        packed.step(
            static_probability=static,
            no_detection_probability=no_detection,
            turn_probability=0.35,
            allow_turn=hidden_index > 0,
        )
        assert position_total_variation(
            packed.position_marginal(), exact.position_marginal()
        ) < 1e-6

    assert packed.cumulative_pruned_mass < 1e-7


def test_pruned_mass_is_recorded_in_branch_evidence() -> None:
    packed = PackedKinematicFilter(k_max=1)
    packed.reset((12, 12), (1, 0))

    result = packed.step(
        static_probability=np.zeros((25, 25), dtype=np.float64),
        no_detection_probability=np.ones((25, 25), dtype=np.float64),
        turn_probability=1.0,
        allow_turn=True,
    )

    assert result["pre_pruning_support"] == 3
    assert np.isclose(result["retained_probability"], 1.0 / 3.0)
    assert np.isclose(packed.cumulative_pruned_mass, 2.0 / 3.0)
    assert np.isclose(packed.branch_log_weight, np.log(1.0 / 3.0))


def test_propagation_overflow_does_not_partially_commit() -> None:
    packed = PackedKinematicFilter(k_max=2)
    packed.reset((12, 12), (1, 0))
    packed._next_codes = np.zeros(1, dtype=np.uint16)
    packed._next_probability = np.zeros(1, dtype=np.float64)
    before = (
        packed.items(),
        packed.branch_log_weight,
        packed.cumulative_retained_probability,
        packed.maximum_step_pruned_mass,
    )

    with pytest.raises(RuntimeError, match="workspace overflow"):
        packed.step(
            static_probability=np.zeros((25, 25), dtype=np.float64),
            no_detection_probability=np.ones((25, 25), dtype=np.float64),
            turn_probability=1.0,
            allow_turn=True,
        )

    assert (
        packed.items(),
        packed.branch_log_weight,
        packed.cumulative_retained_probability,
        packed.maximum_step_pruned_mass,
    ) == before


def test_invalid_probability_grid_fails_before_filter_commit() -> None:
    packed = PackedKinematicFilter(k_max=4)
    packed.reset((12, 12), (1, 0))
    static = np.zeros((25, 25), dtype=np.float64)
    static[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        packed.step(
            static_probability=static,
            no_detection_probability=np.ones((25, 25), dtype=np.float64),
            turn_probability=0.35,
            allow_turn=True,
        )

    assert packed.items() == ((packed.spec.encode((12, 12), (1, 0)), 1.0),)


def test_no_detection_updates_conditional_probability_and_existence_once() -> None:
    posterior, existence, l_no, log_evidence = bayesian_no_detection_update(
        np.asarray([0.75, 0.25]),
        np.asarray([1.0, 0.0]),
        0.8,
    )

    assert np.allclose(posterior, [1.0, 0.0])
    assert np.isclose(l_no, 0.75)
    assert np.isclose(existence, 0.6 / 0.8)
    assert np.isclose(log_evidence, np.log(0.8))


def test_full_packed_pool_is_bounded_and_fully_detached() -> None:
    pool = PackedPosteriorPool(k_max=48)
    pool.fill_fully_detached()

    assert pool.s_max == 5 * 11 * 48
    assert pool.capacity_contract()["fully_detached_safe"] is True
    assert pool.capacity_contract()["direct_index_accumulator"] is True
    assert pool.capacity_contract()["shared_expansion_workspace_size"] == (
        pool.capacity_contract()["shared_expansion_workspace_required"]
    )
    for code in pool.codes:
        pool.spec.decode(int(code))
    assert _deep_size(pool) <= pool.active_state_bytes
    assert pool.active_state_bytes <= 65_536
    assert pool.estimated_mac_per_step <= 5_000_000


def test_packed_pool_overflow_is_atomic() -> None:
    pool = PackedPosteriorPool(k_max=48)
    pool.fill_fully_detached()
    before = (pool.codes.copy(), pool.probability.copy(), pool.counts.copy())

    with pytest.raises(RuntimeError, match="pool overflow"):
        pool.replace_factor_atomic(
            0,
            0,
            np.zeros(49, dtype=np.uint16),
            np.ones(49, dtype=np.float64),
        )

    assert np.array_equal(pool.codes, before[0])
    assert np.array_equal(pool.probability, before[1])
    assert np.array_equal(pool.counts, before[2])


def test_packed_accumulator_is_direct_index_not_quadratic_scan() -> None:
    source = inspect.getsource(PackedKinematicFilter._accumulate)

    assert "flatnonzero" not in source
    assert "_next_index" in source


def test_candidate_filter_source_has_no_evaluation_import() -> None:
    source = inspect.getsource(inspect.getmodule(PackedKinematicFilter))
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    assert not any(name.startswith("cal.evaluation") for name in imported)
