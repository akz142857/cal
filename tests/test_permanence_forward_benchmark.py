"""Tests for the item-3 forward-model permanence benchmark.

The scientific point of the benchmark is that, on the randomized world, an
occupancy belief that models the hidden-maneuver kernel must clearly beat
belief-free constant-velocity extrapolation.  These tests lock that ordering
and the determinism of the harness, plus a smoke check of the GRU baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from cal.evaluation._permanence_gru_baseline import (
    gru_predictor_pairs,
    parameter_count,
)
from cal.evaluation._permanence_slot_baseline import (
    parameter_count as slot_parameter_count,
    slot_predictor_maps,
)
from cal.evaluation.permanence_forward_benchmark import (
    _belief_occupancy,
    _collect,
    _score_maps,
    gru_capacity_sweep,
    run_benchmark,
)


def test_belief_beats_geometric_extrapolation():
    report = run_benchmark(
        [61000, 61001, 61002],
        [61100, 61101, 61102, 61103],
        steps=140,
        warmup=12,
        turn_probability=0.35,
    )
    belief = report["predictors"]["belief"]
    geometric = report["predictors"]["geometric"]
    assert report["evaluation_sample_count"] > 0
    # Full-field ranking: a belief that represents hidden-maneuver uncertainty
    # ranks the true cell above the whole field of empty hidden cells far more
    # often than constant-velocity extrapolation. Chance top-1 ~ 1/candidates.
    assert belief["top1_accuracy"] > geometric["top1_accuracy"]
    assert belief["mrr"] > geometric["mrr"]
    assert belief["top1_accuracy"] > 5.0 / belief["mean_candidate_count"]  # >> chance
    # And it is much better calibrated: constant-velocity extrapolation puts a
    # confident point mass on a single (often now-visible, out-of-field) cell,
    # so its field-normalized categorical NLL explodes.
    assert belief["categorical_nll"] < geometric["categorical_nll"]


def test_first_hidden_transition_cannot_turn():
    visible = np.zeros((25, 25), dtype=bool)
    occupancy = _belief_occupancy(
        (10, 12),
        (1, 0),
        1,
        frozenset(),
        visible,
        0.35,
    )
    assert occupancy == {(11, 12): 1.0}


def test_collected_velocities_are_cardinal_unit_vectors():
    samples = _collect(61104, steps=200, warmup=12, turn_probability=0.35)
    assert samples
    assert all(
        abs(sample.observed_velocity[0]) + abs(sample.observed_velocity[1]) == 1
        for sample in samples
    )


def test_benchmark_is_deterministic():
    kwargs = dict(steps=120, warmup=12, turn_probability=0.35)
    first = run_benchmark([61000, 61001], [61100, 61101], **kwargs)
    second = run_benchmark([61000, 61001], [61100, 61101], **kwargs)
    assert first["predictors"]["belief"] == second["predictors"]["belief"]
    assert first["predictors"]["geometric"] == second["predictors"]["geometric"]


def test_gru_baseline_within_budget_and_runs():
    assert parameter_count() < 100_000  # V2 learnable-parameter budget
    train = _collect(61000, steps=140, warmup=12, turn_probability=0.35)
    evaluation = _collect(61100, steps=140, warmup=12, turn_probability=0.35)
    assert train and evaluation
    pairs = gru_predictor_pairs(train, evaluation, epochs=2)
    assert len(pairs) == len(evaluation)
    for p_pos, p_neg in pairs:
        assert 0.0 <= p_pos <= 1.0
        assert 0.0 <= p_neg <= 1.0


def test_benchmark_reports_argmax_and_occlusion_bins():
    report = run_benchmark(
        [61000, 61001],
        [61100, 61101],
        steps=140,
        warmup=12,
        turn_probability=0.35,
    )
    for predictor in ("belief", "geometric"):
        assert report["predictors"][predictor]["argmax_position_error"] is not None
        assert report["ranking_by_occlusion_length"][predictor]
    # A belief that models maneuvers ranks the true cell above the field more
    # often than extrapolation (geometric's point mass can have a smaller argmax
    # distance yet a worse full-field rank).
    assert (
        report["predictors"]["belief"]["top1_accuracy"]
        > report["predictors"]["geometric"]["top1_accuracy"]
    )
    assert report["predictors"]["belief"]["empty_map_rate"] == 0.0


def test_empty_map_is_penalized_and_reported():
    sample = _collect(61100, steps=140, warmup=12, turn_probability=0.35)[0]
    score = _score_maps([sample], np.zeros((1, 121), dtype=np.float64))
    assert score["argmax_position_error"] == 20.0
    assert score["empty_map_rate"] == 1.0


def test_empty_benchmark_configuration_has_clear_error():
    with pytest.raises(ValueError, match="no valid hidden-object samples"):
        run_benchmark(
            [61000],
            [61100],
            steps=0,
            warmup=12,
            turn_probability=0.35,
        )


def test_slot_baseline_runs_and_shapes():
    assert slot_parameter_count() > 0
    train = _collect(61000, steps=140, warmup=12, turn_probability=0.35)
    evaluation = _collect(61100, steps=140, warmup=12, turn_probability=0.35)
    maps = slot_predictor_maps(train, evaluation, epochs=2)
    assert maps.shape == (len(evaluation), 121)
    assert (maps >= 0.0).all() and (maps <= 1.0).all()


def test_entity_graph_predictor_runs():
    report = run_benchmark(
        [61000],
        [61100],
        steps=140,
        warmup=12,
        turn_probability=0.35,
        include_entity_graph=True,
    )
    entity_graph = report["predictors"]["entity_graph"]
    assert 0.0 <= entity_graph["top1_accuracy"] <= 1.0
    assert 0.0 <= entity_graph["mrr"] <= 1.0
    assert entity_graph["argmax_position_error"] is not None
    assert report["ranking_by_occlusion_length"]["entity_graph"]


def test_gru_capacity_sweep_structure():
    result = gru_capacity_sweep(
        [61000, 61001, 61002],
        [61100, 61101],
        steps=140,
        warmup=12,
        turn_probability=0.35,
        hidden_sizes=(16, 64),
        epochs_grid=(3,),
    )
    assert len(result["runs"]) == 2
    for run in result["runs"]:
        assert run["parameter_count"] > 0
        assert 0.0 <= run["top1_accuracy"] <= 1.0
        assert 0.0 <= run["mrr"] <= 1.0
