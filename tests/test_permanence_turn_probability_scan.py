"""Tests for model-independent hidden-turn difficulty selection."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import cal.evaluation.permanence_turn_probability_scan as turn_scan
from cal.evaluation.permanence_seed_registry import coverage_contract
from cal.evaluation.permanence_turn_probability_scan import (
    build_scan_artifact,
    evaluate_row,
    run_scan,
    selection_contract,
    validate_scan_artifact,
    validate_seed_registry_for_scan,
)


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = (
    ROOT / "experiments" / "V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY_V2.json"
)
SCAN_PATH = (
    ROOT
    / "experiments"
    / "V2_P1_PERMANENCE_TURN_PROBABILITY_DEVELOPMENT_SCAN_V2.json"
)


def _report(*, belief_top1: float, geometric_top1: float, chance: float):
    def predictor(top1: float, brier: float):
        return {
            "complete_seed_count": 16,
            "by_occlusion_length": {
                "6+": {
                    "top1_accuracy": top1,
                    "mean_any_positive_top1_chance": chance,
                    "brier": brier,
                }
            },
        }

    return {
        "evaluation_seed_coverage": {"all_seeds_have_samples": True},
        "episode_binned_predictors": {
            "belief": predictor(belief_top1, 0.2),
            "geometric": predictor(geometric_top1, 0.4),
        },
    }


def test_turn_probability_contract_is_model_independent():
    contract = selection_contract()
    assert contract["candidate_or_learned_model_metrics_must_not_be_read"] is True
    assert "exact_kernel_belief_oracle" in contract["permitted_selection_sources"]
    assert "geometric_shortcut" in contract["permitted_selection_sources"]
    assert contract["selection"] == "first_candidate_passing_all_conditions"
    assert list(contract["candidates_in_ascending_order"]) == sorted(
        contract["candidates_in_ascending_order"]
    )


def test_turn_probability_row_requires_shortcut_failure_and_oracle_headroom():
    accepted = evaluate_row(
        0.35,
        _report(belief_top1=0.30, geometric_top1=0.07, chance=0.04),
        16,
    )
    rejected = evaluate_row(
        0.25,
        _report(belief_top1=0.30, geometric_top1=0.20, chance=0.04),
        16,
    )
    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert rejected["conditions"]["geometric_6plus_near_chance"] is False
    assert set(accepted["conditions"]) == set(selection_contract()["conditions"])
    assert accepted["long_6plus"]["geometric_top1_minus_chance_top1"] == (
        pytest.approx(0.03)
    )


def test_run_scan_never_enables_candidate_or_neural_metrics(monkeypatch):
    calls = []

    def fake_run_benchmark(train_seeds, evaluation_seeds, **kwargs):
        calls.append((train_seeds, evaluation_seeds, kwargs))
        probability = kwargs["turn_probability"]
        geometric = 0.07 if probability >= 0.35 else 0.20
        return _report(belief_top1=0.30, geometric_top1=geometric, chance=0.04)

    monkeypatch.setattr(turn_scan, "run_benchmark", fake_run_benchmark)
    report = run_scan(list(range(40)), list(range(100, 116)))
    assert report["selected_turn_probability"] == 0.35
    assert [call[2]["turn_probability"] for call in calls] == list(
        turn_scan.CANDIDATE_TURN_PROBABILITIES
    )
    assert all(
        call[2]["include_entity_graph"] is False
        and call[2]["include_gru"] is False
        and call[2]["include_slot"] is False
        and call[2]["paired_bootstrap_samples"] == 0
        for call in calls
    )


def _registry():
    return {
        "status": "development_only_non_gated",
        "model_metrics_read": False,
        "coverage_contract": coverage_contract(),
        "selection_digest_sha256": "a" * 64,
        "train_seeds": list(range(40)),
        "evaluation_seeds": list(range(100, 116)),
    }


def test_registry_validation_locks_cross_probability_contract_and_split():
    validate_seed_registry_for_scan(_registry())
    drifted_contract = copy.deepcopy(_registry())
    drifted_contract["coverage_contract"]["coverage_turn_probabilities"] = [0.35]
    with pytest.raises(ValueError, match="coverage contract"):
        validate_seed_registry_for_scan(drifted_contract)
    duplicate_seed = _registry()
    duplicate_seed["evaluation_seeds"][0] = duplicate_seed["train_seeds"][0]
    with pytest.raises(ValueError, match="must be unique"):
        validate_seed_registry_for_scan(duplicate_seed)


def test_scan_artifact_is_digest_bound_and_reproducible(monkeypatch):
    def fake_run_benchmark(train_seeds, evaluation_seeds, **kwargs):
        probability = kwargs["turn_probability"]
        geometric = 0.07 if probability >= 0.35 else 0.20
        return _report(belief_top1=0.30, geometric_top1=geometric, chance=0.04)

    monkeypatch.setattr(turn_scan, "run_benchmark", fake_run_benchmark)
    first = build_scan_artifact(_registry(), registry_path="registry.json")
    second = build_scan_artifact(_registry(), registry_path="registry.json")
    assert first == second
    assert first["selected_turn_probability"] == 0.35
    assert len(first["scan_digest_sha256"]) == 64
    assert first["seed_registry_selection_digest_sha256"] == "a" * 64
    assert first["reproduction_command"].endswith(
        "--seed-registry registry.json"
    )


def test_committed_scan_artifact_is_linked_and_internally_valid():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    artifact = json.loads(SCAN_PATH.read_text(encoding="utf-8"))
    validate_scan_artifact(artifact, registry)
    assert artifact["selected_turn_probability"] == next(
        row["turn_probability"] for row in artifact["rows"] if row["accepted"]
    )
    assert registry["selected_turn_probability"] == artifact[
        "selected_turn_probability"
    ]


def test_scan_artifact_validation_rejects_tampering():
    registry = _registry()
    artifact = {
        "scan_version": turn_scan.SCAN_VERSION,
        "selection_contract": selection_contract(),
        "seed_registry_selection_digest_sha256": registry[
            "selection_digest_sha256"
        ],
        "rows": [],
        "selected_turn_probability": None,
    }
    artifact["rows"] = [
        {
            "turn_probability": probability,
            "accepted": False,
            "conditions": {
                condition: False
                for condition in selection_contract()["conditions"]
            },
        }
        for probability in turn_scan.CANDIDATE_TURN_PROBABILITIES
    ]
    artifact["scan_digest_sha256"] = turn_scan._scan_digest(artifact)
    validate_scan_artifact(artifact, registry)
    artifact["rows"][0]["accepted"] = True
    with pytest.raises(ValueError, match="accepted flag"):
        validate_scan_artifact(artifact, registry)
