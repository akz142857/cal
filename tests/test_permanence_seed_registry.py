"""Regression tests for model-blind permanence seed selection."""

from __future__ import annotations

import json
from pathlib import Path

import cal.evaluation.permanence_seed_registry as seed_registry
from cal.evaluation.permanence_seed_registry import (
    LOCKED_DEVELOPMENT_TRAIN_COUNT,
    LOCKED_PHASE0_EVALUATION_SOURCE_COUNT,
    REQUIRED_OCCLUSION_BINS,
    _selection_digest,
    _selection_provenance,
    audit_seed,
    coverage_contract,
    generate_registry,
    reproduce_registry_artifact,
)


REGISTRY_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY_V4.json"
)


def test_coverage_contract_is_model_blind_and_complete():
    contract = coverage_contract()
    assert contract["model_metrics_must_not_be_read"] is True
    assert contract["selection"] == "first_n_accepted_without_manual_substitution"
    assert tuple(contract["required_occlusion_bins"]) == REQUIRED_OCCLUSION_BINS
    assert contract["episode_bin_groups_minimum_per_bin"] >= 1


def test_seed_audit_accepts_and_rejects_by_event_structure():
    # 62003 qualified before the hidden stream became HMAC-keyed; under the new
    # trajectories it no longer produces enough 6+ episode groups, so the
    # accepted example moved to 62002.  Seed identities are not stable across a
    # world change -- only the audit's structure is.
    rejected = audit_seed(62000)
    accepted = audit_seed(62002)
    assert rejected["accepted"] is False
    assert rejected["rejection_reasons"]
    assert accepted["accepted"] is True
    assert accepted["rejection_reasons"] == []
    assert accepted["layout"]["valid"] is True


def test_probability_audit_explicitly_disables_entity_graph(monkeypatch):
    calls = []

    def fake_collect(seed, **kwargs):
        calls.append((seed, kwargs))
        return []

    monkeypatch.setattr(seed_registry, "_collect", fake_collect)
    result = seed_registry._audit_seed_at_probability(7, 0.35)
    assert result["accepted"] is False
    assert len(calls) == 1
    seed, kwargs = calls[0]
    assert seed == 7
    assert kwargs["attach_entity_graph"] is False
    assert kwargs["turn_probability"] == 0.35


def test_generator_selects_first_qualifying_seeds_deterministically():
    first = generate_registry(
        candidate_start=62000,
        candidate_stop=62020,
        train_count=1,
        evaluation_count=1,
    )
    second = generate_registry(
        candidate_start=62000,
        candidate_stop=62020,
        train_count=1,
        evaluation_count=1,
    )
    assert first["train_seeds"] == [62002]
    assert first["evaluation_seeds"] == [62006]
    assert first["selection_digest_sha256"] == second["selection_digest_sha256"]
    assert first["model_metrics_read"] is False
    scanned = sorted(
        first["train_seeds"]
        + first["evaluation_seeds"]
        + [row["seed"] for row in first["rejected_candidates"]]
    )
    assert scanned == list(
        range(
            first["candidate_range"]["start_inclusive"],
            first["candidate_range"]["last_scanned_inclusive"] + 1,
        )
    )


def test_selection_digest_binds_candidate_range_split_and_audit_evidence():
    common = {
        "contract": coverage_contract(),
        "candidate_range": {
            "start_inclusive": 1,
            "stop_exclusive": 10,
            "last_scanned_inclusive": 4,
        },
        "train_seeds": [2],
        "evaluation_seeds": [4],
        "accepted_seed_audits": [{"seed": 2}, {"seed": 4}],
        "rejected_candidates": [
            {"seed": 1, "rejection_reasons": ["insufficient"]},
            {"seed": 3, "rejection_reasons": ["insufficient"]},
        ],
    }
    original = _selection_digest(_selection_provenance(**common))
    changed_split = {**common, "train_seeds": [4], "evaluation_seeds": [2]}
    changed_range = {
        **common,
        "candidate_range": {**common["candidate_range"], "stop_exclusive": 11},
    }
    changed_audit = {
        **common,
        "accepted_seed_audits": [{"seed": 2}, {"seed": 4, "count": 99}],
    }
    assert original != _selection_digest(_selection_provenance(**changed_split))
    assert original != _selection_digest(_selection_provenance(**changed_range))
    assert original != _selection_digest(_selection_provenance(**changed_audit))


def test_committed_registry_matches_generator_and_all_accepted_seeds_reaudit():
    committed = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert committed["requested_counts"] == {
        "train": LOCKED_DEVELOPMENT_TRAIN_COUNT,
        "evaluation": LOCKED_PHASE0_EVALUATION_SOURCE_COUNT,
    }
    assert committed == reproduce_registry_artifact(committed)
