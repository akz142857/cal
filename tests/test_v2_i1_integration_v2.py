"""Protocol and gate tests for the next-generation I1 runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from threading import Barrier

import pytest

import cal.evaluation.v2_i1_integration_v2 as i1_runner
from cal.evaluation.v2_i1_integration_v2 import (
    DEFAULT_PROTOCOL,
    REQUIRED_I1_GATES,
    _identity_metrics,
    _load_result_evidence,
    _load_frozen_protocol,
    _learner_source_has_no_evaluation_imports,
    _mechanism_gates,
    _require_i1_artifact,
    _require_i1_payload,
    _require_prerequisite,
    _reserve_one_shot,
    _reserve_shared_one_shot,
    _publish_result_evidence,
    run_v2_i1_v2,
)
from cal.model.entity_belief_graph import IntegratedBeliefAgentV2


ROOT = Path(__file__).resolve().parents[1]


def test_v2_protocol_is_hash_locked_and_preserves_holdout() -> None:
    protocol, digest = _load_frozen_protocol(ROOT / DEFAULT_PROTOCOL)
    v1 = json.loads(
        (
            ROOT / "experiments/V2_I1_INTEGRATION_PROTOCOL.json"
        ).read_text(encoding="utf-8")
    )

    assert protocol["protocol_version"] == 4
    assert digest == hashlib.sha256(
        (ROOT / DEFAULT_PROTOCOL).read_bytes()
    ).hexdigest()
    assert (
        protocol["amendment_record"][
            "corrected_calibration_runs_before_amendment"
        ] == 0
    )
    assert (
        protocol["amendment_record"][
            "i1_v2_holdout_runs_before_amendment"
        ]
        == 0
    )
    assert protocol["amendment_record"]["fixed_gates_changed"] is True
    assert (
        protocol["amendment_record"][
            "development_or_holdout_seeds_changed"
        ]
        is False
    )
    assert protocol["amendment_record"][
        "model_or_stage_algorithm_changed"
    ] is True
    assert protocol["base_protocol_sha256"] == (
        protocol["amendment_record"]["prior_protocol_sha256"]
    )
    assert len(protocol["holdout"]["seeds"]) == 16
    assert protocol["holdout"]["seeds"] == v1["holdout"]["seeds"]
    for name, value in v1["fixed_gates"].items():
        assert protocol["fixed_gates"][name] == value
    assert protocol["fixed_gates"][
        "visible_identity_coverage_minimum"
    ] == 0.9
    assert protocol["prerequisite"]["artifact_sha256"] == hashlib.sha256(
        (ROOT / protocol["prerequisite"]["artifact"]).read_bytes()
    ).hexdigest()
    assert not (
        set(protocol["calibration"]["seeds"])
        & set(protocol["validation"]["seeds"])
    )
    assert not (
        set(protocol["validation"]["seeds"])
        & set(protocol["holdout"]["seeds"])
    )


def test_mechanism_gates_keep_original_thresholds() -> None:
    protocol, _ = _load_frozen_protocol(ROOT / DEFAULT_PROTOCOL)
    aggregate = {
        "self_f1": 0.91,
        "identity_consistency": 0.91,
        "visible_identity_coverage": 0.91,
        "distractor_hidden_probability": 0.56,
        "no_action_self_f1": 0.75,
        "time_shuffled_self_f1": 0.75,
        "assume_all_visible_hidden_probability": 0.54,
        "paired_formal_beats_visible_control": 0.75,
    }

    assert all(
        _mechanism_gates(aggregate, protocol["fixed_gates"]).values()
    )


def test_formal_update_api_rejects_privileged_inputs() -> None:
    parameters = set(
        inspect.signature(IntegratedBeliefAgentV2.update).parameters
    )

    assert parameters == {"self", "sensed_occupancy", "action"}
    assert not {
        "visibility",
        "mask",
        "identity",
        "self_label",
        "ground_truth",
    } & parameters
    assert _learner_source_has_no_evaluation_imports()


def test_runner_rejects_non_protocol_output(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError, match="output path does not match"
    ):
        run_v2_i1_v2(
            split="calibration",
            output_path=tmp_path / "elsewhere.json",
        )


def test_identity_metrics_penalize_misses_and_forbid_duplicate_assignment() -> None:
    consistency, coverage = _identity_metrics(
        {"a": {7: 8}, "b": {7: 8}},
        {"a": 10, "b": 10},
        {"a": 8, "b": 8},
    )

    assert consistency == 0.4
    assert coverage == 0.8


def test_strict_artifact_validator_rejects_bare_pass(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "bare.json"
    artifact.write_text('{"passed": true}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="schema"):
        _require_i1_artifact(
            artifact,
            split="calibration",
            protocol_digest="digest",
            seeds=(1,),
            decision="next",
        )


def test_strict_artifact_validator_rejects_incomplete_gate_set(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "partial.json"
    artifact.write_text(
        json.dumps(
            {
                "result_schema_version": 1,
                "experiment": "V2-I1-unified-entity-belief-graph",
                "review_split": "calibration",
                "protocol_sha256": "digest",
                "seeds": [1],
                "passed": True,
                "decision": "next",
                "gates": {"only_one": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="gates are incomplete"):
        _require_i1_artifact(
            artifact,
            split="calibration",
            protocol_digest="digest",
            seeds=(1,),
            decision="next",
        )


def test_strict_artifact_validator_recomputes_claimed_gates() -> None:
    protocol, digest = _load_frozen_protocol(ROOT / DEFAULT_PROTOCOL)
    payload = {
        "result_schema_version": 1,
        "experiment": "V2-I1-unified-entity-belief-graph",
        "review_split": "calibration",
        "protocol_sha256": digest,
        "seeds": [1],
        "passed": True,
        "decision": "next",
        "evaluation_labels_used_for_learning": False,
        "privileged_input": None,
        "conditions": {
            "formal": [
                {
                    "self_f1": 0.0,
                    "identity_consistency": 1.0,
                    "visible_identity_coverage": 1.0,
                    "distractor_hidden_probability": 1.0,
                }
            ],
            "no_action": [{"self_f1": 0.0}],
            "time_shuffled": [{"self_f1": 0.0}],
            "assume_all_visible": [
                {"distractor_hidden_probability": 0.0}
            ],
        },
        "aggregate": {
            "self_f1": 0.0,
            "identity_consistency": 1.0,
            "visible_identity_coverage": 1.0,
            "distractor_hidden_probability": 1.0,
            "no_action_self_f1": 0.0,
            "time_shuffled_self_f1": 0.0,
            "assume_all_visible_hidden_probability": 0.0,
            "paired_formal_beats_visible_control": 1.0,
        },
        "resources": {
            "learnable_parameter_count": 2605,
            "active_state_bytes": 56527,
            "estimated_mac_per_step": 3997392,
            "steps_per_seed": 200,
            "maximum_replays_per_experience": 0,
            "cpu_wall_seconds": 0.0,
        },
        "architecture": {
            "shared_entity_store": True,
            "maximum_global_hypotheses": 5,
            "maximum_entities_per_hypothesis": 11,
        },
        "gates": {name: True for name in REQUIRED_I1_GATES},
        "run_start": {
            "git_dirty": False,
            "git_commit": "commit",
            "source_sha256": "source",
        },
        "provenance": {
            "git_dirty": False,
            "git_commit": "commit",
            "source_sha256": "source",
        },
    }

    with pytest.raises(RuntimeError, match="do not recompute"):
        _require_i1_payload(
            payload,
            split="calibration",
            protocol_digest=digest,
            seeds=(1,),
            decision="next",
            protocol=protocol,
        )


def test_one_shot_reservation_is_atomic_and_non_repeatable(
    tmp_path: Path,
) -> None:
    reservation = tmp_path / "reservation.json"
    _reserve_one_shot(
        reservation,
        split="validation",
        protocol_digest="digest",
        git_commit="commit",
    )

    with pytest.raises(RuntimeError, match="already exists"):
        _reserve_one_shot(
            reservation,
            split="validation",
            protocol_digest="digest",
            git_commit="commit",
        )


def test_prerequisite_is_hash_pinned(tmp_path: Path) -> None:
    protocol, _ = _load_frozen_protocol(ROOT / DEFAULT_PROTOCOL)
    prerequisite = dict(protocol["prerequisite"])
    replacement = tmp_path / "stage.json"
    replacement.write_text("{}\n", encoding="utf-8")
    prerequisite["artifact"] = str(replacement)

    with pytest.raises(RuntimeError, match="digest mismatch"):
        _require_prerequisite(prerequisite)


def test_shared_git_registry_is_cross_clone_and_result_is_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = tmp_path / "origin.git"
    seed_repo = tmp_path / "seed"
    subprocess.run(
        ("git", "init", "--bare", str(origin)),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "init", str(seed_repo)),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=seed_repo,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Test"),
        cwd=seed_repo,
        check=True,
    )
    tracked = seed_repo / "tracked.txt"
    tracked.write_text("source\n", encoding="utf-8")
    subprocess.run(
        ("git", "add", "tracked.txt"), cwd=seed_repo, check=True
    )
    subprocess.run(
        ("git", "commit", "-m", "source"),
        cwd=seed_repo,
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=seed_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ("git", "branch", "-M", "main"),
        cwd=seed_repo,
        check=True,
    )
    subprocess.run(
        ("git", "remote", "add", "origin", str(origin)),
        cwd=seed_repo,
        check=True,
    )
    subprocess.run(
        ("git", "push", "-u", "origin", "main"),
        cwd=seed_repo,
        check=True,
        capture_output=True,
    )
    worktrees = [tmp_path / "work-a", tmp_path / "work-b"]
    for worktree in worktrees:
        subprocess.run(
            ("git", "clone", "--branch", "main", str(origin), str(worktree)),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ("git", "config", "user.email", "same@example.invalid"),
            cwd=worktree,
            check=True,
        )
        subprocess.run(
            ("git", "config", "user.name", "Same"),
            cwd=worktree,
            check=True,
        )

    original_exists = i1_runner._remote_tag_exists
    barrier = Barrier(2)

    def synchronized_exists(
        remote: str,
        tag: str,
        *,
        cwd: str | Path | None = None,
    ) -> bool:
        exists = original_exists(remote, tag, cwd=cwd)
        if tag == "validation-consumed" and not exists:
            barrier.wait(timeout=5)
        return exists

    monkeypatch.setattr(
        i1_runner, "_remote_tag_exists", synchronized_exists
    )

    def attempt(worktree: Path) -> tuple[Path, bool]:
        try:
            _reserve_shared_one_shot(
                remote="origin",
                tag="validation-consumed",
                split="validation",
                protocol_digest="protocol",
                git_commit=commit,
                source_sha256="source",
                cwd=worktree,
            )
        except RuntimeError:
            return worktree, False
        return worktree, True

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = list(executor.map(attempt, worktrees))
    assert sum(success for _, success in attempts) == 1
    winner = next(
        worktree for worktree, success in attempts if success
    )
    loser = next(
        worktree for worktree, success in attempts if not success
    )
    loser_local_tag = subprocess.run(
        (
            "git",
            "rev-parse",
            "--verify",
            "refs/tags/validation-consumed",
        ),
        cwd=loser,
        capture_output=True,
        check=False,
    )
    assert loser_local_tag.returncode != 0

    result_path = winner / "result.json"
    result_path.write_text('{"passed": true}\n', encoding="utf-8")
    _publish_result_evidence(
        result_path,
        remote="origin",
        tag="validation-result",
        split="validation",
        protocol_digest="protocol",
        git_commit=commit,
        source_sha256="source",
        cwd=winner,
    )
    payload, certificate = _load_result_evidence(
        remote="origin",
        tag="validation-result",
        split="validation",
        protocol_digest="protocol",
        cwd=winner,
    )

    assert payload == {"passed": True}
    assert certificate["git_commit"] == commit
