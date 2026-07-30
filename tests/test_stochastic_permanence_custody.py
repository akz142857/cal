from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from threading import Barrier
from typing import Any

import pytest

import cal.evaluation.stochastic_permanence_custody as custody
from cal.evaluation.stochastic_permanence_artifacts import (
    canonical_json_bytes,
    sha256_bytes,
)
from cal.evaluation.stochastic_permanence_custody import (
    CustodyState,
    build_custody_record,
    infrastructure_retry_allowed,
    mark_result_bytes_visible,
    publish_immutable_result_evidence,
    publish_remote_tag_cas,
    reserve_once,
    reserve_remote_split_once,
    transition_custody,
    validate_disjoint_seed_sets,
)


def _locked_record(*, include_holdout: bool = True) -> dict[str, Any]:
    streams = {"train": [1], "development": [2], "validation": [3]}
    if include_holdout:
        streams["holdout"] = [4]
    record = build_custody_record(streams=streams)
    record = transition_custody(
        record,
        CustodyState.BASE_LOCKED,
        evidence={"base_lock_sha256": "a" * 64},
    )
    return transition_custody(
        record,
        CustodyState.CANDIDATE_LOCKED,
        evidence={"candidate_lock_sha256": "b" * 64},
    )


def _terminal_evidence(
    decision: str, *, reservation: str = "d" * 40
) -> dict[str, Any]:
    return {
        "decision": decision,
        "result_sha256": "c" * 64,
        "custody_record_sha256": "e" * 64,
        "git_blob": "f" * 40,
        "reservation_tag_object_sha": reservation,
        "immutable_result_tag_object_sha": "1" * 40,
    }


def _holdout_authorization(record: dict[str, Any]) -> dict[str, Any]:
    validation_evidence = next(
        (
            entry["evidence"]
            for entry in reversed(record["history"])
            if entry.get("to") == CustodyState.VALIDATION_CONSUMED_PASS.value
        ),
        {},
    )
    authorization = {
        "authorization_schema_version": 1,
        "decision": "authorize_holdout_once",
        "split": "holdout",
        "authorized_by": "independent-reviewer",
        "base_lock_sha256": record["base_lock_sha256"],
        "candidate_lock_sha256": record["candidate_lock_sha256"],
        "stream_commitment_sha256": record["stream_commitment_sha256"],
        "validation_result_sha256": validation_evidence.get("result_sha256"),
        "validation_result_tag_object_sha": validation_evidence.get(
            "immutable_result_tag_object_sha"
        ),
        "validation_custody_record_sha256": validation_evidence.get(
            "custody_record_sha256"
        ),
    }
    authorization["authorization_sha256"] = sha256_bytes(
        canonical_json_bytes(authorization)
    )
    return {"separate_human_authorization": authorization}


def _run(
    *arguments: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _prepare_remote(tmp_path: Path) -> tuple[Path, tuple[Path, Path], str]:
    origin = tmp_path / "origin.git"
    seed_repo = tmp_path / "seed"
    _run("git", "init", "--bare", str(origin))
    _run("git", "init", str(seed_repo))
    _run("git", "config", "user.email", "test@example.invalid", cwd=seed_repo)
    _run("git", "config", "user.name", "Test", cwd=seed_repo)
    (seed_repo / "tracked.txt").write_text("source\n", encoding="utf-8")
    _run("git", "add", "tracked.txt", cwd=seed_repo)
    _run("git", "commit", "-m", "source", cwd=seed_repo)
    commit = _run("git", "rev-parse", "HEAD", cwd=seed_repo).stdout.strip()
    _run("git", "branch", "-M", "main", cwd=seed_repo)
    _run("git", "remote", "add", "origin", str(origin), cwd=seed_repo)
    _run("git", "push", "origin", "main", cwd=seed_repo)

    worktrees = (tmp_path / "work-a", tmp_path / "work-b")
    for worktree in worktrees:
        _run(
            "git",
            "clone",
            "--branch",
            "main",
            str(origin),
            str(worktree),
        )
        _run(
            "git",
            "config",
            "user.email",
            "same@example.invalid",
            cwd=worktree,
        )
        _run("git", "config", "user.name", "Same", cwd=worktree)
    return origin, worktrees, commit


def test_seed_streams_must_be_nonempty_integer_unique_and_disjoint() -> None:
    assert validate_disjoint_seed_sets(
        {"train": [2, 1], "development": [3], "historical": [4]}
    ) == {"development": [3], "historical": [4], "train": [1, 2]}

    with pytest.raises(RuntimeError, match="collision") as collision:
        validate_disjoint_seed_sets({"train": [1234567], "validation": [1234567]})
    assert "1234567" not in str(collision.value)
    with pytest.raises(RuntimeError, match="duplicates"):
        validate_disjoint_seed_sets({"train": [1, 1]})
    with pytest.raises(RuntimeError, match="empty"):
        validate_disjoint_seed_sets({"train": []})
    with pytest.raises(TypeError, match="non-integer"):
        validate_disjoint_seed_sets({"train": [1.9]})  # type: ignore[list-item]
    with pytest.raises(TypeError, match="non-integer"):
        validate_disjoint_seed_sets({"train": [True]})
    with pytest.raises(ValueError, match="unsafe seed stream name"):
        validate_disjoint_seed_sets(  # type: ignore[arg-type]
            {"train": [1], 2: [2]}
        )


def test_terminal_transitions_require_visible_immutable_result_evidence() -> None:
    reserved = transition_custody(
        _locked_record(),
        CustodyState.VALIDATION_RESERVED,
        evidence={"reservation_nonce": "before-run"},
    )
    with pytest.raises(RuntimeError, match="terminal result evidence"):
        transition_custody(
            reserved,
            CustodyState.VALIDATION_CONSUMED_PASS,
            evidence={"decision": "pass"},
            result_bytes_visible=True,
        )
    with pytest.raises(RuntimeError, match="matching decision"):
        transition_custody(
            reserved,
            CustodyState.VALIDATION_CONSUMED_PASS,
            evidence=_terminal_evidence("fail"),
            result_bytes_visible=True,
        )
    with pytest.raises(RuntimeError, match="visible result bytes"):
        transition_custody(
            reserved,
            CustodyState.VALIDATION_CONSUMED_PASS,
            evidence=_terminal_evidence("pass"),
        )

    passed = transition_custody(
        reserved,
        CustodyState.VALIDATION_CONSUMED_PASS,
        evidence=_terminal_evidence("pass"),
        result_bytes_visible=True,
    )
    assert passed["result_bytes_visible_by_split"] == {
        "validation": True,
        "holdout": False,
    }
    with pytest.raises(RuntimeError, match="structured human authorization"):
        transition_custody(
            passed,
            CustodyState.HOLDOUT_RESERVED,
            evidence={"separate_human_authorization": True},
        )
    authorization = _holdout_authorization(passed)
    authorization["separate_human_authorization"]["candidate_lock_sha256"] = (
        "0" * 64
    )
    authorization_body = authorization["separate_human_authorization"]
    authorization_body.pop("authorization_sha256")
    authorization_body["authorization_sha256"] = sha256_bytes(
        canonical_json_bytes(authorization_body)
    )
    with pytest.raises(RuntimeError, match="does not bind"):
        transition_custody(
            passed,
            CustodyState.HOLDOUT_RESERVED,
            evidence=authorization,
        )


def test_retry_is_split_scoped_and_stops_at_first_partial_result_byte() -> None:
    reserved = transition_custody(
        _locked_record(),
        CustodyState.VALIDATION_RESERVED,
        evidence={"reservation_nonce": "before-run"},
    )
    assert infrastructure_retry_allowed(reserved)
    assert infrastructure_retry_allowed(reserved, split="validation")
    assert not infrastructure_retry_allowed(reserved, split="holdout")

    partial = mark_result_bytes_visible(
        reserved,
        split="validation",
        evidence={"reason": "stdout received", "byte_count": 1},
    )
    assert not infrastructure_retry_allowed(partial)
    no_decision = transition_custody(
        partial,
        CustodyState.VALIDATION_CONSUMED_NO_DECISION,
        evidence=_terminal_evidence("no_decision"),
    )
    with pytest.raises(RuntimeError, match="illegal"):
        transition_custody(
            no_decision,
            CustodyState.HOLDOUT_RESERVED,
            evidence=_holdout_authorization(no_decision),
        )


def test_record_commitment_history_and_visibility_cannot_be_tampered() -> None:
    record = _locked_record()
    record["streams"]["validation"] = [99]
    with pytest.raises(RuntimeError, match="commitment mismatch"):
        transition_custody(
            record,
            CustodyState.VALIDATION_RESERVED,
            evidence={"reservation_nonce": "n"},
        )

    record = _locked_record()
    record["result_bytes_visible_by_split"]["validation"] = True
    with pytest.raises(RuntimeError, match="does not match history"):
        infrastructure_retry_allowed(record)


def test_local_reservation_is_private_atomic_and_fail_closed(tmp_path: Path) -> None:
    destination = tmp_path / "reservation.json"
    record = build_custody_record(streams={"validation": [1]})

    digest = reserve_once(destination, record)

    assert len(digest) == 64
    assert destination.stat().st_mode & 0o777 == 0o600
    assert json.loads(destination.read_text(encoding="utf-8")) == record
    with pytest.raises(RuntimeError, match="already exists"):
        reserve_once(destination, record)

    target = tmp_path / "actual.json"
    target.write_text("do not overwrite", encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(RuntimeError, match="already exists"):
        reserve_once(symlink, record)
    assert target.read_text(encoding="utf-8") == "do not overwrite"


def test_remote_tag_cas_has_one_cross_clone_winner_and_split_scoped_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin, worktrees, commit = _prepare_remote(tmp_path)
    record = transition_custody(
        _locked_record(),
        CustodyState.VALIDATION_RESERVED,
        evidence={"reservation_nonce": "fixed-before-run"},
    )

    original_remote_oid = custody._remote_tag_oid
    barrier = Barrier(2)

    def synchronized_remote_oid(
        *, remote: str, tag: str, cwd: str | Path
    ) -> str | None:
        oid = original_remote_oid(remote=remote, tag=tag, cwd=cwd)
        if tag == "v2-i1-validation-reserved" and oid is None:
            barrier.wait(timeout=5)
        return oid

    monkeypatch.setattr(custody, "_remote_tag_oid", synchronized_remote_oid)

    def attempt(worktree: Path) -> tuple[Path, dict[str, Any] | None]:
        try:
            publication = reserve_remote_split_once(
                remote="origin",
                tag="v2-i1-validation-reserved",
                split="validation",
                git_commit=commit,
                custody_record=record,
                cwd=worktree,
            )
        except RuntimeError:
            return worktree, None
        return worktree, publication

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = list(executor.map(attempt, worktrees))
    assert sum(publication is not None for _path, publication in attempts) == 1
    winner, reservation = next(
        (path, publication)
        for path, publication in attempts
        if publication is not None
    )
    loser = next(path for path, publication in attempts if publication is None)
    assert reservation is not None
    loser_ref = subprocess.run(
        ("git", "show-ref", "--verify", "refs/tags/v2-i1-validation-reserved"),
        cwd=loser,
        check=False,
        capture_output=True,
    )
    assert loser_ref.returncode != 0

    result_path = tmp_path / "result.json"
    result_path.write_bytes(b'{"decision":"pass"}\n')
    publication = publish_immutable_result_evidence(
        result_path,
        remote="origin",
        tag="v2-i1-validation-result",
        split="validation",
        git_commit=commit,
        custody_record_sha256=reservation["certificate"][
            "custody_record_sha256"
        ],
        reservation_tag_object_sha=reservation["tag_object_sha"],
        cwd=winner,
    )
    stored = subprocess.run(
        ("git", "cat-file", "-p", publication["git_blob"]),
        cwd=winner,
        check=True,
        capture_output=True,
    ).stdout
    assert stored == result_path.read_bytes()
    remote_result_tag = _run(
        "git",
        "ls-remote",
        "--tags",
        str(origin),
        "refs/tags/v2-i1-validation-result",
    ).stdout.split()[0]
    assert remote_result_tag == publication["tag_object_sha"]

    passed = transition_custody(
        record,
        CustodyState.VALIDATION_CONSUMED_PASS,
        evidence={
            "decision": "pass",
            "result_sha256": publication["result_sha256"],
            "custody_record_sha256": reservation["certificate"][
                "custody_record_sha256"
            ],
            "git_blob": publication["git_blob"],
            "reservation_tag_object_sha": reservation["tag_object_sha"],
            "immutable_result_tag_object_sha": publication["tag_object_sha"],
        },
        result_bytes_visible=True,
    )
    holdout = transition_custody(
        passed,
        CustodyState.HOLDOUT_RESERVED,
        evidence=_holdout_authorization(passed),
    )
    assert holdout["result_bytes_visible_by_split"] == {
        "validation": True,
        "holdout": False,
    }
    assert infrastructure_retry_allowed(holdout, split="holdout")
    holdout_reservation = reserve_remote_split_once(
        remote="origin",
        tag="v2-i1-holdout-reserved",
        split="holdout",
        git_commit=commit,
        custody_record=holdout,
        cwd=winner,
    )
    assert holdout_reservation["certificate"]["split"] == "holdout"


def test_git_inputs_and_result_file_boundary_are_rejected_before_publication(
    tmp_path: Path,
) -> None:
    _origin, worktrees, commit = _prepare_remote(tmp_path)
    worktree = worktrees[0]
    with pytest.raises(ValueError, match="unsafe custody tag"):
        publish_remote_tag_cas(
            remote="origin",
            tag="bad tag",
            target=commit,
            certificate={"safe": True},
            cwd=worktree,
        )
    with pytest.raises(RuntimeError, match="full lowercase Git object ID"):
        publish_remote_tag_cas(
            remote="origin",
            tag="safe-tag",
            target="HEAD",
            certificate={"safe": True},
            cwd=worktree,
        )
    with pytest.raises(ValueError, match="unsafe custody registry remote"):
        publish_remote_tag_cas(
            remote="--upload-pack=malicious",
            tag="safe-tag",
            target=commit,
            certificate={"safe": True},
            cwd=worktree,
        )

    reservation = reserve_remote_split_once(
        remote="origin",
        tag="validation-reserved",
        split="validation",
        git_commit=commit,
        custody_record=transition_custody(
            _locked_record(),
            CustodyState.VALIDATION_RESERVED,
            evidence={"reservation_nonce": "n"},
        ),
        cwd=worktree,
    )
    actual = tmp_path / "actual-result.json"
    actual.write_text('{"decision":"pass"}\n', encoding="utf-8")
    symlink = tmp_path / "result-link.json"
    symlink.symlink_to(actual)
    with pytest.raises(RuntimeError, match="regular non-symlink"):
        publish_immutable_result_evidence(
            symlink,
            remote="origin",
            tag="validation-result",
            split="validation",
            git_commit=commit,
            custody_record_sha256=reservation["certificate"][
                "custody_record_sha256"
            ],
            reservation_tag_object_sha=reservation["tag_object_sha"],
            cwd=worktree,
        )


def test_cas_cleans_local_tag_when_post_push_verification_is_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _origin, worktrees, commit = _prepare_remote(tmp_path)
    worktree = worktrees[0]
    original = custody._remote_tag_oid
    calls = 0

    def fail_after_preflight(
        *, remote: str, tag: str, cwd: str | Path
    ) -> str | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(remote=remote, tag=tag, cwd=cwd)
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(custody, "_remote_tag_oid", fail_after_preflight)
    with pytest.raises(RuntimeError, match="registry unavailable"):
        publish_remote_tag_cas(
            remote="origin",
            tag="ambiguous-cas",
            target=commit,
            certificate={"safe": True},
            cwd=worktree,
        )
    local_ref = subprocess.run(
        ("git", "show-ref", "--verify", "refs/tags/ambiguous-cas"),
        cwd=worktree,
        check=False,
        capture_output=True,
    )
    assert local_ref.returncode != 0
