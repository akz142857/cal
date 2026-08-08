"""Fail-closed custody state for stochastic permanence secret splits.

This module implements the state machine, atomic local reservation, shared
expected-missing Git-tag CAS, and immutable Git-blob result evidence.  It never
generates, serializes into public certificates, or prints a secret manifest.
"""

from __future__ import annotations

from enum import StrEnum
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

from cal.evaluation.stochastic_permanence_artifacts import (
    canonical_json_bytes,
    sha256_bytes,
)


class CustodyState(StrEnum):
    DRAFT = "DRAFT"
    BASE_LOCKED = "BASE_LOCKED"
    CANDIDATE_LOCKED = "CANDIDATE_LOCKED"
    VALIDATION_RESERVED = "VALIDATION_RESERVED"
    VALIDATION_CONSUMED_PASS = "VALIDATION_CONSUMED_PASS"
    VALIDATION_CONSUMED_FAIL = "VALIDATION_CONSUMED_FAIL"
    VALIDATION_CONSUMED_NO_DECISION = "VALIDATION_CONSUMED_NO_DECISION"
    HOLDOUT_RESERVED = "HOLDOUT_RESERVED"
    HOLDOUT_CONSUMED = "HOLDOUT_CONSUMED"


_TRANSITIONS: dict[CustodyState, frozenset[CustodyState]] = {
    CustodyState.DRAFT: frozenset({CustodyState.BASE_LOCKED}),
    CustodyState.BASE_LOCKED: frozenset({CustodyState.CANDIDATE_LOCKED}),
    CustodyState.CANDIDATE_LOCKED: frozenset(
        {CustodyState.VALIDATION_RESERVED}
    ),
    CustodyState.VALIDATION_RESERVED: frozenset(
        {
            CustodyState.VALIDATION_CONSUMED_PASS,
            CustodyState.VALIDATION_CONSUMED_FAIL,
            CustodyState.VALIDATION_CONSUMED_NO_DECISION,
        }
    ),
    CustodyState.VALIDATION_CONSUMED_PASS: frozenset(
        {CustodyState.HOLDOUT_RESERVED}
    ),
    CustodyState.HOLDOUT_RESERVED: frozenset({CustodyState.HOLDOUT_CONSUMED}),
    CustodyState.VALIDATION_CONSUMED_FAIL: frozenset(),
    CustodyState.VALIDATION_CONSUMED_NO_DECISION: frozenset(),
    CustodyState.HOLDOUT_CONSUMED: frozenset(),
}

_RESERVED_STATE_BY_SPLIT = {
    "validation": CustodyState.VALIDATION_RESERVED,
    "holdout": CustodyState.HOLDOUT_RESERVED,
}
_VALIDATION_DECISION_BY_STATE = {
    CustodyState.VALIDATION_CONSUMED_PASS: "pass",
    CustodyState.VALIDATION_CONSUMED_FAIL: "fail",
    CustodyState.VALIDATION_CONSUMED_NO_DECISION: "no_decision",
}
_HEX = frozenset("0123456789abcdef")
_REMOTE_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise RuntimeError(f"{name} requires a lowercase SHA-256 digest")
    return value


def _require_object_id(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in _HEX for character in value)
    ):
        raise RuntimeError(f"{name} requires a full lowercase Git object ID")
    return value


def _canonical_mapping_copy(
    value: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"{name} requires a non-empty JSON object")
    try:
        decoded = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} is not canonical JSON") from error
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{name} requires a JSON object")
    return decoded


def validate_disjoint_seed_sets(
    streams: Mapping[str, Sequence[int]],
) -> dict[str, list[int]]:
    """Return canonical seed lists or fail without disclosing seed values."""

    if not isinstance(streams, Mapping) or not streams:
        raise ValueError("at least one seed stream is required")
    if any(
        not isinstance(name, str)
        or not name
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in name
        )
        for name in streams
    ):
        raise ValueError("unsafe seed stream name")
    canonical: dict[str, list[int]] = {}
    owner: dict[int, str] = {}
    for name in sorted(streams):
        source = streams[name]
        if isinstance(source, (str, bytes)):
            raise TypeError(f"seed stream is not an integer sequence: {name}")
        try:
            values = list(source)
        except TypeError as error:
            raise TypeError(
                f"seed stream is not an integer sequence: {name}"
            ) from error
        if not values:
            raise RuntimeError(f"seed stream is empty: {name}")
        if any(type(seed) is not int or seed < 0 for seed in values):
            raise TypeError(
                f"seed stream contains a non-negative non-integer: {name}"
            )
        if len(values) != len(set(values)):
            raise RuntimeError(f"seed stream contains duplicates: {name}")
        canonical[name] = sorted(values)
        for seed in values:
            previous = owner.get(seed)
            if previous is not None:
                raise RuntimeError(
                    f"seed collision between streams {previous} and {name}"
                )
            owner[seed] = name
    return canonical


def _validate_result_evidence(
    evidence: Mapping[str, Any], *, expected_decision: str | None
) -> None:
    decision = evidence.get("decision")
    allowed = {expected_decision} if expected_decision else {
        "pass",
        "fail",
        "no_decision",
    }
    if decision not in allowed:
        raise RuntimeError("terminal transition requires the matching decision")
    _require_sha256(evidence.get("result_sha256"), "terminal result evidence")
    _require_sha256(
        evidence.get("custody_record_sha256"), "terminal custody evidence"
    )
    _require_object_id(evidence.get("git_blob"), "terminal result blob")
    _require_object_id(
        evidence.get("reservation_tag_object_sha"),
        "terminal reservation evidence",
    )
    _require_object_id(
        evidence.get("immutable_result_tag_object_sha"),
        "terminal immutable result evidence",
    )


def _validate_holdout_authorization(
    evidence: Mapping[str, Any], record: Mapping[str, Any]
) -> None:
    authorization = evidence.get("separate_human_authorization")
    if not isinstance(authorization, Mapping):
        raise RuntimeError("holdout requires structured human authorization")
    if (
        authorization.get("authorization_schema_version") != 1
        or authorization.get("decision") != "authorize_holdout_once"
        or authorization.get("split") != "holdout"
        or not isinstance(authorization.get("authorized_by"), str)
        or not authorization["authorized_by"].strip()
    ):
        raise RuntimeError("holdout human authorization is incomplete")
    _require_sha256(
        authorization.get("authorization_sha256"), "holdout authorization"
    )
    authorization_body = dict(authorization)
    claimed_digest = authorization_body.pop("authorization_sha256")
    if sha256_bytes(canonical_json_bytes(authorization_body)) != claimed_digest:
        raise RuntimeError("holdout authorization digest does not match its body")
    for field in (
        "base_lock_sha256",
        "candidate_lock_sha256",
        "stream_commitment_sha256",
    ):
        if authorization.get(field) != record.get(field):
            raise RuntimeError(f"holdout authorization does not bind {field}")
    validation_evidence = next(
        (
            entry.get("evidence")
            for entry in reversed(record.get("history", []))
            if isinstance(entry, Mapping)
            and entry.get("event_type") == "transition"
            and entry.get("to") == CustodyState.VALIDATION_CONSUMED_PASS.value
        ),
        None,
    )
    if not isinstance(validation_evidence, Mapping):
        raise RuntimeError("holdout authorization requires a validation pass")
    validation_bindings = {
        "validation_result_sha256": validation_evidence.get("result_sha256"),
        "validation_result_tag_object_sha": validation_evidence.get(
            "immutable_result_tag_object_sha"
        ),
        "validation_custody_record_sha256": validation_evidence.get(
            "custody_record_sha256"
        ),
    }
    for field, expected in validation_bindings.items():
        if authorization.get(field) != expected:
            raise RuntimeError(f"holdout authorization does not bind {field}")


def _validate_custody_record(record: Mapping[str, Any]) -> None:
    """Reject malformed or internally inconsistent custody records."""

    if (
        not isinstance(record, Mapping)
        or record.get("custody_schema_version") != 2
    ):
        raise RuntimeError("unsupported custody record schema")
    try:
        current = CustodyState(str(record["state"]))
    except (KeyError, ValueError) as error:
        raise RuntimeError("invalid custody state") from error
    streams = validate_disjoint_seed_sets(record.get("streams", {}))
    if record.get("streams") != streams:
        raise RuntimeError("seed streams are not canonical")
    expected_commitment = sha256_bytes(canonical_json_bytes(streams))
    if record.get("stream_commitment_sha256") != expected_commitment:
        raise RuntimeError("seed stream commitment mismatch")

    visibility = record.get("result_bytes_visible_by_split")
    if not isinstance(visibility, Mapping) or set(visibility) != {
        "validation",
        "holdout",
    } or any(type(visibility[split]) is not bool for split in visibility):
        raise RuntimeError("invalid per-split result visibility ledger")

    history = record.get("history")
    if not isinstance(history, list):
        raise RuntimeError("invalid custody history")
    replay_state = CustodyState.DRAFT
    replay_visibility = {"validation": False, "holdout": False}
    replay_base: str | None = None
    replay_candidate: str | None = None
    for entry in history:
        if not isinstance(entry, Mapping):
            raise RuntimeError("invalid custody history entry")
        event_type = entry.get("event_type")
        if event_type == "result_bytes_visible":
            split = entry.get("split")
            if split not in _RESERVED_STATE_BY_SPLIT:
                raise RuntimeError("invalid result visibility split")
            if replay_state is not _RESERVED_STATE_BY_SPLIT[split]:
                raise RuntimeError("result visibility recorded outside reservation")
            if replay_visibility[split]:
                raise RuntimeError("duplicate first-result-byte event")
            if (
                not isinstance(entry.get("evidence"), Mapping)
                or not entry["evidence"]
            ):
                raise RuntimeError("result visibility event requires evidence")
            replay_visibility[split] = True
            continue
        if event_type != "transition":
            raise RuntimeError("unknown custody history event")
        try:
            source = CustodyState(str(entry["from"]))
            target = CustodyState(str(entry["to"]))
        except (KeyError, ValueError) as error:
            raise RuntimeError("invalid custody history transition") from error
        if source is not replay_state or target not in _TRANSITIONS[source]:
            raise RuntimeError("custody history transition chain is invalid")
        evidence = entry.get("evidence")
        if not isinstance(evidence, Mapping) or not evidence:
            raise RuntimeError("custody transition requires evidence")
        visible_now = entry.get("result_bytes_visible")
        if type(visible_now) is not bool:
            raise RuntimeError("invalid transition visibility evidence")
        split: str | None = None
        if target is CustodyState.BASE_LOCKED:
            replay_base = _require_sha256(
                evidence.get("base_lock_sha256"), "base lock transition"
            )
        elif target is CustodyState.CANDIDATE_LOCKED:
            replay_candidate = _require_sha256(
                evidence.get("candidate_lock_sha256"),
                "candidate lock transition",
            )
        elif target is CustodyState.VALIDATION_RESERVED:
            if replay_base is None or replay_candidate is None:
                raise RuntimeError("validation reservation requires both locks")
        elif target in _VALIDATION_DECISION_BY_STATE:
            split = "validation"
            _validate_result_evidence(
                evidence,
                expected_decision=_VALIDATION_DECISION_BY_STATE[target],
            )
        elif target is CustodyState.HOLDOUT_RESERVED:
            _validate_holdout_authorization(evidence, record)
        elif target is CustodyState.HOLDOUT_CONSUMED:
            split = "holdout"
            _validate_result_evidence(evidence, expected_decision=None)
        if split is None and visible_now:
            raise RuntimeError("result bytes may become visible only on consumption")
        if split is not None:
            if not (visible_now or replay_visibility[split]):
                raise RuntimeError("terminal transition requires visible result bytes")
            replay_visibility[split] = True
        replay_state = target

    if replay_state is not current:
        raise RuntimeError("custody state does not match its history")
    if replay_visibility != dict(visibility):
        raise RuntimeError("result visibility ledger does not match history")
    if replay_base != record.get("base_lock_sha256"):
        raise RuntimeError("base lock does not match custody history")
    if replay_candidate != record.get("candidate_lock_sha256"):
        raise RuntimeError("candidate lock does not match custody history")


def build_custody_record(
    *,
    streams: Mapping[str, Sequence[int]],
    base_lock_sha256: str | None = None,
) -> dict[str, Any]:
    canonical = validate_disjoint_seed_sets(streams)
    record: dict[str, Any] = {
        "custody_schema_version": 2,
        "state": CustodyState.DRAFT.value,
        "streams": canonical,
        "stream_commitment_sha256": sha256_bytes(
            canonical_json_bytes(canonical)
        ),
        "base_lock_sha256": None,
        "candidate_lock_sha256": None,
        "history": [],
        "result_bytes_visible_by_split": {
            "validation": False,
            "holdout": False,
        },
    }
    if base_lock_sha256 is not None:
        return transition_custody(
            record,
            CustodyState.BASE_LOCKED,
            evidence={"base_lock_sha256": base_lock_sha256},
        )
    return record


def transition_custody(
    record: Mapping[str, Any],
    target: CustodyState,
    *,
    evidence: Mapping[str, Any],
    result_bytes_visible: bool = False,
) -> dict[str, Any]:
    """Apply one legal transition and append immutable evidence metadata."""

    _validate_custody_record(record)
    current = CustodyState(str(record["state"]))
    if target not in _TRANSITIONS[current]:
        raise RuntimeError(f"illegal custody transition: {current} -> {target}")
    evidence_copy = _canonical_mapping_copy(
        evidence, name="custody transition evidence"
    )
    if type(result_bytes_visible) is not bool:
        raise TypeError("result_bytes_visible must be boolean")

    updated = json.loads(json.dumps(record))
    terminal_split: str | None = None
    if target is CustodyState.BASE_LOCKED:
        updated["base_lock_sha256"] = _require_sha256(
            evidence_copy.get("base_lock_sha256"), "base lock transition"
        )
    elif target is CustodyState.CANDIDATE_LOCKED:
        updated["candidate_lock_sha256"] = _require_sha256(
            evidence_copy.get("candidate_lock_sha256"),
            "candidate lock transition",
        )
    elif target is CustodyState.VALIDATION_RESERVED:
        if not updated.get("base_lock_sha256") or not updated.get(
            "candidate_lock_sha256"
        ):
            raise RuntimeError("validation reservation requires both locks")
    elif target in _VALIDATION_DECISION_BY_STATE:
        terminal_split = "validation"
        _validate_result_evidence(
            evidence_copy,
            expected_decision=_VALIDATION_DECISION_BY_STATE[target],
        )
    elif target is CustodyState.HOLDOUT_RESERVED:
        _validate_holdout_authorization(evidence_copy, updated)
    elif target is CustodyState.HOLDOUT_CONSUMED:
        terminal_split = "holdout"
        _validate_result_evidence(evidence_copy, expected_decision=None)

    if terminal_split is None and result_bytes_visible:
        raise RuntimeError("result bytes may become visible only on consumption")
    if terminal_split is not None:
        already_visible = updated["result_bytes_visible_by_split"][terminal_split]
        if not (already_visible or result_bytes_visible):
            raise RuntimeError("terminal transition requires visible result bytes")
        updated["result_bytes_visible_by_split"][terminal_split] = True
    updated["state"] = target.value
    updated["history"].append(
        {
            "event_type": "transition",
            "from": current.value,
            "to": target.value,
            "evidence": evidence_copy,
            "result_bytes_visible": result_bytes_visible,
        }
    )
    _validate_custody_record(updated)
    return updated


def mark_result_bytes_visible(
    record: Mapping[str, Any],
    *,
    split: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the fail-closed boundary at the first visible result byte."""

    _validate_custody_record(record)
    if split not in _RESERVED_STATE_BY_SPLIT:
        raise ValueError("invalid result visibility split")
    if record.get("state") != _RESERVED_STATE_BY_SPLIT[split].value:
        raise RuntimeError(f"{split} result bytes require an active reservation")
    if record["result_bytes_visible_by_split"][split]:
        raise RuntimeError(f"{split} result bytes were already visible")
    evidence_copy = _canonical_mapping_copy(
        evidence, name="first-result-byte event"
    )
    updated = json.loads(json.dumps(record))
    updated["result_bytes_visible_by_split"][split] = True
    updated["history"].append(
        {
            "event_type": "result_bytes_visible",
            "split": split,
            "evidence": evidence_copy,
        }
    )
    _validate_custody_record(updated)
    return updated


def reserve_once(path: str | Path, record: Mapping[str, Any]) -> str:
    """Atomically create a private local reservation; existence is consuming."""

    _validate_custody_record(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(record)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"reservation already exists: {destination}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Deliberately leave any created path in place: an ambiguous write must
        # consume the one-shot local reservation rather than permit a rerun.
        raise
    return sha256_bytes(raw)


def infrastructure_retry_allowed(
    record: Mapping[str, Any], *, split: str | None = None
) -> bool:
    """Allow only the reserved split to retry before its first result byte."""

    _validate_custody_record(record)
    current = CustodyState(str(record["state"]))
    active_split = next(
        (
            name
            for name, state in _RESERVED_STATE_BY_SPLIT.items()
            if state is current
        ),
        None,
    )
    if active_split is None or (split is not None and split != active_split):
        return False
    return record["result_bytes_visible_by_split"][active_split] is False


def _git(
    *arguments: str,
    cwd: str | Path,
    check: bool = True,
    text: bool = True,
    input_data: bytes | str | None = None,
) -> subprocess.CompletedProcess[Any]:
    result = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        capture_output=True,
        check=False,
        text=text,
        input=input_data,
    )
    if check and result.returncode != 0:
        operation = arguments[0] if arguments else "command"
        # Arguments, stderr, and remote URLs can contain credentials.  Keep the
        # raised boundary useful without reflecting attacker-controlled input.
        raise RuntimeError(
            f"git {operation} failed with exit code {result.returncode}"
        )
    return result


def _validate_remote(remote: str) -> None:
    if (
        not isinstance(remote, str)
        or not remote
        or any(character not in _REMOTE_CHARACTERS for character in remote)
    ):
        raise ValueError("unsafe custody registry remote")


def _validate_tag(tag: str, *, cwd: str | Path) -> None:
    if (
        not isinstance(tag, str)
        or not tag
        or "\x00" in tag
        or "\n" in tag
        or "\r" in tag
    ):
        raise ValueError("unsafe custody tag")
    result = _git(
        "check-ref-format", f"refs/tags/{tag}", cwd=cwd, check=False
    )
    if result.returncode != 0:
        raise ValueError("unsafe custody tag")


def _resolve_object(
    oid: str, *, cwd: str | Path, expected_type: str | None = None
) -> str:
    exact = _require_object_id(oid, "Git target")
    exists = _git("cat-file", "-e", f"{exact}^{{object}}", cwd=cwd, check=False)
    if exists.returncode != 0:
        raise RuntimeError("Git target object does not exist locally")
    if expected_type is not None:
        actual = _git("cat-file", "-t", exact, cwd=cwd).stdout.strip()
        if actual != expected_type:
            raise RuntimeError(f"Git target must be a {expected_type} object")
    return exact


def _read_reservation_certificate(
    tag_oid: str, *, cwd: str | Path
) -> tuple[str, Mapping[str, Any]]:
    raw = _git("cat-file", "-p", tag_oid, cwd=cwd).stdout
    headers, separator, message = raw.partition("\n\n")
    if not separator:
        raise RuntimeError("reservation tag has no certificate payload")
    target: str | None = None
    target_type: str | None = None
    for line in headers.splitlines():
        key, _, value = line.partition(" ")
        if key == "object":
            target = value
        elif key == "type":
            target_type = value
    if target is None or target_type != "commit":
        raise RuntimeError("reservation tag does not target a commit")
    target = _require_object_id(target, "reservation target")
    try:
        certificate = json.loads(message.strip())
    except json.JSONDecodeError as error:
        raise RuntimeError("reservation tag certificate is not JSON") from error
    if not isinstance(certificate, Mapping):
        raise RuntimeError("reservation tag certificate is not an object")
    return target, certificate


def _read_regular_file_no_follow(path: Path) -> bytes:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(
            "result evidence must be a regular non-symlink file"
        ) from error
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                "result evidence must be a regular non-symlink file"
            )
        return stream.read()


def _remote_tag_oid(
    *, remote: str, tag: str, cwd: str | Path
) -> str | None:
    _validate_remote(remote)
    _validate_tag(tag, cwd=cwd)
    result = _git(
        "ls-remote",
        "--tags",
        remote,
        f"refs/tags/{tag}",
        cwd=cwd,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("cannot query shared custody registry")
    expected_ref = f"refs/tags/{tag}"
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise RuntimeError("shared custody registry returned malformed data")
        oid, reference = fields
        if reference == expected_ref:
            return _require_object_id(oid, "remote custody tag")
    return None


def publish_remote_tag_cas(
    *,
    remote: str,
    tag: str,
    target: str,
    certificate: Mapping[str, Any],
    cwd: str | Path,
) -> dict[str, Any]:
    """Publish one annotated tag iff the shared ref is still absent."""

    _validate_remote(remote)
    _validate_tag(tag, cwd=cwd)
    target_oid = _resolve_object(target, cwd=cwd)
    if not isinstance(certificate, Mapping):
        raise TypeError("custody certificate must be a mapping")
    certificate_copy = _canonical_mapping_copy(
        certificate, name="custody certificate"
    )
    if _remote_tag_oid(remote=remote, tag=tag, cwd=cwd) is not None:
        raise RuntimeError(f"shared one-shot custody tag already exists: {tag}")
    local = _git(
        "rev-parse", "--verify", f"refs/tags/{tag}", cwd=cwd, check=False
    )
    if local.returncode == 0:
        raise RuntimeError(f"local custody tag already exists: {tag}")
    payload = {
        **certificate_copy,
        "publication_nonce": uuid4().hex,
    }
    local_created = False
    acquired = False
    try:
        _git(
            "tag",
            "-a",
            tag,
            target_oid,
            "-m",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            cwd=cwd,
        )
        local_created = True
        local_oid = _require_object_id(
            _git("rev-parse", f"refs/tags/{tag}", cwd=cwd).stdout.strip(),
            "local custody tag",
        )
        try:
            _git(
                "push",
                f"--force-with-lease=refs/tags/{tag}:",
                remote,
                f"refs/tags/{tag}",
                cwd=cwd,
            )
        except RuntimeError as error:
            # A transport may report failure after the server accepted the ref.
            # Treat that ambiguous case as success only when the exact tag object
            # is now visible remotely.
            if _remote_tag_oid(remote=remote, tag=tag, cwd=cwd) != local_oid:
                raise RuntimeError(
                    f"shared one-shot custody tag CAS was not acquired: {tag}"
                ) from error
        if _remote_tag_oid(remote=remote, tag=tag, cwd=cwd) != local_oid:
            raise RuntimeError(
                f"shared one-shot custody tag CAS verification failed: {tag}"
            )
        acquired = True
        return {"certificate": payload, "tag_object_sha": local_oid}
    finally:
        if local_created and not acquired:
            _git("tag", "-d", tag, cwd=cwd, check=False)


def reserve_remote_split_once(
    *,
    remote: str,
    tag: str,
    split: str,
    git_commit: str,
    custody_record: Mapping[str, Any],
    cwd: str | Path,
) -> dict[str, Any]:
    """CAS-reserve validation/holdout before any episode or result byte."""

    _validate_custody_record(custody_record)
    if split not in _RESERVED_STATE_BY_SPLIT:
        raise ValueError("only validation/holdout can be remotely reserved")
    if split not in custody_record["streams"]:
        raise RuntimeError(f"custody record has no {split} seed stream")
    expected_state = _RESERVED_STATE_BY_SPLIT[split].value
    if custody_record.get("state") != expected_state:
        raise RuntimeError(f"custody record is not ready for {split} reservation")
    if custody_record["result_bytes_visible_by_split"][split]:
        raise RuntimeError(f"cannot reserve after {split} result bytes became visible")
    commit_oid = _resolve_object(git_commit, cwd=cwd, expected_type="commit")
    record_digest = sha256_bytes(canonical_json_bytes(custody_record))
    return publish_remote_tag_cas(
        remote=remote,
        tag=tag,
        target=commit_oid,
        certificate={
            "certificate_schema_version": 1,
            "certificate_type": "stochastic_permanence_one_shot_reservation",
            "split": split,
            "state": expected_state,
            "custody_record_sha256": record_digest,
            "stream_commitment_sha256": custody_record[
                "stream_commitment_sha256"
            ],
            "base_lock_sha256": custody_record["base_lock_sha256"],
            "candidate_lock_sha256": custody_record["candidate_lock_sha256"],
            "git_commit": commit_oid,
            "status": "reserved_before_first_episode",
        },
        cwd=cwd,
    )


def publish_immutable_result_evidence(
    result_path: str | Path,
    *,
    remote: str,
    tag: str,
    split: str,
    git_commit: str,
    custody_record_sha256: str,
    reservation_tag_object_sha: str,
    cwd: str | Path,
) -> dict[str, Any]:
    """Store exact regular-file bytes as a Git blob and publish their certificate."""

    if split not in _RESERVED_STATE_BY_SPLIT:
        raise ValueError("invalid result-evidence split")
    commit_oid = _resolve_object(git_commit, cwd=cwd, expected_type="commit")
    record_digest = _require_sha256(
        custody_record_sha256, "result custody record"
    )
    reservation_oid = _resolve_object(
        reservation_tag_object_sha, cwd=cwd, expected_type="tag"
    )
    reservation_target, reservation_certificate = _read_reservation_certificate(
        reservation_oid, cwd=cwd
    )
    if (
        reservation_target != commit_oid
        or reservation_certificate.get("certificate_type")
        != "stochastic_permanence_one_shot_reservation"
        or reservation_certificate.get("split") != split
        or reservation_certificate.get("git_commit") != commit_oid
        or reservation_certificate.get("custody_record_sha256") != record_digest
    ):
        raise RuntimeError(
            "immutable result evidence does not match its reservation"
        )
    raw = _read_regular_file_no_follow(Path(result_path))
    if not raw:
        raise RuntimeError("result evidence must not be empty")
    result_sha256 = sha256_bytes(raw)
    blob = _require_object_id(
        _git(
            "hash-object",
            "-w",
            "--stdin",
            cwd=cwd,
            text=False,
            input_data=raw,
        ).stdout.decode().strip(),
        "immutable result blob",
    )
    _resolve_object(blob, cwd=cwd, expected_type="blob")
    stored = _git("cat-file", "-p", blob, cwd=cwd, text=False).stdout
    if stored != raw:
        raise RuntimeError("immutable result blob verification failed")
    publication = publish_remote_tag_cas(
        remote=remote,
        tag=tag,
        target=blob,
        certificate={
            "certificate_schema_version": 1,
            "certificate_type": "stochastic_permanence_immutable_result",
            "split": split,
            "git_commit": commit_oid,
            "custody_record_sha256": record_digest,
            "reservation_tag_object_sha": reservation_oid,
            "result_sha256": result_sha256,
            "git_blob": blob,
        },
        cwd=cwd,
    )
    return {
        **publication,
        "git_blob": blob,
        "result_sha256": result_sha256,
        "reservation_tag_object_sha": reservation_oid,
        "immutable_result_tag_object_sha": publication["tag_object_sha"],
    }
