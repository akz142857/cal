"""V2-I1 protocol v2: unified entity-belief system integration."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from cal.evaluation.v2_artifacts import (
    build_resources,
    constructor_apis_reject_ground_truth,
    load_frozen_protocol,
    require_authorization,
    resources_pass,
)
from cal.evaluation.v2_i1_integration import (
    PERMANENCE_WARMUP,
    WARMUP,
    _IntegratedWorld,
    _global_visibility,
)
from cal.infra.provenance import capture_provenance
from cal.model.entity_belief_graph import IntegratedBeliefAgentV2


DEFAULT_PROTOCOL = Path("experiments/V2_I1_INTEGRATION_PROTOCOL_V4.json")
V2_FROZEN_STATUS = (
    "frozen_before_unified_entity_belief_graph_implementation"
)
V3_FROZEN_STATUS = (
    "frozen_after_prerequisite_portability_fix_before_any_i1_v2_development_run"
)
V4_FROZEN_STATUS = (
    "frozen_after_subagent_review_before_corrected_calibration_or_new_validation"
)
REQUIRED_I1_GATES = {
    "self_identification_pass",
    "identity_consistency_pass",
    "visible_identity_coverage_pass",
    "distractor_permanence_pass",
    "no_action_control_fails",
    "time_shuffle_control_fails",
    "assume_all_visible_control_fails",
    "paired_separation_pass",
    "single_agent_single_stream",
    "shared_entity_store_for_all_outputs",
    "labels_absent_from_learner",
    "resources_pass",
    "architecture_limits_match_protocol",
}


def _episode(
    seed: int,
    *,
    steps: int,
    infer_occlusion: bool = True,
    use_action: bool = True,
    shuffle_lag: int = 0,
) -> dict[str, Any]:
    world = _IntegratedWorld(seed)
    agent = IntegratedBeliefAgentV2(
        infer_occlusion=infer_occlusion,
        use_action=use_action,
        seed=seed + 40_000,
    )
    action_rng = np.random.default_rng(seed + 50_000)
    sensed, _ = world.observe()
    agent.update(sensed, 0)
    executed: list[int] = [0]
    true_positive = false_positive = false_negative = 0
    hidden_probabilities: list[float] = []
    identity_map: dict[str, dict[int, int]] = {"a": {}, "b": {}}
    identity_opportunities: dict[str, int] = {"a": 0, "b": 0}
    identity_detections: dict[str, int] = {"a": 0, "b": 0}
    for step in range(1, steps + 1):
        action = int(action_rng.integers(0, 5))
        sensed, visibility = world.step(action)
        executed.append(action)
        supplied = (
            executed[max(0, len(executed) - 1 - shuffle_lag)]
            if shuffle_lag
            else action
        )
        agent.update(sensed, supplied)
        visible = _global_visibility(visibility, world.grid_size)
        positions = agent.track_positions()
        if step >= WARMUP:
            self_identity = agent.self_track_identity()
            true_self = (
                int(world.self_position[0]),
                int(world.self_position[1]),
            )
            if visible[true_self[1], true_self[0]]:
                predicted = (
                    positions.get(self_identity)
                    if self_identity is not None
                    else None
                )
                if predicted == true_self:
                    true_positive += 1
                else:
                    false_negative += 1
                    if predicted is not None:
                        false_positive += 1
        truth_cells = {
            "self": (
                int(world.self_position[0]),
                int(world.self_position[1]),
            ),
            "a": (
                int(world.distractor_a[0]),
                int(world.distractor_a[1]),
            ),
            "b": (
                int(world.distractor_b[0]),
                int(world.distractor_b[1]),
            ),
        }
        for name, point in (
            ("a", world.distractor_a),
            ("b", world.distractor_b),
        ):
            cell = (int(point[0]), int(point[1]))
            # Binary occupancy cannot distinguish two physical points in one
            # cell. Exclude only those observation-equivalent merge frames;
            # every other visible opportunity remains in the denominator, so
            # a missing track is an explicit identity failure.
            if (
                visible[cell[1], cell[0]]
                and sum(
                    other_cell == cell
                    for other_cell in truth_cells.values()
                )
                == 1
            ):
                identity_opportunities[name] += 1
                matched = [
                    identity
                    for identity, position in positions.items()
                    if position == cell
                ]
                if len(matched) == 1:
                    identity = matched[0]
                    identity_detections[name] += 1
                    identity_map[name][identity] = (
                        identity_map[name].get(identity, 0) + 1
                    )
        if step >= PERMANENCE_WARMUP:
            probability = agent.probability()
            for point in (world.distractor_a, world.distractor_b):
                cell = (int(point[0]), int(point[1]))
                if not visible[cell[1], cell[0]]:
                    hidden_probabilities.append(
                        float(probability[cell[1], cell[0]])
                    )
    denominator = 2 * true_positive + false_positive + false_negative
    identity_consistency, visible_identity_coverage = (
        _identity_metrics(
            identity_map,
            identity_opportunities,
            identity_detections,
        )
    )
    total_opportunities = sum(identity_opportunities.values())
    return {
        "seed": seed,
        "self_f1": (
            2 * true_positive / denominator if denominator else 0.0
        ),
        "identity_consistency": identity_consistency,
        "visible_identity_coverage": visible_identity_coverage,
        "visible_identity_opportunities": total_opportunities,
        "distractor_hidden_probability": (
            mean(hidden_probabilities) if hidden_probabilities else 0.0
        ),
        "hidden_sample_count": len(hidden_probabilities),
    }


def _identity_metrics(
    identity_map: dict[str, dict[int, int]],
    identity_opportunities: dict[str, int],
    identity_detections: dict[str, int],
) -> tuple[float, float]:
    """Score visible identities with misses and one-to-one assignment."""

    total_opportunities = sum(identity_opportunities.values())
    if not total_opportunities:
        return 0.0, 0.0
    identities = sorted(
        set(identity_map["a"]) | set(identity_map["b"])
    )
    best_distinct_identity_matches = 0
    candidates: list[int | None] = [None, *identities]
    for identity_a in candidates:
        for identity_b in candidates:
            if identity_a is not None and identity_a == identity_b:
                continue
            best_distinct_identity_matches = max(
                best_distinct_identity_matches,
                identity_map["a"].get(identity_a, 0)
                + identity_map["b"].get(identity_b, 0),
            )
    total_detections = sum(identity_detections.values())
    return (
        best_distinct_identity_matches / total_opportunities,
        total_detections / total_opportunities,
    )


def _run_conditions(
    seeds: tuple[int, ...],
    *,
    steps: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "formal": [_episode(seed, steps=steps) for seed in seeds],
        "no_action": [
            _episode(seed, steps=steps, use_action=False) for seed in seeds
        ],
        "time_shuffled": [
            _episode(seed, steps=steps, shuffle_lag=5) for seed in seeds
        ],
        "assume_all_visible": [
            _episode(seed, steps=steps, infer_occlusion=False)
            for seed in seeds
        ],
    }


def _aggregate(
    conditions: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    formal = conditions["formal"]
    no_action = conditions["no_action"]
    shuffled = conditions["time_shuffled"]
    visible = conditions["assume_all_visible"]
    return {
        "self_f1": mean(item["self_f1"] for item in formal),
        "identity_consistency": mean(
            item["identity_consistency"] for item in formal
        ),
        "visible_identity_coverage": mean(
            item["visible_identity_coverage"] for item in formal
        ),
        "distractor_hidden_probability": mean(
            item["distractor_hidden_probability"] for item in formal
        ),
        "no_action_self_f1": mean(
            item["self_f1"] for item in no_action
        ),
        "time_shuffled_self_f1": mean(
            item["self_f1"] for item in shuffled
        ),
        "assume_all_visible_hidden_probability": mean(
            item["distractor_hidden_probability"] for item in visible
        ),
        "paired_formal_beats_visible_control": mean(
            1.0
            if formal_item["distractor_hidden_probability"]
            > visible_item["distractor_hidden_probability"]
            else 0.0
            for formal_item, visible_item in zip(formal, visible)
        ),
    }


def _mechanism_gates(
    aggregate: dict[str, float],
    fixed: dict[str, Any],
) -> dict[str, bool]:
    return {
        "self_identification_pass": (
            aggregate["self_f1"]
            >= fixed["self_identification_f1_minimum"]
        ),
        "identity_consistency_pass": (
            aggregate["identity_consistency"]
            >= fixed["identity_consistency_minimum"]
        ),
        "visible_identity_coverage_pass": (
            aggregate["visible_identity_coverage"]
            >= fixed["visible_identity_coverage_minimum"]
        ),
        "distractor_permanence_pass": (
            aggregate["distractor_hidden_probability"]
            >= fixed["distractor_hidden_probability_minimum"]
        ),
        "no_action_control_fails": (
            aggregate["self_f1"] - aggregate["no_action_self_f1"]
            >= fixed["no_action_self_f1_drop_minimum"]
        ),
        "time_shuffle_control_fails": (
            aggregate["self_f1"]
            - aggregate["time_shuffled_self_f1"]
            >= fixed["time_shuffled_self_f1_drop_minimum"]
        ),
        "assume_all_visible_control_fails": (
            aggregate["assume_all_visible_hidden_probability"]
            <= fixed["assume_all_visible_hidden_probability_maximum"]
        ),
        "paired_separation_pass": (
            aggregate["paired_formal_beats_visible_control"]
            >= fixed[
                "paired_formal_beats_visible_control_fraction_minimum"
            ]
        ),
    }


def _load_v3_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    amendment, digest = load_frozen_protocol(
        path, frozen_statuses=V3_FROZEN_STATUS
    )
    base_path = Path(amendment["base_protocol_path"])
    base, base_digest = load_frozen_protocol(
        base_path, frozen_statuses=V2_FROZEN_STATUS
    )
    if base_digest != amendment["base_protocol_sha256"]:
        raise RuntimeError("I1-V2 base protocol digest mismatch")
    if (
        amendment["amendment_record"]["prior_protocol_sha256"]
        != base_digest
    ):
        raise RuntimeError("I1-V3 amendment chain mismatch")
    merged = dict(base)
    merged.update(
        {
            "protocol_name": amendment["protocol_name"],
            "protocol_version": amendment["protocol_version"],
            "status": amendment["status"],
            "base_protocol_path": amendment["base_protocol_path"],
            "base_protocol_sha256": amendment["base_protocol_sha256"],
            "amendment_record_v2": base["amendment_record"],
            "amendment_record": amendment["amendment_record"],
            "prerequisite": amendment["prerequisite"],
        }
    )
    return merged, digest


def _load_frozen_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    amendment, digest = load_frozen_protocol(
        path, frozen_statuses=V4_FROZEN_STATUS
    )
    base, base_digest = _load_v3_protocol(
        Path(amendment["base_protocol_path"])
    )
    if base_digest != amendment["base_protocol_sha256"]:
        raise RuntimeError("I1-V3 base protocol digest mismatch")
    if (
        amendment["amendment_record"]["prior_protocol_sha256"]
        != base_digest
    ):
        raise RuntimeError("I1-V4 amendment chain mismatch")
    prior_result = Path(
        amendment["amendment_record"]["prior_development_result_path"]
    )
    prior_digest = hashlib.sha256(prior_result.read_bytes()).hexdigest()
    if prior_digest != amendment["amendment_record"][
        "prior_development_result_sha256"
    ]:
        raise RuntimeError("I1-V4 prior development artifact mismatch")
    merged = dict(base)
    merged.update(
        {
            "protocol_name": amendment["protocol_name"],
            "protocol_version": amendment["protocol_version"],
            "status": amendment["status"],
            "base_protocol_path": amendment["base_protocol_path"],
            "base_protocol_sha256": amendment["base_protocol_sha256"],
            "amendment_record_v3": base["amendment_record"],
            "amendment_record": amendment["amendment_record"],
            "prerequisite": amendment["prerequisite"],
            "architecture_limits": amendment["architecture_limits"],
            "fixed_gates": amendment["fixed_gates"],
            "calibration": amendment["calibration"],
            "validation": amendment["validation"],
            "holdout": amendment["holdout"],
            "authorization_rules": amendment["authorization_rules"],
            "shared_git_registry": amendment["shared_git_registry"],
        }
    )
    return merged, digest


def _require_prerequisite(prerequisite: dict[str, Any]) -> dict[str, Any]:
    source = Path(prerequisite["artifact"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != prerequisite["artifact_sha256"]:
        raise RuntimeError("I1-V4 prerequisite artifact digest mismatch")
    return require_authorization(
        source,
        expected_name=prerequisite["expected_name"],
        expected_decision=prerequisite["expected_decision"],
    )


def _require_i1_artifact(
    path: str | Path,
    *,
    split: str,
    protocol_digest: str,
    seeds: tuple[int, ...],
    decision: str,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _require_i1_payload(
        payload,
        split=split,
        protocol_digest=protocol_digest,
        seeds=seeds,
        decision=decision,
        protocol=protocol,
    )


def _require_i1_payload(
    payload: dict[str, Any],
    *,
    split: str,
    protocol_digest: str,
    seeds: tuple[int, ...],
    decision: str,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload.get("result_schema_version") != 1:
        raise RuntimeError(f"invalid I1-V4 {split} schema")
    if payload.get("experiment") != (
        "V2-I1-unified-entity-belief-graph"
    ):
        raise RuntimeError(f"invalid I1-V4 {split} experiment")
    if payload.get("review_split") != split:
        raise RuntimeError(f"invalid I1-V4 {split} split")
    if payload.get("protocol_sha256") != protocol_digest:
        raise RuntimeError(f"invalid I1-V4 {split} protocol digest")
    if payload.get("seeds") != list(seeds):
        raise RuntimeError(f"invalid I1-V4 {split} seeds")
    if payload.get("passed") is not True:
        raise RuntimeError(f"I1-V4 {split} did not pass")
    if payload.get("decision") != decision:
        raise RuntimeError(f"I1-V4 {split} did not authorize next split")
    gates = payload.get("gates")
    if (
        not isinstance(gates, dict)
        or set(gates) != REQUIRED_I1_GATES
        or not all(gates.values())
    ):
        raise RuntimeError(f"I1-V4 {split} gates are incomplete")
    if protocol is not None:
        aggregate = payload.get("aggregate")
        conditions = payload.get("conditions")
        resources = payload.get("resources")
        architecture = payload.get("architecture")
        if (
            not isinstance(aggregate, dict)
            or not isinstance(conditions, dict)
            or not isinstance(resources, dict)
            or not isinstance(architecture, dict)
        ):
            raise RuntimeError(
                f"I1-V4 {split} evidence is incomplete"
            )
        agent = IntegratedBeliefAgentV2()
        try:
            aggregate_from_conditions = _aggregate(conditions)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"I1-V4 {split} conditions are invalid"
            ) from error
        if aggregate != aggregate_from_conditions:
            raise RuntimeError(
                f"I1-V4 {split} aggregate does not recompute"
            )
        fixed_resource_values = {
            "learnable_parameter_count": (
                agent.learnable_parameter_count
            ),
            "active_state_bytes": agent.active_state_bytes,
            "estimated_mac_per_step": agent.estimated_mac_per_step,
            "steps_per_seed": protocol["fixed_runtime"][
                "steps_per_seed"
            ],
            "maximum_replays_per_experience": protocol[
                "fixed_runtime"
            ]["maximum_replays_per_experience"],
        }
        if any(
            resources.get(name) != value
            for name, value in fixed_resource_values.items()
        ):
            raise RuntimeError(
                f"I1-V4 {split} resources do not recompute"
            )
        update_parameters = set(
            inspect.signature(
                IntegratedBeliefAgentV2.update
            ).parameters
        )
        recomputed = _mechanism_gates(
            aggregate, protocol["fixed_gates"]
        )
        recomputed.update(
            {
                "single_agent_single_stream": (
                    update_parameters
                    == {"self", "sensed_occupancy", "action"}
                ),
                "shared_entity_store_for_all_outputs": (
                    hasattr(agent, "graph")
                    and not hasattr(agent, "memory")
                    and architecture.get("shared_entity_store") is True
                ),
                "labels_absent_from_learner": (
                    constructor_apis_reject_ground_truth(
                        IntegratedBeliefAgentV2.update
                    )
                    and _learner_source_has_no_evaluation_imports()
                    and payload.get(
                        "evaluation_labels_used_for_learning"
                    )
                    is False
                    and payload.get("privileged_input") is None
                ),
                "resources_pass": resources_pass(
                    resources, protocol["resource_limits"]
                ),
                "architecture_limits_match_protocol": (
                    agent.graph.maximum_hypotheses
                    == protocol["architecture_limits"][
                        "maximum_global_hypotheses"
                    ]
                    == architecture.get(
                        "maximum_global_hypotheses"
                    )
                    and agent.graph.maximum_entities
                    == protocol["architecture_limits"][
                        "maximum_entities_per_hypothesis"
                    ]
                    == architecture.get(
                        "maximum_entities_per_hypothesis"
                    )
                ),
            }
        )
        if gates != recomputed or not all(recomputed.values()):
            raise RuntimeError(
                f"I1-V4 {split} gates do not recompute"
            )
    provenance = payload.get("provenance")
    if (
        not isinstance(provenance, dict)
        or not provenance.get("source_sha256")
        or not provenance.get("git_commit")
        or provenance.get("git_dirty") is not False
    ):
        raise RuntimeError(f"I1-V4 {split} provenance is incomplete")
    start = payload.get("run_start")
    if (
        not isinstance(start, dict)
        or start.get("git_dirty") is not False
        or start.get("git_commit") != provenance.get("git_commit")
        or start.get("source_sha256") != provenance.get("source_sha256")
    ):
        raise RuntimeError(
            f"I1-V4 {split} did not start from its reviewed clean commit"
        )
    return payload


def _learner_source_has_no_evaluation_imports() -> bool:
    source_path = Path(inspect.getfile(IntegratedBeliefAgentV2))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
    return all(
        not module.startswith("cal.evaluation")
        for module in imported_modules
    )


def _git_command(
    *arguments: str,
    cwd: str | Path | None = None,
    check: bool = True,
    text: bool = True,
    input_data: bytes | str | None = None,
) -> subprocess.CompletedProcess[Any]:
    result = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        capture_output=True,
        text=text,
        input=input_data,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = (
            result.stderr.strip()
            if isinstance(result.stderr, str)
            else result.stderr.decode(errors="replace").strip()
        )
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {stderr}"
        )
    return result


def _remote_tag_exists(
    remote: str,
    tag: str,
    *,
    cwd: str | Path | None = None,
) -> bool:
    result = _git_command(
        "ls-remote",
        "--exit-code",
        "--tags",
        remote,
        f"refs/tags/{tag}",
        cwd=cwd,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 2:
        return False
    raise RuntimeError(
        f"cannot query shared I1 registry {remote}: "
        f"{result.stderr.strip()}"
    )


def _publish_annotated_tag(
    *,
    remote: str,
    tag: str,
    target: str,
    certificate: dict[str, Any],
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    local = _git_command(
        "rev-parse",
        "--verify",
        f"refs/tags/{tag}",
        cwd=cwd,
        check=False,
    )
    if local.returncode == 0 or _remote_tag_exists(
        remote, tag, cwd=cwd
    ):
        raise RuntimeError(
            f"shared one-shot I1-V4 tag already exists: {tag}"
        )
    unique_certificate = dict(certificate)
    unique_certificate["publication_nonce"] = uuid4().hex
    _git_command(
        "tag",
        "-a",
        tag,
        target,
        "-m",
        json.dumps(unique_certificate, sort_keys=True),
        cwd=cwd,
    )
    local_oid = _git_command(
        "rev-parse",
        f"refs/tags/{tag}",
        cwd=cwd,
    ).stdout.strip()
    try:
        _git_command(
            "push",
            f"--force-with-lease=refs/tags/{tag}:",
            remote,
            f"refs/tags/{tag}",
            cwd=cwd,
        )
    except RuntimeError as error:
        remote_result = _git_command(
            "ls-remote",
            "--tags",
            remote,
            f"refs/tags/{tag}",
            cwd=cwd,
        )
        remote_entries = {
            reference: oid
            for oid, reference in (
                line.split(maxsplit=1)
                for line in remote_result.stdout.splitlines()
                if line.strip()
            )
        }
        if remote_entries.get(f"refs/tags/{tag}") == local_oid:
            return {
                "certificate": unique_certificate,
                "tag_object_sha": local_oid,
            }
        _git_command("tag", "-d", tag, cwd=cwd, check=False)
        raise RuntimeError(
            f"shared one-shot I1-V4 tag CAS was not acquired: {tag}"
        ) from error
    remote_result = _git_command(
        "ls-remote",
        "--tags",
        remote,
        f"refs/tags/{tag}",
        cwd=cwd,
    )
    remote_entries = {
        reference: oid
        for oid, reference in (
            line.split(maxsplit=1)
            for line in remote_result.stdout.splitlines()
            if line.strip()
        )
    }
    if remote_entries.get(f"refs/tags/{tag}") != local_oid:
        _git_command("tag", "-d", tag, cwd=cwd, check=False)
        raise RuntimeError(
            f"shared one-shot I1-V4 tag CAS was not acquired: {tag}"
        )
    return {
        "certificate": unique_certificate,
        "tag_object_sha": local_oid,
    }


def _reserve_shared_one_shot(
    *,
    remote: str,
    tag: str,
    split: str,
    protocol_digest: str,
    git_commit: str,
    source_sha256: str,
    attempt_id: str | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    certificate = {
        "certificate_schema_version": 1,
        "certificate_type": "one_shot_consumption",
        "split": split,
        "protocol_sha256": protocol_digest,
        "git_commit": git_commit,
        "source_sha256": source_sha256,
        "status": "consumed_before_first_episode",
    }
    if attempt_id is not None:
        certificate["attempt_id"] = attempt_id
    return _publish_annotated_tag(
        remote=remote,
        tag=tag,
        target=git_commit,
        certificate=certificate,
        cwd=cwd,
    )


def _publish_result_evidence(
    path: str | Path,
    *,
    remote: str,
    tag: str,
    split: str,
    protocol_digest: str,
    git_commit: str,
    source_sha256: str,
    extra_certificate: Mapping[str, Any] | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(path)
    raw = source.read_bytes()
    result_sha256 = hashlib.sha256(raw).hexdigest()
    blob = _git_command(
        "hash-object",
        "-w",
        "--stdin",
        cwd=cwd,
        text=False,
        input_data=raw,
    ).stdout.decode().strip()
    publication = _publish_annotated_tag(
        remote=remote,
        tag=tag,
        target=blob,
        certificate={
            **dict(extra_certificate or {}),
            "certificate_schema_version": 1,
            "certificate_type": "immutable_result_evidence",
            "split": split,
            "protocol_sha256": protocol_digest,
            "git_commit": git_commit,
            "source_sha256": source_sha256,
            "result_sha256": result_sha256,
            "git_blob": blob,
        },
        cwd=cwd,
    )
    return {
        **publication,
        "git_blob": blob,
        "result_sha256": result_sha256,
    }


def _load_result_evidence(
    *,
    remote: str,
    tag: str,
    split: str,
    protocol_digest: str,
    cwd: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _remote_tag_exists(remote, tag, cwd=cwd):
        raise RuntimeError(
            f"shared I1-V4 {split} evidence tag is absent"
        )
    _git_command(
        "fetch",
        "--force",
        remote,
        f"refs/tags/{tag}:refs/tags/{tag}",
        cwd=cwd,
    )
    contents = _git_command(
        "for-each-ref",
        "--format=%(contents)",
        f"refs/tags/{tag}",
        cwd=cwd,
    ).stdout.strip()
    try:
        certificate = json.loads(contents)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"invalid shared I1-V4 {split} certificate"
        ) from error
    blob = _git_command(
        "rev-parse",
        f"refs/tags/{tag}^{{}}",
        cwd=cwd,
    ).stdout.strip()
    raw = _git_command(
        "cat-file",
        "-p",
        blob,
        cwd=cwd,
        text=False,
    ).stdout
    if (
        certificate.get("certificate_schema_version") != 1
        or certificate.get("certificate_type")
        != "immutable_result_evidence"
        or certificate.get("split") != split
        or certificate.get("protocol_sha256") != protocol_digest
        or certificate.get("git_blob") != blob
        or certificate.get("result_sha256")
        != hashlib.sha256(raw).hexdigest()
    ):
        raise RuntimeError(
            f"invalid shared I1-V4 {split} evidence certificate"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"invalid shared I1-V4 {split} result blob"
        ) from error
    return payload, certificate


def _reserve_one_shot(
    path: str | Path,
    *,
    split: str,
    protocol_digest: str,
    git_commit: str,
    attempt_id: str | None = None,
    consumption_tag_object_sha: str | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reservation = {
        "split": split,
        "protocol_sha256": protocol_digest,
        "git_commit": git_commit,
        "status": "consumed_before_first_episode",
    }
    if attempt_id is not None:
        reservation["attempt_id"] = attempt_id
    if consumption_tag_object_sha is not None:
        reservation["consumption_tag_object_sha"] = (
            consumption_tag_object_sha
        )
    payload = (
        json.dumps(
            reservation,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise RuntimeError(
            f"one-shot I1-V4 {split} reservation already exists"
        ) from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def run_v2_i1_v2(
    *,
    split: str,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    protocol, protocol_digest = _load_frozen_protocol(protocol_path)
    if split not in {"calibration", "validation", "holdout"}:
        raise ValueError(
            "split must be calibration, validation, or holdout"
        )
    expected_output = Path(protocol[split]["result_path"])
    destination = Path(output_path) if output_path else expected_output
    if destination != expected_output:
        raise RuntimeError("output path does not match the frozen protocol")
    if split in {"validation", "holdout"} and destination.exists():
        raise RuntimeError(f"one-shot I1-V4 {split} already exists")
    prerequisite = protocol["prerequisite"]
    _require_prerequisite(prerequisite)
    seeds = tuple(int(seed) for seed in protocol[split]["seeds"])
    run_start = capture_provenance()
    if split in {"validation", "holdout"} and run_start["git_dirty"]:
        raise RuntimeError(
            f"one-shot I1-V4 {split} requires a reviewed clean commit"
        )
    if split == "validation":
        calibration_seeds = tuple(
            int(seed) for seed in protocol["calibration"]["seeds"]
        )
        calibration = _require_i1_artifact(
            protocol["calibration"]["result_path"],
            split="calibration",
            protocol_digest=protocol_digest,
            seeds=calibration_seeds,
            decision="authorize_one_shot_i1_v4_validation",
            protocol=protocol,
        )
        if (
            calibration["run_start"]["git_commit"]
            != run_start["git_commit"]
            or calibration["run_start"]["source_sha256"]
            != run_start["source_sha256"]
        ):
            raise RuntimeError(
                "calibration and validation must use identical reviewed source"
            )
    elif split == "holdout":
        validation_seeds = tuple(
            int(seed) for seed in protocol["validation"]["seeds"]
        )
        registry = protocol["shared_git_registry"]
        validation_payload, validation_certificate = (
            _load_result_evidence(
                remote=registry["remote"],
                tag=registry["validation_evidence_tag"],
                split="validation",
                protocol_digest=protocol_digest,
            )
        )
        validation = _require_i1_payload(
            validation_payload,
            split="validation",
            protocol_digest=protocol_digest,
            seeds=validation_seeds,
            decision="authorize_one_shot_i1_v4_holdout",
            protocol=protocol,
        )
        if (
            validation["run_start"]["git_commit"]
            != run_start["git_commit"]
            or validation["run_start"]["source_sha256"]
            != run_start["source_sha256"]
            or validation_certificate["git_commit"]
            != run_start["git_commit"]
            or validation_certificate["source_sha256"]
            != run_start["source_sha256"]
        ):
            raise RuntimeError(
                "validation and holdout must use identical reviewed source"
            )
    if split in {"validation", "holdout"}:
        registry = protocol["shared_git_registry"]
        consumption_tag = registry[
            f"{split}_consumption_tag"
        ]
        _reserve_shared_one_shot(
            remote=registry["remote"],
            tag=consumption_tag,
            split=split,
            protocol_digest=protocol_digest,
            git_commit=run_start["git_commit"],
            source_sha256=run_start["source_sha256"],
        )
        _reserve_one_shot(
            protocol[split]["reservation_path"],
            split=split,
            protocol_digest=protocol_digest,
            git_commit=run_start["git_commit"],
        )

    steps = int(protocol["fixed_runtime"]["steps_per_seed"])
    started = perf_counter()
    conditions = _run_conditions(seeds, steps=steps)
    aggregate = _aggregate(conditions)
    fixed = protocol["fixed_gates"]
    gates = _mechanism_gates(aggregate, fixed)

    agent = IntegratedBeliefAgentV2()
    resources = build_resources(agent, steps=steps, started=started)
    update_parameters = set(
        inspect.signature(IntegratedBeliefAgentV2.update).parameters
    )
    gates.update(
        {
            "single_agent_single_stream": (
                update_parameters
                == {"self", "sensed_occupancy", "action"}
            ),
            "shared_entity_store_for_all_outputs": (
                hasattr(agent, "graph")
                and not hasattr(agent, "memory")
            ),
            "labels_absent_from_learner": (
                constructor_apis_reject_ground_truth(
                    IntegratedBeliefAgentV2.update
                )
                and _learner_source_has_no_evaluation_imports()
            ),
            "resources_pass": resources_pass(
                resources, protocol["resource_limits"]
            ),
            "architecture_limits_match_protocol": (
                agent.graph.maximum_hypotheses
                == protocol["architecture_limits"][
                    "maximum_global_hypotheses"
                ]
                and agent.graph.maximum_entities
                == protocol["architecture_limits"][
                    "maximum_entities_per_hypothesis"
                ]
            ),
        }
    )
    passed = all(gates.values())
    passing_decisions = {
        "calibration": "authorize_one_shot_i1_v4_validation",
        "validation": "authorize_one_shot_i1_v4_holdout",
        "holdout": "i1_next_generation_architecture_verified",
    }
    result = {
        "result_schema_version": 1,
        "experiment": "V2-I1-unified-entity-belief-graph",
        "review_split": split,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "holdout_run_count": 1 if split == "holdout" else 0,
        "one_shot_run_count": (
            1 if split in {"validation", "holdout"} else 0
        ),
        "formal_agent_input": (
            "sensed_visible_occupancy_and_executed_action_copy_only"
        ),
        "evaluation_labels_used_for_learning": False,
        "privileged_input": None,
        "architecture": {
            "shared_entity_store": True,
            "maximum_global_hypotheses": agent.graph.maximum_hypotheses,
            "maximum_entities_per_hypothesis": (
                agent.graph.maximum_entities
            ),
            "hard_self_lock": False,
            "visibility_agnostic_reachable_floor": False,
        },
        "seeds": list(seeds),
        "conditions": conditions,
        "aggregate": aggregate,
        "resources": resources,
        "gates": gates,
        "passed": passed,
        "decision": (
            passing_decisions[split]
            if passed
            else "stop_i1_v4_before_next_split"
        ),
        "run_start": {
            "git_commit": run_start["git_commit"],
            "git_dirty": run_start["git_dirty"],
            "source_sha256": run_start["source_sha256"],
        },
        "provenance": capture_provenance(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if split in {"validation", "holdout"}:
        registry = protocol["shared_git_registry"]
        evidence_tag = registry[
            f"{split}_evidence_tag"
        ]
        _publish_result_evidence(
            destination,
            remote=registry["remote"],
            tag=evidence_tag,
            split=split,
            protocol_digest=protocol_digest,
            git_commit=run_start["git_commit"],
            source_sha256=run_start["source_sha256"],
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("calibration", "validation", "holdout"),
        required=True,
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    result = run_v2_i1_v2(
        split=arguments.split,
        protocol_path=arguments.protocol,
        output_path=arguments.output,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
