"""Versioned canonical artifacts for Phase-R capacity conformance.

This module is deliberately separate from the frozen Phase-0 statistics
validator.  Candidate-side capacity layouts may evolve without invalidating a
completed candidate-independent power artifact.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from cal.evaluation.stochastic_permanence_artifacts import (
    _require_mapping,
    _require_nonnegative_int,
    _require_nonnegative_number,
    _validate_source_lock_structure,
    canonical_json_bytes,
    sha256_bytes,
    source_lock,
)


CAPACITY_ARTIFACT_SCHEMA_VERSION = 3
CAPACITY_ARTIFACT_KIND = "stochastic_permanence_capacity_conformance"


def validate_capacity_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("artifact_kind") != CAPACITY_ARTIFACT_KIND:
        raise RuntimeError("capacity artifact kind mismatch")
    if payload.get("artifact_schema_version") != CAPACITY_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("capacity artifact schema mismatch")
    if (
        payload.get("status") != "development_only_non_gated"
        or payload.get("privileged_diagnostic_only") is not True
        or payload.get("candidate_artifact_eligible") is not False
    ):
        raise RuntimeError("capacity artifact privilege/status mismatch")
    turn_probability = _require_nonnegative_number(
        payload.get("turn_probability"), name="turn_probability"
    )
    if turn_probability > 1.0:
        raise RuntimeError("turn_probability must be in [0, 1]")

    contract = _require_mapping(
        payload.get("capacity_contract"), name="capacity contract"
    )
    dimensions = {
        name: _require_nonnegative_int(contract.get(name), name=name)
        for name in ("H_max", "E_max", "K_max", "grid_size", "S_max")
    }
    if any(value < 1 for value in dimensions.values()):
        raise RuntimeError("capacity dimensions must be positive")
    h_max, e_max, k_max = (
        dimensions["H_max"],
        dimensions["E_max"],
        dimensions["K_max"],
    )
    s_max = dimensions["S_max"]
    required_slots = h_max * e_max * k_max
    if (
        contract.get("S_max_minimum") != required_slots
        or contract.get("fully_detached_safe") is not (s_max >= required_slots)
        or contract.get("copy_on_write") is not False
    ):
        raise RuntimeError("fully detached capacity contract mismatch")
    if (
        contract.get("state_code_dtype") != "uint16"
        or contract.get("probability_dtype") != "float32"
        or contract.get("overflow_policy")
        != "atomic_fail_without_partial_commit"
        or contract.get("atomic_staging_scope") != "one_factor"
        or contract.get("update_schedule")
        != "sequential_factor_expansion_into_factor_local_staging_then_validated_slice_commit"
        or contract.get("atomic_overflow_probe_passed") is not True
        or contract.get("python_container_guard_bytes") != 8_192
        or contract.get("pruning_tie_break")
        != "probability_desc_then_state_code_asc"
        or contract.get("cumulative_pruned_mass_definition")
        != "1-product(retained_probability)"
        or contract.get("tv_definition")
        != "0.5*L1(position_marginal_packed,position_marginal_reference)"
    ):
        raise RuntimeError("factor-local atomic capacity contract mismatch")
    grid_size = dimensions["grid_size"]
    arena_low = _require_nonnegative_int(
        contract.get("arena_low"), name="arena_low"
    )
    arena_high = _require_nonnegative_int(
        contract.get("arena_high"), name="arena_high"
    )
    state_capacity = (arena_high - arena_low + 1) ** 2 * 4
    if (
        arena_low > arena_high
        or arena_high >= grid_size
        or contract.get("valid_state_capacity") != state_capacity
        or contract.get("direct_index_capacity") != state_capacity
        or contract.get("direct_index_accumulator") is not True
        or contract.get("shared_expansion_workspace_required") != 12 * k_max
        or contract.get("shared_expansion_workspace_size") != 12 * k_max
    ):
        raise RuntimeError("bounded expansion capacity contract mismatch")

    expected_arrays = {
        "codes": ([s_max], "uint16"),
        "probability": ([s_max], "float32"),
        "scratch_codes": ([k_max], "uint16"),
        "scratch_probability": ([k_max], "float32"),
        "expansion_codes": ([12 * k_max], "uint16"),
        "expansion_probability": ([12 * k_max], "float64"),
        "expansion_index": ([state_capacity], "int16"),
        "counts": ([h_max, e_max], "uint8"),
        "existence": ([h_max, e_max], "float32"),
        "self_probability": ([h_max, e_max + 1], "float32"),
        "hypothesis_weight": ([h_max], "float32"),
        "assignment_workspace": ([h_max, e_max, h_max], "float32"),
        "static_probability": ([grid_size, grid_size], "float16"),
        "front_static_score": ([grid_size, grid_size], "int16"),
        "last_visibility": ([grid_size, grid_size], "bool"),
        "last_sensed": ([grid_size, grid_size], "bool"),
        "rng_state": ([4], "uint64"),
        "kernel_parameters": ([16], "float32"),
        "configuration": ([16], "uint32"),
    }
    dtype_bytes = {
        "bool": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "float16": 2,
        "uint32": 4,
        "float32": 4,
        "uint64": 8,
        "float64": 8,
    }
    arrays = _require_mapping(
        contract.get("persistent_arrays"), name="persistent arrays"
    )
    if set(arrays) != set(expected_arrays):
        raise RuntimeError("persistent array population mismatch")
    array_bytes = 0
    for name, (shape, dtype) in expected_arrays.items():
        descriptor = _require_mapping(arrays[name], name=f"array {name}")
        expected_nbytes = math.prod(shape) * dtype_bytes[dtype]
        if (
            descriptor.get("shape") != shape
            or descriptor.get("dtype") != dtype
            or descriptor.get("nbytes") != expected_nbytes
        ):
            raise RuntimeError(f"persistent array {name} layout mismatch")
        array_bytes += expected_nbytes

    conformance = _require_mapping(
        payload.get("conformance"), name="capacity conformance"
    )
    episodes = conformance.get("episodes")
    errors = conformance.get("errors")
    if not isinstance(episodes, list) or not isinstance(errors, list):
        raise RuntimeError("capacity episodes/errors must be arrays")
    if conformance.get("episode_count") != len(episodes):
        raise RuntimeError("capacity episode count mismatch")
    maximum_pruned = _require_nonnegative_number(
        conformance.get("maximum_cumulative_pruned_mass"),
        name="maximum cumulative pruned mass",
    )
    maximum_tv = _require_nonnegative_number(
        conformance.get("maximum_position_tv"), name="maximum position TV"
    )
    maximum_branch = _require_nonnegative_number(
        conformance.get("maximum_branch_accounting_residual"),
        name="maximum branch residual",
    )
    if episodes:
        recomputed = (
            max(float(item["cumulative_pruned_mass"]) for item in episodes),
            max(float(item["maximum_position_tv"]) for item in episodes),
            max(
                float(item["packed_branch_accounting_residual"])
                for item in episodes
            ),
        )
        if not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(
                (maximum_pruned, maximum_tv, maximum_branch), recomputed
            )
        ):
            raise RuntimeError("capacity conformance maxima mismatch")
    alignment = _require_mapping(
        conformance.get("known_topology_kernel_alignment"),
        name="kernel alignment",
    )
    support_mismatches = _require_nonnegative_int(
        alignment.get("support_mismatch_count"), name="support mismatch count"
    )
    checked_cases = _require_nonnegative_int(
        alignment.get("checked_transition_cases"), name="checked transition cases"
    )
    alignment_l1 = _require_nonnegative_number(
        alignment.get("maximum_probability_l1"), name="kernel alignment L1"
    )

    resources = _require_mapping(payload.get("resources"), name="resources")
    resource_names = (
        "learnable_parameter_count",
        "persistent_array_bytes",
        "declared_active_state_bytes",
        "measured_full_object_deep_size_bytes",
        "estimated_mac_per_step",
        "active_state_limit_bytes",
        "parameter_limit",
        "mac_per_step_limit",
    )
    resource = {
        name: _require_nonnegative_int(resources.get(name), name=name)
        for name in resource_names
    }
    if (
        resources.get("resource_scope")
        != "complete_persistent_candidate_prototype_object_graph"
        or resource["active_state_limit_bytes"] != 65_536
        or resource["parameter_limit"] != 100_000
        or resource["mac_per_step_limit"] != 5_000_000
    ):
        raise RuntimeError("capacity resource contract mismatch")
    guard = _require_nonnegative_int(
        contract.get("python_container_guard_bytes"), name="guard bytes"
    )
    if (
        resource["persistent_array_bytes"] != array_bytes
        or resource["declared_active_state_bytes"] != array_bytes + guard
        or resource["measured_full_object_deep_size_bytes"]
        > resource["declared_active_state_bytes"]
        or resource["learnable_parameter_count"] != 16
        or resource["estimated_mac_per_step"] != h_max * e_max * k_max * 12 * 32
    ):
        raise RuntimeError("capacity resource accounting mismatch")
    research = _require_mapping(
        resources.get("formal_research_budget_contract"),
        name="formal research budget",
    )
    research_ok = dict(research) == {
        "steps_per_seed_maximum": 100_000,
        "train_replay_maximum": 4,
        "cpu_total_seconds_maximum": 7_200,
    }

    provenance = _require_mapping(payload.get("provenance"), name="provenance")
    registry_path = Path(str(provenance.get("registry_path", "")))
    registry_digest = provenance.get("registry_selection_digest_sha256")
    runtime = provenance.get("runtime")
    if (
        not registry_path.as_posix()
        or registry_path.is_absolute()
        or ".." in registry_path.parts
        or not isinstance(registry_digest, str)
        or len(registry_digest) != 64
        or any(character not in "0123456789abcdef" for character in registry_digest)
        or not isinstance(provenance.get("command"), str)
        or not provenance["command"]
        or not isinstance(runtime, Mapping)
        or not all(
            isinstance(runtime.get(name), str) and runtime[name]
            for name in ("python", "platform", "numpy")
        )
        or not math.isclose(
            float(provenance.get("registry_turn_probability", math.nan)),
            turn_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("capacity registry provenance mismatch")
    _validate_source_lock_structure(provenance.get("source_lock"))
    source_files = _require_mapping(
        _require_mapping(
            provenance.get("source_lock"), name="source lock"
        ).get("files"),
        name="source lock files",
    )
    required_sources = {
        "cal/model/stochastic_motion_filter.py",
        "cal/evaluation/stochastic_permanence_kernel_diagnostic.py",
        "cal/evaluation/stochastic_permanence_capacity_artifacts.py",
        "cal/evaluation/stochastic_permanence_artifacts.py",
        "docs/experiments/V2_I1_STOCHASTIC_PERMANENCE_PLAN.md",
        "docs/experiments/V2_I1_PHASE_R_CAPACITY_AMENDMENT_V3.md",
        registry_path.as_posix(),
        "pyproject.toml",
        "uv.lock",
    }
    if not required_sources <= set(source_files):
        raise RuntimeError("capacity source lock is incomplete")

    expected_gates = {
        "episodes_present": bool(episodes),
        "no_filter_errors": not errors,
        "cumulative_pruned_mass": maximum_pruned <= 0.01,
        "maximum_position_tv": maximum_tv <= 0.01,
        "branch_evidence_accounting": maximum_branch <= 1e-12,
        "known_topology_kernel_alignment": (
            checked_cases == 38_400
            and support_mismatches == 0
            and alignment_l1 <= 1e-12
        ),
        "fully_detached_pool_safe": bool(contract["fully_detached_safe"]),
        "shared_expansion_workspace_safe": True,
        "atomic_overflow_safe": True,
        "declared_active_state": resource["declared_active_state_bytes"]
        <= resource["active_state_limit_bytes"],
        "measured_active_state": resource["measured_full_object_deep_size_bytes"]
        <= resource["active_state_limit_bytes"],
        "learnable_parameters": resource["learnable_parameter_count"]
        <= resource["parameter_limit"],
        "mac_per_step": resource["estimated_mac_per_step"]
        <= resource["mac_per_step_limit"],
        "registry_turn_probability": True,
        "formal_research_budget_declared": research_ok,
    }
    gates = _require_mapping(payload.get("gates"), name="capacity gates")
    if dict(gates) != expected_gates:
        raise RuntimeError("capacity gates do not match evidence")
    passed = all(expected_gates.values())
    if (
        payload.get("passed") is not passed
        or payload.get("decision")
        != ("phase_r_go" if passed else "phase_r_no_go")
    ):
        raise RuntimeError("capacity decision mismatch")


def write_capacity_artifact(
    path: str | Path, payload: Mapping[str, Any], *, overwrite: bool = False
) -> str:
    validate_capacity_artifact(payload)
    destination = Path(path)
    sidecar = destination.with_suffix(".sha256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(payload)
    digest = sha256_bytes(raw)
    lock_path = destination.with_name(f".{destination.name}.write.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"artifact write already in progress: {destination}") from error
    os.close(lock_fd)
    temporary: Path | None = None
    temporary_sidecar: Path | None = None
    try:
        if not overwrite and (destination.exists() or sidecar.exists()):
            raise FileExistsError(f"artifact already exists: {destination}")
        artifact_fd, artifact_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        )
        temporary = Path(artifact_name)
        with os.fdopen(artifact_fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        sidecar_fd, sidecar_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{sidecar.name}.", suffix=".tmp"
        )
        temporary_sidecar = Path(sidecar_name)
        with os.fdopen(sidecar_fd, "w", encoding="ascii") as stream:
            stream.write(f"{digest}  {destination.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        temporary_sidecar.replace(sidecar)
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if temporary_sidecar is not None:
            temporary_sidecar.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def load_capacity_artifact(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError("capacity artifact is not canonical JSON")
    digest = sha256_bytes(raw)
    if source.with_suffix(".sha256").read_text(encoding="ascii") != (
        f"{digest}  {source.name}\n"
    ):
        raise RuntimeError("capacity artifact digest sidecar mismatch")
    validate_capacity_artifact(payload)
    return payload, digest
