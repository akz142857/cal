"""Canonical, fail-closed artifacts for stochastic permanence Phase 0/R.

The helpers in this module intentionally do not know any validation or holdout
seeds.  They provide deterministic JSON serialization, digest sidecars, schema
checks, and transitive file locks for development and later custodian-owned
protocol artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


# 3: reference-health gate details dropped the tautological
#    `mean_at_least_development_floor` component, and the score table carries
#    the `belief_free` reference (2026-08-08 review, F1/F5).
ARTIFACT_SCHEMA_VERSION = 3
CAPACITY_ARTIFACT_SCHEMA_VERSION = 2
POWER_SIMULATION_DESIGN_VERSION = (
    "outer_reference_bootstrap_variance_envelope_physical_null_v10"
)
POWER_CANDIDATE_GENERATOR = (
    "bounded_physical_metric_advantage_with_fixed_variance_envelope_"
    "signed_reference_gap_and_feasible_covariance_v10"
)
POWER_GRID_POLICY_VERSION = "bounded_variance_envelope_grid_v6"
POWER_MODEL_INITIALIZATION_POLICY = "fixed_model_seed_17011_no_power_axis_v1"
POWER_GRID_ENDPOINT_VARIANCE_LIMIT = 0.90
POWER_GRID_SCENARIOS = (
    {
        "scenario_name": "low_cov_nominal_variance",
        "covariance_interval": "full_feasible",
        "feasible_covariance_fraction": 0.25,
        "single_initialization_sd_fraction": 0.0,
        "variance_scale_multiplier": 1.0,
    },
    {
        "scenario_name": "low_cov_high_variance",
        "covariance_interval": "full_feasible",
        "feasible_covariance_fraction": 0.25,
        "single_initialization_sd_fraction": 0.0,
        "variance_scale_multiplier": 1.5,
    },
    {
        "scenario_name": "high_cov_nominal_variance",
        "covariance_interval": "full_feasible",
        "feasible_covariance_fraction": 0.75,
        "single_initialization_sd_fraction": 0.0,
        "variance_scale_multiplier": 1.0,
    },
    {
        "scenario_name": "high_cov_high_variance",
        "covariance_interval": "full_feasible",
        "feasible_covariance_fraction": 0.75,
        "single_initialization_sd_fraction": 0.0,
        "variance_scale_multiplier": 1.5,
    },
)
POWER_NULL_GRID_SCENARIOS = (
    {
        "scenario_name": "low_cov_nominal_variance_null",
        "covariance_interval": "full_feasible",
        "feasible_covariance_fraction": 0.25,
        "single_initialization_sd_fraction": 0.0,
        "variance_scale_multiplier": 1.0,
    },
    {
        "scenario_name": "low_cov_high_variance_null",
        "covariance_interval": "full_feasible",
        "feasible_covariance_fraction": 0.25,
        "single_initialization_sd_fraction": 0.0,
        "variance_scale_multiplier": 1.5,
    },
    {
        "scenario_name": "high_cov_nominal_variance_null",
        "covariance_interval": "full_feasible",
        "feasible_covariance_fraction": 0.75,
        "single_initialization_sd_fraction": 0.0,
        "variance_scale_multiplier": 1.0,
    },
    {
        "scenario_name": "high_cov_high_variance_null",
        "covariance_interval": "full_feasible",
        "feasible_covariance_fraction": 0.75,
        "single_initialization_sd_fraction": 0.0,
        "variance_scale_multiplier": 1.5,
    },
)
POWER_METRIC_PHYSICAL_BOUNDS = {
    "top1_accuracy": [0.0, 1.0],
    "categorical_nll": [0.0, float(-math.log(1e-6))],
    "brier": [0.0, 1.0],
    "argmax_position_error": [0.0, 20.0],
}
POWER_SIMULATION_GATE_NAMES = frozenset(
    {
        "bounded_advantage_moment_scenarios_feasible",
        "target_composite_power_familywise_lower_at_least_0.80",
        "binding_contrast_type1_familywise_upper_at_most_0.05",
        "one_sided_ci_coverage_familywise_lower_at_least_0.90",
        "simulated_metrics_within_physical_domain_without_clipping",
        "outer_reference_uncertainty_propagated",
    }
)
POWER_GRID_SELECTION_BASIS = (
    "fixed Cartesian product of feasible-covariance interval fractions "
    "{0.25,0.75} and reference-relative variance-scale multipliers {1.0,1.5}; "
    "power is conditional on preregistered model_seed=17011, so no synthetic "
    "initialization axis participates; no power or coverage result participates "
    "in grid construction"
)
POWER_LOCKED_SIMULATION_TRIALS = 1_024
POWER_LOCKED_BOOTSTRAP_SAMPLES_PER_TRIAL = 2_000
POWER_LOCKED_SIMULATION_SEED = 20_260_731
POWER_LOCKED_TYPE1_SIMULATION_SEED = 20_260_732
POWER_LOCKED_BOOTSTRAP_SEED = 20_260_729
POWER_LOCKED_FORMAL_BOOTSTRAP_SAMPLES = 10_000
# Amended 2026-08-08 (64 -> 150) together with
# `LOCKED_PHASE0_EVALUATION_SOURCE_COUNT`; see that constant for the power
# argument behind the change.
POWER_LOCKED_DEVELOPMENT_SOURCE_COUNT = 150
POWER_CONFIRMATORY_ONE_SIDED_ALPHA = 0.01
POWER_SEED_SAFETY_MULTIPLIER = 2.0
POWER_FAMILYWISE_ALPHA = 0.05
POWER_TARGET_MINIMUM = 0.80
POWER_TYPE1_MAXIMUM = 0.05
POWER_COVERAGE_POINT_MINIMUM = 0.93
POWER_COVERAGE_FAMILYWISE_LOWER_MINIMUM = 0.90
POWER_LOCKED_MINIMUM_POWER_PASSES = 848
POWER_LOCKED_MINIMUM_COVERED = 953
POWER_LOCKED_MAXIMUM_FALSE_REJECTIONS = 29
POWER_TYPE1_CONTRAST_NAMES = frozenset(
    {
        "overall/candidate_vs_geometric/top1",
        "overall/top1_closure_0.30",
        "overall/candidate_vs_geometric/nll",
        "overall/candidate_vs_uniform/nll",
        "overall/nll_closure_0.20",
        "overall/candidate_vs_geometric/brier",
        "overall/candidate_vs_uniform/brier",
        "overall/brier_closure_0.15",
        "2-3/candidate_vs_geometric/top1",
        "4-5/candidate_vs_geometric/top1",
        "4-5/top1_closure_0.20",
        "6+/candidate_vs_geometric/top1",
        "6+/top1_closure_0.40",
        "6+/candidate_vs_old_i1/top1",
        "6+/candidate_vs_uniform/nll",
        "6+/nll_closure_0.20",
        "6+/candidate_vs_uniform/brier",
        "6+/brier_closure_0.15",
        "overall/candidate_vs_geometric/position_error",
        "overall/candidate_vs_old_i1/position_error",
    }
)
POWER_COVERAGE_CONTRAST_NAMES = POWER_TYPE1_CONTRAST_NAMES | frozenset(
    {
        "reference/overall/top1_accuracy",
        "reference/overall/categorical_nll",
        "reference/overall/brier",
        "reference/4-5/top1_accuracy",
        "reference/6+/top1_accuracy",
        "reference/6+/categorical_nll",
        "reference/6+/brier",
    }
)
PHASE0_GATE_NAMES = frozenset(
    {
        "candidate_independent_reference_health",
        "candidate_independent_composite_power",
    }
)
_REQUIRED_BY_KIND: dict[str, frozenset[str]] = {
    "stochastic_permanence_reference_health_power": frozenset(
        {
            "artifact_kind",
            "artifact_schema_version",
            "status",
            "candidate_maps_read",
            "registry_selection_digest_sha256",
            "complete_seed_ids",
            "per_seed_per_bin",
            "reference_health",
            "reference_health_decision",
            "power_design",
            "phase0_gates",
            "passed",
            "decision",
            "provenance",
        }
    ),
    "stochastic_permanence_capacity_conformance": frozenset(
        {
            "artifact_kind",
            "artifact_schema_version",
            "status",
            "privileged_diagnostic_only",
            "candidate_artifact_eligible",
            "turn_probability",
            "capacity_contract",
            "conformance",
            "resources",
            "gates",
            "passed",
            "decision",
            "provenance",
        }
    ),
    "stochastic_permanence_formal_decision": frozenset(
        {
            "artifact_kind",
            "artifact_schema_version",
            "status",
            "complete_seed_ids",
            "per_seed_per_bin",
            "bootstrap",
            "contrasts",
            "gates",
            "passed",
            "decision",
            "locks",
        }
    ),
    "stochastic_permanence_attempt_ledger": frozenset(
        {
            "artifact_kind",
            "artifact_schema_version",
            "status",
            "maximum_attempts",
            "entries",
            "selection",
        }
    ),
}


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the only accepted on-disk JSON representation."""

    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _expected_schema_version(kind: str) -> int:
    if kind == "stochastic_permanence_capacity_conformance":
        return CAPACITY_ARTIFACT_SCHEMA_VERSION
    return ARTIFACT_SCHEMA_VERSION


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be an object")
    return value


def _require_nonnegative_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"{name} must be finite and non-negative")
    return result


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise RuntimeError(f"{name} must be a boolean")
    return value


def _require_finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} must be finite")
    return result


def _require_close(
    value: object,
    expected: float,
    *,
    name: str,
    tolerance: float = 1e-12,
) -> float:
    result = _require_finite_number(value, name=name)
    if not math.isclose(result, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"{name} does not match recomputed evidence")
    return result


def _validate_required_source_lock_structure(
    value: object,
    *,
    required_files: set[str],
) -> None:
    lock = _require_mapping(value, name="source lock")
    if lock.get("algorithm") != "sha256_sorted_relative_path_and_digest_v1":
        raise RuntimeError("source-lock algorithm mismatch")
    files = _require_mapping(lock.get("files"), name="source-lock files")
    if not required_files <= set(files):
        missing = sorted(required_files - set(files))
        raise RuntimeError(f"source lock is missing transitive files: {missing}")
    if lock.get("file_count") != len(files):
        raise RuntimeError("source-lock file count mismatch")
    combined = hashlib.sha256()
    for path in sorted(files):
        digest = files[path]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or path.startswith("../")
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError("source-lock entry is invalid")
        combined.update(path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    if lock.get("combined_sha256") != combined.hexdigest():
        raise RuntimeError("source-lock combined digest mismatch")


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Return P[X <= successes] for X~Binomial(trials, probability)."""

    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_probability = math.log(probability)
    log_failure = math.log1p(-probability)
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(outcome + 1)
        - math.lgamma(trials - outcome + 1)
        + outcome * log_probability
        + (trials - outcome) * log_failure
        for outcome in range(successes + 1)
    ]
    maximum = max(log_terms)
    total = math.exp(maximum) * sum(
        math.exp(value - maximum) for value in log_terms
    )
    return min(1.0, max(0.0, float(total)))


def exact_binomial_lower_bound(
    successes: int,
    trials: int,
    *,
    alpha: float,
) -> float:
    """One-sided Clopper-Pearson lower confidence bound."""

    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial count")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be inside (0, 1)")
    if successes == 0:
        return 0.0
    low, high = 0.0, 1.0
    # At the lower bound, P_p[X >= successes] = alpha.
    for _ in range(96):
        middle = (low + high) / 2.0
        survival = 1.0 - _binomial_cdf(successes - 1, trials, middle)
        if survival < alpha:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def exact_binomial_upper_bound(
    successes: int,
    trials: int,
    *,
    alpha: float,
) -> float:
    """One-sided Clopper-Pearson upper confidence bound."""

    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial count")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be inside (0, 1)")
    if successes == trials:
        return 1.0
    low, high = 0.0, 1.0
    # At the upper bound, P_p[X <= successes] = alpha.
    for _ in range(96):
        middle = (low + high) / 2.0
        cdf = _binomial_cdf(successes, trials, middle)
        if cdf > alpha:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def _validate_source_lock_structure(lock: object) -> None:
    source = _require_mapping(lock, name="source lock")
    if source.get("algorithm") != "sha256_sorted_relative_path_and_digest_v1":
        raise RuntimeError("source lock algorithm mismatch")
    files = _require_mapping(source.get("files"), name="source lock files")
    if not files:
        raise RuntimeError("source lock has no files")
    if source.get("file_count") != len(files):
        raise RuntimeError("source lock file count mismatch")
    if list(files) != sorted(files):
        raise RuntimeError("source lock files are not canonically ordered")
    combined = hashlib.sha256()
    for name, digest in files.items():
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise RuntimeError("source lock contains an invalid relative path")
        if ".." in Path(name).parts:
            raise RuntimeError("source lock path escapes root")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError("source lock contains an invalid digest")
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    if source.get("combined_sha256") != combined.hexdigest():
        raise RuntimeError("source lock combined digest mismatch")


def _validate_capacity_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "development_only_non_gated":
        raise RuntimeError("capacity artifact status mismatch")
    if payload.get("privileged_diagnostic_only") is not True:
        raise RuntimeError("capacity artifact lacks privileged-only marker")
    if payload.get("candidate_artifact_eligible") is not False:
        raise RuntimeError("privileged capacity artifact is candidate-eligible")
    turn_probability = _require_nonnegative_number(
        payload.get("turn_probability"), name="turn_probability"
    )
    if turn_probability > 1.0:
        raise RuntimeError("turn_probability must be in [0, 1]")

    contract = _require_mapping(
        payload.get("capacity_contract"), name="capacity contract"
    )
    required_contract = {
        "H_max",
        "E_max",
        "K_max",
        "grid_size",
        "S_max",
        "S_max_minimum",
        "fully_detached_safe",
        "copy_on_write",
        "arena_low",
        "arena_high",
        "valid_state_capacity",
        "state_code_dtype",
        "probability_dtype",
        "overflow_policy",
        "update_schedule",
        "shared_expansion_workspace_size",
        "shared_expansion_workspace_required",
        "direct_index_accumulator",
        "direct_index_capacity",
        "python_container_guard_bytes",
        "pruning_tie_break",
        "persistent_arrays",
        "cumulative_pruned_mass_definition",
        "tv_definition",
        "atomic_overflow_probe_passed",
    }
    missing_contract = sorted(required_contract - set(contract))
    if missing_contract:
        raise RuntimeError(
            f"capacity contract is missing required fields: {missing_contract}"
        )
    dimensions: dict[str, int] = {}
    for name in (
        "H_max",
        "E_max",
        "K_max",
        "grid_size",
        "S_max",
        "S_max_minimum",
    ):
        value = _require_nonnegative_int(contract.get(name), name=name)
        if value < 1:
            raise RuntimeError(f"capacity contract has invalid {name}")
        dimensions[name] = value
    required_slots = (
        dimensions["H_max"] * dimensions["E_max"] * dimensions["K_max"]
    )
    if dimensions["S_max_minimum"] != required_slots:
        raise RuntimeError("capacity S_max_minimum mismatch")
    if contract.get("fully_detached_safe") is not (
        dimensions["S_max"] >= required_slots
    ):
        raise RuntimeError("capacity fully-detached decision mismatch")
    if contract.get("copy_on_write") is not False:
        raise RuntimeError("Phase R capacity proof must not depend on COW")
    if contract.get("overflow_policy") != "atomic_fail_without_partial_commit":
        raise RuntimeError("capacity overflow policy mismatch")
    if contract.get("atomic_overflow_probe_passed") is not True:
        raise RuntimeError("capacity atomic-overflow probe failed")
    if contract.get("direct_index_accumulator") is not True:
        raise RuntimeError("capacity accumulator is not bounded direct-index")
    if contract.get("shared_expansion_workspace_size") != contract.get(
        "shared_expansion_workspace_required"
    ):
        raise RuntimeError("shared expansion workspace size mismatch")
    if contract.get("direct_index_capacity") != contract.get(
        "valid_state_capacity"
    ):
        raise RuntimeError("direct-index capacity mismatch")
    arena_low = _require_nonnegative_int(contract.get("arena_low"), name="arena_low")
    arena_high = _require_nonnegative_int(
        contract.get("arena_high"), name="arena_high"
    )
    if arena_low > arena_high or arena_high >= dimensions["grid_size"]:
        raise RuntimeError("capacity contract has invalid arena bounds")
    if ((dimensions["grid_size"] ** 2) * 4 - 1) > 65_535:
        raise RuntimeError("capacity state code does not fit uint16")
    expected_state_capacity = (arena_high - arena_low + 1) ** 2 * 4
    if contract.get("valid_state_capacity") != expected_state_capacity:
        raise RuntimeError("valid-state capacity mismatch")
    if contract.get("shared_expansion_workspace_required") != (
        dimensions["K_max"] * 12
    ):
        raise RuntimeError("shared expansion requirement mismatch")
    guard_bytes = _require_nonnegative_int(
        contract.get("python_container_guard_bytes"),
        name="python container guard bytes",
    )
    arrays = _require_mapping(
        contract.get("persistent_arrays"), name="persistent arrays"
    )
    if not arrays:
        raise RuntimeError("capacity artifact has no persistent arrays")
    dtype_bytes = {
        "bool": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "float16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "uint64": 8,
        "float64": 8,
    }
    array_bytes = 0
    array_shapes: dict[str, list[int]] = {}
    for name, descriptor_value in arrays.items():
        descriptor = _require_mapping(
            descriptor_value, name=f"persistent array {name}"
        )
        shape = descriptor.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in shape
        ):
            raise RuntimeError(f"persistent array {name} has invalid shape")
        dtype = descriptor.get("dtype")
        if not isinstance(dtype, str) or dtype not in dtype_bytes:
            raise RuntimeError(f"persistent array {name} has invalid dtype")
        nbytes = descriptor.get("nbytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise RuntimeError(f"persistent array {name} has invalid nbytes")
        element_count = math.prod(shape)
        if nbytes != element_count * dtype_bytes[dtype]:
            raise RuntimeError(f"persistent array {name} byte count mismatch")
        array_bytes += nbytes
        array_shapes[str(name)] = shape
    h_max = dimensions["H_max"]
    e_max = dimensions["E_max"]
    k_max = dimensions["K_max"]
    s_max = dimensions["S_max"]
    grid_size = dimensions["grid_size"]
    expected_shapes = {
        "codes": [s_max],
        "probability": [s_max],
        "scratch_codes": [s_max],
        "scratch_probability": [s_max],
        "expansion_codes": [12 * k_max],
        "expansion_probability": [12 * k_max],
        "expansion_index": [expected_state_capacity],
        "counts": [h_max, e_max],
        "existence": [h_max, e_max],
        "self_probability": [h_max, e_max + 1],
        "hypothesis_weight": [h_max],
        "assignment_workspace": [h_max, e_max, h_max],
        "static_probability": [grid_size, grid_size],
        "front_static_score": [grid_size, grid_size],
        "last_visibility": [grid_size, grid_size],
        "last_sensed": [grid_size, grid_size],
        "rng_state": [4],
        "kernel_parameters": [16],
        "configuration": [16],
    }
    if array_shapes != expected_shapes:
        raise RuntimeError("persistent array layout mismatch")

    conformance = _require_mapping(
        payload.get("conformance"), name="capacity conformance"
    )
    episodes = conformance.get("episodes")
    errors = conformance.get("errors")
    if not isinstance(episodes, list) or not isinstance(errors, list):
        raise RuntimeError("capacity conformance episodes/errors must be arrays")
    if conformance.get("episode_count") != len(episodes):
        raise RuntimeError("capacity conformance episode count mismatch")
    maximum_pruned = _require_nonnegative_number(
        conformance.get("maximum_cumulative_pruned_mass"),
        name="maximum cumulative pruned mass",
    )
    maximum_tv = _require_nonnegative_number(
        conformance.get("maximum_position_tv"), name="maximum position TV"
    )
    branch_residual = _require_nonnegative_number(
        conformance.get("maximum_branch_accounting_residual"),
        name="maximum branch accounting residual",
    )
    alignment = _require_mapping(
        conformance.get("known_topology_kernel_alignment"),
        name="known-topology kernel alignment",
    )
    support_mismatches = alignment.get("support_mismatch_count")
    if (
        isinstance(support_mismatches, bool)
        or not isinstance(support_mismatches, int)
        or support_mismatches < 0
    ):
        raise RuntimeError("kernel alignment has invalid mismatch count")
    alignment_l1 = _require_nonnegative_number(
        alignment.get("maximum_probability_l1"),
        name="kernel alignment maximum L1",
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
    resource_values = {
        name: _require_nonnegative_int(resources.get(name), name=name)
        for name in resource_names
    }
    if resources.get("resource_scope") != (
        "complete_persistent_candidate_prototype_object_graph"
    ):
        raise RuntimeError("capacity resource scope is incomplete")
    if resource_values["persistent_array_bytes"] != array_bytes:
        raise RuntimeError("persistent array byte total mismatch")
    if resource_values["declared_active_state_bytes"] != array_bytes + guard_bytes:
        raise RuntimeError("declared active state does not match arrays plus guard")
    if resource_values["measured_full_object_deep_size_bytes"] > resource_values[
        "declared_active_state_bytes"
    ]:
        raise RuntimeError("declared active state is below measured deep size")
    if resource_values["learnable_parameter_count"] != math.prod(
        expected_shapes["kernel_parameters"]
    ):
        raise RuntimeError("learnable parameter count mismatch")
    expected_mac = h_max * e_max * k_max * 12 * 32
    if resource_values["estimated_mac_per_step"] != expected_mac:
        raise RuntimeError("estimated MAC/step formula mismatch")
    research = _require_mapping(
        resources.get("formal_research_budget_contract"),
        name="formal research budget",
    )
    research_budget_declared = dict(research) == {
        "steps_per_seed_maximum": 100_000,
        "train_replay_maximum": 4,
        "cpu_total_seconds_maximum": 7_200,
    }
    deterministic_work = _require_mapping(
        resources.get("deterministic_diagnostic_work"),
        name="deterministic diagnostic work",
    )
    if "conformance_seconds" in deterministic_work:
        raise RuntimeError("canonical capacity artifact contains wall-clock timing")

    provenance = _require_mapping(payload.get("provenance"), name="provenance")
    for name in ("registry_path", "registry_selection_digest_sha256", "command"):
        if not isinstance(provenance.get(name), str) or not provenance[name]:
            raise RuntimeError(f"capacity provenance has invalid {name}")
    registry_path = Path(provenance["registry_path"])
    if registry_path.is_absolute() or ".." in registry_path.parts:
        raise RuntimeError("capacity provenance registry path escapes root")
    registry_digest = provenance["registry_selection_digest_sha256"]
    if len(registry_digest) != 64 or any(
        character not in "0123456789abcdef" for character in registry_digest
    ):
        raise RuntimeError("capacity provenance registry digest is invalid")
    if not math.isclose(
        float(provenance.get("registry_turn_probability", math.nan)),
        turn_probability,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        registry_turn_probability_gate = False
    else:
        registry_turn_probability_gate = True
    source_lock_value = provenance.get("source_lock")
    _validate_source_lock_structure(source_lock_value)
    source_files = _require_mapping(
        _require_mapping(source_lock_value, name="source lock").get("files"),
        name="source lock files",
    )
    required_source_files = {
        "cal/model/stochastic_motion_filter.py",
        "cal/evaluation/stochastic_permanence_kernel_diagnostic.py",
        "cal/evaluation/stochastic_permanence_artifacts.py",
        "cal/evaluation/permanence_forward_benchmark.py",
        "cal/evaluation/randomized_occlusion_world.py",
        "cal/evaluation/permanence_seed_registry.py",
        "cal/evaluation/stochastic_permanence_custody.py",
        "cal/evaluation/v2_i1_integration.py",
        "cal/evaluation/v2_artifacts.py",
        "cal/infra/provenance.py",
        "cal/model/integrated_agent.py",
        "cal/model/entity_graph.py",
        "cal/model/occupancy.py",
        "docs/experiments/V2_I1_STOCHASTIC_PERMANENCE_PLAN.md",
        "pyproject.toml",
        "uv.lock",
        registry_path.as_posix(),
    }
    missing_sources = sorted(required_source_files - set(source_files))
    if missing_sources:
        raise RuntimeError(
            f"capacity source lock is missing transitive files: {missing_sources}"
        )
    runtime = _require_mapping(provenance.get("runtime"), name="runtime")
    if not all(isinstance(runtime.get(name), str) and runtime[name] for name in (
        "python",
        "platform",
        "numpy",
    )):
        raise RuntimeError("capacity provenance runtime is incomplete")

    gates = _require_mapping(payload.get("gates"), name="capacity gates")
    expected_gates = {
        "episodes_present": bool(episodes),
        "no_filter_errors": not errors,
        "cumulative_pruned_mass": maximum_pruned <= 0.01,
        "maximum_position_tv": maximum_tv <= 0.01,
        "branch_evidence_accounting": branch_residual <= 1e-12,
        "known_topology_kernel_alignment": (
            support_mismatches == 0 and alignment_l1 <= 1e-12
        ),
        "fully_detached_pool_safe": bool(contract["fully_detached_safe"]),
        "shared_expansion_workspace_safe": bool(
            contract["shared_expansion_workspace_size"]
            >= contract["shared_expansion_workspace_required"]
            and contract["direct_index_accumulator"]
        ),
        "atomic_overflow_safe": bool(contract["atomic_overflow_probe_passed"]),
        "declared_active_state": resource_values["declared_active_state_bytes"]
        <= resource_values["active_state_limit_bytes"],
        "measured_active_state": resource_values[
            "measured_full_object_deep_size_bytes"
        ]
        <= resource_values["active_state_limit_bytes"],
        "learnable_parameters": resource_values["learnable_parameter_count"]
        <= resource_values["parameter_limit"],
        "mac_per_step": resource_values["estimated_mac_per_step"]
        <= resource_values["mac_per_step_limit"],
        "registry_turn_probability": registry_turn_probability_gate,
        "formal_research_budget_declared": research_budget_declared,
    }
    for name, expected in expected_gates.items():
        if gates.get(name) is not expected:
            raise RuntimeError(f"capacity gate {name} does not match evidence")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise RuntimeError("capacity gates must be booleans")
    strict_and = all(gates.values())
    if payload.get("passed") is not strict_and:
        raise RuntimeError("capacity decision does not equal strict gate AND")
    expected_decision = "phase_r_go" if strict_and else "phase_r_no_go"
    if payload.get("decision") != expected_decision:
        raise RuntimeError("capacity decision label mismatch")


def _validate_reference_health_power_artifact(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("status") != "development_only_non_gated":
        raise RuntimeError("Phase 0 artifact status mismatch")
    if payload.get("candidate_maps_read") is not False:
        raise RuntimeError("reference-health artifact read candidate maps")
    complete_seed_ids = payload.get("complete_seed_ids")
    if (
        not isinstance(complete_seed_ids, list)
        or len(complete_seed_ids) != POWER_LOCKED_DEVELOPMENT_SOURCE_COUNT
        or any(type(seed) is not int or seed < 0 for seed in complete_seed_ids)
        or complete_seed_ids != sorted(set(complete_seed_ids))
    ):
        raise RuntimeError("Phase 0 complete-seed population is invalid")
    score_table = _require_mapping(
        payload.get("per_seed_per_bin"), name="Phase 0 score table"
    )
    expected_seed_keys = {str(seed) for seed in complete_seed_ids}
    if set(score_table) != {
        "oracle",
        "geometric",
        "belief_free",
        "uniform",
        "old_i1",
    }:
        raise RuntimeError("Phase 0 reference predictor population mismatch")
    if any(
        not isinstance(rows, Mapping) or set(rows) != expected_seed_keys
        for rows in score_table.values()
    ):
        raise RuntimeError("Phase 0 score-table seed population mismatch")

    power_design = _require_mapping(
        payload.get("power_design"), name="power design"
    )
    _require_close(
        power_design.get("alpha_one_sided"),
        POWER_CONFIRMATORY_ONE_SIDED_ALPHA,
        name="confirmatory one-sided alpha",
    )
    _require_close(
        power_design.get("target_power"),
        POWER_TARGET_MINIMUM,
        name="analytic target power",
    )
    simulation = _require_mapping(
        power_design.get("composite_simulation"),
        name="power composite simulation",
    )
    if simulation.get("simulation_design_version") != POWER_SIMULATION_DESIGN_VERSION:
        raise RuntimeError("power simulation design version mismatch")
    if simulation.get("candidate_generator") != POWER_CANDIDATE_GENERATOR:
        raise RuntimeError("power artifact candidate generator mismatch")
    if simulation.get("initialization_error_distribution") != (
        POWER_MODEL_INITIALIZATION_POLICY
    ):
        raise RuntimeError("power initialization distribution mismatch")
    if simulation.get("simulation_rng") != "numpy.PCG64":
        raise RuntimeError("power simulation RNG mismatch")
    if simulation.get("simulation_seed") != POWER_LOCKED_SIMULATION_SEED:
        raise RuntimeError("power simulation seed mismatch")
    if simulation.get("type1_simulation_seed") != (
        POWER_LOCKED_TYPE1_SIMULATION_SEED
    ):
        raise RuntimeError("power Type-I simulation seed mismatch")
    if simulation.get("bootstrap_seed") != POWER_LOCKED_BOOTSTRAP_SEED:
        raise RuntimeError("power bootstrap seed mismatch")
    if simulation.get("formal_runner_bootstrap_samples") != (
        POWER_LOCKED_FORMAL_BOOTSTRAP_SAMPLES
    ):
        raise RuntimeError("formal bootstrap sample count mismatch")
    _require_close(
        simulation.get("familywise_alpha"),
        POWER_FAMILYWISE_ALPHA,
        name="power familywise alpha",
    )
    trials = _require_nonnegative_int(
        simulation.get("trials_per_scenario"), name="power trial count"
    )
    bootstrap_samples = _require_nonnegative_int(
        simulation.get("bootstrap_samples_per_trial"),
        name="power bootstrap sample count",
    )
    if trials < 4 or bootstrap_samples < 2:
        raise RuntimeError("power simulation population is too small")
    acceptance = _require_mapping(
        simulation.get("monte_carlo_acceptance"),
        name="Monte Carlo acceptance contract",
    )
    expected_acceptance = {
        "power_family_size": len(POWER_GRID_SCENARIOS),
        "minimum_power_passes_at_locked_trials": (
            POWER_LOCKED_MINIMUM_POWER_PASSES
        ),
        "coverage_family_size": (
            len(POWER_GRID_SCENARIOS) * len(POWER_COVERAGE_CONTRAST_NAMES)
        ),
        "minimum_covered_at_locked_trials": POWER_LOCKED_MINIMUM_COVERED,
        "type1_family_size": (
            len(POWER_NULL_GRID_SCENARIOS) * len(POWER_TYPE1_CONTRAST_NAMES)
        ),
        "maximum_false_rejections_at_locked_trials": (
            POWER_LOCKED_MAXIMUM_FALSE_REJECTIONS
        ),
    }
    if dict(acceptance) != expected_acceptance:
        raise RuntimeError("Monte Carlo acceptance contract mismatch")
    if not (
        exact_binomial_lower_bound(
            POWER_LOCKED_MINIMUM_POWER_PASSES,
            POWER_LOCKED_SIMULATION_TRIALS,
            alpha=POWER_FAMILYWISE_ALPHA / len(POWER_GRID_SCENARIOS),
        )
        >= POWER_TARGET_MINIMUM
        > exact_binomial_lower_bound(
            POWER_LOCKED_MINIMUM_POWER_PASSES - 1,
            POWER_LOCKED_SIMULATION_TRIALS,
            alpha=POWER_FAMILYWISE_ALPHA / len(POWER_GRID_SCENARIOS),
        )
    ):
        raise RuntimeError("locked power acceptance count is not minimal")
    coverage_family_size = len(POWER_GRID_SCENARIOS) * len(
        POWER_COVERAGE_CONTRAST_NAMES
    )
    if not (
        POWER_LOCKED_MINIMUM_COVERED / POWER_LOCKED_SIMULATION_TRIALS
        >= POWER_COVERAGE_POINT_MINIMUM
        and exact_binomial_lower_bound(
            POWER_LOCKED_MINIMUM_COVERED,
            POWER_LOCKED_SIMULATION_TRIALS,
            alpha=POWER_FAMILYWISE_ALPHA / coverage_family_size,
        )
        >= POWER_COVERAGE_FAMILYWISE_LOWER_MINIMUM
        > exact_binomial_lower_bound(
            POWER_LOCKED_MINIMUM_COVERED - 1,
            POWER_LOCKED_SIMULATION_TRIALS,
            alpha=POWER_FAMILYWISE_ALPHA / coverage_family_size,
        )
    ):
        raise RuntimeError("locked coverage acceptance count is not minimal")
    type1_family_size = len(POWER_NULL_GRID_SCENARIOS) * len(
        POWER_TYPE1_CONTRAST_NAMES
    )
    if not (
        exact_binomial_upper_bound(
            POWER_LOCKED_MAXIMUM_FALSE_REJECTIONS,
            POWER_LOCKED_SIMULATION_TRIALS,
            alpha=POWER_FAMILYWISE_ALPHA / type1_family_size,
        )
        <= POWER_TYPE1_MAXIMUM
        < exact_binomial_upper_bound(
            POWER_LOCKED_MAXIMUM_FALSE_REJECTIONS + 1,
            POWER_LOCKED_SIMULATION_TRIALS,
            alpha=POWER_FAMILYWISE_ALPHA / type1_family_size,
        )
    ):
        raise RuntimeError("locked Type-I acceptance count is not maximal")
    analytic_count = _require_nonnegative_int(
        simulation.get("analytic_starting_seed_count"),
        name="analytic seed count",
    )
    candidate_count = _require_nonnegative_int(
        simulation.get("candidate_seed_count_after_safety_multiplier"),
        name="candidate seed count",
    )
    multiplier = _require_finite_number(
        simulation.get("power_seed_safety_multiplier"),
        name="power seed safety multiplier",
    )
    if multiplier != POWER_SEED_SAFETY_MULTIPLIER or candidate_count != max(
        2, math.ceil(analytic_count * multiplier)
    ):
        raise RuntimeError("power seed-count recommendation mismatch")

    grid_policy = _require_mapping(
        simulation.get("sensitivity_grid_policy"),
        name="power sensitivity-grid policy",
    )
    if grid_policy.get("version") != POWER_GRID_POLICY_VERSION:
        raise RuntimeError("power sensitivity-grid version mismatch")
    if grid_policy.get("selection_basis") != POWER_GRID_SELECTION_BASIS:
        raise RuntimeError("power sensitivity-grid selection rule mismatch")
    if grid_policy.get("metric_physical_bounds") != POWER_METRIC_PHYSICAL_BOUNDS:
        raise RuntimeError("power metric physical bounds mismatch")
    _require_close(
        grid_policy.get("endpoint_variance_fraction_limit"),
        POWER_GRID_ENDPOINT_VARIANCE_LIMIT,
        name="endpoint variance-fraction limit",
    )
    if grid_policy.get("scenarios") != [dict(item) for item in POWER_GRID_SCENARIOS]:
        raise RuntimeError("power sensitivity-grid exact rows mismatch")
    if grid_policy.get("null_scenarios") != [
        dict(item) for item in POWER_NULL_GRID_SCENARIOS
    ]:
        raise RuntimeError("power null-grid exact rows mismatch")
    construction = _require_mapping(
        grid_policy.get("construction_audit"), name="grid construction audit"
    )
    expected_construction = {
        "construction_reads_power_or_coverage": False,
        "parameterization": (
            "fixed_reference_variance_scale_x_feasible_covariance_"
            "interval_fraction_signed_gap_fixed_model_seed_v3"
        ),
        "alternative_grid_is_fixed_cartesian_product": True,
        "null_grid_is_fixed_cartesian_product": True,
        "alternative_scenario_count": len(POWER_GRID_SCENARIOS),
        "null_scenario_count": len(POWER_NULL_GRID_SCENARIOS),
        "passed": True,
    }
    if dict(construction) != expected_construction:
        raise RuntimeError("grid construction audit mismatch")

    results = simulation.get("sensitivity_grid")
    if not isinstance(results, list) or len(results) != len(POWER_GRID_SCENARIOS):
        raise RuntimeError("power sensitivity-grid result population mismatch")
    target_effects = _require_mapping(
        simulation.get("target_effects"), name="power target effects"
    )
    position_target_effect = _require_finite_number(
        simulation.get("position_oracle_gap_target_effect"),
        name="position target effect",
    )
    expected_calibration_labels = {
        f"{scope}/{metric}/init_{sign}"
        for scope in ("2-3", "4-5", "6+")
        for metric in POWER_METRIC_PHYSICAL_BOUNDS
        for sign in ("+0",)
    }
    recomputed_power_passes: list[bool] = []
    recomputed_coverage_passes: list[bool] = []
    recomputed_distribution_passes: list[bool] = []
    recomputed_outer_passes: list[bool] = []
    recomputed_feasibility_passes: list[bool] = []
    for scenario_index, (expected_scenario, result) in enumerate(
        zip(POWER_GRID_SCENARIOS, results, strict=True), start=1
    ):
        result = _require_mapping(result, name="power sensitivity-grid result")
        if result.get("scenario_id") != scenario_index or any(
            result.get(key) != value for key, value in expected_scenario.items()
        ):
            raise RuntimeError("power sensitivity-grid result mismatch")
        feasibility = _require_mapping(
            result.get("bounded_advantage_moment_feasibility"),
            name="bounded moment feasibility",
        )
        failures = feasibility.get("failures")
        calibrations = feasibility.get("calibrations")
        if (
            not isinstance(failures, list)
            or any(not isinstance(item, str) for item in failures)
            or not isinstance(calibrations, Mapping)
            or not set(calibrations).issubset(expected_calibration_labels)
            or (not failures and set(calibrations) != expected_calibration_labels)
        ):
            raise RuntimeError("bounded moment feasibility evidence is incomplete")
        calibrations_complete = set(calibrations) == expected_calibration_labels
        fractions: list[float] = []
        all_coordinates_valid = True
        for label, raw in calibrations.items():
            calibration = _require_mapping(raw, name=f"calibration {label}")
            if (
                calibration.get("requested_correlation") is not None
                or calibration.get("requested_variance_inflation") is not None
            ):
                raise RuntimeError("feasibility-native grid targets legacy moments")
            requested_covariance_fraction = float(
                expected_scenario["feasible_covariance_fraction"]
            )
            requested_variance_multiplier = float(
                expected_scenario["variance_scale_multiplier"]
            )
            _require_close(
                calibration.get("requested_feasible_covariance_fraction"),
                requested_covariance_fraction,
                name=f"{label} requested covariance fraction",
            )
            if calibration.get("covariance_interval") != expected_scenario[
                "covariance_interval"
            ]:
                raise RuntimeError("calibration covariance interval mismatch")
            _require_close(
                calibration.get("requested_variance_scale_multiplier"),
                requested_variance_multiplier,
                name=f"{label} requested variance multiplier",
            )
            correlation_identifiable = calibration.get(
                "correlation_identifiable"
            ) is True
            variance_identifiable = calibration.get(
                "variance_identifiable"
            ) is True
            tolerance = _require_nonnegative_number(
                calibration.get("moment_tolerance"),
                name=f"{label} moment tolerance",
            )
            if tolerance > 1e-6:
                raise RuntimeError("calibration moment tolerance is too loose")
            if correlation_identifiable:
                _require_finite_number(
                    calibration.get("achieved_correlation"),
                    name=f"{label} achieved correlation",
                )
            elif calibration.get("achieved_correlation") is not None:
                raise RuntimeError("unidentifiable calibration reports correlation")
            if variance_identifiable:
                achieved_variance_multiplier = _require_finite_number(
                    calibration.get("achieved_variance_inflation"),
                    name=f"{label} achieved variance inflation",
                )
                target_variance_multiplier = _require_finite_number(
                    calibration.get("target_variance_scale_multiplier"),
                    name=f"{label} target variance multiplier",
                )
                if abs(
                    achieved_variance_multiplier
                    - target_variance_multiplier
                ) > tolerance:
                    raise RuntimeError(
                        "variance-envelope multiplier exceeds tolerance"
                    )
            elif calibration.get("achieved_variance_inflation") is not None:
                raise RuntimeError("unidentifiable calibration reports variance")
            elif calibration.get("target_variance_scale_multiplier") is not None:
                raise RuntimeError(
                    "unidentifiable calibration reports target variance"
                )
            floor_applied = calibration.get("variance_floor_applied")
            ceiling_applied = calibration.get("variance_ceiling_applied")
            if type(floor_applied) is not bool or type(ceiling_applied) is not bool:
                raise RuntimeError("variance clamp flags must be boolean")
            if floor_applied and ceiling_applied:
                raise RuntimeError("variance floor and ceiling cannot both apply")
            if variance_identifiable and (
                floor_applied
                is not (
                    target_variance_multiplier
                    > requested_variance_multiplier + tolerance
                )
                or ceiling_applied
                is not (
                    target_variance_multiplier
                    < requested_variance_multiplier - tolerance
                )
            ):
                raise RuntimeError("variance clamp flags contradict target")
            minimum_sd = _require_nonnegative_number(
                calibration.get("minimum_feasible_sd"),
                name=f"{label} minimum feasible SD",
            )
            maximum_sd = _require_nonnegative_number(
                calibration.get("maximum_margin_sd"),
                name=f"{label} maximum margin SD",
            )
            if minimum_sd > maximum_sd + tolerance:
                raise RuntimeError("variance envelope bounds are inverted")
            if abs(
                _require_finite_number(
                    calibration.get("achieved_feasible_covariance_fraction"),
                    name=f"{label} achieved covariance fraction",
                )
                - requested_covariance_fraction
            ) > tolerance:
                raise RuntimeError("achieved covariance fraction exceeds tolerance")
            covariance_lower = _require_finite_number(
                calibration.get("feasible_covariance_lower"),
                name=f"{label} feasible covariance lower",
            )
            covariance_upper = _require_finite_number(
                calibration.get("feasible_covariance_upper"),
                name=f"{label} feasible covariance upper",
            )
            achieved_covariance = _require_finite_number(
                calibration.get("achieved_covariance"),
                name=f"{label} achieved covariance",
            )
            covariance_scale = max(
                1.0,
                abs(covariance_lower),
                abs(covariance_upper),
                abs(achieved_covariance),
            )
            if not (
                covariance_lower - tolerance * covariance_scale
                <= achieved_covariance
                <= covariance_upper + tolerance * covariance_scale
            ):
                raise RuntimeError("achieved covariance left feasible interval")
            scope, metric, _sign = label.rsplit("/", 2)
            expected_effect = (
                position_target_effect
                if metric == "argmax_position_error"
                else float(_require_mapping(target_effects[metric], name=metric)[scope])
            )
            _require_close(
                calibration.get("requested_effect"),
                expected_effect,
                name=f"{label} requested effect",
            )
            endpoint_fraction = _require_nonnegative_number(
                calibration.get("endpoint_variance_fraction"),
                name=f"{label} endpoint variance fraction",
            )
            all_coordinates_valid &= (
                endpoint_fraction <= POWER_GRID_ENDPOINT_VARIANCE_LIMIT + 1e-12
            )
            fractions.append(endpoint_fraction)
        maximum_fraction = max(fractions, default=None)
        if maximum_fraction is None:
            if feasibility.get("maximum_endpoint_variance_fraction") is not None:
                raise RuntimeError("empty feasibility audit reports a maximum")
        else:
            _require_close(
                feasibility.get("maximum_endpoint_variance_fraction"),
                maximum_fraction,
                name="maximum endpoint variance fraction",
            )
        _require_close(
            feasibility.get("maximum_endpoint_variance_fraction_limit"),
            POWER_GRID_ENDPOINT_VARIANCE_LIMIT,
            name="feasibility endpoint variance limit",
        )
        feasibility_passed = (
            calibrations_complete
            and not failures
            and all_coordinates_valid
            and maximum_fraction is not None
            and maximum_fraction <= POWER_GRID_ENDPOINT_VARIANCE_LIMIT + 1e-12
        )
        if _require_bool(
            feasibility.get("passed"), name="bounded feasibility passed"
        ) is not feasibility_passed:
            raise RuntimeError("bounded feasibility decision mismatch")
        recomputed_feasibility_passes.append(feasibility_passed)
        if not feasibility_passed:
            continue

        outer = _require_mapping(
            result.get("outer_reference_uncertainty"),
            name="outer reference uncertainty",
        )
        if outer.get("method") != (
            "nonparametric_outer_bootstrap_of_development_seeds_"
            "with_per_trial_moment_recalibration"
        ):
            raise RuntimeError("outer reference uncertainty method mismatch")
        if (
            outer.get("source_seed_count") != len(complete_seed_ids)
            or outer.get("outer_draw_size") != len(complete_seed_ids)
            or outer.get("trials") != trials
        ):
            raise RuntimeError("outer reference uncertainty population mismatch")
        outer_failures = outer.get("failures")
        if not isinstance(outer_failures, list) or any(
            not isinstance(item, str) for item in outer_failures
        ):
            raise RuntimeError("outer reference failures are invalid")
        moment_audit = _require_mapping(
            outer.get("moment_audit"), name="outer moment audit"
        )
        successful_trials = _require_nonnegative_int(
            moment_audit.get("successful_trials"),
            name="outer successful trials",
        )
        outer_passed = not outer_failures and successful_trials == trials
        if _require_nonnegative_number(
            moment_audit.get("maximum_endpoint_variance_fraction"),
            name="outer maximum endpoint fraction",
        ) > POWER_GRID_ENDPOINT_VARIANCE_LIMIT + 1e-12:
            raise RuntimeError("outer endpoint variance fraction exceeds limit")
        for key in (
            "maximum_feasible_covariance_fraction_error",
            "maximum_variance_scale_multiplier_error",
        ):
            if _require_nonnegative_number(
                moment_audit.get(key), name=f"outer {key}"
            ) > 1e-6:
                raise RuntimeError("outer moment calibration exceeds tolerance")
        if _require_bool(outer.get("passed"), name="outer uncertainty passed") is not (
            outer_passed
        ):
            raise RuntimeError("outer uncertainty decision mismatch")
        recomputed_outer_passes.append(outer_passed)
        signs = _require_mapping(
            result.get("initialization_sign_counts"),
            name="initialization sign counts",
        )
        if signs != {"0": trials}:
            raise RuntimeError("fixed-model-seed initialization audit mismatch")

        target = _require_mapping(
            result.get("target_composite"), name="target composite"
        )
        passes = _require_nonnegative_int(
            target.get("passes"), name="target composite passes"
        )
        if passes > trials or target.get("trials") != trials:
            raise RuntimeError("target composite count mismatch")
        _require_close(
            target.get("point_power"),
            passes / trials,
            name="target point power",
        )
        if target.get("family_size") != len(POWER_GRID_SCENARIOS):
            raise RuntimeError("power family size mismatch")
        power_lower = exact_binomial_lower_bound(
            passes,
            trials,
            alpha=POWER_FAMILYWISE_ALPHA / len(POWER_GRID_SCENARIOS),
        )
        _require_close(
            target.get("familywise_exact_one_sided_lower"),
            power_lower,
            name="power familywise lower bound",
        )
        _require_close(
            target.get("minimum"),
            POWER_TARGET_MINIMUM,
            name="power minimum",
        )
        power_passed = outer_passed and power_lower >= POWER_TARGET_MINIMUM
        if _require_bool(target.get("passed"), name="target power passed") is not (
            power_passed
        ):
            raise RuntimeError("target power decision mismatch")
        recomputed_power_passes.append(power_passed)

        coverage = _require_mapping(
            result.get("one_sided_ci_coverage"), name="CI coverage"
        )
        by_contrast = _require_mapping(
            coverage.get("by_contrast"), name="CI coverage contrasts"
        )
        if set(by_contrast) != POWER_COVERAGE_CONTRAST_NAMES:
            raise RuntimeError("coverage contrast population mismatch")
        coverage_family_size = len(POWER_GRID_SCENARIOS) * len(
            POWER_COVERAGE_CONTRAST_NAMES
        )
        if coverage.get("family_size") != coverage_family_size:
            raise RuntimeError("coverage family size mismatch")
        _require_close(
            coverage.get("point_minimum"),
            POWER_COVERAGE_POINT_MINIMUM,
            name="coverage point minimum",
        )
        _require_close(
            coverage.get("familywise_lower_minimum"),
            POWER_COVERAGE_FAMILYWISE_LOWER_MINIMUM,
            name="coverage familywise lower minimum",
        )
        coverage_passes: list[bool] = []
        for name, raw in by_contrast.items():
            item = _require_mapping(raw, name=f"coverage {name}")
            count = _require_nonnegative_int(
                item.get("covered"), name=f"{name} covered count"
            )
            if count > trials or item.get("trials") != trials:
                raise RuntimeError("coverage count mismatch")
            point = count / trials
            _require_close(item.get("point"), point, name=f"{name} coverage point")
            lower = exact_binomial_lower_bound(
                count,
                trials,
                alpha=POWER_FAMILYWISE_ALPHA / coverage_family_size,
            )
            _require_close(
                item.get("familywise_exact_one_sided_lower"),
                lower,
                name=f"{name} coverage lower bound",
            )
            _require_close(
                item.get("point_minimum"),
                POWER_COVERAGE_POINT_MINIMUM,
                name=f"{name} point minimum",
            )
            _require_close(
                item.get("familywise_lower_minimum"),
                POWER_COVERAGE_FAMILYWISE_LOWER_MINIMUM,
                name=f"{name} familywise lower minimum",
            )
            item_passed = (
                point >= POWER_COVERAGE_POINT_MINIMUM
                and lower >= POWER_COVERAGE_FAMILYWISE_LOWER_MINIMUM
            )
            if _require_bool(item.get("passed"), name=f"{name} passed") is not (
                item_passed
            ):
                raise RuntimeError("coverage contrast decision mismatch")
            coverage_passes.append(item_passed)
        coverage_passed = outer_passed and all(coverage_passes)
        if _require_bool(
            coverage.get("passed"), name="coverage passed"
        ) is not coverage_passed:
            raise RuntimeError("coverage decision mismatch")
        recomputed_coverage_passes.append(coverage_passed)

        distribution = _require_mapping(
            result.get("physical_metric_distribution"),
            name="physical metric distribution",
        )
        if distribution.get("bounds") != POWER_METRIC_PHYSICAL_BOUNDS:
            raise RuntimeError("physical distribution bounds mismatch")
        observed = _require_mapping(
            distribution.get("observed"), name="physical distribution observed"
        )
        clipping = _require_mapping(
            distribution.get("clipping_counts"), name="metric clipping counts"
        )
        if set(observed) != set(POWER_METRIC_PHYSICAL_BOUNDS) or clipping != {
            metric: 0 for metric in sorted(POWER_METRIC_PHYSICAL_BOUNDS)
        }:
            raise RuntimeError("physical distribution evidence is incomplete")
        distribution_passed = True
        for metric, bounds in POWER_METRIC_PHYSICAL_BOUNDS.items():
            item = _require_mapping(observed[metric], name=f"observed {metric}")
            count = _require_nonnegative_int(
                item.get("count"), name=f"{metric} observed count"
            )
            minimum = _require_finite_number(
                item.get("minimum"), name=f"{metric} observed minimum"
            )
            maximum = _require_finite_number(
                item.get("maximum"), name=f"{metric} observed maximum"
            )
            distribution_passed &= (
                count > 0
                and bounds[0] - 1e-11 <= minimum <= maximum
                and maximum <= bounds[1] + 1e-11
            )
        if _require_bool(
            distribution.get("passed"), name="physical distribution passed"
        ) is not distribution_passed:
            raise RuntimeError("physical distribution decision mismatch")
        recomputed_distribution_passes.append(distribution_passed)

    recomputed_type1_passes: list[bool] = []
    null_results = simulation.get("null_sensitivity_grid")
    all_feasible = all(recomputed_feasibility_passes)
    if not isinstance(null_results, list) or (
        all_feasible and len(null_results) != len(POWER_NULL_GRID_SCENARIOS)
    ) or (not all_feasible and null_results):
        raise RuntimeError("power null-grid result population mismatch")
    type1_family_size = len(POWER_NULL_GRID_SCENARIOS) * len(
        POWER_TYPE1_CONTRAST_NAMES
    )
    for scenario_index, (expected_scenario, raw_result) in enumerate(
        zip(POWER_NULL_GRID_SCENARIOS, null_results), start=1
    ):
        result = _require_mapping(raw_result, name="power null-grid result")
        if result.get("scenario_id") != scenario_index or any(
            result.get(key) != value for key, value in expected_scenario.items()
        ):
            raise RuntimeError("power null-grid result mismatch")
        if result.get("method") != (
            "outer_reference_bootstrap_then_bounded_physical_"
            "component_boundary_recalibration"
        ) or result.get("simulation_seed") != POWER_LOCKED_TYPE1_SIMULATION_SEED:
            raise RuntimeError("physical component-null method mismatch")
        if (
            result.get("outer_source_seed_count") != len(complete_seed_ids)
            or result.get("outer_draw_size") != len(complete_seed_ids)
            or result.get("family_size") != type1_family_size
        ):
            raise RuntimeError("Type-I population or family size mismatch")
        successful_trials = _require_nonnegative_int(
            result.get("successful_trials"), name="Type-I successful trials"
        )
        moment_audit = _require_mapping(
            result.get("outer_moment_audit"), name="Type-I outer moment audit"
        )
        if _require_nonnegative_number(
            moment_audit.get("maximum_endpoint_variance_fraction"),
            name="Type-I maximum endpoint fraction",
        ) > POWER_GRID_ENDPOINT_VARIANCE_LIMIT + 1e-12:
            raise RuntimeError("Type-I endpoint variance fraction exceeds limit")
        for key in (
            "maximum_feasible_covariance_fraction_error",
            "maximum_variance_scale_multiplier_error",
        ):
            if _require_nonnegative_number(
                moment_audit.get(key), name=f"Type-I {key}"
            ) > 1e-6:
                raise RuntimeError("Type-I moment calibration exceeds tolerance")
        failures = result.get("failures")
        if not isinstance(failures, list) or any(
            not isinstance(item, str) for item in failures
        ):
            raise RuntimeError("Type-I generation failures are invalid")
        boundary_tests = _require_mapping(
            result.get("boundary_tests"), name="boundary tests"
        )
        if set(boundary_tests) != POWER_TYPE1_CONTRAST_NAMES:
            raise RuntimeError("Type-I boundary-test population mismatch")
        component_passes: list[bool] = []
        for name, raw in boundary_tests.items():
            item = _require_mapping(raw, name=f"Type-I {name}")
            count = _require_nonnegative_int(
                item.get("false_rejections"), name=f"{name} false rejections"
            )
            if count > trials or item.get("trials") != trials:
                raise RuntimeError("Type-I count mismatch")
            _require_close(
                item.get("point_type1_error"),
                count / trials,
                name=f"{name} point Type-I",
            )
            upper = exact_binomial_upper_bound(
                count,
                trials,
                alpha=POWER_FAMILYWISE_ALPHA / type1_family_size,
            )
            _require_close(
                item.get("familywise_exact_one_sided_upper"),
                upper,
                name=f"{name} Type-I upper bound",
            )
            component_passed = upper <= POWER_TYPE1_MAXIMUM
            if _require_bool(item.get("passed"), name=f"{name} passed") is not (
                component_passed
            ):
                raise RuntimeError("Type-I component decision mismatch")
            component_passes.append(component_passed)

        distribution = _require_mapping(
            result.get("physical_metric_distribution"),
            name="null physical metric distribution",
        )
        if distribution.get("bounds") != POWER_METRIC_PHYSICAL_BOUNDS:
            raise RuntimeError("null physical distribution bounds mismatch")
        observed = _require_mapping(
            distribution.get("observed"), name="null physical distribution observed"
        )
        clipping = _require_mapping(
            distribution.get("clipping_counts"),
            name="null metric clipping counts",
        )
        if set(observed) != set(POWER_METRIC_PHYSICAL_BOUNDS) or clipping != {
            metric: 0 for metric in sorted(POWER_METRIC_PHYSICAL_BOUNDS)
        }:
            raise RuntimeError("null physical distribution evidence is incomplete")
        distribution_passed = True
        for metric, bounds in POWER_METRIC_PHYSICAL_BOUNDS.items():
            item = _require_mapping(
                observed[metric], name=f"null observed {metric}"
            )
            count = _require_nonnegative_int(
                item.get("count"), name=f"null {metric} observed count"
            )
            minimum = _require_finite_number(
                item.get("minimum"), name=f"null {metric} observed minimum"
            )
            maximum = _require_finite_number(
                item.get("maximum"), name=f"null {metric} observed maximum"
            )
            distribution_passed &= (
                count > 0
                and bounds[0] - 1e-11 <= minimum <= maximum
                and maximum <= bounds[1] + 1e-11
            )
        if _require_bool(
            distribution.get("passed"), name="null physical distribution passed"
        ) is not distribution_passed:
            raise RuntimeError("null physical distribution decision mismatch")
        type1_passed = (
            not failures
            and successful_trials == trials
            and distribution_passed
            and all(component_passes)
        )
        if _require_bool(result.get("passed"), name="component null passed") is not (
            type1_passed
        ):
            raise RuntimeError("Type-I decision mismatch")
        recomputed_type1_passes.append(type1_passed)
        recomputed_distribution_passes.append(distribution_passed)
        recomputed_outer_passes.append(not failures and successful_trials == trials)

    gates = _require_mapping(simulation.get("gates"), name="power gates")
    if set(gates) != POWER_SIMULATION_GATE_NAMES or any(
        type(value) is not bool for value in gates.values()
    ):
        raise RuntimeError("power gates must be the exact boolean population")
    early_design_failure = not all(recomputed_feasibility_passes)
    expected_gates = (
        {name: False for name in POWER_SIMULATION_GATE_NAMES}
        if early_design_failure
        else {
            "bounded_advantage_moment_scenarios_feasible": True,
            "target_composite_power_familywise_lower_at_least_0.80": all(
                recomputed_power_passes
            ),
            "binding_contrast_type1_familywise_upper_at_most_0.05": all(
                recomputed_type1_passes
            ),
            "one_sided_ci_coverage_familywise_lower_at_least_0.90": all(
                recomputed_coverage_passes
            ),
            "simulated_metrics_within_physical_domain_without_clipping": all(
                recomputed_distribution_passes
            ),
            "outer_reference_uncertainty_propagated": all(
                recomputed_outer_passes
            ),
        }
    )
    if dict(gates) != expected_gates:
        raise RuntimeError("power gates do not match recomputed evidence")
    simulation_passed = _require_bool(
        simulation.get("passed"), name="power simulation passed"
    )
    if simulation_passed is not all(expected_gates.values()):
        raise RuntimeError("power decision does not equal strict gate AND")
    if simulation.get("decision") != (
        "power_go" if simulation_passed else "power_no_go"
    ):
        raise RuntimeError("power decision label mismatch")
    if simulation_passed and (
        trials != POWER_LOCKED_SIMULATION_TRIALS
        or bootstrap_samples != POWER_LOCKED_BOOTSTRAP_SAMPLES_PER_TRIAL
    ):
        raise RuntimeError("passing power evidence did not use locked MC counts")
    for key in (
        "recommended_validation_seed_count",
        "recommended_holdout_seed_count",
    ):
        expected = candidate_count if simulation_passed else None
        if simulation.get(key) != expected:
            raise RuntimeError(f"power artifact has invalid {key}")
    if power_design.get(
        "composite_bootstrap_simulation_required_before_secret_commitment"
    ) is not False:
        raise RuntimeError("power artifact still requires composite simulation")

    reference_decision = _require_mapping(
        payload.get("reference_health_decision"),
        name="reference-health decision",
    )
    required_reference = _require_mapping(
        reference_decision.get("required"), name="reference-health gates"
    )
    if not required_reference or any(
        not isinstance(detail, Mapping)
        or set(detail) != {"one_sided_99_lower_positive"}
        or any(type(value) is not bool for value in detail.values())
        for detail in required_reference.values()
    ):
        raise RuntimeError("reference-health gate evidence is invalid")
    _verify_reference_health_recomputation(score_table, required_reference)
    reference_passed = all(
        all(detail.values()) for detail in required_reference.values()
    )
    if _require_bool(
        reference_decision.get("passed"), name="reference health passed"
    ) is not reference_passed:
        raise RuntimeError("reference-health decision mismatch")
    phase0_gates = _require_mapping(
        payload.get("phase0_gates"), name="Phase 0 gates"
    )
    if set(phase0_gates) != PHASE0_GATE_NAMES or any(
        type(value) is not bool for value in phase0_gates.values()
    ):
        raise RuntimeError("Phase 0 gates must be the exact boolean population")
    expected_phase0_gates = {
        "candidate_independent_reference_health": reference_passed,
        "candidate_independent_composite_power": simulation_passed,
    }
    if dict(phase0_gates) != expected_phase0_gates:
        raise RuntimeError("Phase 0 gates do not match evidence")
    phase0_passed = _require_bool(payload.get("passed"), name="Phase 0 passed")
    if phase0_passed is not all(expected_phase0_gates.values()):
        raise RuntimeError("Phase 0 decision does not equal strict gate AND")
    if payload.get("decision") != (
        "phase0_go" if phase0_passed else "phase0_no_go"
    ):
        raise RuntimeError("Phase 0 decision label mismatch")

    provenance = _require_mapping(payload.get("provenance"), name="provenance")
    if (
        provenance.get("development_only") is not True
        or provenance.get("candidate_source_imported") is not False
        or provenance.get("candidate_maps_read") is not False
    ):
        raise RuntimeError("Phase 0 provenance isolation mismatch")
    registry_path = provenance.get("registry_path")
    turn_path = provenance.get("turn_probability_selection_artifact")
    if not isinstance(registry_path, str) or not isinstance(turn_path, str):
        raise RuntimeError("Phase 0 provenance paths are missing")
    for key in (
        "registry_sha256",
        "turn_probability_selection_sha256",
        "registry_selection_digest_sha256",
    ):
        value = (
            payload.get(key)
            if key == "registry_selection_digest_sha256"
            else provenance.get(key)
        )
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"Phase 0 provenance {key} is invalid")
    required_sources = {
        "cal/evaluation/stochastic_permanence_phase0.py",
        "cal/evaluation/stochastic_permanence_benchmark.py",
        "cal/evaluation/stochastic_permanence_artifacts.py",
        "cal/evaluation/stochastic_permanence_custody.py",
        "cal/evaluation/permanence_forward_benchmark.py",
        "cal/evaluation/permanence_seed_registry.py",
        "cal/evaluation/permanence_turn_probability_scan.py",
        "cal/evaluation/randomized_occlusion_world.py",
        "cal/evaluation/v2_i1_integration.py",
        "cal/evaluation/v2_artifacts.py",
        "cal/infra/provenance.py",
        "cal/model/integrated_agent.py",
        "cal/model/entity_graph.py",
        "cal/model/entity_belief_graph.py",
        "cal/model/occupancy.py",
        "docs/experiments/V2_I1_STOCHASTIC_PERMANENCE_PLAN.md",
        "pyproject.toml",
        "uv.lock",
        registry_path,
        turn_path,
    }
    _validate_required_source_lock_structure(
        provenance.get("source_lock"), required_files=required_sources
    )
    runtime = _require_mapping(provenance.get("runtime"), name="runtime")
    if not all(
        isinstance(runtime.get(name), str) and runtime[name]
        for name in ("python", "platform", "numpy")
    ):
        raise RuntimeError("Phase 0 runtime provenance is incomplete")
    coverage_provenance = _require_mapping(
        provenance.get("coverage"), name="coverage provenance"
    )
    if (
        coverage_provenance.get("development_seed_count")
        != len(complete_seed_ids)
        or coverage_provenance.get("complete_seed_ids") != complete_seed_ids
        or coverage_provenance.get(
            "all_predictors_same_complete_seed_population"
        )
        is not True
        or coverage_provenance.get("required_occlusion_bins")
        != ["2-3", "4-5", "6+"]
    ):
        raise RuntimeError("Phase 0 coverage provenance mismatch")


def validate_artifact(payload: Mapping[str, Any], *, expected_kind: str) -> None:
    """Validate the stable outer contract for a Phase 0/R artifact."""

    required = _REQUIRED_BY_KIND.get(expected_kind)
    if required is None:
        raise ValueError(f"unknown stochastic permanence artifact: {expected_kind}")
    if payload.get("artifact_kind") != expected_kind:
        raise RuntimeError("artifact kind mismatch")
    if payload.get("artifact_schema_version") != _expected_schema_version(
        expected_kind
    ):
        raise RuntimeError("artifact schema version mismatch")
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"artifact is missing required fields: {missing}")
    if expected_kind == "stochastic_permanence_reference_health_power":
        _validate_reference_health_power_artifact(payload)
    elif expected_kind == "stochastic_permanence_capacity_conformance":
        _validate_capacity_artifact(payload)
    elif expected_kind == "stochastic_permanence_formal_decision":
        complete_seed_ids = payload.get("complete_seed_ids")
        if (
            not isinstance(complete_seed_ids, list)
            or len(complete_seed_ids) < 2
            or any(type(seed) is not int or seed < 0 for seed in complete_seed_ids)
            or complete_seed_ids != sorted(set(complete_seed_ids))
        ):
            raise RuntimeError("formal decision seed population is invalid")
        bootstrap = _require_mapping(
            payload.get("bootstrap"), name="formal bootstrap"
        )
        if (
            bootstrap.get("algorithm") != "numpy.PCG64"
            or bootstrap.get("seed") != POWER_LOCKED_BOOTSTRAP_SEED
            or bootstrap.get("samples") != POWER_LOCKED_FORMAL_BOOTSTRAP_SAMPLES
            or bootstrap.get("quantile_method") != "linear"
        ):
            raise RuntimeError("formal bootstrap contract mismatch")
        _require_close(
            bootstrap.get("quantile"),
            POWER_CONFIRMATORY_ONE_SIDED_ALPHA,
            name="formal bootstrap quantile",
        )
        common_digest = bootstrap.get("common_indices_sha256")
        if (
            not isinstance(common_digest, str)
            or len(common_digest) != 64
            or any(character not in "0123456789abcdef" for character in common_digest)
        ):
            raise RuntimeError("formal common-index digest is invalid")
        contrasts = _require_mapping(
            payload.get("contrasts"), name="formal contrasts"
        )
        if not contrasts:
            raise RuntimeError("formal decision has no contrasts")
        for name, raw in contrasts.items():
            summary = _require_mapping(raw, name=f"formal contrast {name}")
            if set(summary) != {
                "per_seed",
                "mean",
                "one_sided_99_lower",
            }:
                raise RuntimeError("formal contrast schema mismatch")
            per_seed = summary.get("per_seed")
            if (
                not isinstance(per_seed, list)
                or len(per_seed) != len(complete_seed_ids)
            ):
                raise RuntimeError("formal contrast seed population mismatch")
            values = [
                _require_finite_number(value, name=f"formal contrast {name}")
                for value in per_seed
            ]
            _require_close(
                summary.get("mean"),
                sum(values) / len(values),
                name=f"formal contrast {name} mean",
            )
            _require_finite_number(
                summary.get("one_sided_99_lower"),
                name=f"formal contrast {name} lower",
            )
        gates = payload.get("gates")
        if not isinstance(gates, Mapping) or not gates:
            raise RuntimeError("formal decision has no gates")
        if any(type(value) is not bool for value in gates.values()):
            raise RuntimeError("formal decision gates must be booleans")
        passed = _require_bool(
            payload.get("passed"), name="formal decision passed"
        )
        if passed is not all(gates.values()):
            raise RuntimeError("formal decision does not equal strict gate AND")
        if payload.get("decision") != (
            "pass_all_confirmatory_gates" if passed else "stop"
        ):
            raise RuntimeError("formal decision label mismatch")


def write_canonical_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    expected_kind: str,
    overwrite: bool = False,
) -> str:
    """Atomically write canonical JSON and its sibling SHA-256 sidecar."""

    validate_artifact(payload, expected_kind=expected_kind)
    destination = Path(path)
    sidecar = destination.with_suffix(".sha256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(payload)
    digest = sha256_bytes(raw)
    lock_path = destination.with_name(f".{destination.name}.write.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"artifact write already in progress: {destination}"
        ) from error
    os.close(lock_fd)
    temporary: Path | None = None
    temporary_sidecar: Path | None = None
    try:
        if not overwrite and (destination.exists() or sidecar.exists()):
            raise FileExistsError(f"artifact already exists: {destination}")
        artifact_fd, artifact_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(artifact_name)
        with os.fdopen(artifact_fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        sidecar_fd, sidecar_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
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


def load_canonical_artifact(
    path: str | Path,
    *,
    expected_kind: str,
) -> tuple[dict[str, Any], str]:
    """Load an artifact only if bytes, sidecar, and schema all agree."""

    source = Path(path)
    raw = source.read_bytes()
    digest = sha256_bytes(raw)
    sidecar = source.with_suffix(".sha256")
    expected_sidecar = f"{digest}  {source.name}\n"
    try:
        sidecar_text = sidecar.read_text(encoding="ascii")
    except (UnicodeDecodeError, OSError) as error:
        raise RuntimeError("artifact sidecar is missing or invalid") from error
    if sidecar_text != expected_sidecar:
        raise RuntimeError("artifact digest sidecar mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("artifact root must be a JSON object")
    if raw != canonical_json_bytes(payload):
        raise RuntimeError("artifact is not canonical JSON")
    validate_artifact(payload, expected_kind=expected_kind)
    return payload, digest


def _verify_reference_health_recomputation(
    score_table: Mapping[str, Any],
    required_reference: Mapping[str, Any],
) -> None:
    """Re-derive the reference-health gate booleans from the raw score table.

    Structural agreement between ``passed`` and the recorded booleans is not
    enough: without this the recorded booleans are unfalsifiable self-reports,
    so an artifact whose per-seed evidence contradicts its own gates would
    validate clean.  The recomputation is the same one the builder performs,
    driven only by ``per_seed_per_bin`` carried inside the artifact.
    """

    # Imported lazily: the benchmark module imports this one at module scope.
    from cal.evaluation.stochastic_permanence_benchmark import (
        REQUIRED_REFERENCE_GATES,
        build_reference_health,
    )

    health, _ = build_reference_health(score_table)
    expected_keys = {f"{scope}/{metric}" for scope, metric in REQUIRED_REFERENCE_GATES}
    if set(required_reference) != expected_keys:
        raise RuntimeError("reference-health gate population mismatch")
    for scope, metric in REQUIRED_REFERENCE_GATES:
        entry = health[scope][metric]
        recorded = required_reference[f"{scope}/{metric}"]
        recomputed = {
            "one_sided_99_lower_positive": bool(entry["one_sided_99_lower"] > 0.0),
        }
        if dict(recorded) != recomputed:
            raise RuntimeError(
                "reference-health gate does not match recomputed evidence: "
                f"{scope}/{metric}"
            )


def source_lock(paths: Iterable[str | Path], *, root: str | Path) -> dict[str, Any]:
    """Hash an explicit transitive source set using portable relative names."""

    base = Path(root).resolve()
    files: dict[str, str] = {}
    for value in sorted({Path(item).resolve() for item in paths}):
        if not value.is_file():
            raise FileNotFoundError(value)
        try:
            relative = value.relative_to(base)
        except ValueError as error:
            raise RuntimeError(f"source-lock path is outside root: {value}") from error
        files[relative.as_posix()] = sha256_path(value)
    combined = hashlib.sha256()
    for name, digest in files.items():
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {
        "algorithm": "sha256_sorted_relative_path_and_digest_v1",
        "file_count": len(files),
        "files": files,
        "combined_sha256": combined.hexdigest(),
    }


def audit_artifact_source_lock(
    path: str | Path, *, root: str | Path
) -> dict[str, Any]:
    """Report whether an artifact's recorded source lock still matches live files.

    ``verify_source_lock`` raises on any difference, which makes it unusable for
    surveying committed artifacts: a development artifact's lock is a claim
    about the past, so drift after a later source fix is expected rather than
    wrong.  This returns the per-file verdict instead, so callers can require
    that drift be *acknowledged* rather than merely absent.
    """

    payload = json.loads(Path(path).read_bytes())
    provenance = payload.get("provenance")
    lock = provenance.get("source_lock") if isinstance(provenance, Mapping) else None
    if not isinstance(lock, Mapping) or not isinstance(lock.get("files"), Mapping):
        raise RuntimeError(f"artifact has no source lock: {path}")
    base = Path(root).resolve()
    drifted: list[str] = []
    missing: list[str] = []
    for name, digest in sorted(lock["files"].items()):
        target = base / name
        if not target.is_file():
            missing.append(name)
        elif sha256_path(target) != digest:
            drifted.append(name)
    return {
        "artifact": Path(path).name,
        "file_count": int(lock["file_count"]),
        "matched": not drifted and not missing,
        "drifted": drifted,
        "missing": missing,
    }


def verify_source_lock(lock: Mapping[str, Any], *, root: str | Path) -> None:
    base = Path(root).resolve()
    files = lock.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("source lock has no files")
    rebuilt = source_lock((base / name for name in files), root=base)
    if dict(lock) != rebuilt:
        raise RuntimeError("source lock mismatch")
