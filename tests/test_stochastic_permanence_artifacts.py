from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from cal.evaluation.stochastic_permanence_artifacts import (
    CAPACITY_ARTIFACT_SCHEMA_VERSION,
    audit_artifact_source_lock,
    exact_binomial_lower_bound,
    exact_binomial_upper_bound,
    load_canonical_artifact,
    source_lock,
    validate_artifact,
    verify_source_lock,
    write_canonical_artifact,
)
from cal.evaluation.stochastic_permanence_capacity_artifacts import (
    load_capacity_artifact,
    validate_capacity_artifact,
)


def test_exact_binomial_bounds_support_simultaneous_tail_probabilities() -> None:
    alpha = 0.05 / 80
    upper = exact_binomial_upper_bound(0, 256, alpha=alpha)
    assert upper == pytest.approx(1.0 - alpha ** (1.0 / 256.0))
    assert upper < 0.05

    lower = exact_binomial_lower_bound(244, 256, alpha=0.05 / 216)
    assert 0.88 < lower < 0.90


def test_phase0_schema_recomputes_moments_distribution_and_recommendation() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/V2_I1_P1_PHASE0_REFERENCE_HEALTH_POWER_DEVELOPMENT_V11.json"
    )
    payload, _digest = load_canonical_artifact(
        path,
        expected_kind="stochastic_permanence_reference_health_power",
    )
    simulation = payload["power_design"]["composite_simulation"]

    tampered = deepcopy(payload)
    calibration = next(
        iter(
            tampered["power_design"]["composite_simulation"][
                "sensitivity_grid"
            ][0]["bounded_advantage_moment_feasibility"]["calibrations"].values()
        )
    )
    calibration["achieved_feasible_covariance_fraction"] += 0.01
    with pytest.raises(RuntimeError, match="covariance fraction"):
        validate_artifact(
            tampered,
            expected_kind="stochastic_permanence_reference_health_power",
        )

    tampered = deepcopy(payload)
    observed = tampered["power_design"]["composite_simulation"][
        "sensitivity_grid"
    ][0]["physical_metric_distribution"]["observed"]
    observed["top1_accuracy"]["minimum"] = -1.0
    with pytest.raises(RuntimeError, match="physical distribution decision"):
        validate_artifact(
            tampered,
            expected_kind="stochastic_permanence_reference_health_power",
        )

    tampered = deepcopy(payload)
    tampered["power_design"]["composite_simulation"][
        "recommended_validation_seed_count"
    ] = 2
    with pytest.raises(RuntimeError, match="recommended_validation_seed_count"):
        validate_artifact(
            tampered,
            expected_kind="stochastic_permanence_reference_health_power",
        )

    assert simulation["recommended_validation_seed_count"] == 11078
    assert simulation["recommended_holdout_seed_count"] == 11078


def test_phase_r_v3_schema_locks_factor_local_capacity_evidence() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/V2_I1_P1_PHASE_R_CAPACITY_CONFORMANCE_DEVELOPMENT_V5.json"
    )
    payload, _digest = load_capacity_artifact(path)

    assert payload["decision"] == "phase_r_go"
    assert payload["capacity_contract"]["K_max"] == 96
    assert payload["capacity_contract"]["atomic_staging_scope"] == "one_factor"
    assert payload["conformance"]["episode_count"] == 1615

    tampered = deepcopy(payload)
    tampered["capacity_contract"]["persistent_arrays"]["scratch_codes"][
        "shape"
    ] = [payload["capacity_contract"]["S_max"]]
    with pytest.raises(RuntimeError, match="scratch_codes layout"):
        validate_capacity_artifact(tampered)

    tampered = deepcopy(payload)
    tampered["conformance"]["maximum_position_tv"] = 0.0
    with pytest.raises(RuntimeError, match="maxima mismatch"):
        validate_capacity_artifact(tampered)


def _capacity_payload() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    registry_path = (
        "experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY_V2.json"
    )
    lock = source_lock(
        (
            root / "cal/model/stochastic_motion_filter.py",
            root / "cal/evaluation/stochastic_permanence_kernel_diagnostic.py",
            root / "cal/evaluation/stochastic_permanence_artifacts.py",
            root / "cal/evaluation/permanence_forward_benchmark.py",
            root / "cal/evaluation/randomized_occlusion_world.py",
            root / "cal/evaluation/permanence_seed_registry.py",
            root / "cal/evaluation/stochastic_permanence_custody.py",
            root / "cal/evaluation/v2_i1_integration.py",
            root / "cal/evaluation/v2_artifacts.py",
            root / "cal/infra/provenance.py",
            root / "cal/model/integrated_agent.py",
            root / "cal/model/entity_graph.py",
            root / "cal/model/occupancy.py",
            root / "docs/experiments/V2_I1_STOCHASTIC_PERMANENCE_PLAN.md",
            root / "pyproject.toml",
            root / "uv.lock",
            root / registry_path,
        ),
        root=root,
    )
    array_specs = {
        "codes": ([1], "uint16", 2),
        "probability": ([1], "float32", 4),
        "scratch_codes": ([1], "uint16", 2),
        "scratch_probability": ([1], "float32", 4),
        "expansion_codes": ([12], "uint16", 24),
        "expansion_probability": ([12], "float64", 96),
        "expansion_index": ([4], "int16", 8),
        "counts": ([1, 1], "uint8", 1),
        "existence": ([1, 1], "float32", 4),
        "self_probability": ([1, 2], "float32", 8),
        "hypothesis_weight": ([1], "float32", 4),
        "assignment_workspace": ([1, 1, 1], "float32", 4),
        "static_probability": ([1, 1], "float16", 2),
        "front_static_score": ([1, 1], "int16", 2),
        "last_visibility": ([1, 1], "bool", 1),
        "last_sensed": ([1, 1], "bool", 1),
        "rng_state": ([4], "uint64", 32),
        "kernel_parameters": ([16], "float32", 64),
        "configuration": ([16], "uint32", 64),
    }
    persistent_arrays = {
        name: {"shape": shape, "dtype": dtype, "nbytes": nbytes}
        for name, (shape, dtype, nbytes) in array_specs.items()
    }
    persistent_array_bytes = sum(
        descriptor["nbytes"] for descriptor in persistent_arrays.values()
    )
    return {
        "artifact_kind": "stochastic_permanence_capacity_conformance",
        "artifact_schema_version": CAPACITY_ARTIFACT_SCHEMA_VERSION,
        "status": "development_only_non_gated",
        "privileged_diagnostic_only": True,
        "candidate_artifact_eligible": False,
        "turn_probability": 0.35,
        "capacity_contract": {
            "H_max": 1,
            "E_max": 1,
            "K_max": 1,
            "grid_size": 1,
            "S_max": 1,
            "S_max_minimum": 1,
            "fully_detached_safe": True,
            "copy_on_write": False,
            "arena_low": 0,
            "arena_high": 0,
            "valid_state_capacity": 4,
            "state_code_dtype": "uint16",
            "probability_dtype": "float32",
            "overflow_policy": "atomic_fail_without_partial_commit",
            "update_schedule": (
                "sequential_factor_expansion_into_full_scratch_then_array_swap"
            ),
            "shared_expansion_workspace_size": 12,
            "shared_expansion_workspace_required": 12,
            "direct_index_accumulator": True,
            "direct_index_capacity": 4,
            "python_container_guard_bytes": 8_192,
            "pruning_tie_break": "probability_desc_then_state_code_asc",
            "persistent_arrays": persistent_arrays,
            "cumulative_pruned_mass_definition": (
                "1-product(retained_probability)"
            ),
            "tv_definition": (
                "0.5*L1(position_marginal_packed,position_marginal_reference)"
            ),
            "atomic_overflow_probe_passed": True,
        },
        "conformance": {
            "population": "test",
            "episode_count": 1,
            "maximum_cumulative_pruned_mass": 0.0,
            "maximum_position_tv": 0.0,
            "maximum_branch_accounting_residual": 0.0,
            "known_topology_kernel_alignment": {
                "checked_transition_cases": 1,
                "support_mismatch_count": 0,
                "maximum_probability_l1": 0.0,
            },
            "errors": [],
            "episodes": [{}],
        },
        "resources": {
            "learnable_parameter_count": 16,
            "persistent_array_bytes": persistent_array_bytes,
            "declared_active_state_bytes": persistent_array_bytes + 8_192,
            "measured_full_object_deep_size_bytes": persistent_array_bytes,
            "estimated_mac_per_step": 384,
            "resource_scope": (
                "complete_persistent_candidate_prototype_object_graph"
            ),
            "active_state_limit_bytes": 65_536,
            "parameter_limit": 100_000,
            "mac_per_step_limit": 5_000_000,
            "formal_research_budget_contract": {
                "steps_per_seed_maximum": 100_000,
                "train_replay_maximum": 4,
                "cpu_total_seconds_maximum": 7_200,
            },
            "deterministic_diagnostic_work": {
                "episode_count": 1,
                "transition_checkpoint_count": 1,
            },
        },
        "gates": {
            "episodes_present": True,
            "no_filter_errors": True,
            "cumulative_pruned_mass": True,
            "maximum_position_tv": True,
            "branch_evidence_accounting": True,
            "known_topology_kernel_alignment": True,
            "fully_detached_pool_safe": True,
            "shared_expansion_workspace_safe": True,
            "atomic_overflow_safe": True,
            "declared_active_state": True,
            "measured_active_state": True,
            "learnable_parameters": True,
            "mac_per_step": True,
            "registry_turn_probability": True,
            "formal_research_budget_declared": True,
        },
        "passed": True,
        "decision": "phase_r_go",
        "provenance": {
            "registry_path": registry_path,
            "registry_selection_digest_sha256": "0" * 64,
            "registry_turn_probability": 0.35,
            "source_lock": lock,
            "runtime": {
                "python": "3.test",
                "platform": "test",
                "numpy": "test",
            },
            "command": "test-command",
        },
    }


def test_canonical_artifact_round_trip_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capacity.json"
    digest = write_canonical_artifact(
        path,
        _capacity_payload(),
        expected_kind="stochastic_permanence_capacity_conformance",
    )

    loaded, loaded_digest = load_canonical_artifact(
        path,
        expected_kind="stochastic_permanence_capacity_conformance",
    )

    assert loaded == _capacity_payload()
    assert loaded_digest == digest
    with pytest.raises(FileExistsError):
        write_canonical_artifact(
            path,
            _capacity_payload(),
            expected_kind="stochastic_permanence_capacity_conformance",
        )
    path.write_text(json.dumps(loaded, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest"):
        load_canonical_artifact(
            path,
            expected_kind="stochastic_permanence_capacity_conformance",
        )


def test_capacity_artifact_requires_privileged_marker(tmp_path: Path) -> None:
    payload = _capacity_payload()
    payload["privileged_diagnostic_only"] = False

    with pytest.raises(RuntimeError, match="privileged"):
        write_canonical_artifact(
            tmp_path / "bad.json",
            payload,
            expected_kind="stochastic_permanence_capacity_conformance",
        )


def test_capacity_schema_recomputes_resources_and_strict_and(
    tmp_path: Path,
) -> None:
    payload = _capacity_payload()
    payload["resources"]["declared_active_state_bytes"] = 70_000  # type: ignore[index]

    with pytest.raises(RuntimeError, match="declared active state"):
        write_canonical_artifact(
            tmp_path / "bad-resource.json",
            payload,
            expected_kind="stochastic_permanence_capacity_conformance",
        )


def test_capacity_schema_rejects_dimension_and_layout_mismatch(
    tmp_path: Path,
) -> None:
    payload = _capacity_payload()
    payload["capacity_contract"]["S_max"] = 2  # type: ignore[index]

    with pytest.raises(RuntimeError, match="layout mismatch"):
        write_canonical_artifact(
            tmp_path / "bad-layout.json",
            payload,
            expected_kind="stochastic_permanence_capacity_conformance",
        )


def test_capacity_schema_rejects_wall_clock_runtime(tmp_path: Path) -> None:
    payload = _capacity_payload()
    payload["resources"]["deterministic_diagnostic_work"][  # type: ignore[index]
        "conformance_seconds"
    ] = 1.25

    with pytest.raises(RuntimeError, match="wall-clock"):
        write_canonical_artifact(
            tmp_path / "runtime.json",
            payload,
            expected_kind="stochastic_permanence_capacity_conformance",
        )


def test_sidecar_filename_and_format_are_canonical(tmp_path: Path) -> None:
    path = tmp_path / "capacity.json"
    write_canonical_artifact(
        path,
        _capacity_payload(),
        expected_kind="stochastic_permanence_capacity_conformance",
    )
    sidecar = path.with_suffix(".sha256")
    digest = sidecar.read_text(encoding="ascii").split()[0]
    sidecar.write_text(f"{digest}  wrong.json\n", encoding="ascii")

    with pytest.raises(RuntimeError, match="sidecar"):
        load_canonical_artifact(
            path,
            expected_kind="stochastic_permanence_capacity_conformance",
        )


def test_canonical_artifact_bytes_are_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_digest = write_canonical_artifact(
        first,
        _capacity_payload(),
        expected_kind="stochastic_permanence_capacity_conformance",
    )
    second_digest = write_canonical_artifact(
        second,
        _capacity_payload(),
        expected_kind="stochastic_permanence_capacity_conformance",
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_digest == second_digest


def test_source_lock_is_portable_and_fail_closed(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "sub/b.py"
    second.parent.mkdir()
    first.write_text("a = 1\n", encoding="utf-8")
    second.write_text("b = 2\n", encoding="utf-8")
    lock = source_lock((first, second), root=tmp_path)

    assert list(lock["files"]) == ["a.py", "sub/b.py"]
    verify_source_lock(lock, root=tmp_path)
    second.write_text("b = 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_source_lock(lock, root=tmp_path)


# Source files whose drift away from the current-generation development
# artifacts is acknowledged.  Every entry must name the change that caused it,
# and the entry must disappear once the artifact is regenerated.
#
# 2026-08-08 review finding F4: the Phase 0 validator now recomputes the
# reference-health gate booleans from `per_seed_per_bin` instead of trusting
# the recorded ones, so `stochastic_permanence_artifacts.py` no longer matches
# the code state that produced V10 / Phase-R V3.  Both artifacts still validate
# and their recomputed gates agree with what they recorded.
# Add an entry only together with the change that caused it, and clear it again
# when the artifact is regenerated.
#
# 2026-08-08, review finding D1: `_wasted_field_mass` was added to
# `permanence_forward_benchmark.py` after V11/V5 were generated.  It only adds
# a reported diagnostic field to `run_benchmark`'s result; Phase 0 reads
# `episode_binned_predictors` and `leakage_audit` only, and Phase R does not
# call `run_benchmark` at all, so no gate value in either artifact depends on
# it.  Regenerating a four-hour artifact to absorb a diagnostic is not worth
# it; the drift is recorded instead.
ACKNOWLEDGED_SOURCE_DRIFT = frozenset(
    {"cal/evaluation/permanence_forward_benchmark.py"}
)

CURRENT_DEVELOPMENT_ARTIFACTS = (
    "V2_I1_P1_PHASE0_REFERENCE_HEALTH_POWER_DEVELOPMENT_V11.json",
    "V2_I1_P1_PHASE_R_CAPACITY_CONFORMANCE_DEVELOPMENT_V5.json",
)


@pytest.mark.parametrize("name", CURRENT_DEVELOPMENT_ARTIFACTS)
def test_current_development_artifact_source_drift_is_acknowledged(
    name: str,
) -> None:
    """Fail loudly when a locked source changes without acknowledgement.

    The recorded source lock is a claim about which sources produced the
    artifact, and nothing verified it against live files before this test, so
    edits to the permanence stack were previously undetectable (review finding
    F8).  Drift is allowed, but only when it has been written down here.
    """

    root = Path(__file__).resolve().parents[1]
    report = audit_artifact_source_lock(root / "experiments" / name, root=root)

    assert report["missing"] == []
    unacknowledged = set(report["drifted"]) - ACKNOWLEDGED_SOURCE_DRIFT
    assert not unacknowledged, (
        f"{name} was produced by a different source state; regenerate the "
        f"artifact or acknowledge the drift: {sorted(unacknowledged)}"
    )


def test_source_lock_audit_reports_drift_without_raising(tmp_path: Path) -> None:
    module = tmp_path / "pkg/mod.py"
    module.parent.mkdir()
    module.write_text("value = 1\n", encoding="utf-8")
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {"provenance": {"source_lock": source_lock((module,), root=tmp_path)}}
        ),
        encoding="utf-8",
    )

    assert audit_artifact_source_lock(artifact, root=tmp_path)["matched"] is True

    module.write_text("value = 2\n", encoding="utf-8")
    drifted = audit_artifact_source_lock(artifact, root=tmp_path)
    assert drifted["matched"] is False
    assert drifted["drifted"] == ["pkg/mod.py"]
    assert drifted["missing"] == []

    module.unlink()
    removed = audit_artifact_source_lock(artifact, root=tmp_path)
    assert removed["missing"] == ["pkg/mod.py"]
