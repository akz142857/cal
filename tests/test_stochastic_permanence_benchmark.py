from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cal.evaluation.stochastic_permanence_artifacts import (
    sha256_path,
    source_lock,
    validate_artifact,
)
from cal.evaluation.stochastic_permanence_benchmark import (
    PREDICTORS,
    REQUIRED_RESOURCE_GATES,
    _POWER_NULL_SENSITIVITY_GRID,
    _POWER_SENSITIVITY_GRID,
    _bounded_advantage_calibration,
    _sample_bounded_advantage,
    _simulated_score_table,
    append_attempt,
    build_reference_health,
    build_reference_health_power_artifact,
    common_bootstrap_indices,
    evaluate_confirmatory_decision,
    new_attempt_ledger,
    run_candidate_lifecycle,
    score_table_from_episode_binned,
    select_final_attempt,
    validate_candidate_factory,
    validate_complete_seed_population,
)


RESOURCE_GATES = {name: True for name in REQUIRED_RESOURCE_GATES}
ROOT = Path(__file__).resolve().parents[1]


def _valid_power_provenance(seed_ids: list[int]) -> dict[str, object]:
    registry = "experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY_V2.json"
    turn_scan = (
        "experiments/V2_P1_PERMANENCE_TURN_PROBABILITY_DEVELOPMENT_SCAN_V2.json"
    )
    sources = (
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
        registry,
        turn_scan,
    )
    return {
        "development_only": True,
        "candidate_source_imported": False,
        "candidate_maps_read": False,
        "registry_path": registry,
        "registry_sha256": sha256_path(ROOT / registry),
        "turn_probability_selection_artifact": turn_scan,
        "turn_probability_selection_sha256": sha256_path(ROOT / turn_scan),
        "source_lock": source_lock((ROOT / item for item in sources), root=ROOT),
        "runtime": {"python": "test", "platform": "test", "numpy": "test"},
        "command": "test",
        "coverage": {
            "development_seed_count": len(seed_ids),
            "complete_seed_ids": seed_ids,
            "all_predictors_same_complete_seed_population": True,
            "required_occlusion_bins": ["2-3", "4-5", "6+"],
        },
    }


def _metrics(
    *, top1: float, nll: float, brier: float, error: float
) -> dict[str, float]:
    return {
        "top1_accuracy": top1,
        "categorical_nll": nll,
        "brier": brier,
        "argmax_position_error": error,
    }


def _passing_table(seed_count: int = 20) -> dict[str, object]:
    bins = {
        "2-3": {
            "candidate": _metrics(top1=0.70, nll=1.50, brier=0.025, error=2.5),
            "oracle": _metrics(top1=0.80, nll=1.00, brier=0.020, error=1.0),
            "geometric": _metrics(top1=0.60, nll=4.00, brier=0.040, error=8.0),
            "uniform": _metrics(top1=0.10, nll=3.00, brier=0.040, error=6.0),
            "old_i1": _metrics(top1=0.50, nll=4.00, brier=0.080, error=3.0),
        },
        "4-5": {
            "candidate": _metrics(top1=0.55, nll=1.70, brier=0.025, error=2.5),
            "oracle": _metrics(top1=0.70, nll=1.20, brier=0.020, error=1.0),
            "geometric": _metrics(top1=0.30, nll=5.00, brier=0.045, error=8.0),
            "uniform": _metrics(top1=0.10, nll=3.20, brier=0.040, error=6.0),
            "old_i1": _metrics(top1=0.40, nll=5.00, brier=0.080, error=3.0),
        },
        "6+": {
            "candidate": _metrics(top1=0.50, nll=2.00, brier=0.030, error=2.5),
            "oracle": _metrics(top1=0.70, nll=1.50, brier=0.025, error=1.0),
            "geometric": _metrics(top1=0.10, nll=8.00, brier=0.050, error=8.0),
            "uniform": _metrics(top1=0.10, nll=3.50, brier=0.040, error=6.0),
            "old_i1": _metrics(top1=0.40, nll=6.00, brier=0.080, error=3.0),
        },
    }
    return {
        predictor: {
            str(seed): {
                bin_name: deepcopy(values[predictor])
                for bin_name, values in bins.items()
            }
            for seed in range(seed_count)
        }
        for predictor in PREDICTORS
    }


def _reference_table(table: dict[str, object]) -> dict[str, object]:
    return {
        name: deepcopy(table[name])
        for name in ("oracle", "geometric", "uniform", "old_i1")
    }


def _decision(table: dict[str, object]) -> dict[str, object]:
    reference_health, _seed_ids = build_reference_health(_reference_table(table))
    return evaluate_confirmatory_decision(
        table,
        reference_health=reference_health,
        resource_gates=RESOURCE_GATES,
        stability_pass=True,
        locks={},
    )


def test_score_table_accepts_reducer_metrics_beyond_formal_subset() -> None:
    metrics = _metrics(top1=0.5, nll=2.0, brier=0.04, error=3.0)
    metrics.update({"mrr": 0.6, "empty_map_rate": 0.0})
    score = {"seed_bin_scores": {"7": {"2-3": metrics}}}

    table = score_table_from_episode_binned({"oracle": score})

    assert table == {
        "oracle": {
            "7": {
                "2-3": _metrics(top1=0.5, nll=2.0, brier=0.04, error=3.0)
            }
        }
    }


def test_common_bootstrap_indices_are_exactly_reproducible() -> None:
    first = common_bootstrap_indices(8)
    second = common_bootstrap_indices(8)

    assert first.shape == (10_000, 8)
    assert (first == second).all()


def test_bounded_power_calibration_matches_requested_moments() -> None:
    reference = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    scenario = {
        "candidate_reference_correlation": 0.75,
        "single_initialization_sd_fraction": 0.10,
        "variance_inflation": 1.5,
    }

    calibration = _bounded_advantage_calibration(
        reference,
        lower_bounds=np.full(reference.shape, -0.5),
        upper_bounds=np.full(reference.shape, 0.8),
        target_effect=0.45,
        scenario=scenario,
        initialization_sign=1.0,
    )

    assert np.all(
        (-0.5 <= calibration["conditional_advantage_means"])
        & (calibration["conditional_advantage_means"] <= 0.8)
    )
    assert calibration["achieved_correlation"] == pytest.approx(0.75)
    assert calibration["achieved_variance_inflation"] == pytest.approx(1.5)
    assert 0.0 <= calibration["endpoint_variance_fraction"] <= 1.0


def test_feasibility_native_calibration_hits_constructive_coordinates() -> None:
    reference = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    scenario = {
        "covariance_interval": "full_feasible",
        "feasible_covariance_fraction": 0.75,
        "single_initialization_sd_fraction": 0.10,
        "variance_scale_multiplier": 1.5,
    }

    calibration = _bounded_advantage_calibration(
        reference,
        lower_bounds=np.full(reference.shape, -0.5),
        upper_bounds=np.full(reference.shape, 0.8),
        target_effect=0.45,
        scenario=scenario,
        initialization_sign=1.0,
    )

    assert calibration["requested_correlation"] is None
    assert calibration["requested_variance_inflation"] is None
    assert calibration["achieved_feasible_covariance_fraction"] == pytest.approx(
        0.75
    )
    assert calibration["achieved_variance_inflation"] == pytest.approx(1.5)
    assert 0.0 <= calibration["endpoint_variance_fraction"] <= 0.90
    assert calibration["feasible_covariance_upper"] > (
        calibration["feasible_covariance_lower"]
    )


def test_feasibility_native_calibration_accepts_signed_reference_gaps() -> None:
    reference = np.asarray([-0.04, 0.01, 0.03, 0.08, 0.12, 0.18])
    calibration = _bounded_advantage_calibration(
        reference,
        lower_bounds=np.full(reference.shape, -0.5),
        upper_bounds=np.full(reference.shape, 0.5),
        target_effect=0.30,
        scenario={
            "covariance_interval": "full_feasible",
            "feasible_covariance_fraction": 0.25,
            "single_initialization_sd_fraction": 0.0,
            "variance_scale_multiplier": 1.0,
        },
        initialization_sign=0.0,
    )

    assert np.mean(calibration["conditional_advantage_means"]) == pytest.approx(
        0.30 * np.mean(reference)
    )
    assert calibration["achieved_feasible_covariance_fraction"] == pytest.approx(
        0.25
    )


def test_feasibility_native_calibration_resolves_narrow_covariance_interval() -> None:
    reference = np.asarray(
        [
            3.9852576046958217,
            3.335508067326538,
            6.795818351374349,
            9.15696202393178,
            9.17704708535491,
            7.355968973309078,
            4.1523558026142915,
            2.2590750513815254,
        ]
    )
    baseline = np.asarray(
        [
            19.568952906944922,
            19.739990963459608,
            8.729379420665607,
            4.477991511511425,
            19.67652332073002,
            10.348202333481751,
            19.788019259324827,
            10.68992239652504,
        ]
    )
    target_mean = -5.872111743570096
    calibration = _bounded_advantage_calibration(
        reference,
        lower_bounds=baseline - 20.0,
        upper_bounds=baseline,
        target_effect=target_mean / float(reference.mean()),
        target_mean_override=target_mean,
        variance_effect=0.765602916986923,
        scenario={
            "covariance_interval": "full_feasible",
            "feasible_covariance_fraction": 0.25,
            "single_initialization_sd_fraction": 0.0,
            "variance_scale_multiplier": 1.0,
        },
        initialization_sign=0.0,
    )

    interval_width = (
        calibration["feasible_covariance_upper"]
        - calibration["feasible_covariance_lower"]
    )
    assert 0.0 < interval_width < 1e-9
    assert abs(
        calibration["achieved_feasible_covariance_fraction"] - 0.25
    ) <= 1e-9


def test_feasibility_native_calibration_accepts_physical_mean_boundary() -> None:
    reference = np.asarray([0.1, 0.2, 0.3, 0.4])
    lower = np.asarray([0.0, 0.1, 0.2, 0.3])
    calibration = _bounded_advantage_calibration(
        reference,
        lower_bounds=lower,
        upper_bounds=np.ones(reference.shape),
        target_effect=0.0,
        target_mean_override=float(lower.mean()),
        variance_effect=0.45,
        scenario={
            "covariance_interval": "full_feasible",
            "feasible_covariance_fraction": 0.75,
            "single_initialization_sd_fraction": 0.0,
            "variance_scale_multiplier": 1.5,
        },
        initialization_sign=0.0,
    )

    assert calibration["conditional_advantage_means"] == pytest.approx(lower)
    assert calibration["covariance_interval_degenerate"] is True
    assert calibration["variance_floor_applied"] is True
    assert calibration["endpoint_variance_fraction"] == pytest.approx(0.0)


def test_feasibility_native_calibration_accepts_degenerate_outer_gap() -> None:
    reference = np.zeros(6)
    calibration = _bounded_advantage_calibration(
        reference,
        lower_bounds=np.full(reference.shape, -0.5),
        upper_bounds=np.full(reference.shape, 0.5),
        target_effect=0.45,
        scenario={
            "covariance_interval": "full_feasible",
            "feasible_covariance_fraction": 0.75,
            "single_initialization_sd_fraction": 0.10,
            "variance_scale_multiplier": 1.5,
        },
        initialization_sign=1.0,
        variance_effect=0.45,
    )

    assert calibration["covariance_interval_degenerate"] is True
    assert calibration["achieved_feasible_covariance_fraction"] == pytest.approx(
        0.75
    )
    assert calibration["achieved_covariance"] == pytest.approx(0.0)
    assert calibration["achieved_correlation"] is None
    assert calibration["endpoint_variance_fraction"] == pytest.approx(0.0)


def test_component_null_can_set_absolute_mean_when_reference_gap_is_zero() -> None:
    reference = np.zeros(6)
    calibration = _bounded_advantage_calibration(
        reference,
        lower_bounds=np.full(reference.shape, -0.5),
        upper_bounds=np.full(reference.shape, 0.5),
        target_effect=0.0,
        target_mean_override=-0.03,
        scenario={
            "covariance_interval": "full_feasible",
            "feasible_covariance_fraction": 0.25,
            "single_initialization_sd_fraction": 0.0,
            "variance_scale_multiplier": 1.5,
        },
        initialization_sign=0.0,
        variance_effect=0.45,
    )

    assert calibration["mean_advantage"] == pytest.approx(-0.03)
    assert np.mean(calibration["conditional_advantage_means"]) == pytest.approx(
        -0.03
    )
    assert calibration["covariance_interval_degenerate"] is True


def test_bounded_variance_uses_contracted_endpoints_not_rare_full_jumps() -> None:
    calibration = {
        "conditional_advantage_means": np.asarray([0.0]),
        "endpoint_variance_fraction": 0.25,
    }
    generator = np.random.Generator(np.random.PCG64(17))
    samples = {
        _sample_bounded_advantage(
            calibration,
            source_index=0,
            lower_bound=-0.5,
            upper_bound=0.5,
            generator=generator,
            include_independent_innovation=True,
        )
        for _ in range(100)
    }

    assert samples == {-0.25, 0.25}


def test_feasibility_native_power_and_null_grids_are_exact_cartesian_products() -> None:
    triples = [
        (
            scenario["feasible_covariance_fraction"],
            scenario["single_initialization_sd_fraction"],
            scenario["variance_scale_multiplier"],
        )
        for scenario in _POWER_SENSITIVITY_GRID
    ]

    assert triples == [
        (0.25, 0.0, 1.0),
        (0.25, 0.0, 1.5),
        (0.75, 0.0, 1.0),
        (0.75, 0.0, 1.5),
    ]
    assert [
        (
            scenario["feasible_covariance_fraction"],
            scenario["variance_scale_multiplier"],
        )
        for scenario in _POWER_NULL_SENSITIVITY_GRID
    ] == [(0.25, 1.0), (0.25, 1.5), (0.75, 1.0), (0.75, 1.5)]


def test_power_generator_stays_inside_metric_physical_domains() -> None:
    table = _reference_table(_passing_table(seed_count=20))
    for index, seed in enumerate(sorted(table["oracle"], key=int)):
        for scope in ("2-3", "4-5", "6+"):
            table["oracle"][seed][scope]["top1_accuracy"] -= index * 0.001
    effects = {
        metric: {scope: 0.45 for scope in ("2-3", "4-5", "6+")}
        for metric in ("top1_accuracy", "categorical_nll", "brier")
    }
    generated = _simulated_score_table(
        table,
        draw=np.arange(20),
        effects=effects,
        scenario={
            "candidate_reference_correlation": 0.75,
            "single_initialization_sd_fraction": 0.0,
            "variance_inflation": 1.0,
        },
        generator=np.random.Generator(np.random.PCG64(99)),
        initialization_sign=1.0,
        include_independent_innovation=True,
        position_effect=0.70,
    )
    for index, seed in enumerate(sorted(table["oracle"], key=int)):
        for scope in ("2-3", "4-5", "6+"):
            for metric in (
                "top1_accuracy",
                "categorical_nll",
                "brier",
                "argmax_position_error",
            ):
                value = generated["candidate"][str(index)][scope][metric]
                upper = {
                    "top1_accuracy": 1.0,
                    "categorical_nll": 13.815510557964274,
                    "brier": 1.0,
                    "argmax_position_error": 20.0,
                }[metric]
                assert 0.0 <= value <= upper


def test_formal_decision_recomputes_all_gates_with_strict_and() -> None:
    table = _passing_table()
    reference_health, _seed_ids = build_reference_health(_reference_table(table))
    artifact = _decision(table)

    assert artifact["passed"] is True
    assert all(artifact["gates"].values())
    assert artifact["bootstrap"]["quantile"] == 0.01
    validate_artifact(
        artifact, expected_kind="stochastic_permanence_formal_decision"
    )

    failed_table = deepcopy(table)
    for seed in failed_table["candidate"].values():
        for metrics in seed.values():
            metrics["top1_accuracy"] = 0.0
    failed = evaluate_confirmatory_decision(
        failed_table,
        reference_health=reference_health,
        resource_gates=RESOURCE_GATES,
        stability_pass=True,
        locks={},
    )
    assert failed["passed"] is False
    assert failed["decision"] == "stop"

    with pytest.raises(RuntimeError, match="population mismatch"):
        evaluate_confirmatory_decision(
            failed_table,
            reference_health=reference_health,
            resource_gates={
                **RESOURCE_GATES,
                "overall_top1_superiority": True,
            },
            stability_pass=True,
            locks={},
        )
    with pytest.raises(TypeError, match="string/bool"):
        evaluate_confirmatory_decision(
            table,
            reference_health=reference_health,
            resource_gates={
                **RESOURCE_GATES,
                "runtime_budget_respected": "yes",
            },  # type: ignore[dict-item]
            stability_pass=True,
            locks={},
        )


def test_closure_contrasts_have_the_preregistered_signs() -> None:
    result = _decision(_passing_table(seed_count=4))

    top = result["contrasts"]["overall/top1_closure_0.30"]["mean"]
    nll = result["contrasts"]["overall/nll_closure_0.20"]["mean"]
    brier = result["contrasts"]["overall/brier_closure_0.15"]["mean"]
    assert top == pytest.approx(
        np.mean([0.70 - 0.60, 0.55 - 0.30, 0.50 - 0.10])
        - 0.30 * np.mean([0.80 - 0.60, 0.70 - 0.30, 0.70 - 0.10])
    )
    assert nll == pytest.approx(
        np.mean([3.00 - 1.50, 3.20 - 1.70, 3.50 - 2.00])
        - 0.20 * np.mean([3.00 - 1.00, 3.20 - 1.20, 3.50 - 1.50])
    )
    assert brier == pytest.approx(
        np.mean([0.040 - 0.025, 0.040 - 0.025, 0.040 - 0.030])
        - 0.15 * np.mean([0.040 - 0.020, 0.040 - 0.020, 0.040 - 0.025])
    )


def test_reference_floor_and_complete_population_fail_closed() -> None:
    table = _passing_table(seed_count=4)
    reference_health, _seed_ids = build_reference_health(_reference_table(table))
    tampered = deepcopy(reference_health)
    tampered["overall"]["top1_accuracy"]["floor"] = -1.0
    with pytest.raises(RuntimeError, match="reference-health floor"):
        evaluate_confirmatory_decision(
            table,
            reference_health=tampered,
            resource_gates=RESOURCE_GATES,
            stability_pass=True,
            locks={},
        )

    reference = _reference_table(table)
    del reference["oracle"]["3"]
    with pytest.raises(RuntimeError, match="complete seed IDs"):
        validate_complete_seed_population(
            reference,
            required_predictors=("oracle", "geometric", "uniform", "old_i1"),
            expected_seed_ids=(0, 1, 2, 3),
        )


def test_reference_health_and_power_artifact_never_reads_candidate() -> None:
    reference = _reference_table(_passing_table(seed_count=64))
    provenance = _valid_power_provenance(list(range(64)))
    first = build_reference_health_power_artifact(
        reference,
        registry_selection_digest_sha256="c" * 64,
        provenance=provenance,
        power_simulation_trials=4,
        power_bootstrap_samples=16,
    )
    second = build_reference_health_power_artifact(
        reference,
        registry_selection_digest_sha256="c" * 64,
        provenance=provenance,
        power_simulation_trials=4,
        power_bootstrap_samples=16,
    )

    assert first == second
    assert first["candidate_maps_read"] is False
    assert "candidate" not in first["per_seed_per_bin"]
    assert first["power_design"]["analytic_starting_seed_count_maximum"] >= 2
    simulation = first["power_design"]["composite_simulation"]
    if simulation["passed"]:
        assert simulation["recommended_validation_seed_count"] >= 2
    else:
        assert simulation["recommended_validation_seed_count"] is None
        assert simulation["recommended_holdout_seed_count"] is None
    assert (
        "simulated_metrics_within_physical_domain_without_clipping"
        in simulation["gates"]
    )
    assert simulation["sensitivity_grid_policy"]["construction_audit"]
    assert simulation["simulation_design_version"].endswith("_v10")
    assert first["power_design"][
        "composite_bootstrap_simulation_required_before_secret_commitment"
    ] is False
    validate_artifact(
        first,
        expected_kind="stochastic_permanence_reference_health_power",
    )

    tampered = deepcopy(first)
    simulation_tampered = tampered["power_design"]["composite_simulation"]
    simulation_tampered["sensitivity_grid_policy"]["scenarios"][0][
        "feasible_covariance_fraction"
    ] = 9.0
    simulation_tampered["sensitivity_grid"][0][
        "feasible_covariance_fraction"
    ] = 9.0
    with pytest.raises(RuntimeError, match="exact rows"):
        validate_artifact(
            tampered,
            expected_kind="stochastic_permanence_reference_health_power",
        )

    tampered = deepcopy(first)
    tampered["power_design"]["composite_simulation"]["gates"] = {
        "forged": "yes"
    }
    with pytest.raises(RuntimeError, match="exact boolean population"):
        validate_artifact(
            tampered,
            expected_kind="stochastic_permanence_reference_health_power",
        )

    tampered = deepcopy(first)
    tampered["provenance"] = {}
    with pytest.raises(RuntimeError, match="provenance isolation"):
        validate_artifact(
            tampered,
            expected_kind="stochastic_permanence_reference_health_power",
        )


def _append_completed(
    ledger: dict[str, object],
    *,
    index: int,
    result: dict[str, object],
    persistent_bytes: int = 50_000,
    mac_per_step: int = 1_000_000,
) -> dict[str, object]:
    return append_attempt(
        ledger,
        configuration={"index": index},
        train_started=True,
        development_metrics_viewed=True,
        terminal_status="completed",
        terminal_reason="all development gates evaluated",
        source_lock_sha256="a" * 64,
        train_artifact_sha256="b" * 64,
        train_replays=1,
        development_result=result,
        persistent_bytes=persistent_bytes,
        mac_per_step=mac_per_step,
    )


def test_attempt_ledger_is_bounded_hashed_and_configuration_unique() -> None:
    ledger = new_attempt_ledger()
    result = _decision(_passing_table(seed_count=4))
    for index in range(12):
        ledger = _append_completed(ledger, index=index, result=result)

    with pytest.raises(RuntimeError, match="exhausted"):
        _append_completed(ledger, index=12, result=result)
    with pytest.raises(RuntimeError, match="consumed"):
        _append_completed(ledger, index=0, result=result)

    tampered = deepcopy(ledger)
    tampered["entries"][0]["terminal_reason"] = "rewritten"
    with pytest.raises(RuntimeError, match="digest"):
        select_final_attempt(tampered)


def test_attempt_infrastructure_retry_and_unique_selection() -> None:
    ledger = new_attempt_ledger()
    ledger = append_attempt(
        ledger,
        configuration={"index": 7},
        train_started=False,
        development_metrics_viewed=False,
        terminal_status="infrastructure_failure",
        terminal_reason="worker unavailable before candidate start",
        source_lock_sha256=None,
        train_artifact_sha256=None,
        train_replays=0,
        development_result=None,
        persistent_bytes=None,
        mac_per_step=None,
    )
    result = _decision(_passing_table(seed_count=4))
    ledger = _append_completed(
        ledger,
        index=7,
        result=result,
        persistent_bytes=49_000,
        mac_per_step=900_000,
    )
    ledger = _append_completed(
        ledger,
        index=8,
        result=result,
        persistent_bytes=48_000,
        mac_per_step=1_100_000,
    )

    selected = select_final_attempt(ledger)

    assert selected["selection"]["decision"] == "candidate_selected"
    assert selected["selection"]["selected"]["configuration_id"] == selected[
        "entries"
    ][2]["configuration_id"]
    with pytest.raises(RuntimeError, match="already"):
        select_final_attempt(selected)
    with pytest.raises(RuntimeError, match="after final"):
        _append_completed(selected, index=9, result=result)


def test_candidate_factory_rejects_privileged_signature() -> None:
    class GoodFactory:
        def fit_kernel(self, train_sensor_streams: object) -> dict[str, object]:
            return {}

        def new_episode(
            self, frozen_kernel: object, model_seed: int
        ) -> object:
            return object()

    class LeakingFactory(GoodFactory):
        def fit_kernel(
            self, train_sensor_streams: object, truth: object
        ) -> dict[str, object]:
            return {}

    validate_candidate_factory(GoodFactory())
    with pytest.raises(RuntimeError, match="expose only"):
        validate_candidate_factory(LeakingFactory())


def test_candidate_lifecycle_fits_once_freezes_and_resets_each_episode() -> None:
    class Factory:
        def __init__(self) -> None:
            self.fit_calls = 0
            self.seen_train_streams: tuple[object, ...] = ()
            self.kernel_ids: list[int] = []

        def fit_kernel(self, train_sensor_streams: object) -> dict[str, object]:
            self.fit_calls += 1
            self.seen_train_streams = tuple(train_sensor_streams)
            return {"weights": np.asarray([1.0, 2.0]), "nested": {"x": [3]}}

        def new_episode(
            self, frozen_kernel: object, model_seed: int
        ) -> object:
            assert model_seed == 17_011
            self.kernel_ids.append(id(frozen_kernel))
            return {"steps": 0, "kernel": frozen_kernel}

    factory = Factory()
    run = run_candidate_lifecycle(
        factory,
        train_sensor_streams={4: "train-4", 2: "train-2"},
        evaluation_sensor_streams={9: "eval-9", 7: "eval-7"},
        run_episode=lambda candidate, stream: (id(candidate), stream),
    )

    assert factory.fit_calls == 1
    assert factory.seen_train_streams == ("train-2", "train-4")
    assert run.train_seed_ids == (2, 4)
    assert run.evaluation_seed_ids == (7, 9)
    assert len(set(factory.kernel_ids)) == 1
    assert len({result[0] for result in run.episode_results.values()}) == 2
    with pytest.raises(TypeError):
        run.frozen_kernel["nested"]["x"] = (4,)
    with pytest.raises(ValueError):
        run.frozen_kernel["weights"][0] = 0.0
    with pytest.raises(ValueError):
        run.frozen_kernel["weights"].setflags(write=True)
