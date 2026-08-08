"""Formal Phase-0 statistics for stochastic permanence.

Unlike :mod:`permanence_forward_benchmark`, this module contains no world
collector and no predictor implementation.  It consumes locked per-seed,
per-bin sufficient statistics, uses one common seed bootstrap matrix, and
recomputes the complete fail-closed decision.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import json
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from cal.evaluation.stochastic_permanence_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    POWER_CANDIDATE_GENERATOR,
    POWER_GRID_ENDPOINT_VARIANCE_LIMIT,
    POWER_GRID_POLICY_VERSION,
    POWER_GRID_SCENARIOS,
    POWER_GRID_SELECTION_BASIS,
    POWER_NULL_GRID_SCENARIOS,
    POWER_CONFIRMATORY_ONE_SIDED_ALPHA,
    POWER_COVERAGE_CONTRAST_NAMES,
    POWER_FAMILYWISE_ALPHA,
    POWER_COVERAGE_FAMILYWISE_LOWER_MINIMUM,
    POWER_COVERAGE_POINT_MINIMUM,
    POWER_LOCKED_BOOTSTRAP_SAMPLES_PER_TRIAL,
    POWER_LOCKED_MAXIMUM_FALSE_REJECTIONS,
    POWER_LOCKED_MINIMUM_COVERED,
    POWER_LOCKED_MINIMUM_POWER_PASSES,
    POWER_LOCKED_SIMULATION_SEED,
    POWER_LOCKED_SIMULATION_TRIALS,
    POWER_LOCKED_TYPE1_SIMULATION_SEED,
    POWER_METRIC_PHYSICAL_BOUNDS,
    POWER_MODEL_INITIALIZATION_POLICY,
    POWER_SIMULATION_DESIGN_VERSION,
    POWER_SEED_SAFETY_MULTIPLIER,
    POWER_TARGET_MINIMUM,
    POWER_TYPE1_MAXIMUM,
    POWER_TYPE1_CONTRAST_NAMES,
    canonical_json_bytes,
    exact_binomial_lower_bound,
    exact_binomial_upper_bound,
)


BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_729
POWER_BOOTSTRAP_SAMPLES = POWER_LOCKED_BOOTSTRAP_SAMPLES_PER_TRIAL
POWER_SIMULATION_TRIALS = POWER_LOCKED_SIMULATION_TRIALS
POWER_SIMULATION_SEED = POWER_LOCKED_SIMULATION_SEED
POWER_TYPE1_SIMULATION_SEED = POWER_LOCKED_TYPE1_SIMULATION_SEED
POWER_GRID_MAX_ENDPOINT_VARIANCE_FRACTION = POWER_GRID_ENDPOINT_VARIANCE_LIMIT
POWER_SENSITIVITY_GRID_VERSION = POWER_GRID_POLICY_VERSION
BOUNDED_MOMENT_BISECTION_ITERATIONS = 56
MODEL_SEED = 17_011
MAX_CATEGORICAL_NLL = float(-np.log(1e-6))
MAX_BRIER = 1.0
MAX_POSITION_ERROR = 20.0
OCCLUSION_BINS = ("2-3", "4-5", "6+")
PREDICTORS = ("candidate", "oracle", "geometric", "uniform", "old_i1")
METRICS = (
    "top1_accuracy",
    "categorical_nll",
    "brier",
    "argmax_position_error",
)
REFERENCE_FLOOR_FRACTION = 0.25
MAXIMUM_CANDIDATE_ATTEMPTS = 12
MAXIMUM_TRAIN_REPLAYS = 4
ATTEMPT_SELECTION_ALGORITHM = (
    "filter_all_gates_then_maximize_min_overall_normalized_closure_"
    "then_persistent_bytes_then_mac_then_configuration_id_v1"
)
REQUIRED_REFERENCE_GATES = (
    ("overall", "top1_accuracy"),
    ("overall", "categorical_nll"),
    ("overall", "brier"),
    ("4-5", "top1_accuracy"),
    ("6+", "top1_accuracy"),
    ("6+", "categorical_nll"),
    ("6+", "brier"),
)
REQUIRED_RESOURCE_GATES = (
    "parameters_within_budget",
    "persistent_active_state_within_budget",
    "mac_per_step_within_budget",
    "step_budget_respected",
    "replay_budget_respected",
    "runtime_budget_respected",
)
_TYPE1_BOUNDARY_TESTS = {
    "overall/candidate_vs_geometric/top1": (0.0, True),
    "overall/top1_closure_0.30": (0.0, False),
    "overall/candidate_vs_geometric/nll": (0.0, True),
    "overall/candidate_vs_uniform/nll": (0.0, True),
    "overall/nll_closure_0.20": (0.0, False),
    "overall/candidate_vs_geometric/brier": (0.0, True),
    "overall/candidate_vs_uniform/brier": (0.0, True),
    "overall/brier_closure_0.15": (0.0, False),
    "2-3/candidate_vs_geometric/top1": (-0.03, False),
    "4-5/candidate_vs_geometric/top1": (0.0, True),
    "4-5/top1_closure_0.20": (0.0, False),
    "6+/candidate_vs_geometric/top1": (0.0, True),
    "6+/top1_closure_0.40": (0.0, False),
    "6+/candidate_vs_old_i1/top1": (-0.02, False),
    "6+/candidate_vs_uniform/nll": (0.0, True),
    "6+/nll_closure_0.20": (0.0, False),
    "6+/candidate_vs_uniform/brier": (0.0, True),
    "6+/brier_closure_0.15": (0.0, False),
    "overall/candidate_vs_geometric/position_error": (0.0, True),
    "overall/candidate_vs_old_i1/position_error": (-0.25, False),
}

_COMPONENT_BOUNDARY_SPECS: dict[str, dict[str, Any]] = {
    "overall/candidate_vs_geometric/top1": {
        "metric": "top1_accuracy", "scope": "overall", "kind": "baseline"
    },
    "overall/top1_closure_0.30": {
        "metric": "top1_accuracy", "scope": "overall", "kind": "closure",
        "threshold": 0.30,
    },
    "overall/candidate_vs_geometric/nll": {
        "metric": "categorical_nll", "scope": "overall",
        "kind": "geometric_lower",
    },
    "overall/candidate_vs_uniform/nll": {
        "metric": "categorical_nll", "scope": "overall", "kind": "baseline"
    },
    "overall/nll_closure_0.20": {
        "metric": "categorical_nll", "scope": "overall", "kind": "closure",
        "threshold": 0.20,
    },
    "overall/candidate_vs_geometric/brier": {
        "metric": "brier", "scope": "overall", "kind": "geometric_lower"
    },
    "overall/candidate_vs_uniform/brier": {
        "metric": "brier", "scope": "overall", "kind": "baseline"
    },
    "overall/brier_closure_0.15": {
        "metric": "brier", "scope": "overall", "kind": "closure",
        "threshold": 0.15,
    },
    "2-3/candidate_vs_geometric/top1": {
        "metric": "top1_accuracy", "scope": "2-3", "kind": "baseline"
    },
    "4-5/candidate_vs_geometric/top1": {
        "metric": "top1_accuracy", "scope": "4-5", "kind": "baseline"
    },
    "4-5/top1_closure_0.20": {
        "metric": "top1_accuracy", "scope": "4-5", "kind": "closure",
        "threshold": 0.20,
    },
    "6+/candidate_vs_geometric/top1": {
        "metric": "top1_accuracy", "scope": "6+", "kind": "baseline"
    },
    "6+/top1_closure_0.40": {
        "metric": "top1_accuracy", "scope": "6+", "kind": "closure",
        "threshold": 0.40,
    },
    "6+/candidate_vs_old_i1/top1": {
        "metric": "top1_accuracy", "scope": "6+", "kind": "old_i1_top1"
    },
    "6+/candidate_vs_uniform/nll": {
        "metric": "categorical_nll", "scope": "6+", "kind": "baseline"
    },
    "6+/nll_closure_0.20": {
        "metric": "categorical_nll", "scope": "6+", "kind": "closure",
        "threshold": 0.20,
    },
    "6+/candidate_vs_uniform/brier": {
        "metric": "brier", "scope": "6+", "kind": "baseline"
    },
    "6+/brier_closure_0.15": {
        "metric": "brier", "scope": "6+", "kind": "closure",
        "threshold": 0.15,
    },
    "overall/candidate_vs_geometric/position_error": {
        "metric": "argmax_position_error", "scope": "overall",
        "kind": "baseline",
    },
    "overall/candidate_vs_old_i1/position_error": {
        "metric": "argmax_position_error", "scope": "overall",
        "kind": "old_i1_position",
    },
}


@runtime_checkable
class CandidateFactory(Protocol):
    """The only candidate lifecycle accepted by a later formal runner."""

    def fit_kernel(
        self, train_sensor_streams: Sequence[object]
    ) -> Mapping[str, Any]: ...

    def new_episode(
        self, frozen_kernel: Mapping[str, Any], model_seed: int
    ) -> object: ...


def _canonical_seed_population(
    population: Mapping[int, object], *, name: str
) -> tuple[int, ...]:
    if not isinstance(population, Mapping) or not population:
        raise RuntimeError(f"{name} seed population is empty")
    seeds = tuple(population)
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise RuntimeError(f"{name} seed IDs must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"{name} seed IDs are not unique")
    return tuple(sorted(seeds))


def validate_candidate_factory(factory: CandidateFactory) -> None:
    if not isinstance(factory, CandidateFactory):
        raise TypeError("candidate factory does not implement fit/new_episode")
    expected_parameters = {
        "fit_kernel": ("train_sensor_streams",),
        "new_episode": ("frozen_kernel", "model_seed"),
    }
    for method_name, expected in expected_parameters.items():
        signature = inspect.signature(getattr(factory, method_name))
        parameters = tuple(signature.parameters.values())
        if tuple(parameter.name for parameter in parameters) != expected or any(
            parameter.kind
            not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for parameter in parameters
        ):
            raise RuntimeError(
                f"candidate factory {method_name} must expose only {expected}"
            )


def _canonical_kernel_value(value: Any) -> Any:
    """Convert a frozen-kernel value to a deterministic, typed JSON form."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("frozen-kernel mapping keys must be strings")
        return {
            key: _canonical_kernel_value(value[key]) for key in sorted(value)
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("frozen-kernel arrays cannot use object dtype")
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": contiguous.tolist(),
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
    if isinstance(value, np.generic):
        return _canonical_kernel_value(value.item())
    if isinstance(value, (tuple, list)):
        return [_canonical_kernel_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        encoded = [_canonical_kernel_value(item) for item in value]
        return {
            "__set__": sorted(
                encoded, key=lambda item: canonical_json_bytes({"v": item})
            )
        }
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("frozen kernel contains NaN/inf")
        return value
    raise TypeError(f"unsupported frozen-kernel value: {type(value).__name__}")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        source = np.ascontiguousarray(value)
        frozen = np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(
            source.shape
        )
        return frozen
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class CandidateLifecycleRun:
    """Evidence returned by the only fit/freeze/reset orchestration path."""

    frozen_kernel: Mapping[str, Any]
    frozen_kernel_sha256: str
    train_seed_ids: tuple[int, ...]
    evaluation_seed_ids: tuple[int, ...]
    model_seed: int
    episode_results: Mapping[int, Any]


def run_candidate_lifecycle(
    factory: CandidateFactory,
    *,
    train_sensor_streams: Mapping[int, object],
    evaluation_sensor_streams: Mapping[int, object],
    run_episode: Callable[[object, object], Any],
) -> CandidateLifecycleRun:
    """Fit once, freeze once, and construct a fresh candidate per episode.

    Both stream populations are normalized to ascending integer seed order.
    Only sensor/action streams cross the factory boundary; the environment seed
    is used solely as an external key and never passed to ``new_episode``.
    """

    validate_candidate_factory(factory)
    if not callable(run_episode):
        raise TypeError("run_episode must be callable")
    train_seeds = _canonical_seed_population(train_sensor_streams, name="train")
    evaluation_seeds = _canonical_seed_population(
        evaluation_sensor_streams, name="evaluation"
    )
    if set(train_seeds) & set(evaluation_seeds):
        raise RuntimeError("train/evaluation seed populations overlap")
    raw_kernel = factory.fit_kernel(
        tuple(train_sensor_streams[seed] for seed in train_seeds)
    )
    if not isinstance(raw_kernel, Mapping):
        raise TypeError("fit_kernel must return a mapping")
    canonical_kernel = _canonical_kernel_value(raw_kernel)
    kernel_digest = hashlib.sha256(
        canonical_json_bytes({"frozen_kernel": canonical_kernel})
    ).hexdigest()
    frozen_kernel = _deep_freeze(raw_kernel)
    retained_instances: list[object] = []
    results: dict[int, Any] = {}
    for seed in evaluation_seeds:
        instance = factory.new_episode(frozen_kernel, MODEL_SEED)
        if any(instance is previous for previous in retained_instances):
            raise RuntimeError("new_episode reused candidate state across episodes")
        retained_instances.append(instance)
        episode_result = run_episode(instance, evaluation_sensor_streams[seed])
        _canonical_kernel_value(episode_result)
        results[seed] = _deep_freeze(episode_result)
    # Re-encoding proves that no callback changed any reachable kernel array.
    if _canonical_kernel_value(frozen_kernel) != canonical_kernel:
        raise RuntimeError("frozen kernel changed during evaluation")
    return CandidateLifecycleRun(
        frozen_kernel=frozen_kernel,
        frozen_kernel_sha256=kernel_digest,
        train_seed_ids=train_seeds,
        evaluation_seed_ids=evaluation_seeds,
        model_seed=MODEL_SEED,
        episode_results=MappingProxyType(results),
    )


def score_table_from_episode_binned(
    predictors: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Extract the sufficient-statistic table exported by the dev reducer."""

    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for name, score in predictors.items():
        raw = score.get("seed_bin_scores")
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"predictor lacks seed/bin sufficient stats: {name}")
        result[name] = {}
        for seed, bins in raw.items():
            if not isinstance(bins, Mapping):
                raise RuntimeError(f"predictor has invalid seed bins: {name}/{seed}")
            result[name][str(seed)] = {}
            for bin_name, metrics in bins.items():
                if not isinstance(metrics, Mapping) or not set(METRICS).issubset(
                    metrics
                ):
                    raise RuntimeError(
                        f"predictor has invalid metrics: {name}/{seed}/{bin_name}"
                    )
                if any(
                    isinstance(metrics[metric], (bool, np.bool_))
                    or not isinstance(
                        metrics[metric], (int, float, np.integer, np.floating)
                    )
                    for metric in METRICS
                ):
                    raise RuntimeError("predictor sufficient statistic is non-numeric")
                result[name][str(seed)][str(bin_name)] = {
                    metric: float(metrics[metric]) for metric in METRICS
                }
    return result


def _validate_score_table(
    table: Mapping[str, Any], *, required_predictors: Sequence[str]
) -> tuple[str, ...]:
    if set(table) != set(required_predictors):
        raise RuntimeError(
            "predictor population mismatch: "
            f"expected {sorted(required_predictors)}, got {sorted(table)}"
        )
    seed_sets = [set(table[name]) for name in required_predictors]
    if not seed_sets or not seed_sets[0]:
        raise RuntimeError("score table has no complete seeds")
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise RuntimeError("predictors do not share complete seed IDs")
    if any(
        not isinstance(seed, str)
        or not seed.isascii()
        or not seed.isdecimal()
        or seed != str(int(seed))
        for seed in seed_sets[0]
    ):
        raise RuntimeError("score table seed IDs are not canonical integers")
    seed_ids = tuple(sorted(seed_sets[0], key=int))
    for name in required_predictors:
        for seed in seed_ids:
            bins = table[name][seed]
            if set(bins) != set(OCCLUSION_BINS):
                raise RuntimeError(f"missing bin for {name}/{seed}")
            for bin_name in OCCLUSION_BINS:
                metrics = bins[bin_name]
                if set(metrics) != set(METRICS):
                    raise RuntimeError(
                        f"metric population mismatch for {name}/{seed}/{bin_name}"
                    )
                if any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    for value in metrics.values()
                ):
                    raise RuntimeError("score table contains a non-numeric metric")
                values = np.asarray(list(metrics.values()), dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise RuntimeError("score table contains NaN/inf")
                if not 0.0 <= float(metrics["top1_accuracy"]) <= 1.0:
                    raise RuntimeError("top1_accuracy is outside [0, 1]")
                if not 0.0 <= float(metrics["brier"]) <= 1.0:
                    raise RuntimeError("brier is outside [0, 1]")
                if float(metrics["categorical_nll"]) < 0.0:
                    raise RuntimeError("categorical_nll is negative")
                if float(metrics["argmax_position_error"]) < 0.0:
                    raise RuntimeError("argmax_position_error is negative")
    if len(seed_ids) < 2:
        raise RuntimeError("at least two complete seeds are required")
    return seed_ids


def validate_complete_seed_population(
    table: Mapping[str, Any],
    *,
    required_predictors: Sequence[str],
    expected_seed_ids: Sequence[int],
) -> tuple[str, ...]:
    """Require the score table to contain the entire precommitted population."""

    seeds = _validate_score_table(table, required_predictors=required_predictors)
    expected = tuple(expected_seed_ids)
    if (
        len(expected) < 2
        or any(type(seed) is not int or seed < 0 for seed in expected)
        or len(set(expected)) != len(expected)
    ):
        raise RuntimeError("expected seed population is invalid")
    if expected != tuple(sorted(expected)):
        raise RuntimeError("expected seed population is not in ascending order")
    if tuple(int(seed) for seed in seeds) != expected:
        raise RuntimeError(
            "score table does not match the complete precommitted seed population"
        )
    return seeds


def _scope_values(
    table: Mapping[str, Any],
    *,
    predictor: str,
    metric: str,
    scope: str,
    seed_ids: Sequence[str],
) -> np.ndarray:
    if scope == "overall":
        return np.asarray(
            [
                np.mean(
                    [
                        table[predictor][seed][bin_name][metric]
                        for bin_name in OCCLUSION_BINS
                    ]
                )
                for seed in seed_ids
            ],
            dtype=np.float64,
        )
    if scope not in OCCLUSION_BINS:
        raise ValueError(f"unknown score scope: {scope}")
    return np.asarray(
        [table[predictor][seed][scope][metric] for seed in seed_ids],
        dtype=np.float64,
    )


def common_bootstrap_indices(seed_count: int) -> np.ndarray:
    if seed_count < 2:
        raise ValueError("bootstrap requires at least two seeds")
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return generator.integers(
        0,
        seed_count,
        size=(BOOTSTRAP_SAMPLES, seed_count),
        dtype=np.int64,
    )


def _power_bootstrap_indices(seed_count: int, sample_count: int) -> np.ndarray:
    if seed_count < 2 or sample_count < 2:
        raise ValueError("power bootstrap requires at least two seeds/resamples")
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return generator.integers(
        0,
        seed_count,
        size=(sample_count, seed_count),
        dtype=np.int64,
    )


def contrast_summary(values: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size != indices.shape[1]:
        raise ValueError("contrast vector/bootstrap shape mismatch")
    means = vector[indices].mean(axis=1)
    return {
        "per_seed": [float(value) for value in vector],
        "mean": float(vector.mean()),
        "one_sided_99_lower": float(
            np.quantile(
                means,
                POWER_CONFIRMATORY_ONE_SIDED_ALPHA,
                method="linear",
            )
        ),
    }


def _bootstrap_weight_matrix(indices: np.ndarray) -> np.ndarray:
    matrix = np.asarray(indices, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("bootstrap indices are invalid")
    rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), matrix.shape[1])
    weights = np.zeros(matrix.shape, dtype=np.float64)
    np.add.at(weights, (rows, matrix.ravel()), 1.0 / matrix.shape[1])
    return weights


def _reference_vector(
    table: Mapping[str, Any],
    *,
    metric: str,
    scope: str,
    seed_ids: Sequence[str],
) -> np.ndarray:
    if metric == "top1_accuracy":
        return _scope_values(
            table,
            predictor="oracle",
            metric=metric,
            scope=scope,
            seed_ids=seed_ids,
        ) - _scope_values(
            table,
            predictor="geometric",
            metric=metric,
            scope=scope,
            seed_ids=seed_ids,
        )
    return _scope_values(
        table,
        predictor="uniform",
        metric=metric,
        scope=scope,
        seed_ids=seed_ids,
    ) - _scope_values(
        table,
        predictor="oracle",
        metric=metric,
        scope=scope,
        seed_ids=seed_ids,
    )


def build_reference_health(
    table: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    seed_ids = _validate_score_table(
        table,
        required_predictors=("oracle", "geometric", "uniform", "old_i1"),
    )
    indices = common_bootstrap_indices(len(seed_ids))
    result: dict[str, Any] = {}
    for scope in ("overall", *OCCLUSION_BINS):
        result[scope] = {}
        for metric in ("top1_accuracy", "categorical_nll", "brier"):
            reference = _reference_vector(
                table,
                metric=metric,
                scope=scope,
                seed_ids=seed_ids,
            )
            summary = contrast_summary(reference, indices)
            result[scope][metric] = {
                **summary,
                "floor": float(REFERENCE_FLOOR_FRACTION * summary["mean"]),
                "floor_fraction": REFERENCE_FLOOR_FRACTION,
                "reference": (
                    "oracle_minus_geometric"
                    if metric == "top1_accuracy"
                    else "uniform_minus_oracle"
                ),
            }
    return result, seed_ids


def _advantage(
    table: Mapping[str, Any],
    *,
    first: str,
    second: str,
    metric: str,
    scope: str,
    seed_ids: Sequence[str],
) -> np.ndarray:
    first_values = _scope_values(
        table,
        predictor=first,
        metric=metric,
        scope=scope,
        seed_ids=seed_ids,
    )
    second_values = _scope_values(
        table,
        predictor=second,
        metric=metric,
        scope=scope,
        seed_ids=seed_ids,
    )
    return (
        first_values - second_values
        if metric == "top1_accuracy"
        else second_values - first_values
    )


def _closure_contrast(
    table: Mapping[str, Any],
    *,
    metric: str,
    scope: str,
    threshold: float,
    seed_ids: Sequence[str],
) -> np.ndarray:
    baseline = "geometric" if metric == "top1_accuracy" else "uniform"
    gain = _advantage(
        table,
        first="candidate",
        second=baseline,
        metric=metric,
        scope=scope,
        seed_ids=seed_ids,
    )
    return gain - threshold * _reference_vector(
        table,
        metric=metric,
        scope=scope,
        seed_ids=seed_ids,
    )


def _validate_reference_health_contract(
    reference_health: Mapping[str, Any],
) -> None:
    for scope, metric in REQUIRED_REFERENCE_GATES:
        try:
            item = reference_health[scope][metric]
            floor = float(item["floor"])
            development_mean = float(item["mean"])
            fraction = float(item["floor_fraction"])
            reference = str(item["reference"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"invalid reference-health contract for {scope}/{metric}"
            ) from error
        expected_reference = (
            "oracle_minus_geometric"
            if metric == "top1_accuracy"
            else "uniform_minus_oracle"
        )
        if (
            not np.isfinite([floor, development_mean, fraction]).all()
            or development_mean <= 0.0
            or fraction != REFERENCE_FLOOR_FRACTION
            or floor != REFERENCE_FLOOR_FRACTION * development_mean
            or floor <= 0.0
            or reference != expected_reference
        ):
            raise RuntimeError(
                f"invalid reference-health floor for {scope}/{metric}"
            )


def evaluate_confirmatory_decision(
    table: Mapping[str, Any],
    *,
    reference_health: Mapping[str, Any],
    resource_gates: Mapping[str, bool],
    stability_pass: bool,
    locks: Mapping[str, str],
    status: str = "development_only_non_gated",
    _bootstrap_indices_override: np.ndarray | None = None,
    _bootstrap_weights_override: np.ndarray | None = None,
    _include_score_table: bool = True,
) -> dict[str, Any]:
    """Recompute every preregistered gate from per-seed/bin statistics."""

    _validate_reference_health_contract(reference_health)
    if type(stability_pass) is not bool:
        raise TypeError("stability_pass must be a bool")
    if not isinstance(resource_gates, Mapping):
        raise RuntimeError("resource gates must be a mapping")
    if set(resource_gates) != set(REQUIRED_RESOURCE_GATES):
        raise RuntimeError("resource gate population mismatch")
    if any(
        not isinstance(name, str) or not name or type(value) is not bool
        for name, value in resource_gates.items()
    ):
        raise TypeError("resource gates must be non-empty string/bool pairs")
    seed_ids = _validate_score_table(table, required_predictors=PREDICTORS)
    indices = (
        common_bootstrap_indices(len(seed_ids))
        if _bootstrap_indices_override is None
        else np.asarray(_bootstrap_indices_override, dtype=np.int64)
    )
    if (
        indices.ndim != 2
        or indices.shape[1] != len(seed_ids)
        or indices.shape[0] < 1
        or np.any(indices < 0)
        or np.any(indices >= len(seed_ids))
    ):
        raise ValueError("invalid bootstrap-index override")
    weights = (
        _bootstrap_weight_matrix(indices)
        if _bootstrap_weights_override is None
        else np.asarray(_bootstrap_weights_override, dtype=np.float64)
    )
    if (
        weights.shape != indices.shape
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    ):
        raise ValueError("invalid bootstrap-weight override")
    contrasts: dict[str, Any] = {}
    contrast_vectors: dict[str, np.ndarray] = {}

    def record(name: str, vector: np.ndarray) -> dict[str, Any]:
        values = np.asarray(vector, dtype=np.float64)
        if values.ndim != 1 or values.size != len(seed_ids):
            raise ValueError("contrast vector has invalid shape")
        summary = {
            "per_seed": [float(value) for value in values],
            "mean": float(values.mean()),
        }
        contrasts[name] = summary
        contrast_vectors[name] = values
        return summary

    reference_gate_floors: dict[str, float] = {}
    for scope, metric in REQUIRED_REFERENCE_GATES:
        vector = _reference_vector(
            table, metric=metric, scope=scope, seed_ids=seed_ids
        )
        summary = record(f"reference/{scope}/{metric}", vector)
        floor = float(reference_health[scope][metric]["floor"])
        reference_gate_floors[f"reference/{scope}/{metric}"] = floor

    overall_top = record(
        "overall/candidate_vs_geometric/top1",
        _advantage(
            table,
            first="candidate",
            second="geometric",
            metric="top1_accuracy",
            scope="overall",
            seed_ids=seed_ids,
        ),
    )
    overall_top_closure = record(
        "overall/top1_closure_0.30",
        _closure_contrast(
            table,
            metric="top1_accuracy",
            scope="overall",
            threshold=0.30,
            seed_ids=seed_ids,
        ),
    )

    overall_nll_g = record(
        "overall/candidate_vs_geometric/nll",
        _advantage(
            table,
            first="candidate",
            second="geometric",
            metric="categorical_nll",
            scope="overall",
            seed_ids=seed_ids,
        ),
    )
    overall_nll_u = record(
        "overall/candidate_vs_uniform/nll",
        _advantage(
            table,
            first="candidate",
            second="uniform",
            metric="categorical_nll",
            scope="overall",
            seed_ids=seed_ids,
        ),
    )
    overall_nll_closure = record(
        "overall/nll_closure_0.20",
        _closure_contrast(
            table,
            metric="categorical_nll",
            scope="overall",
            threshold=0.20,
            seed_ids=seed_ids,
        ),
    )
    overall_brier_g = record(
        "overall/candidate_vs_geometric/brier",
        _advantage(
            table,
            first="candidate",
            second="geometric",
            metric="brier",
            scope="overall",
            seed_ids=seed_ids,
        ),
    )
    overall_brier_u = record(
        "overall/candidate_vs_uniform/brier",
        _advantage(
            table,
            first="candidate",
            second="uniform",
            metric="brier",
            scope="overall",
            seed_ids=seed_ids,
        ),
    )
    overall_brier_closure = record(
        "overall/brier_closure_0.15",
        _closure_contrast(
            table,
            metric="brier",
            scope="overall",
            threshold=0.15,
            seed_ids=seed_ids,
        ),
    )

    short_top = record(
        "2-3/candidate_vs_geometric/top1",
        _advantage(
            table,
            first="candidate",
            second="geometric",
            metric="top1_accuracy",
            scope="2-3",
            seed_ids=seed_ids,
        ),
    )
    medium_top = record(
        "4-5/candidate_vs_geometric/top1",
        _advantage(
            table,
            first="candidate",
            second="geometric",
            metric="top1_accuracy",
            scope="4-5",
            seed_ids=seed_ids,
        ),
    )
    medium_top_closure = record(
        "4-5/top1_closure_0.20",
        _closure_contrast(
            table,
            metric="top1_accuracy",
            scope="4-5",
            threshold=0.20,
            seed_ids=seed_ids,
        ),
    )
    long_top = record(
        "6+/candidate_vs_geometric/top1",
        _advantage(
            table,
            first="candidate",
            second="geometric",
            metric="top1_accuracy",
            scope="6+",
            seed_ids=seed_ids,
        ),
    )
    long_top_closure = record(
        "6+/top1_closure_0.40",
        _closure_contrast(
            table,
            metric="top1_accuracy",
            scope="6+",
            threshold=0.40,
            seed_ids=seed_ids,
        ),
    )
    long_old = record(
        "6+/candidate_vs_old_i1/top1",
        _advantage(
            table,
            first="candidate",
            second="old_i1",
            metric="top1_accuracy",
            scope="6+",
            seed_ids=seed_ids,
        ),
    )
    long_nll_u = record(
        "6+/candidate_vs_uniform/nll",
        _advantage(
            table,
            first="candidate",
            second="uniform",
            metric="categorical_nll",
            scope="6+",
            seed_ids=seed_ids,
        ),
    )
    long_nll_closure = record(
        "6+/nll_closure_0.20",
        _closure_contrast(
            table,
            metric="categorical_nll",
            scope="6+",
            threshold=0.20,
            seed_ids=seed_ids,
        ),
    )
    long_brier_u = record(
        "6+/candidate_vs_uniform/brier",
        _advantage(
            table,
            first="candidate",
            second="uniform",
            metric="brier",
            scope="6+",
            seed_ids=seed_ids,
        ),
    )
    long_brier_closure = record(
        "6+/brier_closure_0.15",
        _closure_contrast(
            table,
            metric="brier",
            scope="6+",
            threshold=0.15,
            seed_ids=seed_ids,
        ),
    )
    position_g = record(
        "overall/candidate_vs_geometric/position_error",
        _advantage(
            table,
            first="candidate",
            second="geometric",
            metric="argmax_position_error",
            scope="overall",
            seed_ids=seed_ids,
        ),
    )
    position_i = record(
        "overall/candidate_vs_old_i1/position_error",
        _advantage(
            table,
            first="candidate",
            second="old_i1",
            metric="argmax_position_error",
            scope="overall",
            seed_ids=seed_ids,
        ),
    )
    candidate_long = _scope_values(
        table,
        predictor="candidate",
        metric="top1_accuracy",
        scope="6+",
        seed_ids=seed_ids,
    )
    old_long = _scope_values(
        table,
        predictor="old_i1",
        metric="top1_accuracy",
        scope="6+",
        seed_ids=seed_ids,
    )

    ordered_names = list(contrasts)
    vector_matrix = np.column_stack(
        [contrast_vectors[name] for name in ordered_names]
    )
    bootstrap_means = weights @ vector_matrix
    lower_bounds = np.quantile(
        bootstrap_means,
        POWER_CONFIRMATORY_ONE_SIDED_ALPHA,
        axis=0,
        method="linear",
    )
    for index, name in enumerate(ordered_names):
        contrasts[name]["one_sided_99_lower"] = float(lower_bounds[index])
    reference_ok = all(
        contrasts[name]["mean"] >= floor
        and contrasts[name]["one_sided_99_lower"] > 0.0
        for name, floor in reference_gate_floors.items()
    )

    confirmatory_gates = {
        "reference_health": bool(reference_ok),
        "overall_top1_superiority": bool(
            overall_top["one_sided_99_lower"] > 0.0
            and overall_top_closure["one_sided_99_lower"] >= 0.0
        ),
        "overall_nll": bool(
            overall_nll_g["one_sided_99_lower"] > 0.0
            and overall_nll_u["one_sided_99_lower"] > 0.0
            and overall_nll_closure["one_sided_99_lower"] >= 0.0
        ),
        "overall_brier": bool(
            overall_brier_g["one_sided_99_lower"] > 0.0
            and overall_brier_u["one_sided_99_lower"] > 0.0
            and overall_brier_closure["one_sided_99_lower"] >= 0.0
        ),
        "short_top1_protection": bool(
            short_top["one_sided_99_lower"] >= -0.03
        ),
        "medium_top1": bool(
            medium_top["one_sided_99_lower"] > 0.0
            and medium_top_closure["one_sided_99_lower"] >= 0.0
        ),
        "long_top1": bool(
            long_top["one_sided_99_lower"] > 0.0
            and long_top_closure["one_sided_99_lower"] >= 0.0
            and float(candidate_long.mean()) >= float(old_long.mean())
            and long_old["one_sided_99_lower"] >= -0.02
        ),
        "long_nll": bool(
            long_nll_u["one_sided_99_lower"] > 0.0
            and long_nll_closure["one_sided_99_lower"] >= 0.0
        ),
        "long_brier": bool(
            long_brier_u["one_sided_99_lower"] > 0.0
            and long_brier_closure["one_sided_99_lower"] >= 0.0
        ),
        "position_error_superiority": bool(
            position_g["one_sided_99_lower"] > 0.0
        ),
        "position_error_noninferiority": bool(
            position_i["one_sided_99_lower"] >= -0.25
        ),
        "coverage_stability": bool(stability_pass),
    }
    overlap = set(confirmatory_gates) & set(resource_gates)
    if overlap:
        raise RuntimeError(
            f"resource gate names collide with confirmatory gates: {sorted(overlap)}"
        )
    gates = {
        **confirmatory_gates,
        **{name: value for name, value in sorted(resource_gates.items())},
    }
    passed = all(gates.values())
    return {
        "artifact_kind": "stochastic_permanence_formal_decision",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": status,
        "complete_seed_ids": [int(seed) for seed in seed_ids],
        "per_seed_per_bin": deepcopy(table) if _include_score_table else {},
        "bootstrap": {
            "algorithm": "numpy.PCG64",
            "seed": BOOTSTRAP_SEED,
            "samples": int(indices.shape[0]),
            "quantile": POWER_CONFIRMATORY_ONE_SIDED_ALPHA,
            "quantile_method": "linear",
            "common_indices_sha256": hashlib.sha256(
                indices.astype("<i8", copy=False).tobytes(order="C")
            ).hexdigest(),
        },
        "contrasts": contrasts,
        "gates": gates,
        "passed": passed,
        "decision": "pass_all_confirmatory_gates" if passed else "stop",
        "locks": dict(locks),
    }


def new_attempt_ledger() -> dict[str, Any]:
    return {
        "artifact_kind": "stochastic_permanence_attempt_ledger",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "development_only_append_only",
        "maximum_attempts": MAXIMUM_CANDIDATE_ATTEMPTS,
        "maximum_train_replays": MAXIMUM_TRAIN_REPLAYS,
        "model_seed": MODEL_SEED,
        "selection_algorithm": ATTEMPT_SELECTION_ALGORITHM,
        "entries": [],
        "selection": None,
    }


def _require_sha256(value: str | None, *, name: str, optional: bool) -> None:
    if value is None and optional:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{name} must be a lowercase SHA-256")


def _validate_attempt_ledger(ledger: Mapping[str, Any]) -> None:
    if ledger.get("selection_algorithm") != ATTEMPT_SELECTION_ALGORITHM:
        raise RuntimeError("attempt ledger selection algorithm mismatch")
    if ledger.get("model_seed") != MODEL_SEED:
        raise RuntimeError("attempt ledger model seed mismatch")
    if ledger.get("maximum_train_replays") != MAXIMUM_TRAIN_REPLAYS:
        raise RuntimeError("attempt ledger replay policy mismatch")
    if ledger.get("maximum_attempts") != MAXIMUM_CANDIDATE_ATTEMPTS:
        raise RuntimeError("attempt ledger budget mismatch")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("attempt ledger entries are invalid")
    previous: str | None = None
    for sequence, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping) or entry.get("sequence") != sequence:
            raise RuntimeError("attempt ledger sequence is invalid")
        if entry.get("previous_entry_sha256") != previous:
            raise RuntimeError("attempt ledger hash chain is broken")
        payload = {
            key: value for key, value in entry.items() if key != "entry_sha256"
        }
        digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if entry.get("entry_sha256") != digest:
            raise RuntimeError("attempt ledger entry digest mismatch")
        previous = digest


def append_attempt(
    ledger: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
    train_started: bool,
    development_metrics_viewed: bool,
    terminal_status: str,
    terminal_reason: str,
    source_lock_sha256: str | None,
    train_artifact_sha256: str | None,
    train_replays: int,
    development_result: Mapping[str, Any] | None,
    persistent_bytes: int | None,
    mac_per_step: int | None,
) -> dict[str, Any]:
    updated = deepcopy(dict(ledger))
    _validate_attempt_ledger(updated)
    if updated.get("selection") is not None:
        raise RuntimeError("cannot append after final candidate selection")
    if type(train_started) is not bool or type(development_metrics_viewed) is not bool:
        raise TypeError("attempt lifecycle flags must be bool")
    if development_metrics_viewed and not train_started:
        raise RuntimeError("development metrics cannot precede training")
    if not isinstance(terminal_status, str) or not terminal_status:
        raise RuntimeError("terminal_status is required")
    if not isinstance(terminal_reason, str) or not terminal_reason:
        raise RuntimeError("terminal_reason is required")
    if (
        type(train_replays) is not int
        or not 0 <= train_replays <= MAXIMUM_TRAIN_REPLAYS
    ):
        raise RuntimeError("train replay count is outside the locked budget")
    if train_started and train_replays < 1:
        raise RuntimeError("a started training run must record at least one replay")
    if not train_started and train_replays != 0:
        raise RuntimeError("an unstarted run cannot record train replays")
    _require_sha256(
        source_lock_sha256, name="source_lock_sha256", optional=not train_started
    )
    _require_sha256(
        train_artifact_sha256,
        name="train_artifact_sha256",
        optional=not development_metrics_viewed,
    )
    if development_metrics_viewed != (development_result is not None):
        raise RuntimeError(
            "complete development result is required exactly when metrics were viewed"
        )
    if development_metrics_viewed:
        if not isinstance(development_result, Mapping):
            raise RuntimeError("development result must be a mapping")
        gates = development_result.get("gates")
        if not isinstance(gates, Mapping) or not gates or any(
            type(value) is not bool for value in gates.values()
        ):
            raise RuntimeError("development result lacks strict boolean gates")
        if development_result.get("passed") is not all(gates.values()):
            raise RuntimeError("development result does not equal strict gate AND")
        if type(persistent_bytes) is not int or persistent_bytes < 0:
            raise RuntimeError("development result lacks persistent bytes")
        if type(mac_per_step) is not int or mac_per_step < 0:
            raise RuntimeError("development result lacks MAC/step")
    elif persistent_bytes is not None or mac_per_step is not None:
        raise RuntimeError("resource ranking values require a development result")
    entries = updated["entries"]
    config_raw = canonical_json_bytes(configuration)
    config_id = hashlib.sha256(config_raw).hexdigest()
    counts = bool(train_started or development_metrics_viewed)
    same_configuration = [
        entry for entry in entries if entry["configuration_id"] == config_id
    ]
    if any(bool(entry["counts_toward_budget"]) for entry in same_configuration):
        raise RuntimeError("configuration already consumed its candidate attempt")
    if same_configuration and not counts and any(
        entry["terminal_status"] != "infrastructure_failure"
        for entry in same_configuration
    ):
        raise RuntimeError("only infrastructure failures may retry the same ID")
    if not counts and terminal_status != "infrastructure_failure":
        raise RuntimeError(
            "a non-counting entry must be a pre-training infrastructure failure"
        )
    if counts and sum(bool(entry["counts_toward_budget"]) for entry in entries) >= int(
        updated["maximum_attempts"]
    ):
        raise RuntimeError("candidate attempt budget exhausted")
    stored_result = (
        deepcopy(dict(development_result))
        if development_result is not None
        else None
    )
    result_sha256 = (
        hashlib.sha256(canonical_json_bytes(stored_result)).hexdigest()
        if stored_result is not None
        else None
    )
    entry = {
            "sequence": len(entries) + 1,
            "configuration_id": config_id,
            "configuration": deepcopy(dict(configuration)),
            "model_seed": MODEL_SEED,
            "train_started": bool(train_started),
            "development_metrics_viewed": bool(development_metrics_viewed),
            "counts_toward_budget": counts,
            "terminal_status": terminal_status,
            "terminal_reason": terminal_reason,
            "source_lock_sha256": source_lock_sha256,
            "train_artifact_sha256": train_artifact_sha256,
            "train_replays": train_replays,
            "development_result": stored_result,
            "development_result_sha256": result_sha256,
            "persistent_bytes": persistent_bytes,
            "mac_per_step": mac_per_step,
            "previous_entry_sha256": (
                entries[-1]["entry_sha256"] if entries else None
            ),
        }
    entry["entry_sha256"] = hashlib.sha256(
        canonical_json_bytes(entry)
    ).hexdigest()
    entries.append(entry)
    return updated


def _overall_closure_points(result: Mapping[str, Any]) -> dict[str, float]:
    table = result.get("per_seed_per_bin")
    if not isinstance(table, Mapping):
        raise RuntimeError("development result lacks per-seed/bin statistics")
    seed_ids = _validate_score_table(table, required_predictors=PREDICTORS)
    closures: dict[str, float] = {}
    definitions = {
        "top1": ("top1_accuracy", "geometric", "oracle"),
        "nll": ("categorical_nll", "uniform", "oracle"),
        "brier": ("brier", "uniform", "oracle"),
    }
    for name, (metric, baseline, oracle) in definitions.items():
        gain = _advantage(
            table,
            first="candidate",
            second=baseline,
            metric=metric,
            scope="overall",
            seed_ids=seed_ids,
        ).mean()
        reference = _advantage(
            table,
            first=oracle,
            second=baseline,
            metric=metric,
            scope="overall",
            seed_ids=seed_ids,
        ).mean()
        if not np.isfinite([gain, reference]).all() or reference <= 0.0:
            raise RuntimeError(f"development result has invalid {name} reference gap")
        closures[name] = float(gain / reference)
    return closures


def select_final_attempt(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the one locked, deterministic candidate-selection rule once."""

    updated = deepcopy(dict(ledger))
    _validate_attempt_ledger(updated)
    if updated.get("selection") is not None:
        raise RuntimeError("final candidate selection is already recorded")
    ranked: list[tuple[tuple[float, int, int, str], dict[str, Any]]] = []
    eligible_ids: list[str] = []
    for entry in updated.get("entries", []):
        result = entry.get("development_result")
        if (
            not entry.get("counts_toward_budget")
            or not entry.get("development_metrics_viewed")
            or entry.get("terminal_status") != "completed"
            or not isinstance(result, Mapping)
            or result.get("passed") is not True
        ):
            continue
        result_digest = hashlib.sha256(
            canonical_json_bytes(dict(result))
        ).hexdigest()
        if result_digest != entry.get("development_result_sha256"):
            raise RuntimeError("attempt ledger development result digest mismatch")
        gates = result.get("gates")
        if not isinstance(gates, Mapping) or not gates or not all(
            value is True for value in gates.values()
        ):
            continue
        closures = _overall_closure_points(result)
        minimum_normalized_closure = min(
            closures["top1"] / 0.30,
            closures["nll"] / 0.20,
            closures["brier"] / 0.15,
        )
        config_id = str(entry["configuration_id"])
        eligible_ids.append(config_id)
        ranking_key = (
            -float(minimum_normalized_closure),
            int(entry["persistent_bytes"]),
            int(entry["mac_per_step"]),
            config_id,
        )
        ranked.append(
            (
                ranking_key,
                {
                    "configuration_id": config_id,
                    "minimum_normalized_overall_closure": float(
                        minimum_normalized_closure
                    ),
                    "overall_closures": closures,
                    "persistent_bytes": int(entry["persistent_bytes"]),
                    "mac_per_step": int(entry["mac_per_step"]),
                    "development_result_sha256": entry[
                        "development_result_sha256"
                    ],
                    "source_lock_sha256": entry["source_lock_sha256"],
                    "train_artifact_sha256": entry["train_artifact_sha256"],
                },
            )
        )
    ranked.sort(key=lambda item: item[0])
    winner = ranked[0][1] if ranked else None
    updated["selection"] = {
        "algorithm": ATTEMPT_SELECTION_ALGORITHM,
        "eligible_configuration_ids": sorted(eligible_ids),
        "decision": "candidate_selected" if winner is not None else "protocol_no_go",
        "selected": winner,
    }
    return updated


@dataclass(frozen=True)
class PowerPlanningAssumption:
    name: str
    gate_threshold: float
    target_effect: float
    variance_inflation: float = 1.5


def approximate_required_seed_count(
    reference_values: Sequence[float],
    assumption: PowerPlanningAssumption,
    *,
    power: float = 0.80,
    alpha: float = POWER_CONFIRMATORY_ONE_SIDED_ALPHA,
) -> int:
    """Conservative candidate-independent sizing for a paired contrast.

    The target effect must be strictly beyond the decision boundary; power at
    the boundary itself is alpha, not 0.80.  Formal protocol sizing additionally
    runs the composite bootstrap simulation using this analytic value as a
    starting point.
    """

    if assumption.target_effect <= assumption.gate_threshold:
        raise ValueError("power target must exceed the gate threshold")
    vector = np.asarray(reference_values, dtype=np.float64)
    if vector.size < 2 or not np.all(np.isfinite(vector)):
        raise ValueError("reference values are insufficient for power sizing")
    mean_gap = float(vector.mean())
    if mean_gap <= 0.0:
        raise ValueError("reference gap must be positive")
    contrast_delta = (
        assumption.target_effect - assumption.gate_threshold
    ) * mean_gap
    sigma = max(
        float(vector.std(ddof=1)) * assumption.variance_inflation,
        1e-12,
    )
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha)
    z_power = normal.inv_cdf(power)
    return max(2, int(np.ceil(((z_alpha + z_power) * sigma / contrast_delta) ** 2)))


_POWER_TARGET_EFFECTS: dict[str, dict[str, float]] = {
    "top1_accuracy": {"2-3": 0.45, "4-5": 0.35, "6+": 0.55},
    "categorical_nll": {"2-3": 0.35, "4-5": 0.35, "6+": 0.35},
    "brier": {"2-3": 0.30, "4-5": 0.30, "6+": 0.30},
}
_POWER_SENSITIVITY_GRID = tuple(dict(item) for item in POWER_GRID_SCENARIOS)
_POWER_NULL_SENSITIVITY_GRID = tuple(
    dict(item) for item in POWER_NULL_GRID_SCENARIOS
)


def _monte_carlo_acceptance_contract() -> dict[str, int]:
    return {
        "power_family_size": len(_POWER_SENSITIVITY_GRID),
        "minimum_power_passes_at_locked_trials": (
            POWER_LOCKED_MINIMUM_POWER_PASSES
        ),
        "coverage_family_size": (
            len(_POWER_SENSITIVITY_GRID) * len(POWER_COVERAGE_CONTRAST_NAMES)
        ),
        "minimum_covered_at_locked_trials": POWER_LOCKED_MINIMUM_COVERED,
        "type1_family_size": (
            len(_POWER_NULL_SENSITIVITY_GRID) * len(POWER_TYPE1_CONTRAST_NAMES)
        ),
        "maximum_false_rejections_at_locked_trials": (
            POWER_LOCKED_MAXIMUM_FALSE_REJECTIONS
        ),
    }


def _planning_target_effects(
    table: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], float, dict[str, float]]:
    """Choose candidate-independent targets strictly beyond every gate.

    The fixed closure targets alone need not clear old-I1 non-inferiority.  The
    development reference table is therefore allowed to raise (never lower)
    the planning target before any candidate or secret split exists.
    """

    seed_ids = _validate_score_table(
        table, required_predictors=("oracle", "geometric", "uniform", "old_i1")
    )
    effects = deepcopy(_POWER_TARGET_EFFECTS)
    long_reference = _reference_vector(
        table,
        metric="top1_accuracy",
        scope="6+",
        seed_ids=seed_ids,
    )
    old_advantage_over_geometric = _advantage(
        table,
        first="old_i1",
        second="geometric",
        metric="top1_accuracy",
        scope="6+",
        seed_ids=seed_ids,
    )
    required_long_fraction = float(
        old_advantage_over_geometric.mean() / long_reference.mean()
    )
    long_target = max(
        effects["top1_accuracy"]["6+"], required_long_fraction + 0.10
    )
    if long_target >= 1.0:
        raise RuntimeError("old-I1 long Top-1 leaves no feasible oracle-gap target")
    effects["top1_accuracy"]["6+"] = float(long_target)

    oracle_position_advantage = _advantage(
        table,
        first="oracle",
        second="geometric",
        metric="argmax_position_error",
        scope="overall",
        seed_ids=seed_ids,
    )
    old_position_advantage = _advantage(
        table,
        first="old_i1",
        second="geometric",
        metric="argmax_position_error",
        scope="overall",
        seed_ids=seed_ids,
    )
    # C vs old-I1 has advantage e*(G-O) - (G-I).  The formal boundary is
    # -0.25; add 0.10 of oracle gap as a planning buffer beyond that boundary.
    required_position_fraction = float(
        (old_position_advantage.mean() - 0.25)
        / oracle_position_advantage.mean()
    )
    position_target = max(0.70, required_position_fraction + 0.10)
    if position_target >= 1.0:
        raise RuntimeError("old-I1 position error leaves no feasible oracle-gap target")
    diagnostics = {
        "long_top1_old_i1_boundary_fraction": required_long_fraction,
        "position_old_i1_noninferiority_boundary_fraction": (
            required_position_fraction
        ),
    }
    return effects, float(position_target), diagnostics


def _boundary_effects(table: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    seed_ids = _validate_score_table(
        table, required_predictors=("oracle", "geometric", "uniform", "old_i1")
    )
    top_means = {
        scope: float(
            _reference_vector(
                table,
                metric="top1_accuracy",
                scope=scope,
                seed_ids=seed_ids,
            ).mean()
        )
        for scope in OCCLUSION_BINS
    }
    denominator = top_means["2-3"]
    if denominator <= 0.0:
        short_effect = 0.0
    else:
        short_effect = (
            0.30 * sum(top_means.values())
            - 0.20 * top_means["4-5"]
            - 0.40 * top_means["6+"]
        ) / denominator
    return {
        "top1_accuracy": {
            "2-3": float(short_effect),
            "4-5": 0.20,
            "6+": 0.40,
        },
        "categorical_nll": {scope: 0.20 for scope in OCCLUSION_BINS},
        "brier": {scope: 0.15 for scope in OCCLUSION_BINS},
    }


def _bounded_advantage_calibration(
    reference: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    target_effect: float,
    scenario: Mapping[str, float],
    initialization_sign: float,
    variance_effect: float | None = None,
    target_mean_override: float | None = None,
) -> dict[str, Any]:
    """Calibrate a bounded advantage using feasible-native moment coordinates.

    Preregistered scenarios choose a location inside the feasible covariance
    interval at a fixed reference-relative variance scale.  The reference gap
    may be signed per seed as long as its population mean is positive; an
    oracle or baseline need not dominate on every individual seed.
    Correlation and endpoint-variance fraction are therefore achieved
    diagnostics, not potentially impossible targets.
    The legacy exact-correlation path remains only for focused compatibility
    tests and is not used by the preregistered grid.
    """

    vector = np.asarray(reference, dtype=np.float64)
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all():
        raise RuntimeError("reference-gap vector is invalid")
    if (
        lower.shape != vector.shape
        or upper.shape != vector.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower >= upper)
    ):
        raise RuntimeError("candidate advantage bounds are invalid")
    tolerance = 1e-11
    reference_mean = float(vector.mean())
    reference_sd = float(vector.std(ddof=0))
    feasibility_native = "feasible_covariance_fraction" in scenario
    if (
        target_mean_override is None
        and (
            reference_mean < -tolerance
            or (reference_mean <= tolerance and not feasibility_native)
        )
    ):
        raise RuntimeError("bounded power model requires a positive mean gap")
    initialization_fraction = float(
        scenario["single_initialization_sd_fraction"]
    )
    adjusted_effect = float(target_effect)
    if reference_sd > 0.0 and reference_mean > 0.0:
        adjusted_effect += (
            float(initialization_sign)
            * initialization_fraction
            * reference_sd
            / reference_mean
        )
    target_mean = (
        adjusted_effect * reference_mean
        if target_mean_override is None
        else float(target_mean_override)
    )
    if target_mean_override is not None and reference_mean > tolerance:
        adjusted_effect = target_mean / reference_mean
    lower_mean = float(lower.mean())
    upper_mean = float(upper.mean())
    mean_scale = max(1.0, abs(lower_mean), abs(upper_mean), abs(target_mean))
    if (
        target_mean < lower_mean - tolerance * mean_scale
        or target_mean > upper_mean + tolerance * mean_scale
    ):
        raise RuntimeError("requested mean advantage is outside physical bounds")
    target_mean = min(upper_mean, max(lower_mean, target_mean))
    variance_scale_effect = (
        abs(adjusted_effect)
        if variance_effect is None
        else float(variance_effect)
    )
    if not np.isfinite(variance_scale_effect) or variance_scale_effect < 0.0:
        raise RuntimeError("variance-scale effect is invalid")
    widths = upper - lower
    if reference_sd <= tolerance and not feasibility_native:
        target_sd = 0.0
        target = np.full(vector.shape, target_mean, dtype=np.float64)
        if np.any(target < lower) or np.any(target > upper):
            probability = (target_mean - float(lower.mean())) / float(
                widths.mean()
            )
            target = lower + probability * widths
        return {
            "conditional_advantage_means": target,
            "endpoint_variance_fraction": 0.0,
            "requested_effect": float(target_effect),
            "initialization_adjusted_effect": adjusted_effect,
            "requested_correlation": float(
                scenario["candidate_reference_correlation"]
            ),
            "achieved_correlation": None,
            "correlation_identifiable": False,
            "requested_variance_inflation": float(
                scenario["variance_inflation"]
            ),
            "variance_scale_effect": variance_scale_effect,
            "achieved_variance_inflation": None,
            "variance_identifiable": False,
            "mean_advantage": target_mean,
            "advantage_sd": target_sd,
            "moment_tolerance": 1e-8,
        }

    standardized = (
        (vector - reference_mean) / reference_sd
        if reference_sd > tolerance
        else np.zeros_like(vector)
    )

    def means_for(beta: float) -> np.ndarray:
        shifted_lower = lower - beta * standardized
        shifted_upper = upper - beta * standardized
        span = max(1.0, float(widths.max()))
        low = float(shifted_lower.min() - span)
        high = float(shifted_upper.max() + span)
        # This inner solve controls the absolute cross-moment error.  A
        # 42-step solve was accurate in absolute units but could miss the
        # preregistered covariance-fraction tolerance after division by an
        # exceptionally narrow feasible interval in an outer resample.
        for _ in range(BOUNDED_MOMENT_BISECTION_ITERATIONS):
            middle = (low + high) / 2.0
            means = np.clip(middle + beta * standardized, lower, upper)
            current = float(means.mean())
            if current < target_mean:
                low = middle
            else:
                high = middle
        return np.clip(
            ((low + high) / 2.0) + beta * standardized,
            lower,
            upper,
        )

    def cross_for(beta: float) -> tuple[float, np.ndarray]:
        means = means_for(beta)
        return float(np.mean(vector * means)), means

    def moment_bounds_for(
        beta: float,
    ) -> tuple[float, np.ndarray, float, float]:
        cross, means = cross_for(beta)
        deterministic_second = float(np.mean(means * means))
        endpoint_probability = (means - lower) / widths
        endpoint_second = float(
            np.mean(
                (1.0 - endpoint_probability) * lower * lower
                + endpoint_probability * upper * upper
            )
        )
        return cross, means, deterministic_second, endpoint_second

    beta_span = 100.0 * max(1.0, float(widths.max()))
    beta_low, beta_high = -beta_span, beta_span

    if feasibility_native:
        requested_covariance_fraction = float(
            scenario["feasible_covariance_fraction"]
        )
        requested_variance_multiplier = float(
            scenario["variance_scale_multiplier"]
        )
        if not 0.0 < requested_covariance_fraction < 1.0:
            raise RuntimeError(
                "feasible covariance fraction must be strictly inside (0, 1)"
            )
        if requested_variance_multiplier <= 0.0:
            raise RuntimeError("variance-scale multiplier must be positive")
        requested_sd = (
            requested_variance_multiplier
            * variance_scale_effect
            * reference_sd
        )
        _, _, minimum_second, maximum_second_at_minimum = moment_bounds_for(0.0)
        minimum_sd = float(
            np.sqrt(max(0.0, minimum_second - target_mean * target_mean))
        )
        margin_second = minimum_second + (
            POWER_GRID_MAX_ENDPOINT_VARIANCE_FRACTION
            * (maximum_second_at_minimum - minimum_second)
        )
        maximum_margin_sd = float(
            np.sqrt(max(0.0, margin_second - target_mean * target_mean))
        )
        target_sd = min(max(requested_sd, minimum_sd), maximum_margin_sd)
        target_second = target_mean * target_mean + target_sd * target_sd
        second_scale = max(1.0, abs(target_second))

        def variance_feasible(beta: float) -> bool:
            _, _, deterministic_second, endpoint_second = moment_bounds_for(beta)
            variance_range = endpoint_second - deterministic_second
            if variance_range <= tolerance:
                endpoint_fraction = (
                    0.0
                    if abs(target_second - deterministic_second)
                    <= tolerance * second_scale
                    else float("inf")
                )
            else:
                endpoint_fraction = (
                    target_second - deterministic_second
                ) / variance_range
            return (
                deterministic_second
                <= target_second + tolerance * second_scale
                and endpoint_second
                >= target_second - tolerance * second_scale
                and endpoint_fraction
                <= POWER_GRID_MAX_ENDPOINT_VARIANCE_FRACTION + tolerance
            )

        if not variance_feasible(0.0):
            raise RuntimeError("variance envelope has no feasible center")

        def feasible_beta_boundary(direction: float) -> float:
            feasible_beta = 0.0
            infeasible_beta = direction * beta_span
            if variance_feasible(infeasible_beta):
                return infeasible_beta
            for _ in range(52):
                middle = (feasible_beta + infeasible_beta) / 2.0
                if variance_feasible(middle):
                    feasible_beta = middle
                else:
                    infeasible_beta = middle
            return feasible_beta

        beta_low = feasible_beta_boundary(-1.0)
        beta_high = feasible_beta_boundary(1.0)
        cross_low, _ = cross_for(beta_low)
        cross_high, _ = cross_for(beta_high)
        if cross_low > cross_high:
            cross_low, cross_high = cross_high, cross_low
        zero_covariance_cross = target_mean * reference_mean
        covariance_interval = str(scenario["covariance_interval"])
        if covariance_interval == "full_feasible":
            feasible_cross_low = cross_low
            feasible_cross_high = cross_high
        else:
            raise RuntimeError("unknown feasible covariance interval policy")
        target_cross_moment = feasible_cross_low + requested_covariance_fraction * (
            feasible_cross_high - feasible_cross_low
        )
    else:
        correlation = float(scenario["candidate_reference_correlation"])
        inflation = float(scenario["variance_inflation"])
        if not -1.0 <= correlation <= 1.0 or inflation <= 0.0:
            raise RuntimeError("requested power moments are invalid")
        target_sd = inflation * variance_scale_effect * reference_sd
        if target_sd <= tolerance:
            raise RuntimeError("requested advantage variance is unidentified")
        target_second = target_mean * target_mean + target_sd * target_sd
        target_covariance = correlation * target_sd * reference_sd
        target_cross_moment = target_covariance + target_mean * reference_mean
        cross_low, _ = cross_for(beta_low)
        cross_high, _ = cross_for(beta_high)
        if cross_low > cross_high:
            cross_low, cross_high = cross_high, cross_low

    cross_scale = max(1.0, abs(target_cross_moment))
    if not (
        cross_low - tolerance * cross_scale
        <= target_cross_moment
        <= cross_high + tolerance * cross_scale
    ):
        raise RuntimeError("requested candidate/reference covariance is infeasible")
    interval_width = cross_high - cross_low
    if interval_width <= tolerance * max(1.0, abs(cross_high)):
        cross_moment, means = cross_for(0.0)
    else:
        means = means_for(0.0)
        for _ in range(42):
            beta_middle = (beta_low + beta_high) / 2.0
            cross_middle, means = cross_for(beta_middle)
            if cross_middle < target_cross_moment:
                beta_low = beta_middle
            else:
                beta_high = beta_middle
        cross_moment, means = cross_for((beta_low + beta_high) / 2.0)

    deterministic_second = float(np.mean(means * means))
    endpoint_probability = (means - lower) / widths
    endpoint_second = float(
        np.mean(
            (1.0 - endpoint_probability) * lower * lower
            + endpoint_probability * upper * upper
        )
    )
    variance_range = endpoint_second - deterministic_second
    if variance_range <= tolerance:
        if abs(target_second - deterministic_second) > tolerance * max(
            1.0, abs(target_second)
        ):
            raise RuntimeError("requested advantage variance is infeasible")
        variance_fraction = 0.0
    else:
        variance_numerator = target_second - deterministic_second
        variance_tolerance = 1e-8 * max(
            1.0,
            abs(target_second),
            abs(deterministic_second),
            abs(endpoint_second),
        )
        if abs(variance_numerator) <= variance_tolerance:
            variance_numerator = 0.0
        if abs(variance_numerator - variance_range) <= variance_tolerance:
            variance_numerator = variance_range
        variance_fraction = variance_numerator / variance_range
        if not -1e-8 <= variance_fraction <= 1.0 + 1e-8:
            raise RuntimeError(
                "requested advantage variance is infeasible: "
                f"target_second={target_second:.6g}, "
                f"deterministic_second={deterministic_second:.6g}, "
                f"endpoint_second={endpoint_second:.6g}"
            )
        variance_fraction = min(1.0, max(0.0, variance_fraction))

    achieved_second = (
        deterministic_second
        + variance_fraction * (endpoint_second - deterministic_second)
    )
    achieved_variance = max(0.0, achieved_second - target_mean * target_mean)
    achieved_sd = float(np.sqrt(achieved_variance))
    achieved_covariance = cross_moment - target_mean * reference_mean
    achieved_correlation = (
        achieved_covariance / (achieved_sd * reference_sd)
        if achieved_sd > tolerance and reference_sd > tolerance
        else None
    )
    inflation_denominator = variance_scale_effect * reference_sd
    achieved_inflation = (
        achieved_sd / inflation_denominator
        if inflation_denominator > tolerance
        else None
    )
    moment_tolerance = 1e-6
    mean_error = abs(float(means.mean()) - target_mean)
    if mean_error > moment_tolerance * max(1.0, abs(target_mean)):
        raise RuntimeError(
            "bounded advantage moment calibration did not converge: "
            f"mean_error={mean_error:.3e}"
        )
    if feasibility_native:
        feasible_width = feasible_cross_high - feasible_cross_low
        achieved_covariance_fraction = (
            requested_covariance_fraction
            if feasible_width <= tolerance * max(1.0, abs(feasible_cross_high))
            else (cross_moment - feasible_cross_low) / feasible_width
        )
        if (
            abs(
                achieved_covariance_fraction
                - requested_covariance_fraction
            )
            > moment_tolerance
        ):
            raise RuntimeError(
                "feasibility-native covariance calibration did not converge: "
                f"requested={requested_covariance_fraction:.12g}, "
                f"achieved={achieved_covariance_fraction:.12g}, "
                f"interval_width={feasible_width:.12g}, "
                f"cross_error={cross_moment - target_cross_moment:.12g}"
            )
        target_variance_multiplier = (
            None
            if inflation_denominator <= tolerance
            else float(target_sd / inflation_denominator)
        )
        if (
            target_variance_multiplier is not None
            and achieved_inflation is not None
            and abs(achieved_inflation - target_variance_multiplier)
            > moment_tolerance
        ):
            raise RuntimeError(
                "variance-envelope calibration did not converge"
            )
        return {
            "conditional_advantage_means": means,
            "endpoint_variance_fraction": float(variance_fraction),
            "requested_variance_scale_multiplier": (
                requested_variance_multiplier
            ),
            "target_variance_scale_multiplier": (
                target_variance_multiplier
            ),
            "variance_floor_applied": bool(
                target_sd > requested_sd + moment_tolerance
            ),
            "variance_ceiling_applied": bool(
                target_sd < requested_sd - moment_tolerance
            ),
            "minimum_feasible_sd": minimum_sd,
            "maximum_margin_sd": maximum_margin_sd,
            "requested_effect": float(target_effect),
            "initialization_adjusted_effect": adjusted_effect,
            "covariance_interval": covariance_interval,
            "requested_feasible_covariance_fraction": (
                requested_covariance_fraction
            ),
            "achieved_feasible_covariance_fraction": float(
                achieved_covariance_fraction
            ),
            "feasible_covariance_lower": float(
                feasible_cross_low - zero_covariance_cross
            ),
            "feasible_covariance_upper": float(
                feasible_cross_high - zero_covariance_cross
            ),
            "covariance_interval_degenerate": bool(
                feasible_width <= tolerance * max(1.0, abs(feasible_cross_high))
            ),
            "achieved_covariance": float(achieved_covariance),
            "requested_correlation": None,
            "achieved_correlation": (
                None
                if achieved_correlation is None
                else float(achieved_correlation)
            ),
            "correlation_identifiable": achieved_correlation is not None,
            "requested_variance_inflation": None,
            "variance_scale_effect": variance_scale_effect,
            "achieved_variance_inflation": (
                None
                if achieved_inflation is None
                else float(achieved_inflation)
            ),
            "variance_identifiable": achieved_inflation is not None,
            "mean_advantage": target_mean,
            "advantage_sd": achieved_sd,
            "moment_tolerance": moment_tolerance,
        }
    assert achieved_correlation is not None
    assert achieved_inflation is not None
    if (
        abs(achieved_correlation - correlation) > moment_tolerance
        or abs(achieved_inflation - inflation) > moment_tolerance
    ):
        raise RuntimeError(
            "bounded advantage moment calibration did not converge: "
            f"correlation_error={abs(achieved_correlation - correlation):.3e}, "
            f"inflation_error={abs(achieved_inflation - inflation):.3e}"
        )
    return {
        "conditional_advantage_means": means,
        "endpoint_variance_fraction": float(variance_fraction),
        "requested_effect": float(target_effect),
        "initialization_adjusted_effect": adjusted_effect,
        "requested_correlation": correlation,
        "achieved_correlation": float(achieved_correlation),
        "correlation_identifiable": True,
        "requested_variance_inflation": inflation,
        "variance_scale_effect": variance_scale_effect,
        "achieved_variance_inflation": float(achieved_inflation),
        "variance_identifiable": True,
        "mean_advantage": target_mean,
        "advantage_sd": achieved_sd,
        "moment_tolerance": moment_tolerance,
    }


def _sample_bounded_advantage(
    calibration: Mapping[str, Any],
    *,
    source_index: int,
    lower_bound: float,
    upper_bound: float,
    generator: np.random.Generator,
    include_independent_innovation: bool,
) -> float:
    mean = float(calibration["conditional_advantage_means"][source_index])
    if not include_independent_innovation:
        return mean
    variance_fraction = float(calibration["endpoint_variance_fraction"])
    if variance_fraction <= 0.0:
        return mean
    probability = (mean - lower_bound) / (upper_bound - lower_bound)
    endpoint = (
        float(upper_bound)
        if float(generator.random()) < probability
        else float(lower_bound)
    )
    return mean + float(np.sqrt(variance_fraction)) * (endpoint - mean)


def _simulated_score_table(
    reference_table: Mapping[str, Any],
    *,
    draw: np.ndarray,
    effects: Mapping[str, Mapping[str, float]],
    scenario: Mapping[str, float],
    generator: np.random.Generator,
    initialization_sign: float,
    include_independent_innovation: bool,
    position_effect: float,
    variance_effects: Mapping[str, Mapping[str, float]] | None = None,
    position_variance_effect: float | None = None,
    calibration_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    calibration_audit: dict[str, dict[str, Any]] | None = None,
    distribution_audit: dict[str, dict[str, float | int]] | None = None,
) -> dict[str, Any]:
    source_seed_ids = _validate_score_table(
        reference_table,
        required_predictors=("oracle", "geometric", "uniform", "old_i1"),
    )
    selected = np.asarray(draw, dtype=np.int64)
    if selected.ndim != 1 or selected.size < 2:
        raise ValueError("power simulation requires a one-dimensional seed draw")
    if np.any(selected < 0) or np.any(selected >= len(source_seed_ids)):
        raise ValueError("power simulation seed draw is out of range")
    result: dict[str, Any] = {
        predictor: {
            str(index): deepcopy(reference_table[predictor][source_seed_ids[source]])
            for index, source in enumerate(selected)
        }
        for predictor in ("oracle", "geometric", "uniform", "old_i1")
    }
    result["candidate"] = {}
    cache = calibration_cache if calibration_cache is not None else {}
    tolerance = 1e-11
    reference_vectors = {
        (scope, metric): _reference_vector(
            reference_table,
            metric=metric,
            scope=scope,
            seed_ids=source_seed_ids,
        )
        for scope in OCCLUSION_BINS
        for metric in ("top1_accuracy", "categorical_nll", "brier")
    }
    metric_bounds: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for scope in OCCLUSION_BINS:
        for metric in ("top1_accuracy", "categorical_nll", "brier"):
            baseline = "geometric" if metric == "top1_accuracy" else "uniform"
            baseline_values = _scope_values(
                reference_table,
                predictor=baseline,
                metric=metric,
                scope=scope,
                seed_ids=source_seed_ids,
            )
            if metric == "top1_accuracy":
                metric_bounds[(scope, metric)] = (
                    -baseline_values,
                    1.0 - baseline_values,
                )
            elif metric == "categorical_nll":
                metric_bounds[(scope, metric)] = (
                    baseline_values - MAX_CATEGORICAL_NLL,
                    baseline_values,
                )
            else:
                metric_bounds[(scope, metric)] = (
                    baseline_values - MAX_BRIER,
                    baseline_values,
                )
    position_vectors = {
        scope: _advantage(
            reference_table,
            first="oracle",
            second="geometric",
            metric="argmax_position_error",
            scope=scope,
            seed_ids=source_seed_ids,
        )
        for scope in OCCLUSION_BINS
    }
    position_bounds = {
        scope: (
            _scope_values(
                reference_table,
                predictor="geometric",
                metric="argmax_position_error",
                scope=scope,
                seed_ids=source_seed_ids,
            )
            - MAX_POSITION_ERROR,
            _scope_values(
                reference_table,
                predictor="geometric",
                metric="argmax_position_error",
                scope=scope,
                seed_ids=source_seed_ids,
            ),
        )
        for scope in OCCLUSION_BINS
    }
    def calibration(
        *,
        label: str,
        vector: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        effect: float,
        variance_effect: float | None,
    ) -> dict[str, Any]:
        key = (
            label,
            float(effect),
            None if variance_effect is None else float(variance_effect),
            scenario.get("candidate_reference_correlation"),
            scenario.get("covariance_interval"),
            scenario.get("feasible_covariance_fraction"),
            float(scenario["single_initialization_sd_fraction"]),
            scenario.get("variance_inflation"),
            scenario.get("variance_scale_multiplier"),
            float(initialization_sign),
        )
        if key not in cache:
            try:
                cache[key] = _bounded_advantage_calibration(
                    vector,
                    lower_bounds=bounds[0],
                    upper_bounds=bounds[1],
                    target_effect=effect,
                    scenario=scenario,
                    initialization_sign=initialization_sign,
                    variance_effect=variance_effect,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"bounded calibration failed for {label}, effect={effect}, "
                    f"initialization_sign={initialization_sign}: {error}"
                ) from error
        calibrated = cache[key]
        if calibration_audit is not None:
            calibration_audit[
                f"{label}/init_{int(initialization_sign):+d}"
            ] = {
                key: value
                for key, value in calibrated.items()
                if key != "conditional_advantage_means"
            }
        return calibrated

    for output_index, source_index in enumerate(selected):
        source_seed = source_seed_ids[source_index]
        result["candidate"][str(output_index)] = {}
        for scope in OCCLUSION_BINS:
            candidate_metrics: dict[str, float] = {}
            for metric in ("top1_accuracy", "categorical_nll", "brier"):
                reference = reference_vectors[(scope, metric)]
                effect = float(effects[metric][scope])
                calibrated = calibration(
                    label=f"{scope}/{metric}",
                    vector=reference,
                    bounds=metric_bounds[(scope, metric)],
                    effect=effect,
                    variance_effect=(
                        None
                        if variance_effects is None
                        else float(variance_effects[metric][scope])
                    ),
                )
                lower_bound = float(metric_bounds[(scope, metric)][0][source_index])
                upper_bound = float(metric_bounds[(scope, metric)][1][source_index])
                advantage = _sample_bounded_advantage(
                    calibrated,
                    source_index=source_index,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    generator=generator,
                    include_independent_innovation=include_independent_innovation,
                )
                baseline = "geometric" if metric == "top1_accuracy" else "uniform"
                baseline_value = float(
                    reference_table[baseline][source_seed][scope][metric]
                )
                candidate_value = (
                    baseline_value + advantage
                    if metric == "top1_accuracy"
                    else baseline_value - advantage
                )
                physical_high = (
                    1.0
                    if metric in ("top1_accuracy", "brier")
                    else MAX_CATEGORICAL_NLL
                )
                if not -tolerance <= candidate_value <= physical_high + tolerance:
                    raise RuntimeError("simulated metric left its physical domain")
                candidate_metrics[metric] = float(candidate_value)
                if distribution_audit is not None:
                    audit = distribution_audit.setdefault(
                        metric,
                        {"minimum": candidate_value, "maximum": candidate_value, "count": 0},
                    )
                    audit["minimum"] = min(float(audit["minimum"]), candidate_value)
                    audit["maximum"] = max(float(audit["maximum"]), candidate_value)
                    audit["count"] = int(audit["count"]) + 1

            position_reference = position_vectors[scope]
            calibrated_position = calibration(
                label=f"{scope}/argmax_position_error",
                vector=position_reference,
                bounds=position_bounds[scope],
                effect=float(position_effect),
                variance_effect=position_variance_effect,
            )
            position_lower = float(position_bounds[scope][0][source_index])
            position_upper = float(position_bounds[scope][1][source_index])
            position_advantage = _sample_bounded_advantage(
                calibrated_position,
                source_index=source_index,
                lower_bound=position_lower,
                upper_bound=position_upper,
                generator=generator,
                include_independent_innovation=include_independent_innovation,
            )
            geometric_error = float(
                reference_table["geometric"][source_seed][scope][
                    "argmax_position_error"
                ]
            )
            candidate_error = geometric_error - position_advantage
            if not -tolerance <= candidate_error <= MAX_POSITION_ERROR + tolerance:
                raise RuntimeError("simulated position error left its physical domain")
            candidate_metrics["argmax_position_error"] = float(candidate_error)
            if distribution_audit is not None:
                audit = distribution_audit.setdefault(
                    "argmax_position_error",
                    {"minimum": candidate_error, "maximum": candidate_error, "count": 0},
                )
                audit["minimum"] = min(float(audit["minimum"]), candidate_error)
                audit["maximum"] = max(float(audit["maximum"]), candidate_error)
                audit["count"] = int(audit["count"]) + 1
            result["candidate"][str(output_index)][scope] = candidate_metrics
    return result


def _wilson_interval(
    successes: int, total: int, *, z: float
) -> tuple[float, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("invalid binomial count")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, float(center - half_width)), min(
        1.0, float(center + half_width)
    )


def _resampled_reference_table(
    reference_table: Mapping[str, Any],
    draw: np.ndarray,
) -> dict[str, Any]:
    """Materialize an outer-bootstrap empirical population with unique IDs."""

    source_seed_ids = _validate_score_table(
        reference_table,
        required_predictors=("oracle", "geometric", "uniform", "old_i1"),
    )
    selected = np.asarray(draw, dtype=np.int64)
    if (
        selected.ndim != 1
        or selected.size != len(source_seed_ids)
        or np.any(selected < 0)
        or np.any(selected >= len(source_seed_ids))
    ):
        raise ValueError("outer reference draw is invalid")
    return {
        predictor: {
            str(output_index): deepcopy(
                reference_table[predictor][source_seed_ids[source_index]]
            )
            for output_index, source_index in enumerate(selected)
        }
        for predictor in ("oracle", "geometric", "uniform", "old_i1")
    }


def _grid_construction_audit() -> dict[str, Any]:
    """Describe the fixed grid construction without consulting outcomes."""

    return {
        "construction_reads_power_or_coverage": False,
        "parameterization": (
            "fixed_reference_variance_scale_x_feasible_covariance_"
            "interval_fraction_signed_gap_fixed_model_seed_v3"
        ),
        "alternative_grid_is_fixed_cartesian_product": True,
        "null_grid_is_fixed_cartesian_product": True,
        "alternative_scenario_count": len(_POWER_SENSITIVITY_GRID),
        "null_scenario_count": len(_POWER_NULL_SENSITIVITY_GRID),
        "passed": True,
    }


def _component_boundary_mean_advantage(
    reference_table: Mapping[str, Any],
    *,
    name: str,
    boundary: float,
    calibration_scope: str | None = None,
) -> float:
    spec = _COMPONENT_BOUNDARY_SPECS[name]
    metric = str(spec["metric"])
    scope = (
        str(spec["scope"])
        if calibration_scope is None
        else str(calibration_scope)
    )
    seed_ids = _validate_score_table(
        reference_table,
        required_predictors=("oracle", "geometric", "uniform", "old_i1"),
    )
    reference = _reference_vector(
        reference_table, metric=metric, scope=scope, seed_ids=seed_ids
    )
    reference_mean = float(reference.mean())
    kind = str(spec["kind"])
    if kind == "closure":
        return float(spec["threshold"]) * reference_mean + boundary
    if kind == "baseline":
        offset_mean = 0.0
    elif kind == "geometric_lower":
        geometric = _scope_values(
            reference_table,
            predictor="geometric",
            metric=metric,
            scope=scope,
            seed_ids=seed_ids,
        )
        uniform = _scope_values(
            reference_table,
            predictor="uniform",
            metric=metric,
            scope=scope,
            seed_ids=seed_ids,
        )
        offset_mean = float((geometric - uniform).mean())
    elif kind == "old_i1_top1":
        geometric = _scope_values(
            reference_table,
            predictor="geometric",
            metric=metric,
            scope=scope,
            seed_ids=seed_ids,
        )
        old_i1 = _scope_values(
            reference_table,
            predictor="old_i1",
            metric=metric,
            scope=scope,
            seed_ids=seed_ids,
        )
        offset_mean = float((geometric - old_i1).mean())
    elif kind == "old_i1_position":
        geometric = _scope_values(
            reference_table,
            predictor="geometric",
            metric=metric,
            scope=scope,
            seed_ids=seed_ids,
        )
        old_i1 = _scope_values(
            reference_table,
            predictor="old_i1",
            metric=metric,
            scope=scope,
            seed_ids=seed_ids,
        )
        offset_mean = float((old_i1 - geometric).mean())
    else:
        raise RuntimeError(f"unknown component boundary kind: {kind}")
    return boundary - offset_mean


def _physical_component_boundary_vector(
    reference_table: Mapping[str, Any],
    *,
    draw: np.ndarray,
    scenario: Mapping[str, float],
    name: str,
    boundary: float,
    planning_effects: Mapping[str, Mapping[str, float]],
    planning_position_effect: float,
    generator: np.random.Generator,
    calibration_cache: dict[tuple[Any, ...], dict[str, Any]],
    distribution_audit: dict[str, dict[str, float | int]],
) -> np.ndarray:
    """Generate one binding contrast at its physical component null."""

    spec = _COMPONENT_BOUNDARY_SPECS[name]
    metric = str(spec["metric"])
    requested_scope = str(spec["scope"])
    scopes = OCCLUSION_BINS if requested_scope == "overall" else (requested_scope,)
    source_seed_ids = _validate_score_table(
        reference_table,
        required_predictors=("oracle", "geometric", "uniform", "old_i1"),
    )
    selected = np.asarray(draw, dtype=np.int64)
    contrast_by_scope: list[np.ndarray] = []
    truth_contrast_by_scope: list[np.ndarray] = []
    for scope in scopes:
        target_mean_advantage = _component_boundary_mean_advantage(
            reference_table,
            name=name,
            boundary=boundary,
            calibration_scope=scope,
        )
        reference = _reference_vector(
            reference_table,
            metric=metric,
            scope=scope,
            seed_ids=source_seed_ids,
        )
        reference_mean = float(reference.mean())
        mean_effect = (
            target_mean_advantage / reference_mean
            if reference_mean > 1e-11
            else 0.0
        )
        if metric in ("top1_accuracy", "argmax_position_error"):
            baseline_name = "geometric"
        else:
            baseline_name = "uniform"
        baseline = _scope_values(
            reference_table,
            predictor=baseline_name,
            metric=metric,
            scope=scope,
            seed_ids=source_seed_ids,
        )
        physical_high = POWER_METRIC_PHYSICAL_BOUNDS[metric][1]
        if metric == "top1_accuracy":
            lower, upper = -baseline, physical_high - baseline
        else:
            lower, upper = baseline - physical_high, baseline
        variance_effect = (
            planning_position_effect
            if metric == "argmax_position_error"
            else float(planning_effects[metric][scope])
        )
        key = (
            str(scenario["scenario_name"]),
            name,
            scope,
            mean_effect,
            variance_effect,
        )
        if key not in calibration_cache:
            calibration_cache[key] = _bounded_advantage_calibration(
                reference,
                lower_bounds=lower,
                upper_bounds=upper,
                target_effect=mean_effect,
                scenario=scenario,
                initialization_sign=0.0,
                variance_effect=variance_effect,
                target_mean_override=target_mean_advantage,
            )
        calibrated = calibration_cache[key]
        full_means = np.asarray(calibrated["conditional_advantage_means"])
        means = full_means[selected]
        selected_lower = lower[selected]
        selected_upper = upper[selected]
        fraction = float(calibrated["endpoint_variance_fraction"])
        if fraction <= 0.0:
            advantage = means
        else:
            probabilities = (means - selected_lower) / (
                selected_upper - selected_lower
            )
            endpoints = np.where(
                generator.random(selected.size) < probabilities,
                selected_upper,
                selected_lower,
            )
            advantage = means + np.sqrt(fraction) * (endpoints - means)
        selected_baseline = baseline[selected]
        candidate = (
            selected_baseline + advantage
            if metric == "top1_accuracy"
            else selected_baseline - advantage
        )
        low_bound, high_bound = POWER_METRIC_PHYSICAL_BOUNDS[metric]
        if np.any(candidate < low_bound - 1e-11) or np.any(
            candidate > high_bound + 1e-11
        ):
            raise RuntimeError("physical component null left metric bounds")
        audit = distribution_audit.setdefault(
            metric,
            {
                "minimum": float(candidate.min()),
                "maximum": float(candidate.max()),
                "count": 0,
            },
        )
        audit["minimum"] = min(float(audit["minimum"]), float(candidate.min()))
        audit["maximum"] = max(float(audit["maximum"]), float(candidate.max()))
        audit["count"] = int(audit["count"]) + int(candidate.size)

        kind = str(spec["kind"])
        selected_reference = reference[selected]
        if kind == "closure":
            contrast = advantage - float(spec["threshold"]) * selected_reference
            truth_contrast = (
                full_means - float(spec["threshold"]) * reference
            )
        elif kind == "baseline":
            contrast = advantage
            truth_contrast = full_means
        elif kind == "geometric_lower":
            full_geometric = _scope_values(
                reference_table,
                predictor="geometric",
                metric=metric,
                scope=scope,
                seed_ids=source_seed_ids,
            )
            full_uniform = _scope_values(
                reference_table,
                predictor="uniform",
                metric=metric,
                scope=scope,
                seed_ids=source_seed_ids,
            )
            geometric = full_geometric[selected]
            uniform = full_uniform[selected]
            contrast = geometric - uniform + advantage
            truth_contrast = full_geometric - full_uniform + full_means
        elif kind == "old_i1_top1":
            full_geometric = _scope_values(
                reference_table,
                predictor="geometric",
                metric=metric,
                scope=scope,
                seed_ids=source_seed_ids,
            )
            full_old_i1 = _scope_values(
                reference_table,
                predictor="old_i1",
                metric=metric,
                scope=scope,
                seed_ids=source_seed_ids,
            )
            geometric = full_geometric[selected]
            old_i1 = full_old_i1[selected]
            contrast = geometric - old_i1 + advantage
            truth_contrast = full_geometric - full_old_i1 + full_means
        elif kind == "old_i1_position":
            full_geometric = _scope_values(
                reference_table,
                predictor="geometric",
                metric=metric,
                scope=scope,
                seed_ids=source_seed_ids,
            )
            full_old_i1 = _scope_values(
                reference_table,
                predictor="old_i1",
                metric=metric,
                scope=scope,
                seed_ids=source_seed_ids,
            )
            geometric = full_geometric[selected]
            old_i1 = full_old_i1[selected]
            contrast = old_i1 - geometric + advantage
            truth_contrast = full_old_i1 - full_geometric + full_means
        else:
            raise RuntimeError(f"unknown component boundary kind: {kind}")
        contrast_by_scope.append(contrast)
        truth_contrast_by_scope.append(truth_contrast)
    vector = np.mean(np.stack(contrast_by_scope, axis=0), axis=0)
    truth_error = abs(
        float(np.mean(np.stack(truth_contrast_by_scope, axis=0))) - boundary
    )
    if not np.isfinite(vector).all() or truth_error > 1e-8:
        raise RuntimeError("component boundary truth calibration is invalid")
    return vector


def simulate_composite_power(
    reference_table: Mapping[str, Any],
    *,
    reference_health: Mapping[str, Any],
    analytic_starting_seed_count: int,
    trials: int = POWER_SIMULATION_TRIALS,
    bootstrap_samples: int = POWER_BOOTSTRAP_SAMPLES,
    target_effects: Mapping[str, Mapping[str, float]] | None = None,
    position_target_effect: float | None = None,
) -> dict[str, Any]:
    """Run candidate-independent sensitivity simulations through the gate code.

    Candidate sufficient statistics are synthesized only from baseline/oracle
    seed variation and preregistered effect assumptions.  No candidate output
    is accepted by this function.
    """

    if trials < 4:
        raise ValueError("at least four power trials are required")
    recommended_seed_count = max(
        2,
        int(np.ceil(analytic_starting_seed_count * POWER_SEED_SAFETY_MULTIPLIER)),
    )
    source_seed_ids = _validate_score_table(
        reference_table,
        required_predictors=("oracle", "geometric", "uniform", "old_i1"),
    )
    indices = _power_bootstrap_indices(recommended_seed_count, bootstrap_samples)
    truth_indices = np.arange(len(source_seed_ids), dtype=np.int64).reshape(1, -1)
    bootstrap_weights = _bootstrap_weight_matrix(indices)
    truth_weights = _bootstrap_weight_matrix(truth_indices)
    derived_effects, derived_position_effect, target_diagnostics = (
        _planning_target_effects(reference_table)
    )
    planning_effects = (
        derived_effects
        if target_effects is None
        else {
            metric: {scope: float(value) for scope, value in scopes.items()}
            for metric, scopes in target_effects.items()
        }
    )
    planning_position_effect = (
        derived_position_effect
        if position_target_effect is None
        else float(position_target_effect)
    )
    construction_audit = _grid_construction_audit()
    generator = np.random.Generator(np.random.PCG64(POWER_SIMULATION_SEED))
    calibration_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    feasibility_results: list[dict[str, Any]] = []
    full_population_draw = np.arange(len(source_seed_ids), dtype=np.int64)
    for scenario_index, scenario in enumerate(_POWER_SENSITIVITY_GRID):
        audit: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        for initialization_sign in (0.0,):
            try:
                _simulated_score_table(
                    reference_table,
                    draw=full_population_draw,
                    effects=planning_effects,
                    scenario=scenario,
                    generator=generator,
                    initialization_sign=initialization_sign,
                    include_independent_innovation=False,
                    position_effect=planning_position_effect,
                    calibration_cache=calibration_cache,
                    calibration_audit=audit,
                )
            except RuntimeError as error:
                failures.append(str(error))
        variance_fraction_values = [
            float(item["endpoint_variance_fraction"])
            for item in audit.values()
        ]
        maximum_variance_fraction = max(
            variance_fraction_values, default=None
        )
        if (
            maximum_variance_fraction is not None
            and maximum_variance_fraction
            > POWER_GRID_MAX_ENDPOINT_VARIANCE_FRACTION + 1e-12
        ):
            failures.append(
                "endpoint variance-fraction feasibility margin exceeded: "
                f"{maximum_variance_fraction:.12g} > "
                f"{POWER_GRID_MAX_ENDPOINT_VARIANCE_FRACTION:.12g}"
            )
        feasibility_results.append(
            {
                "scenario_id": scenario_index + 1,
                **scenario,
                "bounded_advantage_moment_feasibility": {
                    "passed": not failures,
                    "failures": sorted(set(failures)),
                    "maximum_endpoint_variance_fraction": (
                        maximum_variance_fraction
                    ),
                    "maximum_endpoint_variance_fraction_limit": (
                        POWER_GRID_MAX_ENDPOINT_VARIANCE_FRACTION
                    ),
                    "calibrations": {
                        key: audit[key] for key in sorted(audit)
                    },
                },
            }
        )
    if not construction_audit["passed"] or any(
        not item["bounded_advantage_moment_feasibility"]["passed"]
        for item in feasibility_results
    ):
        gates = {
            "bounded_advantage_moment_scenarios_feasible": False,
            "target_composite_power_familywise_lower_at_least_0.80": False,
            "binding_contrast_type1_familywise_upper_at_most_0.05": False,
            "one_sided_ci_coverage_familywise_lower_at_least_0.90": False,
            "simulated_metrics_within_physical_domain_without_clipping": False,
            "outer_reference_uncertainty_propagated": False,
        }
        return {
            "simulation_design_version": POWER_SIMULATION_DESIGN_VERSION,
            "method": (
                "outer_reference_resampling_same_composite_gate_function_with_"
                "physical_component_nulls"
            ),
            "candidate_generator": POWER_CANDIDATE_GENERATOR,
            "initialization_error_distribution": (
                POWER_MODEL_INITIALIZATION_POLICY
            ),
            "simulation_rng": "numpy.PCG64",
            "simulation_seed": POWER_SIMULATION_SEED,
            "type1_simulation_seed": POWER_TYPE1_SIMULATION_SEED,
            "trials_per_scenario": trials,
            "bootstrap_samples_per_trial": bootstrap_samples,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "formal_runner_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "familywise_alpha": POWER_FAMILYWISE_ALPHA,
            "monte_carlo_acceptance": _monte_carlo_acceptance_contract(),
            "power_seed_safety_multiplier": POWER_SEED_SAFETY_MULTIPLIER,
            "analytic_starting_seed_count": analytic_starting_seed_count,
            "candidate_seed_count_after_safety_multiplier": (
                recommended_seed_count
            ),
            "recommended_validation_seed_count": None,
            "recommended_holdout_seed_count": None,
            "target_effects": deepcopy(planning_effects),
            "position_oracle_gap_target_effect": planning_position_effect,
            "target_derivation": target_diagnostics,
            "sensitivity_grid_policy": {
                "version": POWER_SENSITIVITY_GRID_VERSION,
                "selection_basis": POWER_GRID_SELECTION_BASIS,
                "metric_physical_bounds": deepcopy(POWER_METRIC_PHYSICAL_BOUNDS),
                "endpoint_variance_fraction_limit": (
                    POWER_GRID_MAX_ENDPOINT_VARIANCE_FRACTION
                ),
                "scenarios": [dict(item) for item in _POWER_SENSITIVITY_GRID],
                "null_scenarios": [
                    dict(item) for item in _POWER_NULL_SENSITIVITY_GRID
                ],
                "construction_audit": construction_audit,
            },
            "sensitivity_grid": feasibility_results,
            "null_sensitivity_grid": [],
            "gates": gates,
            "passed": False,
            "decision": "power_no_go",
            "no_go_reason": "requested_bounded_moment_scenario_infeasible",
        }
    scenario_results: list[dict[str, Any]] = []
    all_power_pass = True
    all_coverage_pass = True
    all_distribution_pass = True
    all_outer_uncertainty_pass = True
    power_family_size = len(_POWER_SENSITIVITY_GRID)
    for scenario_index, scenario in enumerate(_POWER_SENSITIVITY_GRID):
        target_gate_counts: dict[str, int] = {}
        target_passes = 0
        coverage_counts: dict[str, int] = {}
        initialization_sign_counts = {"0": 0}
        outer_failures: list[str] = []
        outer_moment_audit = {
            "successful_trials": 0,
            "maximum_endpoint_variance_fraction": 0.0,
            "maximum_feasible_covariance_fraction_error": 0.0,
            "maximum_variance_scale_multiplier_error": 0.0,
        }
        distribution_audit: dict[str, dict[str, float | int]] = {}
        for trial_index in range(trials):
            initialization_sign = 0.0
            initialization_sign_counts["0"] += 1
            outer_draw = generator.integers(
                0,
                len(source_seed_ids),
                size=len(source_seed_ids),
                dtype=np.int64,
            )
            outer_table = _resampled_reference_table(reference_table, outer_draw)
            trial_draw = generator.integers(
                0,
                len(source_seed_ids),
                size=recommended_seed_count,
                dtype=np.int64,
            )
            trial_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
            trial_calibration_audit: dict[str, dict[str, Any]] = {}
            try:
                target_table = _simulated_score_table(
                    outer_table,
                    draw=trial_draw,
                    effects=planning_effects,
                    scenario=scenario,
                    generator=generator,
                    initialization_sign=initialization_sign,
                    include_independent_innovation=True,
                    position_effect=planning_position_effect,
                    calibration_cache=trial_cache,
                    calibration_audit=trial_calibration_audit,
                    distribution_audit=distribution_audit,
                )
                maximum_fraction = max(
                    float(item["endpoint_variance_fraction"])
                    for item in trial_calibration_audit.values()
                )
                if (
                    maximum_fraction
                    > POWER_GRID_MAX_ENDPOINT_VARIANCE_FRACTION + 1e-12
                ):
                    raise RuntimeError(
                        "outer-bootstrap endpoint variance fraction exceeds 0.90"
                    )
                target = evaluate_confirmatory_decision(
                    target_table,
                    reference_health=reference_health,
                    resource_gates={
                        name: True for name in REQUIRED_RESOURCE_GATES
                    },
                    stability_pass=True,
                    locks={},
                    status="candidate_independent_power_simulation",
                    _bootstrap_indices_override=indices,
                    _bootstrap_weights_override=bootstrap_weights,
                    _include_score_table=False,
                )
                truth_table = _simulated_score_table(
                    outer_table,
                    draw=np.arange(len(source_seed_ids), dtype=np.int64),
                    effects=planning_effects,
                    scenario=scenario,
                    generator=generator,
                    initialization_sign=initialization_sign,
                    include_independent_innovation=False,
                    position_effect=planning_position_effect,
                    calibration_cache=trial_cache,
                    calibration_audit=trial_calibration_audit,
                    distribution_audit=distribution_audit,
                )
                truth = evaluate_confirmatory_decision(
                    truth_table,
                    reference_health=reference_health,
                    resource_gates={
                        name: True for name in REQUIRED_RESOURCE_GATES
                    },
                    stability_pass=True,
                    locks={},
                    status="candidate_independent_power_truth",
                    _bootstrap_indices_override=truth_indices,
                    _bootstrap_weights_override=truth_weights,
                    _include_score_table=False,
                )
            except RuntimeError as error:
                outer_failures.append(f"trial_{trial_index + 1}: {error}")
                continue
            outer_moment_audit["successful_trials"] += 1
            outer_moment_audit["maximum_endpoint_variance_fraction"] = max(
                float(outer_moment_audit["maximum_endpoint_variance_fraction"]),
                maximum_fraction,
            )
            for item in trial_calibration_audit.values():
                covariance_error = abs(
                    float(item["achieved_feasible_covariance_fraction"])
                    - float(item["requested_feasible_covariance_fraction"])
                )
                outer_moment_audit[
                    "maximum_feasible_covariance_fraction_error"
                ] = max(
                    float(
                        outer_moment_audit[
                            "maximum_feasible_covariance_fraction_error"
                        ]
                    ),
                    covariance_error,
                )
                if item["target_variance_scale_multiplier"] is not None:
                    variance_error = abs(
                        float(item["achieved_variance_inflation"])
                        - float(item["target_variance_scale_multiplier"])
                    )
                    outer_moment_audit[
                        "maximum_variance_scale_multiplier_error"
                    ] = max(
                        float(
                            outer_moment_audit[
                                "maximum_variance_scale_multiplier_error"
                            ]
                        ),
                        variance_error,
                    )
            target_passes += int(target["passed"])
            for name, passed in target["gates"].items():
                target_gate_counts[name] = target_gate_counts.get(name, 0) + int(
                    passed
                )
            for name, summary in target["contrasts"].items():
                covered = float(summary["one_sided_99_lower"]) <= float(
                    truth["contrasts"][name]["mean"]
                )
                coverage_counts[name] = coverage_counts.get(name, 0) + int(covered)

        power_alpha = POWER_FAMILYWISE_ALPHA / power_family_size
        power_lower = exact_binomial_lower_bound(
            target_passes, trials, alpha=power_alpha
        )
        outer_uncertainty_pass = not outer_failures
        scenario_power_pass = (
            outer_uncertainty_pass and power_lower >= POWER_TARGET_MINIMUM
        )
        gate_power = {
            name: {
                "passes": count,
                "trials": trials,
                "point": count / trials,
            }
            for name, count in sorted(target_gate_counts.items())
        }
        coverage_family_size = len(coverage_counts) * power_family_size
        coverage = {}
        if coverage_family_size:
            coverage_alpha = POWER_FAMILYWISE_ALPHA / coverage_family_size
            for name, count in sorted(coverage_counts.items()):
                lower = exact_binomial_lower_bound(
                    count, trials, alpha=coverage_alpha
                )
                point = count / trials
                coverage[name] = {
                    "covered": count,
                    "trials": trials,
                    "point": point,
                    "familywise_exact_one_sided_lower": lower,
                    "point_minimum": POWER_COVERAGE_POINT_MINIMUM,
                    "familywise_lower_minimum": (
                        POWER_COVERAGE_FAMILYWISE_LOWER_MINIMUM
                    ),
                    "passed": (
                        point >= POWER_COVERAGE_POINT_MINIMUM
                        and lower
                        >= POWER_COVERAGE_FAMILYWISE_LOWER_MINIMUM
                    ),
                }
        scenario_coverage_pass = (
            coverage_family_size > 0
            and not outer_failures
            and all(item["passed"] for item in coverage.values())
        )
        expected_metrics = set(POWER_METRIC_PHYSICAL_BOUNDS)
        scenario_distribution_pass = set(distribution_audit) == expected_metrics
        for metric, bounds in POWER_METRIC_PHYSICAL_BOUNDS.items():
            item = distribution_audit.get(metric, {})
            scenario_distribution_pass &= (
                int(item.get("count", 0)) > 0
                and float(item.get("minimum", float("-inf"))) >= bounds[0] - 1e-11
                and float(item.get("maximum", float("inf"))) <= bounds[1] + 1e-11
            )
        all_power_pass &= scenario_power_pass
        all_coverage_pass &= scenario_coverage_pass
        all_distribution_pass &= scenario_distribution_pass
        all_outer_uncertainty_pass &= outer_uncertainty_pass
        scenario_results.append(
            {
                "scenario_id": scenario_index + 1,
                **scenario,
                "bounded_advantage_moment_feasibility": feasibility_results[
                    scenario_index
                ]["bounded_advantage_moment_feasibility"],
                "outer_reference_uncertainty": {
                    "method": (
                        "nonparametric_outer_bootstrap_of_development_seeds_"
                        "with_per_trial_moment_recalibration"
                    ),
                    "source_seed_count": len(source_seed_ids),
                    "outer_draw_size": len(source_seed_ids),
                    "trials": trials,
                    "failures": outer_failures,
                    "moment_audit": outer_moment_audit,
                    "passed": outer_uncertainty_pass,
                },
                "initialization_sign_counts": initialization_sign_counts,
                "target_composite": {
                    "passes": target_passes,
                    "trials": trials,
                    "point_power": target_passes / trials,
                    "family_size": power_family_size,
                    "familywise_exact_one_sided_lower": power_lower,
                    "minimum": POWER_TARGET_MINIMUM,
                    "gate_power": gate_power,
                    "passed": scenario_power_pass,
                },
                "one_sided_ci_coverage": {
                    "by_contrast": coverage,
                    "family_size": coverage_family_size,
                    "point_minimum": POWER_COVERAGE_POINT_MINIMUM,
                    "familywise_lower_minimum": (
                        POWER_COVERAGE_FAMILYWISE_LOWER_MINIMUM
                    ),
                    "passed": scenario_coverage_pass,
                },
                "physical_metric_distribution": {
                    "bounds": deepcopy(POWER_METRIC_PHYSICAL_BOUNDS),
                    "observed": distribution_audit,
                    "clipping_counts": {
                        metric: 0 for metric in sorted(expected_metrics)
                    },
                    "passed": bool(scenario_distribution_pass),
                },
            }
        )
    type1_generator = np.random.Generator(
        np.random.PCG64(POWER_TYPE1_SIMULATION_SEED)
    )
    type1_family_size = len(_POWER_NULL_SENSITIVITY_GRID) * len(
        _TYPE1_BOUNDARY_TESTS
    )
    null_results: list[dict[str, Any]] = []
    all_type1_pass = True
    for scenario_index, scenario in enumerate(_POWER_NULL_SENSITIVITY_GRID):
        type1_counts = {name: 0 for name in _TYPE1_BOUNDARY_TESTS}
        null_failures: list[str] = []
        null_distribution_audit: dict[str, dict[str, float | int]] = {}
        successful_trials = 0
        null_moment_audit = {
            "maximum_endpoint_variance_fraction": 0.0,
            "maximum_feasible_covariance_fraction_error": 0.0,
            "maximum_variance_scale_multiplier_error": 0.0,
        }
        for trial_index in range(trials):
            outer_draw = type1_generator.integers(
                0,
                len(source_seed_ids),
                size=len(source_seed_ids),
                dtype=np.int64,
            )
            outer_table = _resampled_reference_table(reference_table, outer_draw)
            draw = type1_generator.integers(
                0,
                len(source_seed_ids),
                size=recommended_seed_count,
                dtype=np.int64,
            )
            trial_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
            trial_vectors: dict[str, np.ndarray] = {}
            trial_failed = False
            for name, (boundary, _strict) in _TYPE1_BOUNDARY_TESTS.items():
                try:
                    trial_vectors[name] = _physical_component_boundary_vector(
                        outer_table,
                        draw=draw,
                        scenario=scenario,
                        name=name,
                        boundary=boundary,
                        planning_effects=planning_effects,
                        planning_position_effect=planning_position_effect,
                        generator=type1_generator,
                        calibration_cache=trial_cache,
                        distribution_audit=null_distribution_audit,
                    )
                except RuntimeError as error:
                    null_failures.append(
                        f"trial_{trial_index + 1}/{name}: {error}"
                    )
                    trial_failed = True
                    break
            if trial_failed:
                continue
            for calibration in trial_cache.values():
                endpoint_fraction = float(
                    calibration["endpoint_variance_fraction"]
                )
                if (
                    endpoint_fraction
                    > POWER_GRID_MAX_ENDPOINT_VARIANCE_FRACTION + 1e-12
                ):
                    null_failures.append(
                        f"trial_{trial_index + 1}: null endpoint variance "
                        "fraction exceeds limit"
                    )
                    trial_failed = True
                    break
                null_moment_audit[
                    "maximum_endpoint_variance_fraction"
                ] = max(
                    float(
                        null_moment_audit[
                            "maximum_endpoint_variance_fraction"
                        ]
                    ),
                    endpoint_fraction,
                )
                null_moment_audit[
                    "maximum_feasible_covariance_fraction_error"
                ] = max(
                    float(
                        null_moment_audit[
                            "maximum_feasible_covariance_fraction_error"
                        ]
                    ),
                    abs(
                        float(
                            calibration[
                                "achieved_feasible_covariance_fraction"
                            ]
                        )
                        - float(
                            calibration[
                                "requested_feasible_covariance_fraction"
                            ]
                        )
                    ),
                )
                if calibration["target_variance_scale_multiplier"] is not None:
                    null_moment_audit[
                        "maximum_variance_scale_multiplier_error"
                    ] = max(
                        float(
                            null_moment_audit[
                                "maximum_variance_scale_multiplier_error"
                            ]
                        ),
                        abs(
                            float(calibration["achieved_variance_inflation"])
                            - float(
                                calibration[
                                    "target_variance_scale_multiplier"
                                ]
                            )
                        ),
                    )
            if trial_failed:
                continue
            successful_trials += 1
            for name, (boundary, strict) in _TYPE1_BOUNDARY_TESTS.items():
                lower = float(
                    np.quantile(
                        bootstrap_weights @ trial_vectors[name],
                        POWER_CONFIRMATORY_ONE_SIDED_ALPHA,
                        method="linear",
                    )
                )
                rejected = lower > boundary if strict else lower >= boundary
                type1_counts[name] += int(rejected)

        type1_by_contrast: dict[str, dict[str, Any]] = {}
        for name, count in sorted(type1_counts.items()):
            upper = exact_binomial_upper_bound(
                count,
                trials,
                alpha=POWER_FAMILYWISE_ALPHA / type1_family_size,
            )
            type1_by_contrast[name] = {
                "false_rejections": count,
                "trials": trials,
                "point_type1_error": count / trials,
                "familywise_exact_one_sided_upper": upper,
                "passed": upper <= POWER_TYPE1_MAXIMUM,
            }
        expected_metrics = set(POWER_METRIC_PHYSICAL_BOUNDS)
        null_distribution_pass = (
            set(null_distribution_audit) == expected_metrics
        )
        for metric, bounds in POWER_METRIC_PHYSICAL_BOUNDS.items():
            item = null_distribution_audit.get(metric, {})
            null_distribution_pass &= (
                int(item.get("count", 0)) > 0
                and float(item.get("minimum", float("-inf")))
                >= bounds[0] - 1e-11
                and float(item.get("maximum", float("inf")))
                <= bounds[1] + 1e-11
            )
        null_pass = (
            not null_failures
            and successful_trials == trials
            and null_distribution_pass
            and all(item["passed"] for item in type1_by_contrast.values())
        )
        all_type1_pass &= null_pass
        all_distribution_pass &= null_distribution_pass
        all_outer_uncertainty_pass &= not null_failures
        null_results.append(
            {
                "scenario_id": scenario_index + 1,
                **scenario,
                "method": (
                    "outer_reference_bootstrap_then_bounded_physical_"
                    "component_boundary_recalibration"
                ),
                "simulation_seed": POWER_TYPE1_SIMULATION_SEED,
                "outer_source_seed_count": len(source_seed_ids),
                "outer_draw_size": len(source_seed_ids),
                "successful_trials": successful_trials,
                "outer_moment_audit": null_moment_audit,
                "family_size": type1_family_size,
                "boundary_tests": type1_by_contrast,
                "failures": null_failures,
                "physical_metric_distribution": {
                    "bounds": deepcopy(POWER_METRIC_PHYSICAL_BOUNDS),
                    "observed": null_distribution_audit,
                    "clipping_counts": {
                        metric: 0 for metric in sorted(expected_metrics)
                    },
                    "passed": bool(null_distribution_pass),
                },
                "passed": null_pass,
            }
        )
    gates = {
        "bounded_advantage_moment_scenarios_feasible": all(
            item["bounded_advantage_moment_feasibility"]["passed"]
            for item in feasibility_results
        ),
        "target_composite_power_familywise_lower_at_least_0.80": all_power_pass,
        "binding_contrast_type1_familywise_upper_at_most_0.05": all_type1_pass,
        "one_sided_ci_coverage_familywise_lower_at_least_0.90": (
            all_coverage_pass
        ),
        "simulated_metrics_within_physical_domain_without_clipping": (
            all_distribution_pass
        ),
        "outer_reference_uncertainty_propagated": all_outer_uncertainty_pass,
    }
    passed = all(gates.values())
    return {
        "simulation_design_version": POWER_SIMULATION_DESIGN_VERSION,
        "method": (
            "outer_reference_resampling_same_composite_gate_function_with_"
            "physical_component_nulls"
        ),
        "candidate_generator": POWER_CANDIDATE_GENERATOR,
        "initialization_error_distribution": POWER_MODEL_INITIALIZATION_POLICY,
        "simulation_rng": "numpy.PCG64",
        "simulation_seed": POWER_SIMULATION_SEED,
        "type1_simulation_seed": POWER_TYPE1_SIMULATION_SEED,
        "trials_per_scenario": trials,
        "bootstrap_samples_per_trial": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "formal_runner_bootstrap_samples": BOOTSTRAP_SAMPLES,
        "familywise_alpha": POWER_FAMILYWISE_ALPHA,
        "monte_carlo_acceptance": _monte_carlo_acceptance_contract(),
        "power_seed_safety_multiplier": POWER_SEED_SAFETY_MULTIPLIER,
        "analytic_starting_seed_count": analytic_starting_seed_count,
        "candidate_seed_count_after_safety_multiplier": recommended_seed_count,
        "recommended_validation_seed_count": (
            recommended_seed_count if passed else None
        ),
        "recommended_holdout_seed_count": (
            recommended_seed_count if passed else None
        ),
        "target_effects": deepcopy(planning_effects),
        "position_oracle_gap_target_effect": planning_position_effect,
        "target_derivation": target_diagnostics,
        "sensitivity_grid_policy": {
            "version": POWER_SENSITIVITY_GRID_VERSION,
            "selection_basis": POWER_GRID_SELECTION_BASIS,
            "metric_physical_bounds": deepcopy(POWER_METRIC_PHYSICAL_BOUNDS),
            "endpoint_variance_fraction_limit": (
                POWER_GRID_MAX_ENDPOINT_VARIANCE_FRACTION
            ),
            "scenarios": [dict(item) for item in _POWER_SENSITIVITY_GRID],
            "null_scenarios": [
                dict(item) for item in _POWER_NULL_SENSITIVITY_GRID
            ],
            "construction_audit": construction_audit,
        },
        "sensitivity_grid": scenario_results,
        "null_sensitivity_grid": null_results,
        "gates": gates,
        "passed": passed,
        "decision": "power_go" if passed else "power_no_go",
    }


def build_reference_health_power_artifact(
    table: Mapping[str, Any],
    *,
    registry_selection_digest_sha256: str,
    provenance: Mapping[str, Any],
    power_simulation_trials: int = POWER_SIMULATION_TRIALS,
    power_bootstrap_samples: int = POWER_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    _require_sha256(
        registry_selection_digest_sha256,
        name="registry_selection_digest_sha256",
        optional=False,
    )
    if provenance.get("candidate_source_imported") is not False:
        raise RuntimeError("power provenance does not prove candidate-source isolation")
    if provenance.get("candidate_maps_read") is not False:
        raise RuntimeError("power provenance does not prove candidate-map isolation")
    health, seed_ids = build_reference_health(table)
    target_effects, position_target_effect, target_diagnostics = (
        _planning_target_effects(table)
    )
    planning = {
        "method": "paired_normal_starting_size_before_composite_bootstrap_simulation",
        "alpha_one_sided": POWER_CONFIRMATORY_ONE_SIDED_ALPHA,
        "target_power": 0.80,
        "variance_inflation": 1.5,
        "warning": (
            "A target exactly on a gate boundary cannot have 0.80 power; "
            "targets are fixed above their decision thresholds."
        ),
        "assumptions": {},
    }
    assumptions = (
        PowerPlanningAssumption("overall_top1", 0.30, 0.45),
        PowerPlanningAssumption("overall_nll", 0.20, 0.35),
        PowerPlanningAssumption("overall_brier", 0.15, 0.30),
        PowerPlanningAssumption("medium_top1", 0.20, 0.35),
        PowerPlanningAssumption(
            "long_top1", 0.40, target_effects["top1_accuracy"]["6+"]
        ),
        PowerPlanningAssumption("long_nll", 0.20, 0.35),
        PowerPlanningAssumption("long_brier", 0.15, 0.30),
    )
    lookup = {
        "overall_top1": ("overall", "top1_accuracy"),
        "overall_nll": ("overall", "categorical_nll"),
        "overall_brier": ("overall", "brier"),
        "medium_top1": ("4-5", "top1_accuracy"),
        "long_top1": ("6+", "top1_accuracy"),
        "long_nll": ("6+", "categorical_nll"),
        "long_brier": ("6+", "brier"),
    }
    recommendations: list[int] = []
    for assumption in assumptions:
        scope, metric = lookup[assumption.name]
        values = health[scope][metric]["per_seed"]
        count = approximate_required_seed_count(values, assumption)
        planning["assumptions"][assumption.name] = {
            "scope": scope,
            "metric": metric,
            "gate_threshold": assumption.gate_threshold,
            "target_effect": assumption.target_effect,
            "variance_inflation": assumption.variance_inflation,
            "analytic_starting_seed_count": count,
        }
        recommendations.append(count)
    planning["analytic_starting_seed_count_maximum"] = max(recommendations)
    planning["target_derivation"] = target_diagnostics
    planning["composite_simulation"] = simulate_composite_power(
        table,
        reference_health=health,
        analytic_starting_seed_count=max(recommendations),
        trials=power_simulation_trials,
        bootstrap_samples=power_bootstrap_samples,
        target_effects=target_effects,
        position_target_effect=position_target_effect,
    )
    planning["composite_bootstrap_simulation_required_before_secret_commitment"] = False
    reference_gate_details = {
        f"{scope}/{metric}": {
            "mean_at_least_development_floor": bool(
                health[scope][metric]["mean"] >= health[scope][metric]["floor"]
            ),
            "one_sided_99_lower_positive": bool(
                health[scope][metric]["one_sided_99_lower"] > 0.0
            ),
        }
        for scope, metric in REQUIRED_REFERENCE_GATES
    }
    reference_health_passed = all(
        all(detail.values()) for detail in reference_gate_details.values()
    )
    phase0_gates = {
        "candidate_independent_reference_health": reference_health_passed,
        "candidate_independent_composite_power": bool(
            planning["composite_simulation"]["passed"]
        ),
    }
    return {
        "artifact_kind": "stochastic_permanence_reference_health_power",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "development_only_non_gated",
        "candidate_maps_read": False,
        "registry_selection_digest_sha256": registry_selection_digest_sha256,
        "complete_seed_ids": [int(seed) for seed in seed_ids],
        "per_seed_per_bin": deepcopy(table),
        "reference_health": health,
        "reference_health_decision": {
            "required": reference_gate_details,
            "passed": reference_health_passed,
        },
        "power_design": planning,
        "phase0_gates": phase0_gates,
        "passed": all(phase0_gates.values()),
        "decision": (
            "phase0_go" if all(phase0_gates.values()) else "phase0_no_go"
        ),
        "provenance": dict(provenance),
    }
