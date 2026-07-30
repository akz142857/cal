"""Generate the development-only Phase-0 reference-health/power artifact.

The runner accepts only the model-blind development registry, recomputes the
oracle/geometric/uniform/old-I1 sufficient statistics, and never constructs or
reads a new candidate.  It does not generate validation or holdout seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Any, Sequence

import numpy as np

from cal.evaluation.permanence_forward_benchmark import run_benchmark
from cal.evaluation.permanence_seed_registry import (
    coverage_contract,
    reproduce_registry_artifact,
)
from cal.evaluation.permanence_turn_probability_scan import (
    validate_scan_artifact,
)
from cal.evaluation.stochastic_permanence_artifacts import (
    sha256_path,
    source_lock,
    write_canonical_artifact,
)
from cal.evaluation.stochastic_permanence_benchmark import (
    POWER_BOOTSTRAP_SAMPLES,
    POWER_SIMULATION_TRIALS,
    build_reference_health_power_artifact,
    score_table_from_episode_binned,
    validate_complete_seed_population,
)
from cal.evaluation.stochastic_permanence_custody import (
    validate_disjoint_seed_sets,
)


DEFAULT_REGISTRY = Path(
    "experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY.json"
)
DEFAULT_OUTPUT = Path(
    "experiments/V2_I1_P1_PHASE0_REFERENCE_HEALTH_POWER_DEVELOPMENT.json"
)


def _load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("status") != "development_only_non_gated":
        raise RuntimeError("Phase 0 accepts only a development-only registry")
    if registry.get("model_metrics_read") is not False:
        raise RuntimeError("registry does not attest model_metrics_read=false")
    if registry.get("coverage_contract") != coverage_contract():
        raise RuntimeError("registry coverage contract is stale or incomplete")
    digest = str(registry.get("selection_digest_sha256", ""))
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError("registry lacks a selection SHA-256")
    if "selected_turn_probability" not in registry:
        raise RuntimeError("registry does not bind a turn probability")
    validate_disjoint_seed_sets(
        {
            "train": registry["train_seeds"],
            "development": registry["evaluation_seeds"],
        }
    )
    _validate_registry_population_and_digest(registry)
    if reproduce_registry_artifact(registry) != registry:
        raise RuntimeError("registry selection digest or reproduction mismatch")
    return registry


def _validate_registry_population_and_digest(registry: dict[str, Any]) -> None:
    train = registry.get("train_seeds")
    evaluation = registry.get("evaluation_seeds")
    if not isinstance(train, list) or not isinstance(evaluation, list):
        raise RuntimeError("registry seed populations are missing")
    combined = [*train, *evaluation]
    if (
        len(train) < 1
        or len(evaluation) < 2
        or any(type(seed) is not int or seed < 0 for seed in combined)
        or len(set(combined)) != len(combined)
        or combined != sorted(combined)
    ):
        raise RuntimeError("registry seed population is incomplete or non-canonical")
    requested = registry.get("requested_counts")
    if requested != {"train": len(train), "evaluation": len(evaluation)}:
        raise RuntimeError("registry requested counts do not match seed populations")
    candidate_range = registry.get("candidate_range")
    if not isinstance(candidate_range, dict):
        raise RuntimeError("registry candidate range is missing")
    start = candidate_range.get("start_inclusive")
    stop = candidate_range.get("stop_exclusive")
    last = candidate_range.get("last_scanned_inclusive")
    if (
        type(start) is not int
        or type(stop) is not int
        or type(last) is not int
        or not start <= last < stop
        or last != combined[-1]
    ):
        raise RuntimeError("registry candidate range is invalid")
    rejected_count = registry.get("rejected_candidate_count")
    if (
        type(rejected_count) is not int
        or rejected_count != last - start + 1 - len(combined)
    ):
        raise RuntimeError("registry scan-prefix count is inconsistent")
    selected_probability = registry["selected_turn_probability"]
    allowed_probabilities = registry["coverage_contract"][
        "coverage_turn_probabilities"
    ]
    if selected_probability not in allowed_probabilities:
        raise RuntimeError("selected turn probability is outside the coverage contract")


def run_phase0(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    workspace_root: str | Path | None = None,
    power_simulation_trials: int = POWER_SIMULATION_TRIALS,
    power_bootstrap_samples: int = POWER_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    registry_source = Path(registry_path)
    if not registry_source.is_absolute():
        registry_source = root / registry_source
    registry = _load_registry(registry_source)
    turn_scan_source = (
        root / registry["turn_probability_selection_artifact"]
    ).resolve()
    turn_scan_source.relative_to(root)
    turn_scan = json.loads(turn_scan_source.read_text(encoding="utf-8"))
    validate_scan_artifact(turn_scan, registry)
    report = run_benchmark(
        [int(seed) for seed in registry["train_seeds"]],
        [int(seed) for seed in registry["evaluation_seeds"]],
        steps=int(registry["coverage_contract"]["steps"]),
        warmup=int(registry["coverage_contract"]["warmup"]),
        turn_probability=float(registry["selected_turn_probability"]),
        include_gru=False,
        include_slot=False,
        include_entity_graph=True,
        paired_bootstrap_samples=0,
    )
    episode_binned = report["episode_binned_predictors"]
    uniform = report["leakage_audit"]["baselines"]["uniform_field"][
        "episode_binned_primary"
    ]
    score_table = score_table_from_episode_binned(
        {
            "oracle": episode_binned["belief"],
            "geometric": episode_binned["geometric"],
            "uniform": uniform,
            "old_i1": episode_binned["entity_graph"],
        }
    )
    complete_seed_ids = tuple(
        sorted(int(seed) for seed in registry["evaluation_seeds"])
    )
    validate_complete_seed_population(
        score_table,
        required_predictors=("oracle", "geometric", "uniform", "old_i1"),
        expected_seed_ids=complete_seed_ids,
    )
    plan_path = root / "docs/experiments/V2_I1_STOCHASTIC_PERMANENCE_PLAN.md"
    source_paths = (
        Path(__file__),
        root / "cal/evaluation/stochastic_permanence_benchmark.py",
        root / "cal/evaluation/stochastic_permanence_artifacts.py",
        root / "cal/evaluation/stochastic_permanence_custody.py",
        root / "cal/evaluation/permanence_forward_benchmark.py",
        root / "cal/evaluation/permanence_seed_registry.py",
        root / "cal/evaluation/permanence_turn_probability_scan.py",
        root / "cal/evaluation/randomized_occlusion_world.py",
        root / "cal/evaluation/v2_i1_integration.py",
        root / "cal/evaluation/v2_artifacts.py",
        root / "cal/infra/provenance.py",
        root / "cal/model/integrated_agent.py",
        root / "cal/model/entity_graph.py",
        root / "cal/model/entity_belief_graph.py",
        root / "cal/model/occupancy.py",
        plan_path,
        registry_source,
        turn_scan_source,
        root / "pyproject.toml",
        root / "uv.lock",
    )
    relative_registry = registry_source.relative_to(root).as_posix()
    provenance = {
        "development_only": True,
        "candidate_source_imported": False,
        "candidate_maps_read": False,
        "registry_path": relative_registry,
        "registry_sha256": sha256_path(registry_source),
        "turn_probability_selection_artifact": turn_scan_source.relative_to(
            root
        ).as_posix(),
        "turn_probability_selection_sha256": sha256_path(turn_scan_source),
        "source_lock": source_lock(source_paths, root=root),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "command": (
            "python -m cal.evaluation.stochastic_permanence_phase0 "
            f"--registry {relative_registry} "
            f"--simulation-trials {power_simulation_trials} "
            f"--power-bootstrap-samples {power_bootstrap_samples}"
        ),
        "coverage": {
            "development_seed_count": len(complete_seed_ids),
            "complete_seed_ids": list(complete_seed_ids),
            "all_predictors_same_complete_seed_population": True,
            "required_occlusion_bins": ["2-3", "4-5", "6+"],
        },
    }
    return build_reference_health_power_artifact(
        score_table,
        registry_selection_digest_sha256=registry[
            "selection_digest_sha256"
        ],
        provenance=provenance,
        power_simulation_trials=power_simulation_trials,
        power_bootstrap_samples=power_bootstrap_samples,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--simulation-trials", type=int, default=POWER_SIMULATION_TRIALS
    )
    parser.add_argument(
        "--power-bootstrap-samples", type=int, default=POWER_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    artifact = run_phase0(
        registry_path=arguments.registry,
        power_simulation_trials=arguments.simulation_trials,
        power_bootstrap_samples=arguments.power_bootstrap_samples,
    )
    digest = write_canonical_artifact(
        arguments.output,
        artifact,
        expected_kind="stochastic_permanence_reference_health_power",
        overwrite=arguments.overwrite,
    )
    print(
        json.dumps(
            {
                "passed": artifact["passed"],
                "decision": artifact["decision"],
                "recommended_validation_seed_count": artifact["power_design"][
                    "composite_simulation"
                ]["recommended_validation_seed_count"],
                "recommended_holdout_seed_count": artifact["power_design"][
                    "composite_simulation"
                ]["recommended_holdout_seed_count"],
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
