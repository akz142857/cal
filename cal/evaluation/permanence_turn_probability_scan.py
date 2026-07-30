"""Model-independent hidden-turn difficulty scan for the P1 benchmark.

Only the exact-kernel belief oracle, geometric shortcut, event coverage, and
runner-computed random chance participate.  No learned candidate or neural
baseline is constructed.  The smallest candidate probability satisfying every
predeclared condition is selected, preventing difficulty tuning around the
current entity graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any

from cal.evaluation.permanence_forward_benchmark import run_benchmark
from cal.evaluation.permanence_seed_registry import (
    COVERAGE_TURN_PROBABILITIES,
    coverage_contract,
)

SCAN_VERSION = 1
CANDIDATE_TURN_PROBABILITIES = COVERAGE_TURN_PROBABILITIES
LONG_SHORTCUT_CHANCE_MARGIN_MAX = 0.05
LONG_ORACLE_CHANCE_ADVANTAGE_MIN = 0.15


def selection_contract() -> dict[str, Any]:
    return {
        "candidate_or_learned_model_metrics_must_not_be_read": True,
        "permitted_selection_sources": [
            "environment_event_coverage",
            "runner_any_positive_chance",
            "exact_kernel_belief_oracle",
            "geometric_shortcut",
        ],
        "candidates_in_ascending_order": list(CANDIDATE_TURN_PROBABILITIES),
        "selection": "first_candidate_passing_all_conditions",
        "conditions": {
            "all_evaluation_seeds_have_samples": {
                "operator": "equals",
                "threshold": True,
            },
            "all_evaluation_seeds_have_all_three_bins": {
                "operator": "equals",
                "threshold": True,
            },
            "geometric_6plus_near_chance": {
                "metric": "geometric_top1_minus_chance_top1",
                "operator": "less_than_or_equal",
                "threshold": LONG_SHORTCUT_CHANCE_MARGIN_MAX,
            },
            "oracle_6plus_has_headroom": {
                "metric": "belief_top1_minus_chance_top1",
                "operator": "greater_than_or_equal",
                "threshold": LONG_ORACLE_CHANCE_ADVANTAGE_MIN,
            },
            "oracle_6plus_brier_below_geometric": {
                "metric": "belief_brier_minus_geometric_brier",
                "operator": "less_than",
                "threshold": 0.0,
            },
        },
    }


def evaluate_row(
    turn_probability: float,
    report: dict[str, object],
    evaluation_seed_count: int,
) -> dict[str, Any]:
    belief = report["episode_binned_predictors"]["belief"]
    geometric = report["episode_binned_predictors"]["geometric"]
    belief_long = belief["by_occlusion_length"]["6+"]
    geometric_long = geometric["by_occlusion_length"]["6+"]
    chance = belief_long["mean_any_positive_top1_chance"]
    if geometric_long["mean_any_positive_top1_chance"] != chance:
        raise ValueError("belief/geometric rows must use the same runner chance")
    geometric_advantage = geometric_long["top1_accuracy"] - chance
    oracle_advantage = belief_long["top1_accuracy"] - chance
    brier_difference = belief_long["brier"] - geometric_long["brier"]
    conditions = {
        "all_evaluation_seeds_have_samples": report["evaluation_seed_coverage"][
            "all_seeds_have_samples"
        ],
        "all_evaluation_seeds_have_all_three_bins": (
            belief["complete_seed_count"] == evaluation_seed_count
            and geometric["complete_seed_count"] == evaluation_seed_count
        ),
        "geometric_6plus_near_chance": (
            geometric_advantage <= LONG_SHORTCUT_CHANCE_MARGIN_MAX
        ),
        "oracle_6plus_has_headroom": (
            oracle_advantage >= LONG_ORACLE_CHANCE_ADVANTAGE_MIN
        ),
        "oracle_6plus_brier_below_geometric": brier_difference < 0.0,
    }
    contract_condition_names = set(selection_contract()["conditions"])
    if set(conditions) != contract_condition_names:
        raise AssertionError("row conditions drifted from selection contract")
    return {
        "turn_probability": turn_probability,
        "accepted": all(conditions.values()),
        "conditions": conditions,
        "long_6plus": {
            "chance_top1": chance,
            "belief_top1": belief_long["top1_accuracy"],
            "geometric_top1": geometric_long["top1_accuracy"],
            "belief_brier": belief_long["brier"],
            "geometric_brier": geometric_long["brier"],
            "geometric_top1_minus_chance_top1": geometric_advantage,
            "belief_top1_minus_chance_top1": oracle_advantage,
            "belief_brier_minus_geometric_brier": brier_difference,
        },
    }


def validate_seed_registry_for_scan(registry: dict[str, Any]) -> None:
    """Reject registry drift or a split that could invalidate a model-blind scan."""

    if registry.get("status") != "development_only_non_gated":
        raise ValueError("only a development-only non-gated registry is accepted")
    if registry.get("model_metrics_read") is not False:
        raise ValueError("seed registry must attest model_metrics_read=false")
    if registry.get("coverage_contract") != coverage_contract():
        raise ValueError("seed registry coverage contract does not match runner")
    train_seeds = [int(seed) for seed in registry.get("train_seeds", [])]
    evaluation_seeds = [
        int(seed) for seed in registry.get("evaluation_seeds", [])
    ]
    if not train_seeds or not evaluation_seeds:
        raise ValueError("seed registry must contain non-empty train/evaluation splits")
    all_seeds = train_seeds + evaluation_seeds
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("seed registry train/evaluation seeds must be unique")
    digest = registry.get("selection_digest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("seed registry must contain a SHA-256 selection digest")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(
            "seed registry selection digest must be hexadecimal"
        ) from error


def run_scan(train_seeds: list[int], evaluation_seeds: list[int]) -> dict[str, Any]:
    rows = []
    for probability in CANDIDATE_TURN_PROBABILITIES:
        report = run_benchmark(
            train_seeds,
            evaluation_seeds,
            steps=200,
            warmup=12,
            turn_probability=probability,
            include_gru=False,
            include_slot=False,
            include_entity_graph=False,
            paired_bootstrap_samples=0,
        )
        rows.append(evaluate_row(probability, report, len(evaluation_seeds)))
    accepted = [row for row in rows if row["accepted"]]
    return {
        "status": "development_only_non_gated",
        "selection_contract": selection_contract(),
        "rows": rows,
        "selected_turn_probability": (
            accepted[0]["turn_probability"] if accepted else None
        ),
    }


def _scan_digest(artifact_without_digest: dict[str, Any]) -> str:
    payload = json.dumps(
        artifact_without_digest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_scan_artifact(
    artifact: dict[str, Any], registry: dict[str, Any]
) -> None:
    """Validate digest linkage and first-passing selection without rerunning."""

    validate_seed_registry_for_scan(registry)
    if artifact.get("scan_version") != SCAN_VERSION:
        raise ValueError("scan artifact version does not match runner")
    if artifact.get("selection_contract") != selection_contract():
        raise ValueError("scan artifact selection contract does not match runner")
    if artifact.get("seed_registry_selection_digest_sha256") != registry.get(
        "selection_digest_sha256"
    ):
        raise ValueError("scan artifact is linked to a different seed registry")
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise ValueError("scan artifact rows must be a list")
    probabilities = [row.get("turn_probability") for row in rows]
    if probabilities != list(CANDIDATE_TURN_PROBABILITIES):
        raise ValueError("scan artifact rows do not match candidate order")
    condition_names = set(selection_contract()["conditions"])
    for row in rows:
        if set(row.get("conditions", {})) != condition_names:
            raise ValueError("scan artifact row conditions do not match contract")
        if bool(row.get("accepted")) != all(row["conditions"].values()):
            raise ValueError("scan artifact accepted flag disagrees with conditions")
    accepted = [row for row in rows if row["accepted"]]
    expected_selection = accepted[0]["turn_probability"] if accepted else None
    if artifact.get("selected_turn_probability") != expected_selection:
        raise ValueError("scan artifact did not select the first passing candidate")
    registry_selection = registry.get("selected_turn_probability")
    if registry_selection is not None and registry_selection != expected_selection:
        raise ValueError("seed registry selected probability disagrees with scan")
    digest = artifact.get("scan_digest_sha256")
    payload = dict(artifact)
    payload.pop("scan_digest_sha256", None)
    if digest != _scan_digest(payload):
        raise ValueError("scan artifact digest does not match its contents")


def build_scan_artifact(
    registry: dict[str, Any], *, registry_path: str
) -> dict[str, Any]:
    """Run and bind the complete model-independent difficulty scan."""

    validate_seed_registry_for_scan(registry)
    report = run_scan(
        [int(seed) for seed in registry["train_seeds"]],
        [int(seed) for seed in registry["evaluation_seeds"]],
    )
    command = shlex.join(
        [
            "uv",
            "run",
            "python",
            "-m",
            "cal.evaluation.permanence_turn_probability_scan",
            "--seed-registry",
            registry_path,
        ]
    )
    artifact = {
        "scan_version": SCAN_VERSION,
        "status": report["status"],
        "generator": "cal.evaluation.permanence_turn_probability_scan",
        "reproduction_command": command,
        "seed_registry_path": registry_path,
        "seed_registry_selection_digest_sha256": registry[
            "selection_digest_sha256"
        ],
        "selection_contract": report["selection_contract"],
        "rows": report["rows"],
        "selected_turn_probability": report["selected_turn_probability"],
    }
    artifact["scan_digest_sha256"] = _scan_digest(artifact)
    validate_scan_artifact(artifact, registry)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-registry",
        type=Path,
        default=Path("experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY.json"),
    )
    args = parser.parse_args()
    registry = json.loads(args.seed_registry.read_text(encoding="utf-8"))
    report = build_scan_artifact(registry, registry_path=str(args.seed_registry))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
