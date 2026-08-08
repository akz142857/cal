"""Model-blind seed/event coverage registry for the P1 permanence benchmark.

This module never constructs a candidate model and never reads predictor
scores.  It scans candidate seeds in ascending order and accepts the first N
whose environment/event structure satisfies the preregistered coverage rule.
The same algorithm can later be run by a holdout custodian over a secret seed
stream; only the development registry is materialized in this repository.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any

from cal.evaluation.permanence_forward_benchmark import (
    _collect,
    _occlusion_length_bin,
)
from cal.evaluation.randomized_occlusion_world import RandomizedOcclusionWorld
from cal.evaluation.v2_i1_integration import WARMUP

REGISTRY_VERSION = 2
DEFAULT_STEPS = 200
LOCKED_DEVELOPMENT_TRAIN_COUNT = 40
# Amended 2026-08-08 (64 -> 150).  Replacing the top-1 closure reference with
# `belief_free` -- calibrated smearing rather than a point-mass extrapolation --
# shrinks the oracle-versus-reference effect, and the 4-5 occlusion bin has the
# highest per-seed variance of the three.  At 64 seeds its one-sided 99% lower
# bound sat at -0.005 with a mean gap of +0.061: an underpowered bin, not an
# absent effect.  The analytic requirement is ~75 seeds; 150 applies the same
# 2.0 safety factor this program already uses for the confirmatory stage
# (1315 analytic -> 2630 recommended).
LOCKED_PHASE0_EVALUATION_SOURCE_COUNT = 150
COVERAGE_TURN_PROBABILITIES = (0.15, 0.25, 0.35, 0.45, 0.55)
MIN_ACCEPTED_SINGLE_HIDDEN_SAMPLES = 12
MIN_EPISODE_BIN_GROUPS = 2
REQUIRED_OCCLUSION_BINS = ("2-3", "4-5", "6+")


def coverage_contract() -> dict[str, Any]:
    """Return the complete model-independent acceptance contract."""

    return {
        "model_metrics_must_not_be_read": True,
        "scan_order": "ascending_integer_seed",
        "selection": "first_n_accepted_without_manual_substitution",
        "steps": DEFAULT_STEPS,
        "warmup": WARMUP,
        "coverage_turn_probabilities": list(COVERAGE_TURN_PROBABILITIES),
        "accepted_single_hidden_samples_minimum": (
            MIN_ACCEPTED_SINGLE_HIDDEN_SAMPLES
        ),
        "episode_bin_groups_minimum_per_bin": MIN_EPISODE_BIN_GROUPS,
        "required_occlusion_bins": list(REQUIRED_OCCLUSION_BINS),
        "event_filters": [
            "known focus track",
            "complete hidden tracks for every positive",
            "no actor collision",
            "no duplicate hidden positive",
            "single hidden object for duration-labelled primary score",
        ],
        "layout_requirements": [
            "connected free arena",
            "camera reachable",
            "at least three hidden and three visible free cells",
            "at least three movable free cells",
            "at least one visible-hidden boundary edge",
        ],
    }


def _audit_seed_at_probability(seed: int, turn_probability: float) -> dict[str, Any]:
    collection_audit: Counter[str] = Counter()
    samples = _collect(
        seed,
        steps=DEFAULT_STEPS,
        warmup=WARMUP,
        turn_probability=turn_probability,
        attach_entity_graph=False,
        audit=collection_audit,
    )
    single_hidden = [sample for sample in samples if sample.hidden_object_count == 1]
    episode_bin_groups: dict[str, set[tuple[str, int]]] = {
        bin_name: set() for bin_name in REQUIRED_OCCLUSION_BINS
    }
    for sample in single_hidden:
        episode_bin_groups[_occlusion_length_bin(sample.hidden_steps)].add(
            (sample.focus_object, sample.focus_occlusion_id)
        )
    group_counts = {
        bin_name: len(episode_bin_groups[bin_name])
        for bin_name in REQUIRED_OCCLUSION_BINS
    }
    rejection_reasons: list[str] = []
    if len(single_hidden) < MIN_ACCEPTED_SINGLE_HIDDEN_SAMPLES:
        rejection_reasons.append("insufficient_single_hidden_samples")
    for bin_name, count in group_counts.items():
        if count < MIN_EPISODE_BIN_GROUPS:
            rejection_reasons.append(f"insufficient_episode_groups_{bin_name}")
    return {
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "turn_probability": turn_probability,
        "accepted_sample_count": len(samples),
        "accepted_single_hidden_sample_count": len(single_hidden),
        "episode_bin_group_counts": group_counts,
        "collection_exclusions": {
            key: value
            for key, value in sorted(collection_audit.items())
            if key.startswith("skipped_")
        },
    }


def audit_seed(seed: int) -> dict[str, Any]:
    """Require model-blind event coverage at every candidate difficulty."""

    layout = RandomizedOcclusionWorld(seed).layout_diagnostics()
    by_probability = {
        str(probability): _audit_seed_at_probability(seed, probability)
        for probability in COVERAGE_TURN_PROBABILITIES
    }
    rejection_reasons: list[str] = []
    if not bool(layout["valid"]):
        rejection_reasons.append("layout_invalid")
    for probability, audit in by_probability.items():
        rejection_reasons.extend(
            f"p{probability}_{reason}" for reason in audit["rejection_reasons"]
        )
    return {
        "seed": int(seed),
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "layout": layout,
        "by_turn_probability": by_probability,
    }


def _selection_digest(provenance: dict[str, Any]) -> str:
    """Bind the contract, scan prefix, split, and structural audit evidence."""

    payload = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selection_provenance(
    *,
    contract: dict[str, Any],
    candidate_range: dict[str, int | None],
    train_seeds: list[int],
    evaluation_seeds: list[int],
    accepted_seed_audits: list[dict[str, Any]],
    rejected_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return every input/output needed to substantiate first-N selection."""

    return {
        "registry_version": REGISTRY_VERSION,
        "coverage_contract": contract,
        "candidate_range": candidate_range,
        "requested_counts": {
            "train": len(train_seeds),
            "evaluation": len(evaluation_seeds),
        },
        "train_seeds": train_seeds,
        "evaluation_seeds": evaluation_seeds,
        "accepted_seed_audits": accepted_seed_audits,
        "rejected_candidates": rejected_candidates,
    }


def generate_registry(
    *,
    candidate_start: int,
    candidate_stop: int,
    train_count: int,
    evaluation_count: int,
) -> dict[str, Any]:
    """Select the first requested qualifying seeds without score inspection."""

    if candidate_stop <= candidate_start:
        raise ValueError("candidate_stop must exceed candidate_start")
    if train_count < 1 or evaluation_count < 1:
        raise ValueError("train_count and evaluation_count must be positive")
    requested = train_count + evaluation_count
    accepted: list[int] = []
    accepted_audits: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    last_scanned: int | None = None
    for seed in range(candidate_start, candidate_stop):
        last_scanned = seed
        audit = audit_seed(seed)
        if audit["accepted"]:
            accepted.append(seed)
            accepted_audits.append(audit)
            if len(accepted) == requested:
                break
        else:
            rejected.append(
                {
                    "seed": seed,
                    "rejection_reasons": audit["rejection_reasons"],
                }
            )
    if len(accepted) != requested:
        raise RuntimeError(
            f"candidate range yielded {len(accepted)} of {requested} required seeds"
        )
    contract = coverage_contract()
    candidate_range: dict[str, int | None] = {
        "start_inclusive": candidate_start,
        "stop_exclusive": candidate_stop,
        "last_scanned_inclusive": last_scanned,
    }
    train_seeds = accepted[:train_count]
    evaluation_seeds = accepted[train_count:]
    provenance = _selection_provenance(
        contract=contract,
        candidate_range=candidate_range,
        train_seeds=train_seeds,
        evaluation_seeds=evaluation_seeds,
        accepted_seed_audits=accepted_audits,
        rejected_candidates=rejected,
    )
    return {
        "registry_version": REGISTRY_VERSION,
        "status": "development_only_non_gated",
        "model_metrics_read": False,
        "candidate_range": candidate_range,
        "requested_counts": {
            "train": train_count,
            "evaluation": evaluation_count,
        },
        "coverage_contract": contract,
        "selection_digest_sha256": _selection_digest(provenance),
        "train_seeds": train_seeds,
        "evaluation_seeds": evaluation_seeds,
        "accepted_seed_audits": accepted_audits,
        "rejected_candidates": rejected,
        "holdout_policy": (
            "Do not publish or scan holdout seeds here. A custodian must apply "
            "this locked algorithm to a secret committed seed stream."
        ),
    }


def _registry_reproduction_command(
    registry: dict[str, Any],
    *,
    selected_turn_probability: float | None,
    turn_probability_selection_artifact: str | None,
) -> str:
    candidate_range = registry["candidate_range"]
    requested_counts = registry["requested_counts"]
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "cal.evaluation.permanence_seed_registry",
        "--candidate-start",
        str(candidate_range["start_inclusive"]),
        "--candidate-stop",
        str(candidate_range["stop_exclusive"]),
        "--train-count",
        str(requested_counts["train"]),
        "--evaluation-count",
        str(requested_counts["evaluation"]),
        "--summary",
    ]
    if selected_turn_probability is not None:
        command.extend(
            ["--selected-turn-probability", str(selected_turn_probability)]
        )
    if turn_probability_selection_artifact is not None:
        command.extend(
            [
                "--turn-probability-selection-artifact",
                turn_probability_selection_artifact,
            ]
        )
    return shlex.join(command)


def summarize_registry(
    registry: dict[str, Any],
    *,
    selected_turn_probability: float | None = None,
    turn_probability_selection_artifact: str | None = None,
) -> dict[str, Any]:
    """Create the compact, exactly reproducible committed registry artifact."""

    reason_counts = Counter(
        reason
        for rejected in registry["rejected_candidates"]
        for reason in rejected["rejection_reasons"]
    )
    output: dict[str, Any] = {
        "registry_version": registry["registry_version"],
        "status": registry["status"],
        "model_metrics_read": registry["model_metrics_read"],
        "generator": "cal.evaluation.permanence_seed_registry",
        "reproduction_command": _registry_reproduction_command(
            registry,
            selected_turn_probability=selected_turn_probability,
            turn_probability_selection_artifact=(
                turn_probability_selection_artifact
            ),
        ),
        "candidate_range": registry["candidate_range"],
        "requested_counts": registry["requested_counts"],
        "coverage_contract": registry["coverage_contract"],
        "selection_digest_sha256": registry["selection_digest_sha256"],
    }
    if selected_turn_probability is not None:
        output["selected_turn_probability"] = selected_turn_probability
    if turn_probability_selection_artifact is not None:
        output["turn_probability_selection_artifact"] = (
            turn_probability_selection_artifact
        )
    output.update(
        {
            "train_seeds": registry["train_seeds"],
            "evaluation_seeds": registry["evaluation_seeds"],
            "rejected_candidate_count": len(registry["rejected_candidates"]),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
            "holdout_policy": registry["holdout_policy"],
        }
    )
    return output


def reproduce_registry_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Regenerate a compact artifact using only its declared provenance."""

    candidate_range = artifact["candidate_range"]
    requested_counts = artifact["requested_counts"]
    registry = generate_registry(
        candidate_start=int(candidate_range["start_inclusive"]),
        candidate_stop=int(candidate_range["stop_exclusive"]),
        train_count=int(requested_counts["train"]),
        evaluation_count=int(requested_counts["evaluation"]),
    )
    return summarize_registry(
        registry,
        selected_turn_probability=artifact.get("selected_turn_probability"),
        turn_probability_selection_artifact=artifact.get(
            "turn_probability_selection_artifact"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-start", type=int, default=62_000)
    parser.add_argument("--candidate-stop", type=int, default=70_000)
    parser.add_argument(
        "--train-count", type=int, default=LOCKED_DEVELOPMENT_TRAIN_COUNT
    )
    parser.add_argument(
        "--evaluation-count",
        type=int,
        default=LOCKED_PHASE0_EVALUATION_SOURCE_COUNT,
    )
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--selected-turn-probability", type=float)
    parser.add_argument("--turn-probability-selection-artifact")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry = generate_registry(
        candidate_start=args.candidate_start,
        candidate_stop=args.candidate_stop,
        train_count=args.train_count,
        evaluation_count=args.evaluation_count,
    )
    if args.summary:
        output = summarize_registry(
            registry,
            selected_turn_probability=args.selected_turn_probability,
            turn_probability_selection_artifact=(
                args.turn_probability_selection_artifact
            ),
        )
    else:
        output = registry
    rendered = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
