"""Aggregate matched-dimension body probes across model seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from calmodel.evaluation.multiseed import summarize_values
from calmodel.infra.provenance import capture_provenance

DEFAULT_CONDITIONS = ("gru", "no_action", "feedforward")
DEFAULT_SEEDS = (5, 6, 7, 8, 9)


def aggregate_body_probe_results(
    root_directory: str | Path,
    *,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Read per-seed body probes and persist aggregate paired statistics."""

    root = Path(root_directory)
    if not conditions or not seeds:
        raise ValueError("conditions and seeds must be non-empty")
    records = {}
    for condition in conditions:
        condition_records = []
        for seed in seeds:
            path = (
                root
                / condition
                / f"seed-{int(seed):03d}"
                / "probe-summary.json"
            )
            payload = _read_json(path)
            body_iou = float(payload["test"]["iou"])
            raw_sensor_iou = float(payload["raw_sensor_probe"]["iou"])
            condition_records.append(
                {
                    "seed": int(seed),
                    "body_iou": body_iou,
                    "body_f1": float(payload["test"]["f1"]),
                    "unseen_pose_iou": float(
                        payload["unseen_pose_test"]["iou"]
                    ),
                    "raw_sensor_iou": raw_sensor_iou,
                    "representation_gain_over_raw_sensor": (
                        body_iou - raw_sensor_iou
                    ),
                    "original_representation_size": int(
                        payload["original_representation_size"]
                    ),
                    "probe_representation_size": int(
                        payload["probe_representation_size"]
                    ),
                    "path": str(path),
                }
            )
        records[condition] = condition_records

    metrics = (
        "body_iou",
        "body_f1",
        "unseen_pose_iou",
        "raw_sensor_iou",
        "representation_gain_over_raw_sensor",
    )
    aggregates = {
        condition: {
            metric: summarize_values(
                [float(record[metric]) for record in condition_records]
            )
            for metric in metrics
        }
        for condition, condition_records in records.items()
    }
    reference = conditions[0]
    reference_by_seed = {
        int(record["seed"]): record for record in records[reference]
    }
    paired_differences = {}
    for condition in conditions[1:]:
        comparison_by_seed = {
            int(record["seed"]): record for record in records[condition]
        }
        paired_differences[f"{reference}_minus_{condition}"] = {
            metric: summarize_values(
                [
                    float(reference_by_seed[int(seed)][metric])
                    - float(comparison_by_seed[int(seed)][metric])
                    for seed in seeds
                ]
            )
            for metric in metrics
        }

    summary = {
        "result_schema_version": 1,
        "seeds": [int(seed) for seed in seeds],
        "conditions": list(conditions),
        "records": records,
        "aggregates": aggregates,
        "paired_differences": paired_differences,
        "provenance": capture_provenance(),
    }
    destination = root / "body-probe-multiseed-summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate matched-dimension body probes."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    arguments = parser.parse_args(argv)
    summary = aggregate_body_probe_results(
        arguments.root,
        conditions=arguments.conditions,
        seeds=arguments.seeds,
    )
    for condition, metrics in summary["aggregates"].items():
        body = metrics["body_iou"]
        print(
            f"{condition}: body IoU "
            f"{body['mean']:.4f}±{body['sample_std']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
