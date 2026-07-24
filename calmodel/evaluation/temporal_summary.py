"""Aggregate persistent-state temporal probes across model seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from calmodel.evaluation.multiseed import summarize_values
from calmodel.infra.provenance import capture_provenance

DEFAULT_CONDITIONS = ("gru", "no_action", "feedforward")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_LENGTHS = (2, 4, 8, 12)


def aggregate_temporal_probe_results(
    root_directory: str | Path,
    *,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Read per-seed probe summaries and persist aggregate comparisons."""

    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be non-empty and unique")
    if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    root = Path(root_directory)
    records: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        condition_records = []
        for seed in seeds:
            path = (
                root
                / condition
                / f"seed-{int(seed):03d}"
                / "temporal-probe-summary.json"
            )
            payload = _read_json(path)
            condition_records.append(
                {
                    "seed": int(seed),
                    "body_iou": float(_mapping(payload, "test")["iou"]),
                    "body_f1": float(_mapping(payload, "test")["f1"]),
                    "fixed_iou": float(
                        _mapping(payload, "fixed_position_baseline")["iou"]
                    ),
                    "path": str(path),
                }
            )
        records[condition] = condition_records

    aggregates = {
        condition: {
            metric: summarize_values(
                [float(record[metric]) for record in condition_records]
            )
            for metric in ("body_iou", "body_f1", "fixed_iou")
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
            for metric in ("body_iou", "body_f1")
        }

    first_payload = _read_json(
        Path(records[reference][0]["path"])
    )
    summary = {
        "result_schema_version": 1,
        "seeds": [int(seed) for seed in seeds],
        "conditions": list(conditions),
        "blackout": first_payload.get("blackout"),
        "action_policy": first_payload.get("action_policy"),
        "records": records,
        "aggregates": aggregates,
        "paired_differences": paired_differences,
        "provenance": capture_provenance(),
    }
    destination = root / "temporal-multiseed-summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def aggregate_temporal_length_curve(
    root_directory: str | Path,
    *,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Aggregate a set of otherwise identical blackout lengths."""

    if not lengths or any(int(length) <= 0 for length in lengths):
        raise ValueError("lengths must be positive and non-empty")
    root = Path(root_directory)
    by_length = {}
    for length in lengths:
        resolved_length = int(length)
        length_root = root / f"length-{resolved_length:02d}"
        by_length[str(resolved_length)] = aggregate_temporal_probe_results(
            length_root,
            conditions=conditions,
            seeds=seeds,
        )

    curves = {
        condition: [
            {
                "length": int(length),
                "body_iou": by_length[str(int(length))]["aggregates"][
                    condition
                ]["body_iou"],
            }
            for length in lengths
        ]
        for condition in conditions
    }
    comparison_names = tuple(
        by_length[str(int(lengths[0]))]["paired_differences"]
    )
    paired_curves = {
        name: [
            {
                "length": int(length),
                "body_iou": by_length[str(int(length))][
                    "paired_differences"
                ][name]["body_iou"],
            }
            for length in lengths
        ]
        for name in comparison_names
    }
    action_curve = paired_curves.get("gru_minus_no_action", ())
    passing_lengths = [
        int(item["length"])
        for item in action_curve
        if float(item["body_iou"]["ci95_low"]) > 0.0
    ]
    action_slopes = _paired_length_slopes(
        by_length,
        lengths=lengths,
        reference="gru",
        comparison="no_action",
    )
    summary = {
        "result_schema_version": 1,
        "lengths": [int(length) for length in lengths],
        "conditions": list(conditions),
        "curves": curves,
        "paired_curves": paired_curves,
        "action_integration_passing_lengths": passing_lengths,
        "action_integration_length_trend": {
            "body_iou_slope_per_step": summarize_values(action_slopes),
            "paired_seed_slopes": action_slopes,
        },
        "provenance": capture_provenance(),
    }
    destination = root / "temporal-length-summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _paired_length_slopes(
    by_length: Mapping[str, Mapping[str, Any]],
    *,
    lengths: Sequence[int],
    reference: str,
    comparison: str,
) -> list[float]:
    x_values = [float(length) for length in lengths]
    x_mean = sum(x_values) / len(x_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0.0:
        raise ValueError("at least two distinct lengths are required")
    seed_differences: dict[int, list[float]] = {}
    for length in lengths:
        records = by_length[str(int(length))]["records"]
        reference_by_seed = {
            int(item["seed"]): float(item["body_iou"])
            for item in records[reference]
        }
        comparison_by_seed = {
            int(item["seed"]): float(item["body_iou"])
            for item in records[comparison]
        }
        for seed, value in reference_by_seed.items():
            seed_differences.setdefault(seed, []).append(
                value - comparison_by_seed[seed]
            )
    slopes = []
    for differences in seed_differences.values():
        y_mean = sum(differences) / len(differences)
        slopes.append(
            sum(
                (x_value - x_mean) * (y_value - y_mean)
                for x_value, y_value in zip(x_values, differences)
            )
            / denominator
        )
    return slopes


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed temporal body probes."
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
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
    )
    arguments = parser.parse_args(argv)
    if arguments.lengths:
        summary = aggregate_temporal_length_curve(
            arguments.root,
            lengths=arguments.lengths,
            conditions=arguments.conditions,
            seeds=arguments.seeds,
        )
        print(
            "action integration passing lengths: "
            f"{summary['action_integration_passing_lengths']}"
        )
    else:
        summary = aggregate_temporal_probe_results(
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
