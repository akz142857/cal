"""Recoverable multi-seed runner and aggregate statistics for M1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

import yaml

from calmodel.evaluation.body_probe import run_probe_experiment
from calmodel.infra.provenance import capture_provenance
from calmodel.learning.trainer import run_experiment

DEFAULT_EXPERIMENTS = (
    Path("experiments/baseline.yaml"),
    Path("experiments/feedforward.yaml"),
    Path("experiments/no_action.yaml"),
    Path("experiments/no_touch.yaml"),
    Path("experiments/no_proprioception.yaml"),
    Path("experiments/shuffled_modalities.yaml"),
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def run_multiseed_suite(
    experiment_paths: Sequence[str | Path] = DEFAULT_EXPERIMENTS,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    output_directory: str | Path = "results/M1-multiseed",
) -> dict[str, Any]:
    """Run missing jobs, reuse complete jobs, and aggregate paired results."""

    if len(set(int(seed) for seed in seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be non-empty and unique")
    output = Path(output_directory)
    config_directory = output / "configs"
    config_directory.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, str]] = {}

    for experiment_path in experiment_paths:
        source = Path(experiment_path)
        source_text = source.read_text(encoding="utf-8")
        base_payload = yaml.safe_load(source_text)
        if not isinstance(base_payload, dict):
            raise ValueError(f"{source} must contain a mapping")
        name = str(base_payload.get("name", source.stem))
        sources[name] = {
            "path": str(source),
            "sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        }
        records[name] = []

        for seed in seeds:
            resolved_seed = int(seed)
            payload = dict(base_payload)
            payload["seed"] = resolved_seed
            materialized = config_directory / f"{name}-seed{resolved_seed:03d}.yaml"
            materialized.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            run_directory = output / name / f"seed-{resolved_seed:03d}"
            prediction_path = run_directory / "summary.json"
            checkpoint_path = run_directory / "checkpoint.pt"
            probe_directory = run_directory / "probe"
            probe_path = probe_directory / "probe-summary.json"

            if prediction_path.exists() and checkpoint_path.exists():
                prediction = _read_json(prediction_path)
                prediction_status = "reused"
            else:
                prediction = run_experiment(
                    materialized,
                    output_directory=run_directory,
                )
                prediction_status = "ran"
            if probe_path.exists():
                probe = _read_json(probe_path)
                probe_status = "reused"
            else:
                probe = run_probe_experiment(
                    checkpoint_path,
                    materialized,
                    output_directory=probe_directory,
                )
                probe_status = "ran"

            record = {
                "seed": resolved_seed,
                "prediction_status": prediction_status,
                "probe_status": probe_status,
                "prediction_loss": float(
                    prediction["best_validation_loss"]
                ),
                "body_iou": float(probe["test"]["iou"]),
                "body_f1": float(probe["test"]["f1"]),
                "unseen_pose_iou": float(
                    probe["unseen_pose_test"]["iou"]
                ),
                "parameter_count": int(prediction["parameter_count"]),
                "run_directory": str(run_directory),
            }
            records[name].append(record)
            print(
                f"{name} seed={resolved_seed}: "
                f"prediction={record['prediction_loss']:.4f}, "
                f"IoU={record['body_iou']:.4f} "
                f"({prediction_status}/{probe_status})",
                flush=True,
            )

    aggregates = {
        name: {
            metric: summarize_values(
                [float(record[metric]) for record in condition_records]
            )
            for metric in (
                "prediction_loss",
                "body_iou",
                "body_f1",
                "unseen_pose_iou",
            )
        }
        for name, condition_records in records.items()
    }
    paired_controls = _paired_control_failures(records)
    summary = {
        "result_schema_version": 1,
        "seeds": [int(seed) for seed in seeds],
        "sources": sources,
        "records": records,
        "aggregates": aggregates,
        "paired_control_failures": paired_controls,
        "provenance": capture_provenance(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "multiseed-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    """Return sample statistics and a two-sided 95% t interval."""

    if not values:
        raise ValueError("cannot summarize empty values")
    count = len(values)
    mean = fmean(values)
    sample_std = stdev(values) if count > 1 else 0.0
    critical = _t_critical_95(count - 1)
    margin = critical * sample_std / (count**0.5) if count > 1 else 0.0
    return {
        "count": count,
        "mean": mean,
        "sample_std": sample_std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "minimum": min(values),
        "maximum": max(values),
    }


def _paired_control_failures(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if "baseline" not in records:
        return {}
    baseline = {
        int(record["seed"]): record for record in records["baseline"]
    }
    comparisons = {}
    for name, condition in records.items():
        if name == "baseline":
            continue
        paired = [
            (baseline[int(record["seed"])], record)
            for record in condition
            if int(record["seed"]) in baseline
        ]
        if not paired:
            continue
        prediction_failures = sum(
            float(full["prediction_loss"])
            >= float(control["prediction_loss"])
            for full, control in paired
        )
        probe_failures = sum(
            float(full["body_iou"]) <= float(control["body_iou"])
            for full, control in paired
        )
        comparisons[name] = {
            "paired_seeds": len(paired),
            "prediction_failure_rate": prediction_failures / len(paired),
            "body_iou_failure_rate": probe_failures / len(paired),
            "prediction_difference_full_minus_control": summarize_values(
                [
                    float(full["prediction_loss"])
                    - float(control["prediction_loss"])
                    for full, control in paired
                ]
            ),
            "body_iou_difference_full_minus_control": summarize_values(
                [
                    float(full["body_iou"]) - float(control["body_iou"])
                    for full, control in paired
                ]
            ),
            "mean_prediction_difference_full_minus_control": fmean(
                float(full["prediction_loss"])
                - float(control["prediction_loss"])
                for full, control in paired
            ),
            "mean_body_iou_difference_full_minus_control": fmean(
                float(full["body_iou"]) - float(control["body_iou"])
                for full, control in paired
            ),
        }
    return comparisons


def _t_critical_95(degrees_of_freedom: int) -> float:
    # Exact common small-sample values; asymptotic normal value thereafter.
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }
    return table.get(degrees_of_freedom, 1.96)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and aggregate the five-seed Cal M1 suite."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/M1-multiseed"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--experiments",
        type=Path,
        nargs="+",
        default=list(DEFAULT_EXPERIMENTS),
    )
    arguments = parser.parse_args(argv)
    summary = run_multiseed_suite(
        arguments.experiments,
        seeds=arguments.seeds,
        output_directory=arguments.output,
    )
    for name, metrics in summary["aggregates"].items():
        prediction = metrics["prediction_loss"]
        probe = metrics["body_iou"]
        print(
            f"{name}: loss={prediction['mean']:.4f}±"
            f"{prediction['sample_std']:.4f}, "
            f"IoU={probe['mean']:.4f}±{probe['sample_std']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
