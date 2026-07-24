"""Aggregate body-adaptation curves across independently trained models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from cal.evaluation.m1_summary import _aggregate_adaptation
from cal.infra.provenance import capture_provenance

DEFAULT_SEEDS = (5, 6, 7, 8, 9)


def aggregate_adaptation_results(
    root_directory: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Read per-seed curves and persist paired adaptation statistics."""

    if not seeds:
        raise ValueError("seeds must be non-empty")
    root = Path(root_directory)
    records = []
    payloads = []
    for seed in seeds:
        path = root / f"seed-{int(seed):03d}" / "adaptation-summary.json"
        payload = _read_json(path)
        payloads.append(payload)
        records.append({"seed": int(seed), "path": str(path)})
    variants = _aggregate_adaptation(payloads)
    variant_gates = {
        name: {
            "changed_body_improved": (
                float(metrics["final_improvement"]["ci95_low"]) > 0.0
            ),
            "original_body_retained": (
                float(metrics["original_loss_change"]["ci95_high"]) <= 0.0
            ),
        }
        for name, metrics in variants.items()
    }
    summary = {
        "result_schema_version": 1,
        "seeds": [int(seed) for seed in seeds],
        "records": records,
        "variants": variants,
        "variant_gates": variant_gates,
        "adaptation_gate_passed": (
            bool(variant_gates)
            and all(
                gates["changed_body_improved"]
                and gates["original_body_retained"]
                for gates in variant_gates.values()
            )
        ),
        "provenance": capture_provenance(),
    }
    destination = root / "adaptation-multiseed-summary.json"
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
        description="Aggregate multi-seed body adaptation curves."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    arguments = parser.parse_args(argv)
    summary = aggregate_adaptation_results(
        arguments.root,
        seeds=arguments.seeds,
    )
    for name, metrics in summary["variants"].items():
        improvement = metrics["final_improvement"]
        print(
            f"{name}: improvement "
            f"{improvement['mean']:.4f}±"
            f"{improvement['sample_std']:.4f}"
        )
    print(f"adaptation gate={summary['adaptation_gate_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
