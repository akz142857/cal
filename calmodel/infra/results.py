"""Build a searchable index over persisted experiment result summaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SUMMARY_NAMES = {
    "summary.json": "prediction",
    "probe-summary.json": "body_probe",
    "adaptation-summary.json": "adaptation",
    "cause-probe-summary.json": "cause_probe",
    "multiseed-summary.json": "multiseed",
    "M1-stage-summary.json": "stage_summary",
    "M1b-stage-summary.json": "m1b_stage_summary",
    "temporal-probe-summary.json": "temporal_probe",
    "temporal-multiseed-summary.json": "temporal_multiseed",
    "temporal-length-summary.json": "temporal_length_curve",
    "body-probe-multiseed-summary.json": "body_probe_multiseed",
    "adaptation-multiseed-summary.json": "adaptation_multiseed",
    "M1c-A-screen-summary.json": "m1c_screen",
    "M1c-B-screen-summary.json": "m1c_screen",
    "M1c-C-screen-summary.json": "m1c_screen",
    "V2-identifiability-summary.json": "v2_identifiability",
    "V2-diagnostic-ceiling-summary.json": "v2_diagnostic_ceiling",
    "V2-causal-sufficiency-summary.json": "v2_causal_sufficiency",
    "V2-audit-summary.json": "v2_audit",
    "V2-M1-summary.json": "v2_stage",
    "V2-M2-summary.json": "v2_stage",
    "V2-M2-probabilistic-development-summary.json": "v2_m2_development",
    "V2-M2-hard-map-development-summary.json": "v2_m2_ablation",
    "V2-M2-nearest-development-summary.json": "v2_m2_ablation",
    "V2-M2-probabilistic-holdout-summary.json": "v2_m2_holdout",
    "V2-M2-probabilistic-review-summary.json": "v2_m2_review",
    "V2-M3-summary.json": "v2_stage",
    "V2-M3-body-graph-development-summary.json": "v2_m3_development",
    "V2-M3-body-graph-development-no-causal-likelihood.json": "v2_m3_ablation",
    "V2-M3-body-graph-holdout-summary.json": "v2_m3_holdout",
    "V2-M3-body-graph-review-summary.json": "v2_m3_review",
    "V2-M1-M3-integrated-development-summary.json": "v2_m1_m3_development_v1",
    "V2-M1-M3-integrated-v2-development-summary.json": "v2_m1_m3_development",
    "V2-M1-M3-integrated-v2-confirmation-summary.json": "v2_m1_m3_confirmation",
    "V2-M1-M3-integrated-confirmation-review-summary.json": "v2_m1_m3_confirmation_review",
    "V2-M4-summary.json": "v2_stage",
    "V2-stage-summary.json": "v2_stage_summary",
}


def build_result_index(
    results_directory: str | Path = "results",
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(results_directory)
    entries = []
    if root.exists():
        for source in sorted(root.rglob("*.json")):
            kind = SUMMARY_NAMES.get(source.name)
            if (
                kind is None
                and source.name.startswith("M1")
                and source.name.endswith("-screen-summary.json")
            ):
                kind = "m1_mechanism_screen"
            if kind is None and source.parent.name == "M1-efficiency":
                kind = "efficiency"
            if kind is None or source == output_path:
                continue
            payload = _read_json(source)
            entries.append(
                {
                    "kind": kind,
                    "path": str(source),
                    "name": payload.get(
                        "name",
                        payload.get("candidate", payload.get("experiment")),
                    ),
                    "checkpoint": payload.get("checkpoint"),
                    "config": payload.get(
                        "config",
                        payload.get("config_path"),
                    ),
                    "source_sha256": (
                        payload.get("provenance", {}).get("source_sha256")
                        if isinstance(payload.get("provenance"), dict)
                        else None
                    ),
                }
            )
    index = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_directory": str(root),
        "entry_count": len(entries),
        "entries": entries,
    }
    destination = (
        Path(output_path) if output_path is not None else root / "INDEX.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index Cal experiment summaries."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    index = build_result_index(
        arguments.results,
        output_path=arguments.output,
    )
    print(f"indexed {index['entry_count']} result summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
