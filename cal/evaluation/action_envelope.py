"""Evaluate all-action causal envelope quality without training a model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from cal.env.body import BodyConfig
from cal.env.world import WorldConfig
from cal.evaluation.body_probe import _replay_body_labels
from cal.evaluation.metrics import segmentation_metrics
from cal.infra.provenance import capture_provenance
from cal.learning.dataset import (
    CounterfactualTrajectorySequenceDataset,
    collect_trajectories,
)


def evaluate_action_envelope(
    config_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(Path(config_path).read_text())
    environment = payload["environment"]
    probe = payload["probe"]
    image_size = tuple(environment["image_size"])
    world = WorldConfig(
        image_size=image_size,
        object_count=int(environment.get("object_count", 3)),
        object_radius=float(environment.get("object_radius", 0.045)),
        external_object_motion_probability=float(
            environment.get("external_object_motion_probability", 0.0)
        ),
        distractor_body_count=int(
            environment.get("distractor_body_count", 0)
        ),
        distractor_body_motion_probability=float(
            environment.get("distractor_body_motion_probability", 0.0)
        ),
    )
    seeds = tuple(int(seed) for seed in probe["train_seeds"])
    steps = int(probe.get("steps_per_seed", 128))
    trajectories = collect_trajectories(
        world,
        BodyConfig(),
        seeds,
        steps_per_seed=steps,
    )
    envelopes = []
    labels = []
    subset_violations = 0
    empty_steps = 0
    for trajectory in trajectories:
        sample = CounterfactualTrajectorySequenceDataset(
            (trajectory,),
            sequence_length=len(trajectory),
        )[0]
        envelope = sample["action_envelope"]
        envelopes.append(envelope)
        subset_violations += int((envelope > sample["vision"]).sum())
        empty_steps += int(
            (
                envelope.flatten(start_dim=1).sum(dim=1) == 0
            ).sum()
        )
        masks, _ = _replay_body_labels(
            trajectory,
            target_offset="current",
        )
        labels.append(torch.tensor(masks).unsqueeze(1).float())
    prediction = torch.cat(envelopes)
    target = torch.cat(labels)
    logits = torch.logit(prediction.clamp(1e-5, 1.0 - 1e-5))
    metrics = segmentation_metrics(logits, target)
    gates = {
        "iou_at_least_0_45": metrics.iou >= 0.45,
        "recall_at_least_0_60": metrics.recall >= 0.60,
        "precision_at_least_0_65": metrics.precision >= 0.65,
        "every_step_nonempty": empty_steps == 0,
        "strict_current_vision_subset": subset_violations == 0,
    }
    summary = {
        "result_schema_version": 1,
        "metrics": asdict(metrics),
        "empty_steps": empty_steps,
        "subset_violations": subset_violations,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "run_M1f_model_screen"
            if all(gates.values())
            else "stop_M1f_before_model_training"
        ),
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = evaluate_action_envelope(
        arguments.config,
        output_path=arguments.output,
    )
    print(
        f"iou={result['metrics']['iou']:.4f}; "
        f"recall={result['metrics']['recall']:.4f}; "
        f"precision={result['metrics']['precision']:.4f}; "
        f"passed={result['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
