"""Measure a frozen model's fit to the all-action causal envelope."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from calmodel.env.body import BodyConfig
from calmodel.env.world import WorldConfig
from calmodel.evaluation.metrics import segmentation_metrics
from calmodel.infra.provenance import capture_provenance
from calmodel.learning.dataset import (
    CounterfactualTrajectorySequenceDataset,
    collect_trajectories,
)
from calmodel.learning.trainer import load_checkpoint


def evaluate_envelope_fit(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(Path(config_path).read_text())
    environment = payload["environment"]
    data = payload["data"]
    probe = payload["probe"]
    world = WorldConfig(
        image_size=tuple(environment["image_size"]),
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
    trajectories = collect_trajectories(
        world,
        BodyConfig(),
        tuple(int(seed) for seed in probe["validation_seeds"]),
        steps_per_seed=int(probe.get("steps_per_seed", 128)),
        action_repeat_probability=float(
            data.get("action_repeat_probability", 0.0)
        ),
    )
    model = load_checkpoint(checkpoint_path)
    model.eval()
    logits = []
    targets = []
    prediction_effects = []
    action_effects = []
    causal_effect_logits = []
    causal_effect_targets = []
    with torch.no_grad():
        for trajectory in trajectories:
            sample = CounterfactualTrajectorySequenceDataset(
                (trajectory,),
                sequence_length=len(trajectory),
            )[0]
            output = model(
                vision=sample["vision"].unsqueeze(0),
                proprioception=sample["proprioception"].unsqueeze(0),
                touch=sample["touch"].unsqueeze(0),
                actions=sample["actions"].unsqueeze(0),
            )
            envelope_logits = (
                output.causal_envelope_logits
                if output.causal_envelope_logits is not None
                else output.object_slot_logits
                if output.object_slot_logits is not None
                else output.global_ownership_logits
                if output.global_ownership_logits is not None
                else output.spatial_ownership_logits
                if output.spatial_ownership_logits is not None
                else output.ownership_logits
            )
            if envelope_logits is None:
                raise ValueError("model has no ownership head")
            logits.append(envelope_logits.squeeze(0))
            targets.append(sample["action_envelope"])
            prediction_effects.append(
                (
                    torch.sigmoid(output.vision_logits.squeeze(0))
                    - sample["vision"]
                ).abs()
            )
            action_effects.append(sample["action_effect"])
            if output.causal_action_effect_logits is not None:
                predicted = output.causal_action_effect_logits.squeeze(0)
                causal_effect_logits.append(
                    predicted.reshape(
                        -1,
                        1,
                        predicted.shape[-2],
                        predicted.shape[-1],
                    )
                )
                expected = sample["all_action_effects"]
                causal_effect_targets.append(
                    expected.reshape(
                        -1,
                        1,
                        expected.shape[-2],
                        expected.shape[-1],
                    )
                )
    metrics = segmentation_metrics(torch.cat(logits), torch.cat(targets))
    prediction_effect_metrics = segmentation_metrics(
        torch.logit(
            torch.cat(prediction_effects).clamp(1e-5, 1.0 - 1e-5)
        ),
        torch.cat(action_effects),
    )
    causal_action_effect_metrics = (
        segmentation_metrics(
            torch.cat(causal_effect_logits),
            torch.cat(causal_effect_targets),
        )
        if causal_effect_logits
        else None
    )
    gates = {
        "iou_at_least_0_45": metrics.iou >= 0.45,
        "recall_at_least_0_60": metrics.recall >= 0.60,
        "precision_at_least_0_65": metrics.precision >= 0.65,
    }
    summary = {
        "result_schema_version": 1,
        "candidate": str(payload.get("name", Path(config_path).stem)),
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "metrics": asdict(metrics),
        "prediction_effect_metrics": asdict(prediction_effect_metrics),
        "causal_action_effect_metrics": (
            asdict(causal_action_effect_metrics)
            if causal_action_effect_metrics is not None
            else None
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "run_body_screen"
            if all(gates.values())
            else "stop_before_body_screen"
        ),
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_envelope_fit(
        args.checkpoint,
        args.config,
        output_path=args.output,
    )
    metrics = result["metrics"]
    print(
        f"iou={metrics['iou']:.4f}; recall={metrics['recall']:.4f}; "
        f"precision={metrics['precision']:.4f}; passed={result['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
