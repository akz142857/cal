"""Evaluate whether the predictor prefers the observed action's future."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch import Tensor

from cal.env.body import BodyConfig
from cal.env.world import WorldConfig
from cal.infra.provenance import capture_provenance
from cal.learning.dataset import (
    ACTION_VOCABULARY,
    TrajectorySequenceDataset,
    collect_trajectories,
)
from cal.learning.trainer import LossConfig, load_checkpoint


@dataclass(frozen=True, slots=True)
class CounterfactualMetrics:
    samples: int
    correct_action_loss: float
    wrong_action_loss: float
    loss_margin_wrong_minus_correct: float
    correct_preference_rate: float


def evaluate_counterfactual(
    model: torch.nn.Module,
    dataset: TrajectorySequenceDataset,
    *,
    device: str = "cpu",
    loss_config: LossConfig | None = None,
) -> CounterfactualMetrics:
    """Compare per-window losses under recorded and deterministically wrong actions."""

    if not hasattr(model, "forward"):
        raise TypeError("model must be callable")
    resolved = loss_config or LossConfig()
    torch_model = model
    torch_model.to(torch.device(device))
    torch_model.eval()
    correct_losses: list[Tensor] = []
    wrong_losses: list[Tensor] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = {
            key: value.unsqueeze(0).to(torch.device(device))
            for key, value in sample.items()
        }
        with torch.no_grad():
            correct = torch_model(
                vision=batch["vision"],
                proprioception=batch["proprioception"],
                touch=batch["touch"],
                actions=batch["actions"],
            )
            wrong = torch_model(
                vision=batch["vision"],
                proprioception=batch["proprioception"],
                touch=batch["touch"],
                actions=(batch["actions"] + 1) % len(ACTION_VOCABULARY),
            )
            correct_losses.append(
                _per_sample_loss(correct, batch, resolved).cpu()
            )
            wrong_losses.append(
                _per_sample_loss(wrong, batch, resolved).cpu()
            )
    correct_values = torch.cat(correct_losses)
    wrong_values = torch.cat(wrong_losses)
    return CounterfactualMetrics(
        samples=len(correct_values),
        correct_action_loss=float(correct_values.mean()),
        wrong_action_loss=float(wrong_values.mean()),
        loss_margin_wrong_minus_correct=float(
            (wrong_values - correct_values).mean()
        ),
        correct_preference_rate=float(
            (correct_values < wrong_values).float().mean()
        ),
    )


def run_counterfactual_experiment(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration must be a mapping")
    environment = _mapping(payload, "environment")
    data = payload.get("counterfactual", payload.get("data"))
    if not isinstance(data, dict):
        raise ValueError("counterfactual or data configuration is required")
    image_size = tuple(int(value) for value in environment["image_size"])
    world = WorldConfig(
        image_size=image_size,  # type: ignore[arg-type]
        object_count=int(environment.get("object_count", 3)),
        object_radius=float(environment.get("object_radius", 0.045)),
        body_visual_value=float(environment.get("body_visual_value", 1.0)),
        object_visual_value=float(environment.get("object_visual_value", 1.0)),
        external_object_motion_probability=float(
            environment.get("external_object_motion_probability", 0.0)
        ),
        external_object_motion_distance=float(
            environment.get("external_object_motion_distance", 0.08)
        ),
        distractor_body_count=int(
            environment.get("distractor_body_count", 0)
        ),
        distractor_body_motion_probability=float(
            environment.get(
                "distractor_body_motion_probability",
                0.0,
            )
        ),
        seed=int(payload.get("seed", 0)),
    )
    seeds = tuple(
        int(value)
        for value in data.get("test_seeds", data.get("validation_seeds", ()))
    )
    if not seeds:
        raise ValueError("counterfactual test_seeds cannot be empty")
    trajectories = collect_trajectories(
        world,
        BodyConfig(),
        seeds,
        steps_per_seed=int(data.get("steps_per_seed", 128)),
    )
    dataset = TrajectorySequenceDataset(
        trajectories,
        sequence_length=int(data.get("sequence_length", 16)),
        stride=int(data.get("stride", 16)),
    )
    model = load_checkpoint(
        checkpoint_path,
        device=str(data.get("device", "cpu")),
    )
    metrics = evaluate_counterfactual(
        model,
        dataset,
        device=str(data.get("device", "cpu")),
    )
    summary = {
        "result_schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "seed_count": len(seeds),
        "dataset_sequences": len(dataset),
        "metrics": asdict(metrics),
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _per_sample_loss(
    prediction: Any,
    batch: Mapping[str, Tensor],
    config: LossConfig,
) -> Tensor:
    vision = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction.vision_logits,
        batch["next_vision"],
        pos_weight=torch.tensor(
            config.vision_positive_weight,
            device=prediction.vision_logits.device,
        ),
        reduction="none",
    ).mean(dim=tuple(range(1, prediction.vision_logits.ndim)))
    proprioception = torch.nn.functional.mse_loss(
        prediction.proprioception,
        batch["next_proprioception"],
        reduction="none",
    ).mean(dim=tuple(range(1, prediction.proprioception.ndim)))
    touch = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction.touch_logits,
        batch["next_touch"],
        reduction="none",
    ).mean(dim=tuple(range(1, prediction.touch_logits.ndim)))
    return (
        config.vision * vision
        + config.proprioception * proprioception
        + config.touch * touch
    )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate correct-action versus wrong-action prediction."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summary = run_counterfactual_experiment(
        arguments.checkpoint,
        arguments.config,
        output_path=arguments.output,
    )
    print(json.dumps(summary["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
