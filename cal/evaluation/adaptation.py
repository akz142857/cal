"""Zero-shot and finite-experience adaptation to changed bodies and sensors."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.optim import Adam

from cal.env.body import BodyConfig
from cal.env.world import WorldConfig
from cal.learning.dataset import (
    SeedSplit,
    TrajectorySequenceDataset,
    collect_trajectories,
)
from cal.learning.trainer import (
    LossConfig,
    PredictionMetrics,
    evaluate_model,
    load_checkpoint,
    make_data_loader,
    train_epoch,
)
from cal.infra.provenance import capture_provenance
from cal.model.predictors import SensorimotorPredictor


@dataclass(frozen=True, slots=True)
class AdaptationVariant:
    name: str
    world_config: WorldConfig
    body_config: BodyConfig


@dataclass(frozen=True, slots=True)
class AdaptationConfig:
    train_seeds: tuple[int, ...] = (300, 301)
    test_seeds: tuple[int, ...] = (350, 351)
    original_test_seeds: tuple[int, ...] = (600, 601)
    experience_budgets: tuple[int, ...] = (0, 32, 64, 128, 256)
    test_steps_per_seed: int = 128
    sequence_length: int = 8
    stride: int = 8
    optimization_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 5e-4
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.train_seeds or not self.test_seeds:
            raise ValueError("adaptation train and test seeds cannot be empty")
        if set(self.train_seeds) & set(self.test_seeds):
            raise ValueError("adaptation train and test seeds must be disjoint")
        if not self.experience_budgets or self.experience_budgets[0] != 0:
            raise ValueError("experience_budgets must begin with zero")
        if any(
            current >= following
            for current, following in zip(
                self.experience_budgets,
                self.experience_budgets[1:],
            )
        ):
            raise ValueError("experience_budgets must be strictly increasing")
        if min(
            self.test_steps_per_seed,
            self.sequence_length,
            self.stride,
            self.optimization_epochs,
            self.batch_size,
        ) <= 0:
            raise ValueError("adaptation dimensions must be positive")


@dataclass(frozen=True, slots=True)
class AdaptationPoint:
    requested_experience_steps: int
    actual_experience_steps: int
    changed_body: PredictionMetrics
    original_body: PredictionMetrics


@dataclass(frozen=True, slots=True)
class VariantResult:
    name: str
    curve: tuple[AdaptationPoint, ...]


def standard_variants(base_world: WorldConfig) -> tuple[AdaptationVariant, ...]:
    """Body and sensor changes required by M1.7."""

    return (
        AdaptationVariant(
            "long_upper_arm",
            base_world,
            BodyConfig(link_lengths=(0.28, 0.18)),
        ),
        AdaptationVariant(
            "long_forearm",
            base_world,
            BodyConfig(link_lengths=(0.22, 0.24)),
        ),
        AdaptationVariant(
            "larger_action_step",
            base_world,
            BodyConfig(angle_step=0.28),
        ),
        AdaptationVariant(
            "elbow_disabled",
            base_world,
            BodyConfig(elbow_enabled=False),
        ),
        AdaptationVariant(
            "touch_dropout_50",
            replace(base_world, touch_dropout_probability=0.5),
            BodyConfig(),
        ),
        AdaptationVariant(
            "proprioception_noise",
            replace(base_world, proprioception_noise_std=0.1),
            BodyConfig(),
        ),
    )


def evaluate_variant_adaptation(
    base_model: SensorimotorPredictor,
    base_world: WorldConfig,
    variant: AdaptationVariant,
    *,
    config: AdaptationConfig | None = None,
    loss_config: LossConfig | None = None,
) -> VariantResult:
    """Evaluate independent fine-tuning runs at increasing experience budgets."""

    resolved = config or AdaptationConfig()
    device = torch.device(resolved.device)
    changed_test = _dataset(
        variant.world_config,
        variant.body_config,
        resolved.test_seeds,
        steps_per_seed=resolved.test_steps_per_seed,
        sequence_length=resolved.sequence_length,
        stride=resolved.stride,
    )
    original_test = _dataset(
        base_world,
        BodyConfig(),
        resolved.original_test_seeds,
        steps_per_seed=resolved.test_steps_per_seed,
        sequence_length=resolved.sequence_length,
        stride=resolved.stride,
    )
    changed_loader = make_data_loader(
        changed_test,
        batch_size=resolved.batch_size,
        shuffle=False,
        seed=resolved.seed,
    )
    original_loader = make_data_loader(
        original_test,
        batch_size=resolved.batch_size,
        shuffle=False,
        seed=resolved.seed,
    )
    curve: list[AdaptationPoint] = []

    for budget in resolved.experience_budgets:
        torch.manual_seed(resolved.seed)
        model = copy.deepcopy(base_model).to(device)
        actual_steps = 0
        if budget > 0:
            steps_per_seed = max(
                resolved.sequence_length,
                budget // len(resolved.train_seeds),
            )
            actual_steps = steps_per_seed * len(resolved.train_seeds)
            adaptation_data = _dataset(
                variant.world_config,
                variant.body_config,
                resolved.train_seeds,
                steps_per_seed=steps_per_seed,
                sequence_length=resolved.sequence_length,
                stride=resolved.stride,
            )
            adaptation_loader = make_data_loader(
                adaptation_data,
                batch_size=resolved.batch_size,
                shuffle=True,
                seed=resolved.seed,
            )
            optimizer = Adam(model.parameters(), lr=resolved.learning_rate)
            for _ in range(resolved.optimization_epochs):
                train_epoch(
                    model,
                    adaptation_loader,
                    optimizer,
                    device=device,
                    loss_config=loss_config,
                )

        curve.append(
            AdaptationPoint(
                requested_experience_steps=budget,
                actual_experience_steps=actual_steps,
                changed_body=evaluate_model(
                    model,
                    changed_loader,
                    device=device,
                    loss_config=loss_config,
                ),
                original_body=evaluate_model(
                    model,
                    original_loader,
                    device=device,
                    loss_config=loss_config,
                ),
            )
        )
    return VariantResult(name=variant.name, curve=tuple(curve))


def run_adaptation_experiment(
    checkpoint_path: str | Path,
    experiment_path: str | Path,
    *,
    output_directory: str | Path,
) -> dict[str, Any]:
    source = Path(experiment_path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration must be a mapping")
    environment = _mapping(payload, "environment")
    adaptation_data = _mapping(payload, "adaptation")
    image_size = tuple(int(value) for value in environment["image_size"])
    base_world = WorldConfig(
        image_size=image_size,  # type: ignore[arg-type]
        object_count=int(environment.get("object_count", 3)),
        object_radius=float(environment.get("object_radius", 0.045)),
        body_visual_value=float(
            environment.get("body_visual_value", 1.0)
        ),
        object_visual_value=float(
            environment.get("object_visual_value", 1.0)
        ),
        vision_noise_probability=float(
            environment.get("vision_noise_probability", 0.0)
        ),
        proprioception_noise_std=float(
            environment.get("proprioception_noise_std", 0.0)
        ),
        touch_dropout_probability=float(
            environment.get("touch_dropout_probability", 0.0)
        ),
        external_object_motion_probability=float(
            environment.get(
                "external_object_motion_probability",
                0.0,
            )
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
    seed_split = SeedSplit(
        train=tuple(int(value) for value in adaptation_data["train_seeds"]),
        validation=tuple(
            int(value) for value in adaptation_data["test_seeds"]
        ),
        test=tuple(
            int(value) for value in adaptation_data["original_test_seeds"]
        ),
    )
    config = AdaptationConfig(
        train_seeds=seed_split.train,
        test_seeds=seed_split.validation,
        original_test_seeds=seed_split.test,
        experience_budgets=tuple(
            int(value) for value in adaptation_data["experience_budgets"]
        ),
        test_steps_per_seed=int(
            adaptation_data.get("test_steps_per_seed", 128)
        ),
        sequence_length=int(adaptation_data.get("sequence_length", 8)),
        stride=int(adaptation_data.get("stride", 8)),
        optimization_epochs=int(
            adaptation_data.get("optimization_epochs", 10)
        ),
        batch_size=int(adaptation_data.get("batch_size", 16)),
        learning_rate=float(adaptation_data.get("learning_rate", 5e-4)),
        seed=int(payload.get("seed", 0)),
        device=str(adaptation_data.get("device", "cpu")),
    )
    model = load_checkpoint(checkpoint_path, device=config.device)
    results = [
        evaluate_variant_adaptation(
            model,
            base_world,
            variant,
            config=config,
        )
        for variant in standard_variants(base_world)
    ]
    summary = {
        "result_schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "config": str(experiment_path),
        "adaptation_config": asdict(config),
        "variants": [
            {
                "name": result.name,
                "curve": [
                    {
                        "requested_experience_steps": point.requested_experience_steps,
                        "actual_experience_steps": point.actual_experience_steps,
                        "changed_body": asdict(point.changed_body),
                        "original_body": asdict(point.original_body),
                    }
                    for point in result.curve
                ],
            }
            for result in results
        ],
        "provenance": capture_provenance(),
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "adaptation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _dataset(
    world_config: WorldConfig,
    body_config: BodyConfig,
    seeds: Sequence[int],
    *,
    steps_per_seed: int,
    sequence_length: int,
    stride: int,
) -> TrajectorySequenceDataset:
    return TrajectorySequenceDataset(
        collect_trajectories(
            world_config,
            body_config,
            seeds,
            steps_per_seed=steps_per_seed,
        ),
        sequence_length=sequence_length,
        stride=stride,
    )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure Cal adaptation to changed bodies and sensors."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/body-adaptation"),
    )
    arguments = parser.parse_args(argv)
    summary = run_adaptation_experiment(
        arguments.checkpoint,
        arguments.config,
        output_directory=arguments.output,
    )
    for variant in summary["variants"]:
        start = variant["curve"][0]["changed_body"]["total"]
        end = variant["curve"][-1]["changed_body"]["total"]
        print(f"{variant['name']}: {start:.4f} -> {end:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
