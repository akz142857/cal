"""Training and evaluation for action-conditioned sensory prediction."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from cal.env.body import BodyConfig
from cal.env.world import WorldConfig
from cal.infra.provenance import capture_provenance
from cal.learning.dataset import (
    ACTION_VOCABULARY,
    CounterfactualTrajectorySequenceDataset,
    PairedTrajectorySequenceDataset,
    SeedSplit,
    SequenceTransformConfig,
    TrajectorySequenceDataset,
    collect_trajectories,
    collect_paired_trajectories,
)
from cal.model.predictors import (
    PredictorConfig,
    SensorimotorPrediction,
    SensorimotorPredictor,
)

Batch = Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Weights used for all learned and non-learned prediction baselines."""

    vision: float = 1.0
    proprioception: float = 1.0
    touch: float = 0.25
    vision_positive_weight: float = 4.0
    action_swap_weight: float = 0.0
    action_swap_margin: float = 0.01
    inverse_dynamics_weight: float = 0.0
    control_vision_delta_weight: float = 0.0
    control_delta_change_weight: float = 9.0
    paired_control_consistency_weight: float = 0.0
    action_effect_weight: float = 0.0
    action_effect_positive_weight: float = 8.0
    ownership_weight: float = 0.0
    ownership_positive_weight: float = 8.0
    ownership_target: str = "ownership_target"
    ownership_dice_weight: float = 0.0
    part_slot_weight: float = 0.0
    part_slot_positive_weight: float = 8.0
    part_slot_target: str = "ownership_target"
    spatial_ownership_weight: float = 0.0
    spatial_ownership_positive_weight: float = 1.0
    spatial_ownership_dice_weight: float = 0.0
    spatial_ownership_target: str = "action_envelope"
    global_ownership_weight: float = 0.0
    global_ownership_positive_weight: float = 1.0
    global_ownership_dice_weight: float = 0.0
    global_ownership_target: str = "action_envelope"
    object_slot_weight: float = 0.0
    object_slot_positive_weight: float = 1.0
    object_slot_dice_weight: float = 0.0
    object_slot_target: str = "action_envelope"
    causal_effect_weight: float = 0.0
    causal_effect_positive_weight: float = 1.0
    causal_effect_dice_weight: float = 0.0
    causal_envelope_weight: float = 0.0
    causal_envelope_positive_weight: float = 1.0
    causal_envelope_dice_weight: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.vision,
            self.proprioception,
            self.touch,
            self.vision_positive_weight,
            self.action_swap_weight,
            self.action_swap_margin,
            self.inverse_dynamics_weight,
            self.control_vision_delta_weight,
            self.control_delta_change_weight,
            self.paired_control_consistency_weight,
            self.action_effect_weight,
            self.action_effect_positive_weight,
            self.ownership_weight,
            self.ownership_positive_weight,
            self.ownership_dice_weight,
            self.part_slot_weight,
            self.part_slot_positive_weight,
            self.spatial_ownership_weight,
            self.spatial_ownership_positive_weight,
            self.spatial_ownership_dice_weight,
            self.global_ownership_weight,
            self.global_ownership_positive_weight,
            self.global_ownership_dice_weight,
            self.object_slot_weight,
            self.object_slot_positive_weight,
            self.object_slot_dice_weight,
            self.causal_effect_weight,
            self.causal_effect_positive_weight,
            self.causal_effect_dice_weight,
            self.causal_envelope_weight,
            self.causal_envelope_positive_weight,
            self.causal_envelope_dice_weight,
        ) < 0.0:
            raise ValueError("loss weights cannot be negative")
        if self.part_slot_target not in {
            "ownership_target",
            "action_envelope",
        }:
            raise ValueError("unknown part slot target")
        if self.ownership_target not in {
            "ownership_target",
            "action_envelope",
        }:
            raise ValueError("unknown ownership target")
        if self.spatial_ownership_target not in {
            "ownership_target",
            "action_envelope",
        }:
            raise ValueError("unknown spatial ownership target")
        if self.global_ownership_target not in {
            "ownership_target",
            "action_envelope",
        }:
            raise ValueError("unknown global ownership target")
        if self.object_slot_target not in {
            "ownership_target",
            "action_envelope",
        }:
            raise ValueError("unknown object slot target")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimizer, batching, reproducibility, and stopping configuration."""

    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    checkpoint_selection: str = "best"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip < 0.0:
            raise ValueError("regularization values cannot be negative")
        if self.checkpoint_selection not in {"best", "final"}:
            raise ValueError("checkpoint_selection must be best or final")


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    """Mean losses for one full dataset pass."""

    total: float
    vision_bce: float
    proprioception_mse: float
    touch_bce: float
    samples: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    train: PredictionMetrics
    validation: PredictionMetrics


@dataclass(frozen=True, slots=True)
class TrainingResult:
    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    checkpoint_path: Path | None


@dataclass(frozen=True, slots=True)
class MeanObservationBaseline:
    """Dataset target means used as a constant prediction baseline."""

    vision_probability: Tensor
    proprioception: Tensor
    touch_probability: Tensor


class _MetricAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.vision = 0.0
        self.proprioception = 0.0
        self.touch = 0.0
        self.samples = 0
        self.started_at = time.perf_counter()

    def add(self, losses: Mapping[str, Tensor], sample_count: int) -> None:
        self.total += float(losses["total"].detach()) * sample_count
        self.vision += float(losses["vision_bce"].detach()) * sample_count
        self.proprioception += (
            float(losses["proprioception_mse"].detach()) * sample_count
        )
        self.touch += float(losses["touch_bce"].detach()) * sample_count
        self.samples += sample_count

    def finish(self) -> PredictionMetrics:
        if self.samples == 0:
            raise ValueError("cannot compute metrics for an empty loader")
        return PredictionMetrics(
            total=self.total / self.samples,
            vision_bce=self.vision / self.samples,
            proprioception_mse=self.proprioception / self.samples,
            touch_bce=self.touch / self.samples,
            samples=self.samples,
            duration_seconds=time.perf_counter() - self.started_at,
        )


def make_data_loader(
    dataset: TrajectorySequenceDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Tensor]]:
    """Create a deterministic single-process loader."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def prediction_losses(
    prediction: SensorimotorPrediction,
    batch: Batch,
    config: LossConfig | None = None,
) -> dict[str, Tensor]:
    """Compute the three modality losses and their weighted total."""

    resolved = config or LossConfig()
    positive_weight = torch.tensor(
        resolved.vision_positive_weight,
        device=prediction.vision_logits.device,
    )
    vision_bce = F.binary_cross_entropy_with_logits(
        prediction.vision_logits,
        batch["next_vision"],
        pos_weight=positive_weight,
    )
    proprioception_mse = F.mse_loss(
        prediction.proprioception,
        batch["next_proprioception"],
    )
    touch_bce = F.binary_cross_entropy_with_logits(
        prediction.touch_logits,
        batch["next_touch"],
    )
    total = (
        resolved.vision * vision_bce
        + resolved.proprioception * proprioception_mse
        + resolved.touch * touch_bce
    )
    return {
        "total": total,
        "vision_bce": vision_bce,
        "proprioception_mse": proprioception_mse,
        "touch_bce": touch_bce,
    }


def train_epoch(
    model: SensorimotorPredictor,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    loss_config: LossConfig | None = None,
    gradient_clip: float = 1.0,
) -> PredictionMetrics:
    model.train()
    accumulator = _MetricAccumulator()
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(model, batch)
        losses = prediction_losses(prediction, batch, loss_config)
        resolved_loss = loss_config or LossConfig()
        if resolved_loss.action_swap_weight > 0.0:
            wrong_batch = dict(batch)
            wrong_batch["actions"] = _wrong_actions(batch["actions"])
            wrong_prediction = _predict(model, wrong_batch)
            wrong_losses = prediction_losses(
                wrong_prediction,
                batch,
                loss_config,
            )
            losses["total"] = losses["total"] + (
                resolved_loss.action_swap_weight
                * F.relu(
                    resolved_loss.action_swap_margin
                    + losses["total"]
                    - wrong_losses["total"]
                )
            )
        if (
            resolved_loss.inverse_dynamics_weight > 0.0
            and prediction.inverse_action_logits is not None
            and prediction.inverse_action_logits.numel() > 0
        ):
            inverse_loss = F.cross_entropy(
                prediction.inverse_action_logits.reshape(-1, prediction.inverse_action_logits.shape[-1]),
                batch["actions"][:, :-1].reshape(-1),
            )
            losses["total"] = losses["total"] + (
                resolved_loss.inverse_dynamics_weight * inverse_loss
            )
        if resolved_loss.control_vision_delta_weight > 0.0:
            losses["total"] = losses["total"] + (
                resolved_loss.control_vision_delta_weight
                * control_vision_delta_loss(
                    prediction,
                    batch,
                    change_weight=resolved_loss.control_delta_change_weight,
                )
            )
        if resolved_loss.paired_control_consistency_weight > 0.0:
            paired_prediction = _predict(model, _paired_batch(batch))
            if (
                prediction.control_state is None
                or paired_prediction.control_state is None
            ):
                raise ValueError(
                    "paired control consistency requires control states"
                )
            losses["total"] = losses["total"] + (
                resolved_loss.paired_control_consistency_weight
                * F.mse_loss(
                    prediction.control_state,
                    paired_prediction.control_state,
                )
            )
        if resolved_loss.action_effect_weight > 0.0:
            if prediction.action_effect_logits is None:
                raise ValueError(
                    "action effect loss requires an action effect head"
                )
            if "action_effect" not in batch:
                raise ValueError(
                    "action effect loss requires counterfactual targets"
                )
            effect_loss = F.binary_cross_entropy_with_logits(
                prediction.action_effect_logits,
                batch["action_effect"],
                pos_weight=torch.tensor(
                    resolved_loss.action_effect_positive_weight,
                    device=prediction.action_effect_logits.device,
                ),
            )
            losses["total"] = losses["total"] + (
                resolved_loss.action_effect_weight * effect_loss
            )
        if resolved_loss.ownership_weight > 0.0:
            if prediction.ownership_logits is None:
                raise ValueError(
                    "ownership loss requires an ownership head"
                )
            if resolved_loss.ownership_target not in batch:
                raise ValueError(
                    "ownership loss requires counterfactual targets"
                )
            ownership_loss = F.binary_cross_entropy_with_logits(
                prediction.ownership_logits,
                batch[resolved_loss.ownership_target],
                pos_weight=torch.tensor(
                    resolved_loss.ownership_positive_weight,
                    device=prediction.ownership_logits.device,
                ),
            )
            losses["total"] = losses["total"] + (
                resolved_loss.ownership_weight * ownership_loss
            )
            if resolved_loss.ownership_dice_weight > 0.0:
                probability = torch.sigmoid(prediction.ownership_logits)
                target = batch[resolved_loss.ownership_target]
                dimensions = tuple(range(2, probability.ndim))
                overlap = 2.0 * (probability * target).sum(dim=dimensions)
                scale = probability.sum(dim=dimensions) + target.sum(
                    dim=dimensions
                )
                dice_loss = 1.0 - ((overlap + 1.0) / (scale + 1.0)).mean()
                losses["total"] = losses["total"] + (
                    resolved_loss.ownership_dice_weight * dice_loss
                )
        if resolved_loss.part_slot_weight > 0.0:
            if prediction.part_slot_logits is None:
                raise ValueError("part slot loss requires part slots")
            if resolved_loss.part_slot_target not in batch:
                raise ValueError(
                    "part slot loss requires counterfactual targets"
                )
            slot_loss = F.binary_cross_entropy_with_logits(
                prediction.part_slot_logits,
                batch[resolved_loss.part_slot_target],
                pos_weight=torch.tensor(
                    resolved_loss.part_slot_positive_weight,
                    device=prediction.part_slot_logits.device,
                ),
            )
            losses["total"] = losses["total"] + (
                resolved_loss.part_slot_weight * slot_loss
            )
        if resolved_loss.spatial_ownership_weight > 0.0:
            if prediction.spatial_ownership_logits is None:
                raise ValueError(
                    "spatial ownership loss requires a spatial head"
                )
            target = batch[resolved_loss.spatial_ownership_target]
            spatial_loss = F.binary_cross_entropy_with_logits(
                prediction.spatial_ownership_logits,
                target,
                pos_weight=torch.tensor(
                    resolved_loss.spatial_ownership_positive_weight,
                    device=prediction.spatial_ownership_logits.device,
                ),
            )
            losses["total"] = losses["total"] + (
                resolved_loss.spatial_ownership_weight * spatial_loss
            )
            if resolved_loss.spatial_ownership_dice_weight > 0.0:
                probability = torch.sigmoid(
                    prediction.spatial_ownership_logits
                )
                dimensions = tuple(range(2, probability.ndim))
                overlap = 2.0 * (probability * target).sum(dim=dimensions)
                scale = probability.sum(dim=dimensions) + target.sum(
                    dim=dimensions
                )
                dice_loss = 1.0 - (
                    (overlap + 1.0) / (scale + 1.0)
                ).mean()
                losses["total"] = losses["total"] + (
                    resolved_loss.spatial_ownership_dice_weight
                    * dice_loss
                )
        if resolved_loss.global_ownership_weight > 0.0:
            if prediction.global_ownership_logits is None:
                raise ValueError(
                    "global ownership loss requires a global head"
                )
            target = batch[resolved_loss.global_ownership_target]
            global_loss = F.binary_cross_entropy_with_logits(
                prediction.global_ownership_logits,
                target,
                pos_weight=torch.tensor(
                    resolved_loss.global_ownership_positive_weight,
                    device=prediction.global_ownership_logits.device,
                ),
            )
            losses["total"] = losses["total"] + (
                resolved_loss.global_ownership_weight * global_loss
            )
            if resolved_loss.global_ownership_dice_weight > 0.0:
                probability = torch.sigmoid(
                    prediction.global_ownership_logits
                )
                dimensions = tuple(range(2, probability.ndim))
                overlap = 2.0 * (probability * target).sum(dim=dimensions)
                scale = probability.sum(dim=dimensions) + target.sum(
                    dim=dimensions
                )
                dice_loss = 1.0 - (
                    (overlap + 1.0) / (scale + 1.0)
                ).mean()
                losses["total"] = losses["total"] + (
                    resolved_loss.global_ownership_dice_weight * dice_loss
                )
        if resolved_loss.object_slot_weight > 0.0:
            if prediction.object_slot_logits is None:
                raise ValueError(
                    "object slot loss requires competitive object slots"
                )
            target = batch[resolved_loss.object_slot_target]
            object_slot_loss = F.binary_cross_entropy_with_logits(
                prediction.object_slot_logits,
                target,
                pos_weight=torch.tensor(
                    resolved_loss.object_slot_positive_weight,
                    device=prediction.object_slot_logits.device,
                ),
            )
            losses["total"] = losses["total"] + (
                resolved_loss.object_slot_weight * object_slot_loss
            )
            if resolved_loss.object_slot_dice_weight > 0.0:
                probability = torch.sigmoid(prediction.object_slot_logits)
                dimensions = tuple(range(2, probability.ndim))
                overlap = 2.0 * (probability * target).sum(dim=dimensions)
                scale = probability.sum(dim=dimensions) + target.sum(
                    dim=dimensions
                )
                dice_loss = 1.0 - (
                    (overlap + 1.0) / (scale + 1.0)
                ).mean()
                losses["total"] = losses["total"] + (
                    resolved_loss.object_slot_dice_weight * dice_loss
                )
        if resolved_loss.causal_effect_weight > 0.0:
            if prediction.causal_action_effect_logits is None:
                raise ValueError(
                    "causal effect loss requires an action-basis head"
                )
            target = batch["all_action_effects"]
            effect_loss = F.binary_cross_entropy_with_logits(
                prediction.causal_action_effect_logits,
                target,
                pos_weight=torch.tensor(
                    resolved_loss.causal_effect_positive_weight,
                    device=prediction.causal_action_effect_logits.device,
                ),
            )
            losses["total"] = losses["total"] + (
                resolved_loss.causal_effect_weight * effect_loss
            )
            if resolved_loss.causal_effect_dice_weight > 0.0:
                probability = torch.sigmoid(
                    prediction.causal_action_effect_logits
                )
                dimensions = tuple(range(3, probability.ndim))
                overlap = 2.0 * (probability * target).sum(dim=dimensions)
                scale = probability.sum(dim=dimensions) + target.sum(
                    dim=dimensions
                )
                dice_loss = 1.0 - (
                    (overlap + 1.0) / (scale + 1.0)
                ).mean()
                losses["total"] = losses["total"] + (
                    resolved_loss.causal_effect_dice_weight * dice_loss
                )
        if resolved_loss.causal_envelope_weight > 0.0:
            if prediction.causal_envelope_logits is None:
                raise ValueError(
                    "causal envelope loss requires an action-basis head"
                )
            target = batch["action_envelope"]
            envelope_loss = F.binary_cross_entropy_with_logits(
                prediction.causal_envelope_logits,
                target,
                pos_weight=torch.tensor(
                    resolved_loss.causal_envelope_positive_weight,
                    device=prediction.causal_envelope_logits.device,
                ),
            )
            losses["total"] = losses["total"] + (
                resolved_loss.causal_envelope_weight * envelope_loss
            )
            if resolved_loss.causal_envelope_dice_weight > 0.0:
                probability = torch.sigmoid(
                    prediction.causal_envelope_logits
                )
                dimensions = tuple(range(2, probability.ndim))
                overlap = 2.0 * (probability * target).sum(dim=dimensions)
                scale = probability.sum(dim=dimensions) + target.sum(
                    dim=dimensions
                )
                dice_loss = 1.0 - (
                    (overlap + 1.0) / (scale + 1.0)
                ).mean()
                losses["total"] = losses["total"] + (
                    resolved_loss.causal_envelope_dice_weight * dice_loss
                )
        losses["total"].backward()
        if gradient_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        accumulator.add(losses, batch["vision"].shape[0])
    return accumulator.finish()


@torch.no_grad()
def evaluate_model(
    model: SensorimotorPredictor,
    loader: DataLoader[dict[str, Tensor]],
    *,
    device: torch.device,
    loss_config: LossConfig | None = None,
) -> PredictionMetrics:
    model.eval()
    accumulator = _MetricAccumulator()
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        losses = prediction_losses(_predict(model, batch), batch, loss_config)
        accumulator.add(losses, batch["vision"].shape[0])
    return accumulator.finish()


def fit_model(
    model: SensorimotorPredictor,
    train_loader: DataLoader[dict[str, Tensor]],
    validation_loader: DataLoader[dict[str, Tensor]],
    *,
    training_config: TrainingConfig | None = None,
    loss_config: LossConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Train a predictor and retain the best validation checkpoint."""

    config = training_config or TrainingConfig()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    resolved_path = Path(checkpoint_path) if checkpoint_path is not None else None
    history: list[EpochRecord] = []
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            loss_config=loss_config,
            gradient_clip=config.gradient_clip,
        )
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device=device,
            loss_config=loss_config,
        )
        history.append(
            EpochRecord(
                epoch=epoch,
                train=train_metrics,
                validation=validation_metrics,
            )
        )
        if (
            config.checkpoint_selection == "final"
            or validation_metrics.total < best_loss
        ):
            best_loss = validation_metrics.total
            best_epoch = epoch
            if resolved_path is not None:
                save_checkpoint(
                    model,
                    resolved_path,
                    epoch=epoch,
                    validation_loss=best_loss,
                )

    return TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        checkpoint_path=resolved_path,
    )


def save_checkpoint(
    model: SensorimotorPredictor,
    path: str | Path,
    *,
    epoch: int,
    validation_loss: float,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
        },
        destination,
    )
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> SensorimotorPredictor:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    config_data = dict(payload["model_config"])
    config_data["image_size"] = tuple(config_data["image_size"])
    model = SensorimotorPredictor(PredictorConfig(**config_data))
    model.load_state_dict(payload["model_state"])
    model.to(torch.device(device))
    return model


@torch.no_grad()
def evaluate_copy_baseline(
    loader: DataLoader[dict[str, Tensor]],
    *,
    loss_config: LossConfig | None = None,
) -> PredictionMetrics:
    """Evaluate the assumption that every modality remains unchanged."""

    accumulator = _MetricAccumulator()
    for batch in loader:
        prediction = _baseline_prediction(
            vision_probability=batch["vision"],
            proprioception=batch["proprioception"],
            touch_probability=batch["touch"],
        )
        losses = prediction_losses(prediction, batch, loss_config)
        accumulator.add(losses, batch["vision"].shape[0])
    return accumulator.finish()


@torch.no_grad()
def fit_mean_baseline(
    loader: DataLoader[dict[str, Tensor]],
) -> MeanObservationBaseline:
    """Fit one constant prediction from training target means."""

    vision_sum: Tensor | None = None
    proprioception_sum: Tensor | None = None
    touch_sum: Tensor | None = None
    step_count = 0
    for batch in loader:
        batch_steps = batch["next_vision"].shape[0] * batch["next_vision"].shape[1]
        current_vision = batch["next_vision"].sum(dim=(0, 1), keepdim=True)
        current_proprioception = batch["next_proprioception"].sum(
            dim=(0, 1), keepdim=True
        )
        current_touch = batch["next_touch"].sum(dim=(0, 1), keepdim=True)
        vision_sum = (
            current_vision if vision_sum is None else vision_sum + current_vision
        )
        proprioception_sum = (
            current_proprioception
            if proprioception_sum is None
            else proprioception_sum + current_proprioception
        )
        touch_sum = (
            current_touch if touch_sum is None else touch_sum + current_touch
        )
        step_count += batch_steps
    if step_count == 0 or vision_sum is None:
        raise ValueError("cannot fit mean baseline on empty data")
    assert proprioception_sum is not None and touch_sum is not None
    return MeanObservationBaseline(
        vision_probability=vision_sum / step_count,
        proprioception=proprioception_sum / step_count,
        touch_probability=touch_sum / step_count,
    )


@torch.no_grad()
def evaluate_mean_baseline(
    baseline: MeanObservationBaseline,
    loader: DataLoader[dict[str, Tensor]],
    *,
    loss_config: LossConfig | None = None,
) -> PredictionMetrics:
    accumulator = _MetricAccumulator()
    for batch in loader:
        batch_size, time = batch["vision"].shape[:2]
        prediction = _baseline_prediction(
            vision_probability=baseline.vision_probability.expand(
                batch_size, time, -1, -1, -1
            ),
            proprioception=baseline.proprioception.expand(
                batch_size, time, -1
            ),
            touch_probability=baseline.touch_probability.expand(
                batch_size, time, -1
            ),
        )
        losses = prediction_losses(prediction, batch, loss_config)
        accumulator.add(losses, batch_size)
    return accumulator.finish()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _predict(model: SensorimotorPredictor, batch: Batch) -> SensorimotorPrediction:
    return model(
        vision=batch["vision"],
        proprioception=batch["proprioception"],
        touch=batch["touch"],
        actions=batch["actions"],
    )


def _wrong_actions(actions: Tensor) -> Tensor:
    """Create an action-swapped negative without simulator labels."""

    return (actions + 1) % len(ACTION_VOCABULARY)


def control_vision_delta_loss(
    prediction: SensorimotorPrediction,
    batch: Batch,
    *,
    change_weight: float = 9.0,
) -> Tensor:
    """Predict signed visual change while emphasizing changed pixels."""

    if prediction.control_vision_delta is None:
        raise ValueError("model does not expose a control vision delta")
    if change_weight < 0.0:
        raise ValueError("change_weight cannot be negative")
    target = batch["next_vision"] - batch["vision"]
    weights = 1.0 + change_weight * target.abs()
    squared_error = (prediction.control_vision_delta - target).square()
    return (weights * squared_error).sum() / weights.sum()


def _move_batch(batch: Batch, device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _paired_batch(batch: Batch) -> dict[str, Tensor]:
    names = ("vision", "proprioception", "touch", "actions")
    if any(f"pair_{name}" not in batch for name in names):
        raise ValueError("batch does not contain a paired trajectory")
    return {name: batch[f"pair_{name}"] for name in names}


def _baseline_prediction(
    *,
    vision_probability: Tensor,
    proprioception: Tensor,
    touch_probability: Tensor,
) -> SensorimotorPrediction:
    vision_logits = torch.logit(vision_probability.clamp(1e-4, 1.0 - 1e-4))
    touch_logits = torch.logit(touch_probability.clamp(1e-4, 1.0 - 1e-4))
    batch, time = vision_probability.shape[:2]
    empty_representation = torch.empty(
        batch,
        time,
        0,
        device=vision_probability.device,
    )
    return SensorimotorPrediction(
        vision_logits=vision_logits,
        proprioception=proprioception,
        touch_logits=touch_logits,
        representation=empty_representation,
        final_state=torch.empty(
            1,
            batch,
            0,
            device=vision_probability.device,
        ),
    )


def run_experiment(
    config_path: str | Path,
    *,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Run one configured baseline experiment and persist its evidence."""

    source = Path(config_path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration must be a mapping")

    environment_data = _mapping(payload, "environment")
    model_data = _mapping(payload, "model")
    data_data = _mapping(payload, "data")
    training_data = _mapping(payload, "training")

    image_size = tuple(int(value) for value in environment_data["image_size"])
    world_config = WorldConfig(
        image_size=image_size,  # type: ignore[arg-type]
        object_count=int(environment_data.get("object_count", 3)),
        object_radius=float(environment_data.get("object_radius", 0.045)),
        body_visual_value=float(
            environment_data.get("body_visual_value", 1.0)
        ),
        object_visual_value=float(
            environment_data.get("object_visual_value", 1.0)
        ),
        vision_noise_probability=float(
            environment_data.get("vision_noise_probability", 0.0)
        ),
        proprioception_noise_std=float(
            environment_data.get("proprioception_noise_std", 0.0)
        ),
        touch_dropout_probability=float(
            environment_data.get("touch_dropout_probability", 0.0)
        ),
        external_object_motion_probability=float(
            environment_data.get(
                "external_object_motion_probability",
                0.0,
            )
        ),
        external_object_motion_distance=float(
            environment_data.get("external_object_motion_distance", 0.08)
        ),
        distractor_body_count=int(
            environment_data.get("distractor_body_count", 0)
        ),
        distractor_body_motion_probability=float(
            environment_data.get(
                "distractor_body_motion_probability",
                0.0,
            )
        ),
        seed=int(payload.get("seed", 0)),
    )
    body_config = BodyConfig()
    seed_split = SeedSplit(
        train=tuple(int(value) for value in data_data["train_seeds"]),
        validation=tuple(
            int(value) for value in data_data["validation_seeds"]
        ),
        test=tuple(int(value) for value in data_data["test_seeds"]),
        transfer=tuple(int(value) for value in data_data.get("transfer_seeds", ())),
    )
    steps_per_seed = int(data_data["steps_per_seed"])
    action_repeat_probability = float(
        data_data.get("action_repeat_probability", 0.0)
    )
    sequence_length = int(data_data["sequence_length"])
    stride = int(data_data.get("stride", sequence_length))
    transform = SequenceTransformConfig(
        temporal_shuffle=bool(
            environment_data.get("shuffle_modalities", False)
        ),
        randomize_actions=bool(
            environment_data.get("randomize_actions", False)
        ),
        sensor_blackout_probability=float(
            environment_data.get("input_blackout_probability", 0.0)
        ),
        sensor_blackout_span=int(
            environment_data.get("input_blackout_span", 1)
        ),
        seed=int(payload.get("seed", 0)),
    )
    if bool(data_data.get("paired_scene_consistency", False)):
        train_dataset = PairedTrajectorySequenceDataset(
            collect_paired_trajectories(
                world_config,
                body_config,
                seed_split.train,
                steps_per_seed=steps_per_seed,
                action_repeat_probability=action_repeat_probability,
            ),
            sequence_length=sequence_length,
            stride=stride,
            transform=transform,
        )
        validation_dataset = PairedTrajectorySequenceDataset(
            collect_paired_trajectories(
                world_config,
                body_config,
                seed_split.validation,
                steps_per_seed=steps_per_seed,
                action_repeat_probability=action_repeat_probability,
            ),
            sequence_length=sequence_length,
            stride=stride,
            transform=transform,
        )
    else:
        dataset_type = (
            CounterfactualTrajectorySequenceDataset
            if bool(data_data.get("counterfactual_action_effect", False))
            else TrajectorySequenceDataset
        )
        train_dataset = dataset_type(
            collect_trajectories(
                world_config,
                body_config,
                seed_split.train,
                steps_per_seed=steps_per_seed,
                action_repeat_probability=action_repeat_probability,
            ),
            sequence_length=sequence_length,
            stride=stride,
            transform=transform,
        )
        validation_dataset = dataset_type(
            collect_trajectories(
                world_config,
                body_config,
                seed_split.validation,
                steps_per_seed=steps_per_seed,
                action_repeat_probability=action_repeat_probability,
            ),
            sequence_length=sequence_length,
            stride=stride,
            transform=transform,
        )
    training_config = TrainingConfig(
        epochs=int(training_data.get("epochs", 10)),
        batch_size=int(training_data.get("batch_size", 16)),
        learning_rate=float(training_data.get("learning_rate", 1e-3)),
        weight_decay=float(training_data.get("weight_decay", 0.0)),
        gradient_clip=float(training_data.get("gradient_clip", 1.0)),
        seed=int(payload.get("seed", 0)),
        device=str(training_data.get("device", "cpu")),
        checkpoint_selection=str(
            training_data.get("checkpoint_selection", "best")
        ),
    )
    train_loader = make_data_loader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=training_config.seed,
    )
    train_evaluation_loader = make_data_loader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        seed=training_config.seed,
    )
    validation_loader = make_data_loader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        seed=training_config.seed,
    )
    predictor_config = PredictorConfig(
        image_size=image_size,  # type: ignore[arg-type]
        hidden_size=int(model_data.get("hidden_size", 128)),
        core_type=str(model_data.get("recurrent_core", "gru")),
        residual_prediction=bool(
            model_data.get("residual_prediction", True)
        ),
        copy_logit_strength=float(
            model_data.get("copy_logit_strength", 3.0)
        ),
        use_vision=bool(environment_data.get("include_vision", True)),
        use_proprioception=bool(
            environment_data.get("include_proprioception", True)
        ),
        use_touch=bool(environment_data.get("include_touch", True)),
        use_action=bool(environment_data.get("include_action", True)),
        use_motion=bool(model_data.get("use_motion", False)),
        motion_size=int(model_data.get("motion_size", 8)),
        motion_passthrough=bool(
            model_data.get("motion_passthrough", False)
        ),
        use_inverse_dynamics=bool(
            model_data.get("use_inverse_dynamics", False)
        ),
        control_state_size=int(model_data.get("control_state_size", 0)),
        use_control_vision_delta=bool(
            model_data.get("use_control_vision_delta", False)
        ),
        action_effect_size=int(model_data.get("action_effect_size", 0)),
        ownership_state_size=int(
            model_data.get("ownership_state_size", 0)
        ),
        ownership_recurrent=bool(
            model_data.get("ownership_recurrent", True)
        ),
        ownership_vision_support=bool(
            model_data.get("ownership_vision_support", False)
        ),
        ownership_copy_logit_strength=float(
            model_data.get("ownership_copy_logit_strength", 3.0)
        ),
        ownership_effect_attention=bool(
            model_data.get("ownership_effect_attention", False)
        ),
        ownership_effect_attention_strength=float(
            model_data.get("ownership_effect_attention_strength", 2.0)
        ),
        ownership_prediction_effect_attention=bool(
            model_data.get(
                "ownership_prediction_effect_attention",
                False,
            )
        ),
        part_slot_count=int(model_data.get("part_slot_count", 0)),
        part_slot_size=int(model_data.get("part_slot_size", 0)),
        part_slot_recurrent=bool(
            model_data.get("part_slot_recurrent", True)
        ),
        spatial_ownership_channels=int(
            model_data.get("spatial_ownership_channels", 0)
        ),
        spatial_ownership_action_size=int(
            model_data.get("spatial_ownership_action_size", 4)
        ),
        spatial_ownership_recurrent=bool(
            model_data.get("spatial_ownership_recurrent", True)
        ),
        spatial_ownership_vision_support=bool(
            model_data.get("spatial_ownership_vision_support", True)
        ),
        spatial_ownership_hard_support=bool(
            model_data.get("spatial_ownership_hard_support", False)
        ),
        global_ownership_query_size=int(
            model_data.get("global_ownership_query_size", 0)
        ),
        global_ownership_token_size=int(
            model_data.get("global_ownership_token_size", 16)
        ),
        global_ownership_action_size=int(
            model_data.get("global_ownership_action_size", 8)
        ),
        global_ownership_recurrent=bool(
            model_data.get("global_ownership_recurrent", True)
        ),
        global_ownership_hard_support=bool(
            model_data.get("global_ownership_hard_support", True)
        ),
        global_ownership_self_attention=bool(
            model_data.get("global_ownership_self_attention", False)
        ),
        object_slot_count=int(model_data.get("object_slot_count", 0)),
        object_slot_size=int(model_data.get("object_slot_size", 16)),
        object_slot_iterations=int(
            model_data.get("object_slot_iterations", 2)
        ),
        object_slot_action_size=int(
            model_data.get("object_slot_action_size", 4)
        ),
        object_slot_ownership_size=int(
            model_data.get("object_slot_ownership_size", 8)
        ),
        object_slot_recurrent=bool(
            model_data.get("object_slot_recurrent", True)
        ),
        object_slot_hard_support=bool(
            model_data.get("object_slot_hard_support", True)
        ),
        object_slot_exclusive_ownership=bool(
            model_data.get("object_slot_exclusive_ownership", False)
        ),
        causal_effect_channels=int(
            model_data.get("causal_effect_channels", 0)
        ),
        causal_effect_action_size=int(
            model_data.get("causal_effect_action_size", 4)
        ),
        causal_effect_recurrent=bool(
            model_data.get("causal_effect_recurrent", True)
        ),
        causal_effect_hard_support=bool(
            model_data.get("causal_effect_hard_support", True)
        ),
    )
    torch.manual_seed(training_config.seed)
    model = SensorimotorPredictor(predictor_config)
    loss_config = LossConfig(
        action_swap_weight=float(
            training_data.get("action_swap_weight", 0.0)
        ),
        action_swap_margin=float(
            training_data.get("action_swap_margin", 0.01)
        ),
        inverse_dynamics_weight=float(
            training_data.get("inverse_dynamics_weight", 0.0)
        ),
        control_vision_delta_weight=float(
            training_data.get("control_vision_delta_weight", 0.0)
        ),
        control_delta_change_weight=float(
            training_data.get("control_delta_change_weight", 9.0)
        ),
        paired_control_consistency_weight=float(
            training_data.get("paired_control_consistency_weight", 0.0)
        ),
        action_effect_weight=float(
            training_data.get("action_effect_weight", 0.0)
        ),
        action_effect_positive_weight=float(
            training_data.get("action_effect_positive_weight", 8.0)
        ),
        ownership_weight=float(
            training_data.get("ownership_weight", 0.0)
        ),
        ownership_positive_weight=float(
            training_data.get("ownership_positive_weight", 8.0)
        ),
        ownership_target=str(
            training_data.get("ownership_target", "ownership_target")
        ),
        ownership_dice_weight=float(
            training_data.get("ownership_dice_weight", 0.0)
        ),
        part_slot_weight=float(
            training_data.get("part_slot_weight", 0.0)
        ),
        part_slot_positive_weight=float(
            training_data.get("part_slot_positive_weight", 8.0)
        ),
        part_slot_target=str(
            training_data.get("part_slot_target", "ownership_target")
        ),
        spatial_ownership_weight=float(
            training_data.get("spatial_ownership_weight", 0.0)
        ),
        spatial_ownership_positive_weight=float(
            training_data.get("spatial_ownership_positive_weight", 1.0)
        ),
        spatial_ownership_dice_weight=float(
            training_data.get("spatial_ownership_dice_weight", 0.0)
        ),
        spatial_ownership_target=str(
            training_data.get(
                "spatial_ownership_target",
                "action_envelope",
            )
        ),
        global_ownership_weight=float(
            training_data.get("global_ownership_weight", 0.0)
        ),
        global_ownership_positive_weight=float(
            training_data.get("global_ownership_positive_weight", 1.0)
        ),
        global_ownership_dice_weight=float(
            training_data.get("global_ownership_dice_weight", 0.0)
        ),
        global_ownership_target=str(
            training_data.get(
                "global_ownership_target",
                "action_envelope",
            )
        ),
        object_slot_weight=float(
            training_data.get("object_slot_weight", 0.0)
        ),
        object_slot_positive_weight=float(
            training_data.get("object_slot_positive_weight", 1.0)
        ),
        object_slot_dice_weight=float(
            training_data.get("object_slot_dice_weight", 0.0)
        ),
        object_slot_target=str(
            training_data.get("object_slot_target", "action_envelope")
        ),
        causal_effect_weight=float(
            training_data.get("causal_effect_weight", 0.0)
        ),
        causal_effect_positive_weight=float(
            training_data.get("causal_effect_positive_weight", 1.0)
        ),
        causal_effect_dice_weight=float(
            training_data.get("causal_effect_dice_weight", 0.0)
        ),
        causal_envelope_weight=float(
            training_data.get("causal_envelope_weight", 0.0)
        ),
        causal_envelope_positive_weight=float(
            training_data.get("causal_envelope_positive_weight", 1.0)
        ),
        causal_envelope_dice_weight=float(
            training_data.get("causal_envelope_dice_weight", 0.0)
        ),
    )
    copy_metrics = evaluate_copy_baseline(
        validation_loader,
        loss_config=loss_config,
    )
    mean_baseline = fit_mean_baseline(train_evaluation_loader)
    mean_metrics = evaluate_mean_baseline(
        mean_baseline,
        validation_loader,
        loss_config=loss_config,
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result = fit_model(
        model,
        train_loader,
        validation_loader,
        training_config=training_config,
        loss_config=loss_config,
        checkpoint_path=output / "checkpoint.pt",
    )
    summary = {
        "result_schema_version": 1,
        "name": str(payload.get("name", source.stem)),
        "config_path": str(source),
        "parameter_count": count_trainable_parameters(model),
        "train_sequences": len(train_dataset),
        "validation_sequences": len(validation_dataset),
        "seed_split": asdict(seed_split),
        "copy_baseline": asdict(copy_metrics),
        "mean_baseline": asdict(mean_metrics),
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "history": [
            {
                "epoch": item.epoch,
                "train": asdict(item.train),
                "validation": asdict(item.validation),
            }
            for item in result.history
        ],
        "provenance": capture_provenance(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "config.yaml").write_text(
        source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return summary


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a Cal sensorimotor prediction baseline."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baseline"),
    )
    arguments = parser.parse_args(argv)
    summary = run_experiment(arguments.config, output_directory=arguments.output)
    print(
        f"best validation loss {summary['best_validation_loss']:.6f} "
        f"at epoch {summary['best_epoch']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
