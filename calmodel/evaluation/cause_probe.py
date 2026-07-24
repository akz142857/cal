"""Probe whether frozen states distinguish self-only from exogenous change.

An external event moves one anonymous object after the body's commanded
motion.  The binary event flag is regenerated from the simulator and is never
stored in a learner trajectory or used to train the prediction model.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam

from calmodel.env.body import BodyConfig
from calmodel.env.world import BodyDiscoveryWorld, WorldConfig
from calmodel.learning.dataset import (
    ACTION_TO_INDEX,
    ACTION_VOCABULARY,
    SeedSplit,
    TrajectorySequenceDataset,
    collect_trajectories,
)
from calmodel.learning.replay import Trajectory
from calmodel.learning.trainer import load_checkpoint
from calmodel.infra.provenance import capture_provenance
from calmodel.model.predictors import SensorimotorPredictor


@dataclass(frozen=True, slots=True)
class CauseData:
    """Frozen features and privileged exogenous-event labels."""

    pre_representations: Tensor
    post_representations: Tensor
    raw_transitions: Tensor
    actions: Tensor
    visual_change: Tensor
    labels: Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.pre_representations,
            self.post_representations,
            self.raw_transitions,
            self.actions,
        )
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("cause feature tensors must be rank two")
        if self.visual_change.ndim != 1 or self.labels.ndim != 1:
            raise ValueError("visual_change and labels must be rank one")
        if len({tensor.shape[0] for tensor in (*tensors, self.visual_change, self.labels)}) != 1:
            raise ValueError("all cause tensors must have equal samples")
        if self.labels.dtype != torch.float32:
            raise ValueError("labels must use float32")
        if any(tensor.requires_grad for tensor in (*tensors, self.visual_change, self.labels)):
            raise ValueError("cause data must be detached")

    def __len__(self) -> int:
        return self.labels.shape[0]


@dataclass(frozen=True, slots=True)
class CauseProbeConfig:
    epochs: int = 100
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0.0:
            raise ValueError("epochs and learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    samples: int
    positives: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    bce: float


@dataclass(frozen=True, slots=True)
class ClassifierResult:
    classifier: nn.Linear
    best_epoch: int
    validation_bce: float
    test: ClassificationMetrics


@torch.no_grad()
def extract_cause_data(
    model: SensorimotorPredictor,
    trajectories: Sequence[Trajectory],
    *,
    device: str = "cpu",
) -> CauseData:
    """Align an event at t→t+1 with states before and after its observation."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    pre_states: list[Tensor] = []
    post_states: list[Tensor] = []
    raw_transitions: list[Tensor] = []
    actions: list[Tensor] = []
    visual_change: list[Tensor] = []
    labels: list[Tensor] = []

    for trajectory in trajectories:
        if len(trajectory) < 2:
            raise ValueError("cause trajectories require at least two steps")
        sequence = TrajectorySequenceDataset(
            (trajectory,),
            sequence_length=len(trajectory),
        )[0]
        output = model(
            vision=sequence["vision"].unsqueeze(0).to(resolved_device),
            proprioception=sequence["proprioception"].unsqueeze(0).to(
                resolved_device
            ),
            touch=sequence["touch"].unsqueeze(0).to(resolved_device),
            actions=sequence["actions"].unsqueeze(0).to(resolved_device),
        )
        representation = output.representation.squeeze(0).detach().cpu()
        # The final transition has no subsequent recurrent state in this
        # trajectory, so it is omitted symmetrically from every feature set.
        pre_states.append(representation[:-1])
        post_states.append(representation[1:])

        transition_features: list[Tensor] = []
        action_features: list[Tensor] = []
        changes: list[float] = []
        for experience in trajectory.experiences[:-1]:
            current = _flatten_observation(experience.observation)
            following = _flatten_observation(experience.next_observation)
            action = torch.zeros(len(ACTION_VOCABULARY), dtype=torch.float32)
            action[ACTION_TO_INDEX[experience.action]] = 1.0
            transition_features.append(torch.cat((current, following, action)))
            action_features.append(action)
            current_vision = torch.tensor(
                experience.observation.vision,
                dtype=torch.float32,
            )
            next_vision = torch.tensor(
                experience.next_observation.vision,
                dtype=torch.float32,
            )
            changes.append(float((next_vision - current_vision).abs().sum()))
        raw_transitions.append(torch.stack(transition_features))
        actions.append(torch.stack(action_features))
        visual_change.append(torch.tensor(changes, dtype=torch.float32))
        labels.append(
            torch.tensor(
                _replay_external_labels(trajectory)[:-1],
                dtype=torch.float32,
            )
        )

    return CauseData(
        pre_representations=torch.cat(pre_states),
        post_representations=torch.cat(post_states),
        raw_transitions=torch.cat(raw_transitions),
        actions=torch.cat(actions),
        visual_change=torch.cat(visual_change),
        labels=torch.cat(labels),
    )


def balance_cause_data(data: CauseData, *, seed: int) -> CauseData:
    """Deterministically downsample the majority class within one split."""

    positive = torch.nonzero(data.labels == 1.0, as_tuple=False).flatten()
    negative = torch.nonzero(data.labels == 0.0, as_tuple=False).flatten()
    count = min(len(positive), len(negative))
    if count == 0:
        raise ValueError("cause data must contain both classes")
    generator = torch.Generator().manual_seed(seed)
    positive = positive[torch.randperm(len(positive), generator=generator)[:count]]
    negative = negative[torch.randperm(len(negative), generator=generator)[:count]]
    selected = torch.cat((positive, negative))
    selected = selected[
        torch.randperm(len(selected), generator=generator)
    ]
    return CauseData(
        pre_representations=data.pre_representations[selected],
        post_representations=data.post_representations[selected],
        raw_transitions=data.raw_transitions[selected],
        actions=data.actions[selected],
        visual_change=data.visual_change[selected],
        labels=data.labels[selected],
    )


def train_linear_classifier(
    train_features: Tensor,
    train_labels: Tensor,
    validation_features: Tensor,
    validation_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    *,
    config: CauseProbeConfig | None = None,
) -> ClassifierResult:
    """Fit a linear readout without updating the prediction model."""

    resolved = config or CauseProbeConfig()
    feature_sizes = {
        train_features.shape[1],
        validation_features.shape[1],
        test_features.shape[1],
    }
    if len(feature_sizes) != 1:
        raise ValueError("classifier feature sizes differ")
    torch.manual_seed(resolved.seed)
    device = torch.device(resolved.device)
    classifier = nn.Linear(train_features.shape[1], 1).to(device)
    optimizer = Adam(
        classifier.parameters(),
        lr=resolved.learning_rate,
        weight_decay=resolved.weight_decay,
    )
    train_x = train_features.to(device)
    train_y = train_labels.to(device)
    validation_x = validation_features.to(device)
    validation_y = validation_labels.to(device)
    best_epoch = 0
    best_bce = float("inf")
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, resolved.epochs + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(train_x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y)
        loss.backward()
        optimizer.step()
        classifier.eval()
        with torch.no_grad():
            validation_bce = float(
                F.binary_cross_entropy_with_logits(
                    classifier(validation_x).squeeze(-1),
                    validation_y,
                )
            )
        if validation_bce < best_bce:
            best_bce = validation_bce
            best_epoch = epoch
            best_state = copy.deepcopy(classifier.state_dict())

    assert best_state is not None
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_features.to(device)).squeeze(-1).cpu()
    return ClassifierResult(
        classifier=classifier,
        best_epoch=best_epoch,
        validation_bce=best_bce,
        test=classification_metrics(test_logits, test_labels),
    )


def classification_metrics(logits: Tensor, labels: Tensor) -> ClassificationMetrics:
    """Compute binary metrics at the fixed zero-logit threshold."""

    if logits.shape != labels.shape or logits.ndim != 1:
        raise ValueError("logits and labels must be equal rank-one tensors")
    predicted = logits >= 0.0
    truth = labels >= 0.5
    true_positive = int((predicted & truth).sum())
    true_negative = int((~predicted & ~truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(positives, 1)
    true_negative_rate = true_negative / max(negatives, 1)
    return ClassificationMetrics(
        samples=len(labels),
        positives=positives,
        accuracy=(true_positive + true_negative) / len(labels),
        balanced_accuracy=(recall + true_negative_rate) / 2.0,
        precision=precision,
        recall=recall,
        f1=2.0 * precision * recall / max(precision + recall, 1e-12),
        bce=float(F.binary_cross_entropy_with_logits(logits, labels)),
    )


def visual_change_baseline(
    validation: CauseData,
    test: CauseData,
) -> tuple[float, ClassificationMetrics]:
    """Select a visual-change threshold on validation and apply it to test."""

    unique_scores = torch.unique(validation.visual_change).sort().values
    if len(unique_scores) == 1:
        thresholds = unique_scores
    else:
        thresholds = torch.cat(
            (
                unique_scores[:1] - 1e-6,
                (unique_scores[:-1] + unique_scores[1:]) / 2.0,
                unique_scores[-1:] + 1e-6,
            )
        )
    best_threshold = float(thresholds[0])
    best_accuracy = -1.0
    for candidate in thresholds:
        predicted = validation.visual_change >= candidate
        accuracy = float(
            (predicted == (validation.labels >= 0.5)).float().mean()
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(candidate)
    # A score minus threshold is a logit-equivalent decision value. BCE is
    # descriptive only because the score is not probability calibrated.
    return best_threshold, classification_metrics(
        test.visual_change - best_threshold,
        test.labels,
    )


def run_cause_probe_experiment(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    output_directory: str | Path,
    external_motion_distance: float | None = None,
) -> dict[str, Any]:
    source = Path(config_path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration must be a mapping")
    environment = _mapping(payload, "environment")
    cause = _mapping(payload, "cause_probe")
    image_size = tuple(int(value) for value in environment["image_size"])
    world_config = WorldConfig(
        image_size=image_size,  # type: ignore[arg-type]
        object_count=int(environment.get("object_count", 3)),
        object_radius=float(environment.get("object_radius", 0.045)),
        body_visual_value=float(environment.get("body_visual_value", 1.0)),
        object_visual_value=float(environment.get("object_visual_value", 1.0)),
        external_object_motion_probability=float(
            cause.get("external_motion_probability", 0.5)
        ),
        external_object_motion_distance=float(
            cause.get("external_motion_distance", 0.08)
            if external_motion_distance is None
            else external_motion_distance
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
    split = SeedSplit(
        train=tuple(int(value) for value in cause["train_seeds"]),
        validation=tuple(int(value) for value in cause["validation_seeds"]),
        test=tuple(int(value) for value in cause["test_seeds"]),
    )
    steps = int(cause.get("steps_per_seed", 256))
    model = load_checkpoint(
        checkpoint_path,
        device=str(cause.get("device", "cpu")),
    )
    trajectories = {
        "train": collect_trajectories(
            world_config,
            BodyConfig(),
            split.train,
            steps_per_seed=steps,
        ),
        "validation": collect_trajectories(
            world_config,
            BodyConfig(),
            split.validation,
            steps_per_seed=steps,
        ),
        "test": collect_trajectories(
            world_config,
            BodyConfig(),
            split.test,
            steps_per_seed=steps,
        ),
    }
    raw_data = {
        name: extract_cause_data(model, items, device=str(cause.get("device", "cpu")))
        for name, items in trajectories.items()
    }
    data = {
        name: balance_cause_data(
            item,
            seed=int(payload.get("seed", 0)) + index,
        )
        for index, (name, item) in enumerate(raw_data.items())
    }
    probe_config = CauseProbeConfig(
        epochs=int(cause.get("epochs", 100)),
        learning_rate=float(cause.get("learning_rate", 1e-2)),
        weight_decay=float(cause.get("weight_decay", 0.0)),
        seed=int(payload.get("seed", 0)),
        device=str(cause.get("device", "cpu")),
    )
    feature_names = (
        "post_representations",
        "pre_representations",
        "raw_transitions",
        "actions",
    )
    results = {}
    for feature_name in feature_names:
        trained = train_linear_classifier(
            getattr(data["train"], feature_name),
            data["train"].labels,
            getattr(data["validation"], feature_name),
            data["validation"].labels,
            getattr(data["test"], feature_name),
            data["test"].labels,
            config=probe_config,
        )
        results[feature_name] = {
            "best_epoch": trained.best_epoch,
            "validation_bce": trained.validation_bce,
            "test": asdict(trained.test),
        }
    threshold, visual_metrics = visual_change_baseline(
        data["validation"],
        data["test"],
    )
    results["visual_change_threshold"] = {
        "threshold": threshold,
        "test": asdict(visual_metrics),
    }
    summary = {
        "result_schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "definition": (
            "0=self-commanded body change only; "
            "1=self-commanded body change plus exogenous object motion"
        ),
        "seed_split": asdict(split),
        "world_config": asdict(world_config),
        "probe_config": asdict(probe_config),
        "samples_before_balancing": {
            name: {
                "total": len(item),
                "positives": int(item.labels.sum()),
            }
            for name, item in raw_data.items()
        },
        "samples_after_balancing": {
            name: len(item) for name, item in data.items()
        },
        "results": results,
        "provenance": capture_provenance(),
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cause-probe-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _flatten_observation(observation: Any) -> Tensor:
    vision = torch.tensor(observation.vision, dtype=torch.float32).flatten()
    proprioception = torch.tensor(
        observation.proprioception,
        dtype=torch.float32,
    )
    touch = torch.tensor(observation.touch, dtype=torch.float32)
    return torch.cat((vision, proprioception, touch))


def _replay_external_labels(trajectory: Trajectory) -> list[bool]:
    world = BodyDiscoveryWorld(
        trajectory.world_config,
        trajectory.body_config,
    )
    current = world.reset(trajectory.seed)
    labels: list[bool] = []
    for experience in trajectory.experiences:
        if current != experience.observation:
            raise RuntimeError("cause replay diverged before event")
        result = world.step(experience.action)
        current = result.observation
        if current != experience.next_observation:
            raise RuntimeError("cause replay diverged after event")
        labels.append(result.external_motion)
    return labels


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe frozen states for self-only versus external change."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cause-probe"),
    )
    parser.add_argument("--external-distance", type=float)
    arguments = parser.parse_args(argv)
    summary = run_cause_probe_experiment(
        arguments.checkpoint,
        arguments.config,
        output_directory=arguments.output,
        external_motion_distance=arguments.external_distance,
    )
    for name, result in summary["results"].items():
        print(
            f"{name}: "
            f"{result['test']['balanced_accuracy']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
