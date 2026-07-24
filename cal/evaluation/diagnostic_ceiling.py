"""Small supervised diagnostics for the information ceiling in V2.

Body masks are intentionally used here. These models are evaluation-only and
their parameters are never exposed to the formal learner.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from cal.env.body import BodyAction, BodyConfig
from cal.env.sensors import BinaryMask, VisionFrame
from cal.env.world import BodyDiscoveryWorld, WorldConfig
from cal.evaluation.metrics import segmentation_metrics
from cal.infra.provenance import capture_provenance

DIAGNOSTIC_MODES = (
    "frame",
    "video",
    "video_action",
    "causal_evidence",
)


@dataclass(frozen=True, slots=True)
class DiagnosticSequence:
    """Video, actions, causal effects, and isolated evaluation labels."""

    seed: int
    visions: tuple[VisionFrame, ...]
    actions: tuple[BodyAction, ...]
    all_action_effects: Tensor
    body_masks: tuple[BinaryMask, ...]

    def __post_init__(self) -> None:
        length = len(self.visions)
        if not (
            length
            == len(self.actions)
            == len(self.body_masks)
            == self.all_action_effects.shape[0]
        ):
            raise ValueError("diagnostic sequence fields must have equal length")
        if self.all_action_effects.ndim != 4:
            raise ValueError("action effects must be [time, action, H, W]")


class DiagnosticHistoryDataset(Dataset[dict[str, Tensor]]):
    """Fixed histories with labels confined to this evaluation dataset."""

    def __init__(
        self,
        sequences: Sequence[DiagnosticSequence],
        *,
        history_length: int,
        mode: str,
    ) -> None:
        if not sequences:
            raise ValueError("diagnostic dataset requires sequences")
        if history_length <= 0:
            raise ValueError("history length must be positive")
        if mode not in DIAGNOSTIC_MODES:
            raise ValueError("unknown diagnostic mode")
        self.sequences = tuple(sequences)
        self.history_length = history_length
        self.mode = mode
        self.index = tuple(
            (sequence_index, end)
            for sequence_index, sequence in enumerate(self.sequences)
            for end in range(history_length - 1, len(sequence.visions))
        )
        if not self.index:
            raise ValueError("no diagnostic history windows")

    @property
    def input_channels(self) -> int:
        if self.mode == "frame":
            return 1
        if self.mode == "video":
            return self.history_length
        if self.mode == "video_action":
            return self.history_length + (
                (self.history_length - 1) * len(tuple(BodyAction))
            )
        return self.history_length * len(tuple(BodyAction))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sequence_index, end = self.index[index]
        sequence = self.sequences[sequence_index]
        start = end - self.history_length + 1
        vision = torch.tensor(
            sequence.visions[start : end + 1],
            dtype=torch.float32,
        )
        if self.mode == "frame":
            inputs = vision[-1:].clone()
        elif self.mode == "video":
            inputs = vision
        elif self.mode == "video_action":
            height, width = vision.shape[-2:]
            transition_actions = sequence.actions[start:end]
            one_hot = F.one_hot(
                torch.tensor(
                    [_action_index(action) for action in transition_actions],
                    dtype=torch.long,
                ),
                num_classes=len(tuple(BodyAction)),
            ).float()
            action_planes = one_hot.reshape(-1, 1, 1).expand(
                -1,
                height,
                width,
            )
            inputs = torch.cat((vision, action_planes), dim=0)
        else:
            effects = sequence.all_action_effects[start : end + 1]
            inputs = effects.flatten(start_dim=0, end_dim=1)
        target = torch.tensor(
            sequence.body_masks[end],
            dtype=torch.float32,
        ).unsqueeze(0)
        return {
            "inputs": inputs,
            "target": target,
            "seed": torch.tensor(sequence.seed, dtype=torch.long),
        }


class SmallMaskDiagnostic(nn.Module):
    """A fixed small CNN used only to test whether information is present."""

    def __init__(self, input_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("diagnostic channels must be positive")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


def collect_diagnostic_sequences(
    world_config: WorldConfig,
    body_config: BodyConfig,
    seeds: Sequence[int],
    *,
    steps_per_seed: int,
) -> tuple[DiagnosticSequence, ...]:
    """Collect random video and all-action visual effects for diagnostics."""

    if steps_per_seed <= 0:
        raise ValueError("steps_per_seed must be positive")
    sequences = []
    for seed in seeds:
        resolved_seed = int(seed)
        world = BodyDiscoveryWorld(
            replace(world_config, seed=resolved_seed),
            body_config,
        )
        current = world.reset(resolved_seed)
        visions = []
        actions = []
        effects = []
        masks = []
        for _ in range(steps_per_seed):
            visions.append(current.vision)
            masks.append(world.evaluation_snapshot().masks.body)
            effects.append(_all_action_effects(world))
            action = world.sample_action()
            actions.append(action)
            current = world.step(action).observation
        sequences.append(
            DiagnosticSequence(
                seed=resolved_seed,
                visions=tuple(visions),
                actions=tuple(actions),
                all_action_effects=torch.stack(effects),
                body_masks=tuple(masks),
            )
        )
    if not sequences:
        raise ValueError("at least one diagnostic seed is required")
    return tuple(sequences)


def train_diagnostic(
    model: SmallMaskDiagnostic,
    train_dataset: DiagnosticHistoryDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Fit one fixed diagnostic without checkpointing its parameters."""

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("invalid diagnostic training configuration")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = Adam(model.parameters(), lr=learning_rate)
    started = time.perf_counter()
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        samples = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["inputs"])
            target = batch["target"]
            bce = F.binary_cross_entropy_with_logits(logits, target)
            probability = torch.sigmoid(logits)
            dimensions = tuple(range(1, probability.ndim))
            overlap = 2.0 * (probability * target).sum(dim=dimensions)
            scale = probability.sum(dim=dimensions) + target.sum(
                dim=dimensions
            )
            dice = 1.0 - ((overlap + 1.0) / (scale + 1.0)).mean()
            loss = bce + dice
            loss.backward()
            optimizer.step()
            batch_size_actual = target.shape[0]
            total += float(loss.detach()) * batch_size_actual
            samples += batch_size_actual
        history.append({"epoch": epoch, "loss": total / samples})
    return {
        "history": history,
        "duration_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate_diagnostic(
    model: SmallMaskDiagnostic,
    dataset: DiagnosticHistoryDataset,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate aggregate and per-seed segmentation information ceilings."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits = []
    targets = []
    seeds = []
    model.eval()
    for batch in loader:
        logits.append(model(batch["inputs"]))
        targets.append(batch["target"])
        seeds.append(batch["seed"])
    all_logits = torch.cat(logits)
    all_targets = torch.cat(targets)
    all_seeds = torch.cat(seeds)
    aggregate = asdict(segmentation_metrics(all_logits, all_targets))
    per_seed = {}
    for seed in sorted(int(value) for value in all_seeds.unique()):
        selected = all_seeds == seed
        per_seed[str(seed)] = asdict(
            segmentation_metrics(
                all_logits[selected],
                all_targets[selected],
            )
        )
    return {
        "aggregate": aggregate,
        "per_seed": per_seed,
    }


def diagnostic_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def diagnostic_mac_per_step(
    model: SmallMaskDiagnostic,
    image_size: tuple[int, int],
) -> int:
    """Exact convolution MAC proxy for one diagnostic frame."""

    width, height = image_size
    hidden = model.hidden_channels
    return width * height * (
        model.input_channels * hidden * 3 * 3
        + hidden * hidden * 3 * 3
        + hidden
    )


def run_diagnostic_ceiling(
    *,
    output_path: str | Path,
    train_seeds: Sequence[int] = tuple(range(16)),
    validation_seeds: Sequence[int] = tuple(range(100, 104)),
    test_seeds: Sequence[int] = tuple(range(200, 208)),
    steps_per_seed: int = 128,
    history_length: int = 16,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_channels: int = 16,
    world_config: WorldConfig | None = None,
    body_config: BodyConfig | None = None,
) -> dict[str, Any]:
    """Run all four fixed supervised information diagnostics."""

    resolved_world = world_config or WorldConfig(
        image_size=(16, 16),
        object_count=0,
        distractor_body_count=2,
        distractor_body_motion_probability=1.0,
    )
    resolved_body = body_config or BodyConfig()
    groups = {
        "train": tuple(int(seed) for seed in train_seeds),
        "validation": tuple(int(seed) for seed in validation_seeds),
        "test": tuple(int(seed) for seed in test_seeds),
    }
    if any(not seeds for seeds in groups.values()):
        raise ValueError("all diagnostic seed groups must be non-empty")
    if (
        set(groups["train"]) & set(groups["validation"])
        or set(groups["train"]) & set(groups["test"])
        or set(groups["validation"]) & set(groups["test"])
    ):
        raise ValueError("diagnostic seed groups must be disjoint")
    collection_started = time.perf_counter()
    sequences = {
        name: collect_diagnostic_sequences(
            resolved_world,
            resolved_body,
            seeds,
            steps_per_seed=steps_per_seed,
        )
        for name, seeds in groups.items()
    }
    collection_duration = time.perf_counter() - collection_started
    diagnostics = {}
    for mode_index, mode in enumerate(DIAGNOSTIC_MODES):
        datasets = {
            name: DiagnosticHistoryDataset(
                items,
                history_length=history_length,
                mode=mode,
            )
            for name, items in sequences.items()
        }
        torch.manual_seed(10_007 + mode_index)
        model = SmallMaskDiagnostic(
            datasets["train"].input_channels,
            hidden_channels=hidden_channels,
        )
        training = train_diagnostic(
            model,
            datasets["train"],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=20_011 + mode_index,
        )
        parameters = diagnostic_parameter_count(model)
        mac = diagnostic_mac_per_step(model, resolved_world.image_size)
        diagnostics[mode] = {
            "diagnostic_only": True,
            "input_channels": datasets["train"].input_channels,
            "parameter_count": parameters,
            "mac_per_step": mac,
            "resource_gates": {
                "parameter_limit": (
                    50_000 if mode == "frame" else 100_000
                ),
                "parameter_passed": parameters
                <= (50_000 if mode == "frame" else 100_000),
                "mac_limit": 5_000_000,
                "mac_passed": mac <= 5_000_000,
                "duration_limit_seconds": 7_200.0,
                "duration_passed": training["duration_seconds"] <= 7_200.0,
            },
            "training": training,
            "train": evaluate_diagnostic(
                model,
                datasets["train"],
                batch_size=batch_size,
            ),
            "validation": evaluate_diagnostic(
                model,
                datasets["validation"],
                batch_size=batch_size,
            ),
            "test": evaluate_diagnostic(
                model,
                datasets["test"],
                batch_size=batch_size,
            ),
        }
    frame_seed_iou = diagnostics["frame"]["test"]["per_seed"]
    comparisons = {}
    for mode in ("video", "video_action", "causal_evidence"):
        candidate_seed_iou = diagnostics[mode]["test"]["per_seed"]
        differences = [
            candidate_seed_iou[seed]["iou"] - frame_seed_iou[seed]["iou"]
            for seed in sorted(frame_seed_iou, key=int)
        ]
        comparisons[f"{mode}_minus_frame"] = {
            "paired_seed_differences": differences,
            "mean_iou_difference": sum(differences) / len(differences),
            "all_seeds_positive": all(value > 0.0 for value in differences),
            "at_least_0_05": (
                sum(differences) / len(differences) >= 0.05
            ),
        }
    summary = {
        "result_schema_version": 1,
        "audit": "v2_diagnostic_ceiling",
        "diagnostic_only": True,
        "world_config": asdict(resolved_world),
        "body_config": asdict(resolved_body),
        "seed_groups": groups,
        "steps_per_seed": steps_per_seed,
        "history_length": history_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "collection_duration_seconds": collection_duration,
        "diagnostics": diagnostics,
        "comparisons": comparisons,
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _all_action_effects(world: BodyDiscoveryWorld) -> Tensor:
    noop = deepcopy(world).step(BodyAction.NOOP).observation.vision
    effects = []
    for action in world.actions:
        branch = deepcopy(world).step(action).observation.vision
        effects.append(
            torch.tensor(branch, dtype=torch.float32)
            .sub(torch.tensor(noop, dtype=torch.float32))
            .abs()
        )
    return torch.stack(effects)


def _action_index(action: BodyAction) -> int:
    return tuple(BodyAction).index(action)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run isolated V2 supervised information diagnostics."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/V2-diagnostic-ceiling-summary.json"),
    )
    parser.add_argument("--steps-per-seed", type=int, default=128)
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    arguments = parser.parse_args(argv)
    result = run_diagnostic_ceiling(
        output_path=arguments.output,
        steps_per_seed=arguments.steps_per_seed,
        history_length=arguments.history_length,
        epochs=arguments.epochs,
    )
    for mode, diagnostic in result["diagnostics"].items():
        metrics = diagnostic["test"]["aggregate"]
        print(
            f"{mode}: IoU={metrics['iou']:.4f}; "
            f"F1={metrics['f1']:.4f}; "
            f"parameters={diagnostic['parameter_count']}; "
            f"MAC={diagnostic['mac_per_step']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
