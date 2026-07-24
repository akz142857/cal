"""Probe body state during contiguous multimodal sensory blackouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

import torch
import yaml

from cal.env.body import BodyAction, BodyConfig
from cal.env.world import BodyDiscoveryWorld, WorldConfig
from cal.evaluation.body_probe import (
    ProbeConfig,
    ProbeData,
    _replay_body_labels,
    evaluate_fixed_position_baseline,
    evaluate_probe,
    train_linear_probe,
)
from cal.infra.provenance import capture_provenance
from cal.learning.dataset import (
    SeedSplit,
    TrajectorySequenceDataset,
    collect_trajectories,
)
from cal.learning.replay import Trajectory, record_action_trajectory
from cal.learning.trainer import load_checkpoint
from cal.model.predictors import SensorimotorPredictor


@dataclass(frozen=True, slots=True)
class BlackoutConfig:
    period: int = 16
    start: int = 8
    length: int = 4

    def __post_init__(self) -> None:
        if min(self.period, self.length) <= 0:
            raise ValueError("blackout period and length must be positive")
        if self.start < 0 or self.start + self.length > self.period:
            raise ValueError("blackout interval must fit inside its period")


@torch.no_grad()
def extract_blackout_probe_data(
    model: SensorimotorPredictor,
    trajectories: Sequence[Trajectory],
    *,
    blackout: BlackoutConfig | None = None,
    device: str = "cpu",
    representation_source: str = "representation",
) -> ProbeData:
    """Return frozen states only at steps with all sensory modalities hidden."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    resolved = blackout or BlackoutConfig()
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    representations = []
    visions = []
    masks: list[list[list[bool]]] = []
    poses: list[tuple[float, float]] = []

    for trajectory in trajectories:
        sample = TrajectorySequenceDataset(
            (trajectory,),
            sequence_length=len(trajectory),
        )[0]
        selected = _blackout_mask(len(trajectory), resolved)
        vision = sample["vision"].clone()
        proprioception = sample["proprioception"].clone()
        touch = sample["touch"].clone()
        vision[selected] = 0.0
        proprioception[selected] = 0.0
        touch[selected] = 0.0
        output = model(
            vision=vision.unsqueeze(0).to(resolved_device),
            proprioception=proprioception.unsqueeze(0).to(resolved_device),
            touch=touch.unsqueeze(0).to(resolved_device),
            actions=sample["actions"].unsqueeze(0).to(resolved_device),
        )
        representation = {
            "representation": output.representation,
            "control_state": output.control_state,
            "action_effect_state": output.action_effect_state,
            "ownership_state": output.ownership_state,
            "ownership_mask": (
                torch.sigmoid(output.ownership_logits).flatten(start_dim=2)
                if output.ownership_logits is not None
                else None
            ),
            "part_slot_mask": (
                output.part_slot_mask.flatten(start_dim=2)
                if output.part_slot_mask is not None
                else None
            ),
            "spatial_ownership_mask": (
                output.spatial_ownership_mask.flatten(start_dim=2)
                if output.spatial_ownership_mask is not None
                else None
            ),
            "global_ownership_mask": (
                output.global_ownership_mask.flatten(start_dim=2)
                if output.global_ownership_mask is not None
                else None
            ),
            "object_slot_mask": (
                output.object_slot_mask.flatten(start_dim=2)
                if output.object_slot_mask is not None
                else None
            ),
            "causal_envelope_mask": (
                output.causal_envelope_mask.flatten(start_dim=2)
                if output.causal_envelope_mask is not None
                else None
            ),
        }.get(representation_source)
        if representation is None:
            raise ValueError(
                f"model does not expose {representation_source}"
            )
        representations.append(
            representation.squeeze(0)[selected].detach().cpu()
        )
        visions.append(vision[selected].detach().cpu())
        trajectory_masks, trajectory_poses = _replay_body_labels(
            trajectory,
            target_offset="next",
        )
        selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
        for index in selected_indices.tolist():
            masks.append(trajectory_masks[index])
            poses.append(trajectory_poses[index])

    return ProbeData(
        representations=torch.cat(representations),
        body_masks=torch.tensor(masks, dtype=torch.float32).unsqueeze(1),
        visions=torch.cat(visions),
        poses=torch.tensor(poses, dtype=torch.float32),
    )


def run_temporal_probe_experiment(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    output_directory: str | Path,
    blackout_override: BlackoutConfig | None = None,
    policy_override: str | None = None,
    body_override: BodyConfig | None = None,
) -> dict[str, Any]:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration must be a mapping")
    environment = _mapping(payload, "environment")
    probe = _mapping(payload, "probe")
    temporal = payload.get("temporal_probe", {})
    if not isinstance(temporal, dict):
        raise ValueError("temporal_probe must be a mapping")
    image_size = tuple(int(value) for value in environment["image_size"])
    world = WorldConfig(
        image_size=image_size,  # type: ignore[arg-type]
        object_count=int(environment.get("object_count", 3)),
        object_radius=float(environment.get("object_radius", 0.045)),
        body_visual_value=float(environment.get("body_visual_value", 1.0)),
        object_visual_value=float(environment.get("object_visual_value", 1.0)),
        external_object_motion_probability=float(
            probe.get("evaluation_external_motion_probability", 0.0)
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
    split = SeedSplit(
        train=tuple(int(value) for value in probe["train_seeds"]),
        validation=tuple(int(value) for value in probe["validation_seeds"]),
        test=tuple(int(value) for value in probe["test_seeds"]),
    )
    steps = int(probe.get("steps_per_seed", 128))
    blackout = blackout_override or BlackoutConfig(
        period=int(temporal.get("period", 16)),
        start=int(temporal.get("start", 8)),
        length=int(temporal.get("length", 4)),
    )
    policy = policy_override or str(temporal.get("policy", "random"))
    if policy not in {"random", "persistent", "intervention"}:
        raise ValueError(
            "temporal probe policy must be random, persistent, or intervention"
        )
    body = body_override or _body_config(temporal)
    representation_source = str(
        temporal.get(
            "representation_source",
            probe.get("representation_source", "representation"),
        )
    )
    model = load_checkpoint(
        checkpoint_path,
        device=str(probe.get("device", "cpu")),
    )
    data = {
        name: extract_blackout_probe_data(
            model,
            _collect_probe_trajectories(
                world,
                seeds,
                steps=steps,
                policy=policy,
                block_length=blackout.period,
                intervention_start=blackout.start,
                body_config=body,
            ),
            blackout=blackout,
            device=str(probe.get("device", "cpu")),
            representation_source=representation_source,
        )
        for name, seeds in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
    }
    probe_config = ProbeConfig(
        epochs=int(probe.get("epochs", 100)),
        batch_size=int(probe.get("batch_size", 64)),
        learning_rate=float(probe.get("learning_rate", 1e-2)),
        weight_decay=float(probe.get("weight_decay", 0.0)),
        seed=int(payload.get("seed", 0)),
        device=str(probe.get("device", "cpu")),
    )
    trained = train_linear_probe(
        data["train"],
        data["validation"],
        config=probe_config,
    )
    test_metrics = evaluate_probe(
        trained.probe,
        data["test"],
        batch_size=probe_config.batch_size,
        device=probe_config.device,
    )
    fixed = evaluate_fixed_position_baseline(
        data["train"],
        data["test"],
    )
    summary = {
        "result_schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "blackout": asdict(blackout),
        "action_policy": policy,
        "body_config": asdict(body),
        "representation_source": representation_source,
        "seed_split": asdict(split),
        "train_samples": len(data["train"]),
        "validation_samples": len(data["validation"]),
        "test_samples": len(data["test"]),
        "best_epoch": trained.best_epoch,
        "test": asdict(test_metrics),
        "fixed_position_baseline": asdict(fixed),
        "provenance": capture_provenance(),
    }
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "temporal-probe-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _blackout_mask(length: int, config: BlackoutConfig) -> torch.Tensor:
    steps = torch.arange(length)
    position = steps % config.period
    return (position >= config.start) & (
        position < config.start + config.length
    )


def _collect_probe_trajectories(
    world_config: WorldConfig,
    seeds: Sequence[int],
    *,
    steps: int,
    policy: str,
    block_length: int,
    intervention_start: int | None = None,
    body_config: BodyConfig | None = None,
) -> tuple[Trajectory, ...]:
    """Collect random or block-persistent action rollouts for the probe."""

    if policy == "random":
        return collect_trajectories(
            world_config,
            body_config or BodyConfig(),
            seeds,
            steps_per_seed=steps,
        )
    if policy not in {"persistent", "intervention"}:
        raise ValueError(
            "temporal probe policy must be random, persistent, or intervention"
        )
    if steps <= 0 or block_length <= 0:
        raise ValueError("steps and block length must be positive")
    if policy == "intervention" and (
        intervention_start is None
        or intervention_start <= 0
        or intervention_start >= block_length
    ):
        raise ValueError("intervention start must fall inside each block")
    if not seeds:
        raise ValueError("at least one seed is required")

    cycle = (
        BodyAction.SHOULDER_INCREASE,
        BodyAction.SHOULDER_DECREASE,
        BodyAction.ELBOW_INCREASE,
        BodyAction.ELBOW_DECREASE,
    )
    trajectories = []
    resolved_body = body_config or BodyConfig()
    for seed in seeds:
        resolved_seed = int(seed)
        generator = Random(resolved_seed + 104_729)
        if policy == "persistent":
            block_actions: list[BodyAction] = []
            for _ in range((steps + block_length - 1) // block_length):
                candidates = tuple(
                    action
                    for action in cycle
                    if not block_actions or action is not block_actions[-1]
                )
                block_actions.append(generator.choice(candidates))
            actions = tuple(
                block_actions[step // block_length]
                for step in range(steps)
            )
        else:
            assert intervention_start is not None
            scheduled: list[BodyAction] = []
            previous: BodyAction | None = None
            for _ in range((steps + block_length - 1) // block_length):
                visible = generator.choice(
                    tuple(action for action in cycle if action is not previous)
                )
                hidden = generator.choice(
                    tuple(action for action in cycle if action is not visible)
                )
                scheduled.extend([visible] * intervention_start)
                scheduled.extend(
                    [hidden] * (block_length - intervention_start)
                )
                previous = hidden
            actions = tuple(scheduled[:steps])
        world = BodyDiscoveryWorld(
            replace(world_config, seed=resolved_seed),
            resolved_body,
        )
        trajectories.append(
            record_action_trajectory(world, actions, seed=resolved_seed)
        )
    return tuple(trajectories)


def _body_config(temporal: Mapping[str, Any]) -> BodyConfig:
    value = temporal.get("body", {})
    if not isinstance(value, dict):
        raise ValueError("temporal_probe.body must be a mapping")
    defaults = BodyConfig()
    return BodyConfig(
        link_lengths=tuple(
            float(item)
            for item in value.get("link_lengths", defaults.link_lengths)
        ),  # type: ignore[arg-type]
        angle_step=float(value.get("angle_step", defaults.angle_step)),
    )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe body representation during sensory blackouts."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--period", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument(
        "--policy",
        choices=("random", "persistent", "intervention"),
    )
    parser.add_argument("--first-link", type=float)
    parser.add_argument("--second-link", type=float)
    parser.add_argument("--angle-step", type=float)
    arguments = parser.parse_args(argv)
    override = None
    if any(
        value is not None
        for value in (arguments.period, arguments.start, arguments.length)
    ):
        override = BlackoutConfig(
            period=arguments.period or 16,
            start=arguments.start if arguments.start is not None else 8,
            length=arguments.length or 4,
        )
    summary = run_temporal_probe_experiment(
        arguments.checkpoint,
        arguments.config,
        output_directory=arguments.output,
        blackout_override=override,
        policy_override=arguments.policy,
        body_override=(
            BodyConfig(
                link_lengths=(
                    arguments.first_link
                    if arguments.first_link is not None
                    else BodyConfig().link_lengths[0],
                    arguments.second_link
                    if arguments.second_link is not None
                    else BodyConfig().link_lengths[1],
                ),
                angle_step=(
                    arguments.angle_step
                    if arguments.angle_step is not None
                    else BodyConfig().angle_step
                ),
            )
            if any(
                value is not None
                for value in (
                    arguments.first_link,
                    arguments.second_link,
                    arguments.angle_step,
                )
            )
            else None
        ),
    )
    print(
        f"blackout body IoU {summary['test']['iou']:.4f}; "
        f"fixed {summary['fixed_position_baseline']['iou']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
