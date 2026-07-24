"""Tensor datasets built from unlabeled sensorimotor trajectories."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from calmodel.env.body import BodyAction, BodyConfig
from calmodel.env.world import BodyDiscoveryWorld, WorldConfig
from calmodel.learning.replay import (
    Experience,
    Trajectory,
    iter_compressed_experiences,
    read_compressed_trajectory_header,
    record_random_trajectory,
)

ACTION_VOCABULARY: tuple[BodyAction, ...] = tuple(BodyAction)
ACTION_TO_INDEX = {
    action: index for index, action in enumerate(ACTION_VOCABULARY)
}


@dataclass(frozen=True, slots=True)
class SeedSplit:
    """Disjoint scene seeds for train, validation, test, and transfer."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    transfer: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        groups = {
            "train": set(self.train),
            "validation": set(self.validation),
            "test": set(self.test),
            "transfer": set(self.transfer),
        }
        names = tuple(groups)
        for index, first_name in enumerate(names):
            for second_name in names[index + 1 :]:
                overlap = groups[first_name] & groups[second_name]
                if overlap:
                    raise ValueError(
                        f"seed groups {first_name} and {second_name} overlap: "
                        f"{sorted(overlap)}"
                    )
        if not self.train:
            raise ValueError("train seeds cannot be empty")
        if not self.validation:
            raise ValueError("validation seeds cannot be empty")
        if not self.test:
            raise ValueError("test seeds cannot be empty")


DEFAULT_SEED_SPLIT = SeedSplit(
    train=tuple(range(0, 16)),
    validation=tuple(range(100, 104)),
    test=tuple(range(200, 204)),
    transfer=tuple(range(300, 304)),
)


@dataclass(frozen=True, slots=True)
class SequenceTransformConfig:
    """Deterministic negative-control transformations for M1 ablations."""

    temporal_shuffle: bool = False
    randomize_actions: bool = False
    sensor_blackout_probability: float = 0.0
    sensor_blackout_span: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.sensor_blackout_probability <= 1.0:
            raise ValueError("sensor_blackout_probability must be in [0, 1]")
        if self.sensor_blackout_span <= 0:
            raise ValueError("sensor_blackout_span must be positive")


class TrajectorySequenceDataset(Dataset[dict[str, Tensor]]):
    """Fixed-length windows for action-conditioned next-sensory prediction."""

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        *,
        sequence_length: int,
        stride: int | None = None,
        transform: SequenceTransformConfig | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        resolved_stride = sequence_length if stride is None else stride
        if resolved_stride <= 0:
            raise ValueError("stride must be positive")
        if not trajectories:
            raise ValueError("at least one trajectory is required")

        first_size = trajectories[0].world_config.image_size
        if any(item.world_config.image_size != first_size for item in trajectories):
            raise ValueError("all trajectories must use the same image size")

        self.trajectories = tuple(trajectories)
        self.sequence_length = sequence_length
        self.stride = resolved_stride
        self.image_size = first_size
        self.transform = transform or SequenceTransformConfig()
        windows: list[tuple[int, int]] = []
        for trajectory_index, trajectory in enumerate(self.trajectories):
            final_start = len(trajectory) - sequence_length
            for start in range(0, final_start + 1, resolved_stride):
                windows.append((trajectory_index, start))
        if not windows:
            raise ValueError("no trajectory is long enough for one sequence")
        self._windows = tuple(windows)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        trajectory_index, start = self._windows[index]
        experiences = self.trajectories[trajectory_index].experiences[
            start : start + self.sequence_length
        ]
        sample = _experiences_to_sample(experiences)
        return self._apply_transform(sample, index)

    def _apply_transform(
        self,
        sample: dict[str, Tensor],
        index: int,
    ) -> dict[str, Tensor]:
        return _apply_sequence_transform(
            sample,
            index=index,
            transform=self.transform,
        )


class CounterfactualTrajectorySequenceDataset(TrajectorySequenceDataset):
    """Windows augmented by action-vs-NOOP sensory intervention effects."""

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        *,
        sequence_length: int,
        stride: int | None = None,
        transform: SequenceTransformConfig | None = None,
    ) -> None:
        super().__init__(
            trajectories,
            sequence_length=sequence_length,
            stride=stride,
            transform=transform,
        )
        targets = tuple(
            _counterfactual_action_targets(trajectory)
            for trajectory in self.trajectories
        )
        self._action_effects = tuple(item[0] for item in targets)
        self._action_envelopes = tuple(item[1] for item in targets)
        self._all_action_effects = tuple(item[2] for item in targets)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = super().__getitem__(index)
        trajectory_index, start = self._windows[index]
        action_effect = self._action_effects[trajectory_index][
            start : start + self.sequence_length
        ]
        sample["action_effect"] = action_effect
        dilated = torch.nn.functional.max_pool2d(
            action_effect,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        sample["ownership_target"] = dilated * sample["next_vision"]
        sample["action_envelope"] = self._action_envelopes[
            trajectory_index
        ][start : start + self.sequence_length]
        sample["all_action_effects"] = self._all_action_effects[
            trajectory_index
        ][start : start + self.sequence_length]
        return sample


def _counterfactual_action_targets(
    trajectory: Trajectory,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build actual-action effects and all-action current occupancy envelopes."""

    world = BodyDiscoveryWorld(
        trajectory.world_config,
        trajectory.body_config,
    )
    current = world.reset(trajectory.seed)
    effects = []
    envelopes = []
    all_action_effects = []
    for experience in trajectory.experiences:
        if current != experience.observation:
            raise RuntimeError("counterfactual replay diverged before action")
        current_vision = torch.tensor(
            experience.observation.vision,
            dtype=torch.float32,
        )
        noop_observation = deepcopy(world).step(BodyAction.NOOP).observation
        noop_vision = torch.tensor(
            noop_observation.vision,
            dtype=torch.float32,
        )
        branch_effects = []
        for action in ACTION_VOCABULARY:
            branch_observation = deepcopy(world).step(action).observation
            branch_vision = torch.tensor(
                branch_observation.vision,
                dtype=torch.float32,
            )
            branch_effects.append((branch_vision - noop_vision).abs())
        all_action_effects.append(
            torch.stack(branch_effects).unsqueeze(1)
        )
        envelope = torch.stack(branch_effects).amax(dim=0).unsqueeze(0)
        envelope = torch.nn.functional.max_pool2d(
            envelope,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        envelopes.append(envelope * current_vision.unsqueeze(0))
        current = world.step(experience.action).observation
        if current != experience.next_observation:
            raise RuntimeError("counterfactual replay diverged after action")
        actual_vision = torch.tensor(current.vision, dtype=torch.float32)
        effects.append((actual_vision - noop_vision).abs().unsqueeze(0))
    return (
        torch.stack(effects),
        torch.stack(envelopes),
        torch.stack(all_action_effects),
    )


class CompressedTrajectorySequenceIterableDataset(
    IterableDataset[dict[str, Tensor]]
):
    """Stream fixed windows from gzip JSONL trajectories."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        sequence_length: int,
        stride: int | None = None,
        transform: SequenceTransformConfig | None = None,
    ) -> None:
        super().__init__()
        if not paths:
            raise ValueError("at least one compressed trajectory is required")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        resolved_stride = sequence_length if stride is None else stride
        if resolved_stride <= 0:
            raise ValueError("stride must be positive")
        self.paths = tuple(Path(path) for path in paths)
        headers = tuple(
            read_compressed_trajectory_header(path) for path in self.paths
        )
        image_sizes = {header.world_config.image_size for header in headers}
        if len(image_sizes) != 1:
            raise ValueError("all trajectories must use the same image size")
        self.image_size = headers[0].world_config.image_size
        self.sequence_length = sequence_length
        self.stride = resolved_stride
        self.transform = transform or SequenceTransformConfig()

    def __iter__(self) -> Iterable[dict[str, Tensor]]:
        worker = get_worker_info()
        paths = (
            self.paths
            if worker is None
            else self.paths[worker.id :: worker.num_workers]
        )
        sample_index = worker.id if worker is not None else 0
        sample_step = worker.num_workers if worker is not None else 1
        for path in paths:
            buffer: list[Experience] = []
            skip = 0
            for experience in iter_compressed_experiences(path):
                if skip > 0:
                    skip -= 1
                    continue
                buffer.append(experience)
                if len(buffer) < self.sequence_length:
                    continue
                sample = _experiences_to_sample(buffer)
                yield _apply_sequence_transform(
                    sample,
                    index=sample_index,
                    transform=self.transform,
                )
                sample_index += sample_step
                if self.stride < self.sequence_length:
                    buffer = buffer[self.stride :]
                else:
                    skip = self.stride - self.sequence_length
                    buffer = []


def _experiences_to_sample(
    experiences: Sequence[Experience],
) -> dict[str, Tensor]:
    return {
        "vision": torch.tensor(
            [[item.observation.vision] for item in experiences],
            dtype=torch.float32,
        ),
        "proprioception": torch.tensor(
            [item.observation.proprioception for item in experiences],
            dtype=torch.float32,
        ),
        "touch": torch.tensor(
            [item.observation.touch for item in experiences],
            dtype=torch.float32,
        ),
        "actions": torch.tensor(
            [ACTION_TO_INDEX[item.action] for item in experiences],
            dtype=torch.long,
        ),
        "next_vision": torch.tensor(
            [[item.next_observation.vision] for item in experiences],
            dtype=torch.float32,
        ),
        "next_proprioception": torch.tensor(
            [item.next_observation.proprioception for item in experiences],
            dtype=torch.float32,
        ),
        "next_touch": torch.tensor(
            [item.next_observation.touch for item in experiences],
            dtype=torch.float32,
        ),
    }


def _apply_sequence_transform(
    sample: dict[str, Tensor],
    *,
    index: int,
    transform: SequenceTransformConfig,
) -> dict[str, Tensor]:
    generator = torch.Generator()
    generator.manual_seed(transform.seed + 104_729 * index)
    if transform.temporal_shuffle:
        length = sample["actions"].shape[0]
        for name in ("vision", "proprioception", "touch"):
            permutation = torch.randperm(length, generator=generator)
            sample[name] = sample[name][permutation]
    if transform.randomize_actions:
        sample["actions"] = torch.randint(
            low=0,
            high=len(ACTION_VOCABULARY),
            size=sample["actions"].shape,
            generator=generator,
            dtype=torch.long,
        )
    if transform.sensor_blackout_probability > 0.0:
        length = sample["actions"].shape[0]
        blackout = _sensor_blackout_mask(
            length,
            probability=transform.sensor_blackout_probability,
            span=transform.sensor_blackout_span,
            generator=generator,
        )
        for name in ("vision", "proprioception", "touch"):
            sample[name][blackout] = 0.0
    return sample


class PairedTrajectorySequenceDataset(Dataset[dict[str, Tensor]]):
    """Aligned windows from scenes that share actions but not object layouts."""

    def __init__(
        self,
        trajectory_pairs: Sequence[tuple[Trajectory, Trajectory]],
        *,
        sequence_length: int,
        stride: int | None = None,
        transform: SequenceTransformConfig | None = None,
    ) -> None:
        if not trajectory_pairs:
            raise ValueError("at least one trajectory pair is required")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        resolved_stride = sequence_length if stride is None else stride
        if resolved_stride <= 0:
            raise ValueError("stride must be positive")
        self.trajectory_pairs = tuple(trajectory_pairs)
        self.sequence_length = sequence_length
        self.stride = resolved_stride
        self.transform = transform or SequenceTransformConfig()
        windows = []
        for pair_index, (first, second) in enumerate(self.trajectory_pairs):
            if len(first) != len(second):
                raise ValueError("paired trajectories must have equal length")
            if tuple(item.action for item in first.experiences) != tuple(
                item.action for item in second.experiences
            ):
                raise ValueError("paired trajectories must share actions")
            final_start = len(first) - sequence_length
            for start in range(0, final_start + 1, resolved_stride):
                windows.append((pair_index, start, False))
                windows.append((pair_index, start, True))
        if not windows:
            raise ValueError("no trajectory pair is long enough")
        self._windows = tuple(windows)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        pair_index, start, reverse = self._windows[index]
        first, second = self.trajectory_pairs[pair_index]
        primary, partner = (second, first) if reverse else (first, second)
        primary_sample = _experiences_to_sample(
            primary.experiences[start : start + self.sequence_length]
        )
        partner_sample = _experiences_to_sample(
            partner.experiences[start : start + self.sequence_length]
        )
        primary_sample = _apply_sequence_transform(
            primary_sample,
            index=index,
            transform=self.transform,
        )
        partner_sample = _apply_sequence_transform(
            partner_sample,
            index=index,
            transform=self.transform,
        )
        return {
            **primary_sample,
            **{
                f"pair_{name}": value
                for name, value in partner_sample.items()
            },
        }


def _sensor_blackout_mask(
    length: int,
    *,
    probability: float,
    span: int,
    generator: torch.Generator,
) -> Tensor:
    """Select independent steps or an approximately fixed share of spans."""

    if span <= 0 or span > length:
        raise ValueError("blackout span must be between one and sequence length")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("blackout probability must be in [0, 1]")
    if probability == 0.0:
        return torch.zeros(length, dtype=torch.bool)
    if probability == 1.0:
        return torch.ones(length, dtype=torch.bool)
    if span == 1:
        return torch.rand(length, generator=generator) < probability

    blackout = torch.zeros(length, dtype=torch.bool)
    span_count = max(1, round(length * probability / span))
    candidates = torch.randperm(
        length - span + 1,
        generator=generator,
    )
    selected = 0
    for candidate in candidates.tolist():
        interval = slice(candidate, candidate + span)
        if bool(blackout[interval].any()):
            continue
        blackout[interval] = True
        selected += 1
        if selected == span_count:
            break
    return blackout


def collect_trajectories(
    world_config: WorldConfig,
    body_config: BodyConfig,
    seeds: Iterable[int],
    *,
    steps_per_seed: int,
    action_repeat_probability: float = 0.0,
) -> tuple[Trajectory, ...]:
    """Collect independent random rollouts for a set of scene seeds."""

    if steps_per_seed <= 0:
        raise ValueError("steps_per_seed must be positive")
    trajectories = []
    for seed in seeds:
        seeded_config = replace(world_config, seed=int(seed))
        world = BodyDiscoveryWorld(seeded_config, body_config)
        trajectories.append(
            record_random_trajectory(
                world,
                steps_per_seed,
                seed=int(seed),
                action_repeat_probability=action_repeat_probability,
            )
        )
    if not trajectories:
        raise ValueError("at least one seed is required")
    return tuple(trajectories)


def collect_paired_trajectories(
    world_config: WorldConfig,
    body_config: BodyConfig,
    seeds: Iterable[int],
    *,
    steps_per_seed: int,
    action_repeat_probability: float = 0.0,
) -> tuple[tuple[Trajectory, Trajectory], ...]:
    """Collect scene pairs with an identical random action schedule."""

    resolved_seeds = tuple(int(seed) for seed in seeds)
    if not resolved_seeds or len(resolved_seeds) % 2 != 0:
        raise ValueError("paired collection requires an even number of seeds")
    pairs = []
    for pair_index in range(0, len(resolved_seeds), 2):
        first_seed = resolved_seeds[pair_index]
        second_seed = resolved_seeds[pair_index + 1]
        action_seed = 1_000_003 + first_seed
        pair = []
        for scene_seed in (first_seed, second_seed):
            world = BodyDiscoveryWorld(
                replace(world_config, seed=scene_seed),
                body_config,
            )
            pair.append(
                record_random_trajectory(
                    world,
                    steps_per_seed,
                    seed=scene_seed,
                    action_repeat_probability=action_repeat_probability,
                    action_seed=action_seed,
                )
            )
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


def action_from_index(index: int) -> BodyAction:
    """Decode an action index using the stable dataset vocabulary."""

    try:
        return ACTION_VOCABULARY[index]
    except IndexError as error:
        raise ValueError(f"unknown action index: {index}") from error
