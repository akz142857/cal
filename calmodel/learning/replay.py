"""Trajectory storage and deterministic replay for reproducible experiments.

The serialized format intentionally contains only learner-facing observations,
actions, and simulator configuration. Evaluation masks and semantic labels are
never recorded here.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterator, Mapping, Sequence

from calmodel.env.body import BodyAction, BodyConfig
from calmodel.env.sensors import SensorObservation
from calmodel.env.world import BodyDiscoveryWorld, WorldConfig

FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Experience:
    """One action-conditioned prediction example."""

    index: int
    observation: SensorObservation
    action: BodyAction
    next_observation: SensorObservation


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A complete, reproducible learner-facing rollout."""

    seed: int
    world_config: WorldConfig
    body_config: BodyConfig
    initial_observation: SensorObservation
    experiences: tuple[Experience, ...]
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported trajectory format version: {self.format_version}"
            )
        expected = self.initial_observation
        for expected_index, experience in enumerate(self.experiences, start=1):
            if experience.index != expected_index:
                raise ValueError("experience indexes must be contiguous from one")
            if experience.observation != expected:
                raise ValueError("trajectory observations are not temporally continuous")
            expected = experience.next_observation

    def __len__(self) -> int:
        return len(self.experiences)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Outcome of replaying a stored trajectory."""

    world: BodyDiscoveryWorld
    observations: tuple[SensorObservation, ...]


@dataclass(frozen=True, slots=True)
class CompressedTrajectoryHeader:
    """Metadata needed to stream experiences without loading the trajectory."""

    seed: int
    world_config: WorldConfig
    body_config: BodyConfig
    initial_observation: SensorObservation
    format_version: int = FORMAT_VERSION


class ReplayMismatchError(RuntimeError):
    """Raised when deterministic replay diverges from recorded experience."""


def record_random_trajectory(
    world: BodyDiscoveryWorld,
    steps: int,
    *,
    seed: int | None = None,
    action_repeat_probability: float = 0.0,
    action_seed: int | None = None,
) -> Trajectory:
    """Collect a random-action trajectory using the world's seeded generator."""

    if steps < 0:
        raise ValueError("steps cannot be negative")
    if not 0.0 <= action_repeat_probability <= 1.0:
        raise ValueError("action_repeat_probability must be in [0, 1]")

    resolved_seed = world.config.seed if seed is None else seed
    resolved_action_seed = (
        resolved_seed if action_seed is None else action_seed
    )
    action_generator = Random(resolved_action_seed + 97_409)
    repeat_generator = Random(resolved_action_seed + 130_363)
    current = world.reset(resolved_seed)
    experiences: list[Experience] = []
    for index in range(1, steps + 1):
        if (
            experiences
            and repeat_generator.random() < action_repeat_probability
        ):
            action = experiences[-1].action
        else:
            action = (
                world.sample_action()
                if action_seed is None
                else action_generator.choice(world.actions)
            )
        result = world.step(action)
        experiences.append(
            Experience(
                index=index,
                observation=current,
                action=action,
                next_observation=result.observation,
            )
        )
        current = result.observation

    return Trajectory(
        seed=resolved_seed,
        world_config=world.config,
        body_config=world.body.config,
        initial_observation=(
            experiences[0].observation if experiences else current
        ),
        experiences=tuple(experiences),
    )


def record_action_trajectory(
    world: BodyDiscoveryWorld,
    actions: Sequence[BodyAction | str],
    *,
    seed: int | None = None,
) -> Trajectory:
    """Collect a trajectory from an explicit action schedule."""

    resolved_seed = world.config.seed if seed is None else seed
    current = world.reset(resolved_seed)
    experiences: list[Experience] = []
    for index, action in enumerate(actions, start=1):
        resolved_action = BodyAction(action)
        result = world.step(resolved_action)
        experiences.append(
            Experience(
                index=index,
                observation=current,
                action=resolved_action,
                next_observation=result.observation,
            )
        )
        current = result.observation

    return Trajectory(
        seed=resolved_seed,
        world_config=world.config,
        body_config=world.body.config,
        initial_observation=(
            experiences[0].observation if experiences else current
        ),
        experiences=tuple(experiences),
    )


def replay_trajectory(
    trajectory: Trajectory,
    *,
    strict: bool = True,
) -> ReplayResult:
    """Recreate a trajectory and optionally verify every recorded observation."""

    world = BodyDiscoveryWorld(
        config=trajectory.world_config,
        body_config=trajectory.body_config,
    )
    current = world.reset(trajectory.seed)
    observations = [current]
    if strict and current != trajectory.initial_observation:
        raise ReplayMismatchError("initial observation does not match recording")

    for experience in trajectory.experiences:
        if strict and current != experience.observation:
            raise ReplayMismatchError(
                f"observation before step {experience.index} does not match"
            )
        result = world.step(experience.action)
        current = result.observation
        observations.append(current)
        if strict and current != experience.next_observation:
            raise ReplayMismatchError(
                f"observation after step {experience.index} does not match"
            )

    return ReplayResult(world=world, observations=tuple(observations))


def save_trajectory(trajectory: Trajectory, path: str | Path) -> Path:
    """Write one trajectory as readable, versioned JSON."""

    destination = Path(path)
    if destination.suffix == ".gz":
        return save_compressed_trajectory(trajectory, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_trajectory_to_dict(trajectory), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination


def load_trajectory(path: str | Path) -> Trajectory:
    """Load and validate a trajectory written by :func:`save_trajectory`."""

    source = Path(path)
    if source.suffix == ".gz":
        return load_compressed_trajectory(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("trajectory root must be a JSON object")
    return _trajectory_from_dict(payload)


def save_compressed_trajectory(
    trajectory: Trajectory,
    path: str | Path,
) -> Path:
    """Write a gzip-compressed JSONL trajectory for incremental reading."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "record_type": "header",
        "format_version": trajectory.format_version,
        "seed": trajectory.seed,
        "world_config": asdict(trajectory.world_config),
        "body_config": asdict(trajectory.body_config),
        "initial_observation": _observation_to_dict(
            trajectory.initial_observation
        ),
    }
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for experience in trajectory.experiences:
            record = {
                "record_type": "experience",
                "index": experience.index,
                "observation": _observation_to_dict(
                    experience.observation
                ),
                "action": experience.action.value,
                "next_observation": _observation_to_dict(
                    experience.next_observation
                ),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return destination


def read_compressed_trajectory_header(
    path: str | Path,
) -> CompressedTrajectoryHeader:
    """Read only the first compressed record."""

    source = Path(path)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        line = handle.readline()
    if not line:
        raise ValueError("compressed trajectory is empty")
    return _compressed_header_from_dict(json.loads(line))


def iter_compressed_experiences(
    path: str | Path,
) -> Iterator[Experience]:
    """Yield validated experiences while retaining only one record."""

    source = Path(path)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            raise ValueError("compressed trajectory is empty")
        header = _compressed_header_from_dict(json.loads(first))
        expected = header.initial_observation
        expected_index = 1
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("stream record must be a JSON object")
            if payload.get("record_type") != "experience":
                raise ValueError("unexpected compressed trajectory record")
            experience = _experience_from_dict(payload)
            if experience.index != expected_index:
                raise ValueError(
                    "experience indexes must be contiguous from one"
                )
            if experience.observation != expected:
                raise ValueError(
                    "trajectory observations are not temporally continuous"
                )
            yield experience
            expected = experience.next_observation
            expected_index += 1


def load_compressed_trajectory(path: str | Path) -> Trajectory:
    """Materialize a compressed stream when random access is required."""

    header = read_compressed_trajectory_header(path)
    return Trajectory(
        seed=header.seed,
        world_config=header.world_config,
        body_config=header.body_config,
        initial_observation=header.initial_observation,
        experiences=tuple(iter_compressed_experiences(path)),
        format_version=header.format_version,
    )


def verify_compressed_trajectory(path: str | Path) -> int:
    """Strictly replay a compressed stream without materializing it."""

    header = read_compressed_trajectory_header(path)
    world = BodyDiscoveryWorld(header.world_config, header.body_config)
    current = world.reset(header.seed)
    if current != header.initial_observation:
        raise ReplayMismatchError("initial observation does not match recording")
    count = 0
    for experience in iter_compressed_experiences(path):
        if current != experience.observation:
            raise ReplayMismatchError(
                f"observation before step {experience.index} does not match"
            )
        current = world.step(experience.action).observation
        if current != experience.next_observation:
            raise ReplayMismatchError(
                f"observation after step {experience.index} does not match"
            )
        count += 1
    return count


def _compressed_header_from_dict(
    payload: Any,
) -> CompressedTrajectoryHeader:
    if not isinstance(payload, dict) or payload.get("record_type") != "header":
        raise ValueError("first compressed record must be a header")
    materialized = _trajectory_from_dict(
        {
            **payload,
            "experiences": [],
        }
    )
    return CompressedTrajectoryHeader(
        seed=materialized.seed,
        world_config=materialized.world_config,
        body_config=materialized.body_config,
        initial_observation=materialized.initial_observation,
        format_version=materialized.format_version,
    )


def _trajectory_to_dict(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "format_version": trajectory.format_version,
        "seed": trajectory.seed,
        "world_config": asdict(trajectory.world_config),
        "body_config": asdict(trajectory.body_config),
        "initial_observation": _observation_to_dict(
            trajectory.initial_observation
        ),
        "experiences": [
            {
                "index": experience.index,
                "observation": _observation_to_dict(experience.observation),
                "action": experience.action.value,
                "next_observation": _observation_to_dict(
                    experience.next_observation
                ),
            }
            for experience in trajectory.experiences
        ],
    }


def _trajectory_from_dict(payload: Mapping[str, Any]) -> Trajectory:
    world_data = _require_mapping(payload, "world_config")
    body_data = _require_mapping(payload, "body_config")
    experience_data = payload.get("experiences")
    if not isinstance(experience_data, list):
        raise ValueError("experiences must be a JSON array")

    world_config = WorldConfig(
        image_size=_number_pair(world_data, "image_size", integer=True),
        object_count=int(world_data["object_count"]),
        object_radius=float(world_data["object_radius"]),
        body_visual_value=float(world_data.get("body_visual_value", 1.0)),
        object_visual_value=float(world_data.get("object_visual_value", 1.0)),
        vision_noise_probability=float(
            world_data.get("vision_noise_probability", 0.0)
        ),
        proprioception_noise_std=float(
            world_data.get("proprioception_noise_std", 0.0)
        ),
        touch_dropout_probability=float(
            world_data.get("touch_dropout_probability", 0.0)
        ),
        external_object_motion_probability=float(
            world_data.get("external_object_motion_probability", 0.0)
        ),
        external_object_motion_distance=float(
            world_data.get("external_object_motion_distance", 0.08)
        ),
        distractor_body_count=int(
            world_data.get("distractor_body_count", 0)
        ),
        distractor_body_motion_probability=float(
            world_data.get("distractor_body_motion_probability", 0.0)
        ),
        seed=int(world_data["seed"]),
    )
    body_config = BodyConfig(
        base=_number_pair(body_data, "base"),
        link_lengths=_number_pair(body_data, "link_lengths"),
        shoulder_limits=_number_pair(body_data, "shoulder_limits"),
        elbow_limits=_number_pair(body_data, "elbow_limits"),
        angle_step=float(body_data["angle_step"]),
        link_radius=float(body_data["link_radius"]),
        joint_radius=float(body_data["joint_radius"]),
        tip_radius=float(body_data["tip_radius"]),
        shoulder_enabled=bool(body_data.get("shoulder_enabled", True)),
        elbow_enabled=bool(body_data.get("elbow_enabled", True)),
    )
    experiences = tuple(
        _experience_from_dict(item) for item in experience_data
    )
    return Trajectory(
        format_version=int(payload["format_version"]),
        seed=int(payload["seed"]),
        world_config=world_config,
        body_config=body_config,
        initial_observation=_observation_from_dict(
            _require_mapping(payload, "initial_observation")
        ),
        experiences=experiences,
    )


def _experience_from_dict(payload: Any) -> Experience:
    if not isinstance(payload, dict):
        raise ValueError("each experience must be a JSON object")
    return Experience(
        index=int(payload["index"]),
        observation=_observation_from_dict(
            _require_mapping(payload, "observation")
        ),
        action=BodyAction(str(payload["action"])),
        next_observation=_observation_from_dict(
            _require_mapping(payload, "next_observation")
        ),
    )


def _observation_to_dict(observation: SensorObservation) -> dict[str, Any]:
    return {
        "vision": [list(row) for row in observation.vision],
        "proprioception": list(observation.proprioception),
        "touch": list(observation.touch),
    }


def _observation_from_dict(payload: Mapping[str, Any]) -> SensorObservation:
    vision_data = payload.get("vision")
    proprioception_data = payload.get("proprioception")
    touch_data = payload.get("touch")
    if not isinstance(vision_data, list) or not vision_data:
        raise ValueError("vision must be a non-empty array")
    if not all(isinstance(row, list) and row for row in vision_data):
        raise ValueError("vision rows must be non-empty arrays")
    row_widths = {len(row) for row in vision_data}
    if len(row_widths) != 1:
        raise ValueError("vision rows must have equal width")
    if not isinstance(proprioception_data, list) or len(proprioception_data) != 4:
        raise ValueError("proprioception must contain four values")
    if not isinstance(touch_data, list) or len(touch_data) != 2:
        raise ValueError("touch must contain two values")

    return SensorObservation(
        vision=tuple(
            tuple(float(value) for value in row) for row in vision_data
        ),
        proprioception=tuple(
            float(value) for value in proprioception_data
        ),  # type: ignore[arg-type]
        touch=tuple(bool(value) for value in touch_data),  # type: ignore[arg-type]
    )


def _require_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _number_pair(
    payload: Mapping[str, Any],
    key: str,
    *,
    integer: bool = False,
) -> tuple[Any, Any]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must contain two numbers")
    if len(value) != 2:
        raise ValueError(f"{key} must contain two numbers")
    converter = int if integer else float
    return converter(value[0]), converter(value[1])


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a trajectory from the command line."""

    parser = argparse.ArgumentParser(
        description="Record a deterministic random Cal trajectory."
    )
    parser.add_argument(
        "output",
        type=Path,
        help="destination .json or compressed .jsonl.gz path",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--objects", type=int, default=3)
    arguments = parser.parse_args(argv)

    world = BodyDiscoveryWorld(
        WorldConfig(
            image_size=(arguments.width, arguments.height),
            object_count=arguments.objects,
            seed=arguments.seed,
        )
    )
    trajectory = record_random_trajectory(
        world,
        arguments.steps,
        seed=arguments.seed,
    )
    destination = save_trajectory(trajectory, arguments.output)
    print(f"recorded {len(trajectory)} steps to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
