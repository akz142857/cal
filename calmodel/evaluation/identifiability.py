"""Audit whether visual-action histories uniquely identify the body.

This module is evaluation-only. Privileged body masks and poses are used to
measure ambiguity, never exposed to a learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from random import Random
from typing import Any, Iterable, Sequence

from calmodel.env.body import (
    ArticulatedBody,
    BodyAction,
    BodyConfig,
    BodyState,
)
from calmodel.env.sensors import (
    BinaryMask,
    VisionConfig,
    VisionFrame,
    VisionSensor,
)
from calmodel.env.world import BodyDiscoveryWorld, WorldConfig
from calmodel.infra.provenance import capture_provenance

DEFAULT_HISTORY_LENGTHS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True, slots=True)
class AuditSample:
    """One current state and its observable history signature."""

    signature: str
    seed: int
    step: int
    body_mask: BinaryMask
    body_state: BodyState


@dataclass(frozen=True, slots=True)
class AuditSequence:
    """Learner-visible frames/actions with evaluation-only state alongside."""

    seed: int
    visions: tuple[VisionFrame, ...]
    actions: tuple[BodyAction, ...]
    body_masks: tuple[BinaryMask, ...]
    body_states: tuple[BodyState, ...]

    def __post_init__(self) -> None:
        length = len(self.visions)
        if length == 0:
            raise ValueError("audit sequence cannot be empty")
        if not (
            len(self.actions)
            == len(self.body_masks)
            == len(self.body_states)
            == length
        ):
            raise ValueError("audit sequence fields must have equal length")


def history_signature(
    visions: Sequence[VisionFrame],
    actions: Sequence[BodyAction],
    *,
    end: int,
    length: int,
) -> str:
    """Hash the exact binary video and intervening action history."""

    if length <= 0:
        raise ValueError("history length must be positive")
    if end < length - 1 or end >= len(visions):
        raise ValueError("history window is outside the sequence")
    if len(actions) < len(visions) - 1:
        raise ValueError("actions must cover transitions between frames")
    digest = hashlib.sha256()
    start = end - length + 1
    for frame_index in range(start, end + 1):
        frame = visions[frame_index]
        digest.update(len(frame).to_bytes(2, "little"))
        digest.update(
            (len(frame[0]) if frame else 0).to_bytes(2, "little")
        )
        digest.update(
            bytes(
                1 if value >= 0.5 else 0
                for row in frame
                for value in row
            )
        )
        if frame_index < end:
            digest.update(
                bytes((_action_index(actions[frame_index]),))
            )
    return digest.hexdigest()


def collect_audit_sequences(
    world_config: WorldConfig,
    body_config: BodyConfig,
    seeds: Iterable[int],
    *,
    steps_per_seed: int,
    policy: str,
) -> tuple[AuditSequence, ...]:
    """Collect random or active visual-probe histories."""

    if steps_per_seed <= 0:
        raise ValueError("steps_per_seed must be positive")
    if policy not in {"random", "active_visual"}:
        raise ValueError("policy must be random or active_visual")
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
        masks = []
        states = []
        previous_action: BodyAction | None = None
        active_counts = {action: 0 for action in world.actions}
        for _ in range(steps_per_seed):
            snapshot = world.evaluation_snapshot()
            visions.append(current.vision)
            masks.append(snapshot.masks.body)
            states.append(snapshot.body_state)
            if policy == "random":
                action = world.sample_action()
            else:
                action = select_active_visual_action(
                    world,
                    previous_action=previous_action,
                    action_counts=active_counts,
                )
                active_counts[action] += 1
            actions.append(action)
            current = world.step(action).observation
            previous_action = action
        sequences.append(
            AuditSequence(
                seed=resolved_seed,
                visions=tuple(visions),
                actions=tuple(actions),
                body_masks=tuple(masks),
                body_states=tuple(states),
            )
        )
    if not sequences:
        raise ValueError("at least one seed is required")
    return tuple(sequences)


def select_active_visual_action(
    world: BodyDiscoveryWorld,
    *,
    previous_action: BodyAction | None = None,
    action_counts: Mapping[BodyAction, int] | None = None,
) -> BodyAction:
    """Choose a balanced intervention without querying counterfactual branches."""

    del world  # The policy deliberately cannot inspect simulator branches.
    counts = action_counts or {}
    candidates = [action for action in BodyAction if action is not BodyAction.NOOP]
    return min(
        candidates,
        key=lambda action: (
            counts.get(action, 0),
            action == previous_action,
            _action_index(action),
        ),
    )


def build_audit_samples(
    sequences: Sequence[AuditSequence],
    *,
    history_length: int,
) -> tuple[AuditSample, ...]:
    """Convert collected sequences into exact observable equivalence samples."""

    samples = []
    for sequence in sequences:
        for end in range(history_length - 1, len(sequence.visions)):
            samples.append(
                AuditSample(
                    signature=history_signature(
                        sequence.visions,
                        sequence.actions,
                        end=end,
                        length=history_length,
                    ),
                    seed=sequence.seed,
                    step=end,
                    body_mask=sequence.body_masks[end],
                    body_state=sequence.body_states[end],
                )
            )
    return tuple(samples)


def analyze_equivalence_classes(
    samples: Sequence[AuditSample],
    *,
    example_limit: int = 20,
) -> dict[str, Any]:
    """Measure ambiguity and a pixel-majority empirical Bayes ceiling."""

    groups: dict[str, list[AuditSample]] = {}
    for sample in samples:
        groups.setdefault(sample.signature, []).append(sample)
    collision_groups = [items for items in groups.values() if len(items) > 1]
    cross_seed_groups = [
        items
        for items in collision_groups
        if len({item.seed for item in items}) > 1
    ]
    ambiguous_groups = [
        items
        for items in collision_groups
        if len({_mask_bytes(item.body_mask) for item in items}) > 1
    ]
    cross_seed_ambiguous = [
        items
        for items in ambiguous_groups
        if len({item.seed for item in items}) > 1
    ]
    collision_samples = sum(len(items) for items in collision_groups)
    ambiguous_samples = sum(len(items) for items in ambiguous_groups)
    majority_all = _majority_metrics(tuple(groups.values()))
    majority_collisions = _majority_metrics(collision_groups)
    majority_ambiguous = _majority_metrics(ambiguous_groups)
    mask_ious = []
    for items in collision_groups:
        majority = _majority_mask(items)
        mask_ious.extend(_mask_iou(item.body_mask, majority) for item in items)
    examples = []
    for items in cross_seed_ambiguous[:example_limit]:
        distinct = []
        seen_masks = set()
        for item in items:
            encoded = _mask_bytes(item.body_mask)
            if encoded in seen_masks:
                continue
            seen_masks.add(encoded)
            distinct.append(item)
            if len(distinct) == 2:
                break
        if len(distinct) < 2:
            continue
        first, second = distinct
        examples.append(
            {
                "signature": first.signature,
                "mask_iou": _mask_iou(
                    first.body_mask,
                    second.body_mask,
                ),
                "first": _sample_description(first),
                "second": _sample_description(second),
            }
        )
    sample_count = len(samples)
    return {
        "sample_count": sample_count,
        "equivalence_class_count": len(groups),
        "collision_class_count": len(collision_groups),
        "cross_seed_collision_class_count": len(cross_seed_groups),
        "ambiguous_class_count": len(ambiguous_groups),
        "cross_seed_ambiguous_class_count": len(cross_seed_ambiguous),
        "collision_sample_count": collision_samples,
        "ambiguous_sample_count": ambiguous_samples,
        "collision_sample_rate": (
            collision_samples / sample_count if sample_count else 0.0
        ),
        "ambiguity_sample_rate": (
            ambiguous_samples / sample_count if sample_count else 0.0
        ),
        "majority_ceiling_all": majority_all,
        "majority_ceiling_collisions": majority_collisions,
        "majority_ceiling_ambiguous": majority_ambiguous,
        "mask_iou_to_class_majority": _distribution(mask_ious),
        "ambiguous_examples": examples,
    }


def run_identifiability_audit(
    *,
    output_path: str | Path,
    seeds: Sequence[int] = tuple(range(64)),
    steps_per_seed: int = 128,
    history_lengths: Sequence[int] = DEFAULT_HISTORY_LENGTHS,
    world_config: WorldConfig | None = None,
    body_config: BodyConfig | None = None,
) -> dict[str, Any]:
    """Run random and active-policy identifiability audits."""

    resolved_world = world_config or WorldConfig(
        image_size=(16, 16),
        object_count=0,
        distractor_body_count=2,
        distractor_body_motion_probability=1.0,
    )
    resolved_body = body_config or BodyConfig()
    if not history_lengths or min(history_lengths) <= 0:
        raise ValueError("history lengths must be positive")
    if max(history_lengths) > steps_per_seed:
        raise ValueError("history length cannot exceed trajectory length")
    started = time.perf_counter()
    policy_results = {}
    for policy in ("random", "active_visual"):
        sequences = collect_audit_sequences(
            resolved_world,
            resolved_body,
            seeds,
            steps_per_seed=steps_per_seed,
            policy=policy,
        )
        policy_results[policy] = {
            str(length): analyze_equivalence_classes(
                build_audit_samples(
                    sequences,
                    history_length=int(length),
                )
            )
            for length in history_lengths
        }
    permutation_symmetry = permutation_symmetry_audit(
        resolved_body,
        image_size=resolved_world.image_size,
        history_lengths=history_lengths,
    )
    summary = {
        "result_schema_version": 1,
        "audit": "v2_identifiability",
        "diagnostic_only": True,
        "learnable_parameter_count": 0,
        "world_config": asdict(resolved_world),
        "body_config": asdict(resolved_body),
        "seeds": [int(seed) for seed in seeds],
        "steps_per_seed": steps_per_seed,
        "history_lengths": [int(length) for length in history_lengths],
        "policies": policy_results,
        "permutation_symmetry": permutation_symmetry,
        "duration_seconds": time.perf_counter() - started,
        "resource_gates": {
            "parameter_limit": 100_000,
            "parameter_passed": True,
            "cpu_only": True,
            "duration_limit_seconds": 7_200.0,
            "duration_passed": (time.perf_counter() - started) <= 7_200.0,
        },
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def permutation_symmetry_audit(
    body_config: BodyConfig,
    *,
    image_size: tuple[int, int],
    history_lengths: Sequence[int],
    case_count: int = 64,
    seed: int = 91_019,
) -> dict[str, Any]:
    """Construct exact histories invariant to swapping self with a mirror arm.

    Two visually identical worlds are paired. Arm A is self in the first and
    arm B is self in the second. A and B receive the same learner action, so
    the complete visual-action history is identical while the privileged body
    mask generally differs. This is a counterexample to universal
    identifiability, not an estimate of its frequency under random motion.
    """

    if case_count <= 0:
        raise ValueError("case count must be positive")
    if not history_lengths or min(history_lengths) <= 0:
        raise ValueError("history lengths must be positive")
    sensor = VisionSensor(
        VisionConfig(width=image_size[0], height=image_size[1])
    )
    random = Random(seed)
    max_length = max(history_lengths)
    schedules = tuple(BodyAction)
    samples_by_length: dict[int, list[AuditSample]] = {
        int(length): [] for length in history_lengths
    }
    exact_history_pairs = {str(length): 0 for length in history_lengths}
    for case_index in range(case_count):
        arms = [
            ArticulatedBody(body_config),
            ArticulatedBody(body_config),
            ArticulatedBody(body_config),
        ]
        for arm in arms:
            arm.reset(
                BodyState(
                    shoulder_angle=random.uniform(
                        *body_config.shoulder_limits
                    ),
                    elbow_angle=random.uniform(*body_config.elbow_limits),
                )
            )
        visions = []
        first_masks = []
        second_masks = []
        states_a = []
        states_b = []
        actions = []
        for step in range(max_length):
            arm_a, arm_b, arm_c = arms
            first_vision = sensor.observe(
                arm_a,
                (),
                (arm_b, arm_c),
            )
            second_vision = sensor.observe(
                arm_b,
                (),
                (arm_a, arm_c),
            )
            if first_vision != second_vision:
                raise RuntimeError("arm permutation changed visual occupancy")
            visions.append(first_vision)
            first_masks.append(sensor.evaluation_masks(arm_a, ()).body)
            second_masks.append(sensor.evaluation_masks(arm_b, ()).body)
            states_a.append(arm_a.state)
            states_b.append(arm_b.state)
            action = schedules[(case_index + step) % len(schedules)]
            actions.append(action)
            # The external arm mirrors the learner action. This is allowed as
            # a possible exogenous history and creates the symmetry proof.
            arm_a.apply(action)
            arm_b.apply(action)
            arm_c.apply(
                schedules[(case_index + 2 * step + 1) % len(schedules)]
            )
        for length in history_lengths:
            end = int(length) - 1
            signature = history_signature(
                visions,
                actions,
                end=end,
                length=int(length),
            )
            first = AuditSample(
                signature=signature,
                seed=case_index * 2,
                step=end,
                body_mask=first_masks[end],
                body_state=states_a[end],
            )
            second = AuditSample(
                signature=signature,
                seed=case_index * 2 + 1,
                step=end,
                body_mask=second_masks[end],
                body_state=states_b[end],
            )
            samples_by_length[int(length)].extend((first, second))
            exact_history_pairs[str(length)] += 1
    return {
        "description": (
            "exact counterexamples formed by swapping self with an "
            "externally mirrored isomorphic arm"
        ),
        "case_count": case_count,
        "exact_history_pairs": exact_history_pairs,
        "histories": {
            str(length): analyze_equivalence_classes(samples)
            for length, samples in samples_by_length.items()
        },
    }


def _majority_metrics(
    groups: Sequence[Sequence[AuditSample]],
) -> dict[str, Any]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    sample_count = 0
    for items in groups:
        if not items:
            continue
        predicted = _majority_mask(items)
        for item in items:
            sample_count += 1
            for expected_row, predicted_row in zip(
                item.body_mask,
                predicted,
                strict=True,
            ):
                for expected, selected in zip(
                    expected_row,
                    predicted_row,
                    strict=True,
                ):
                    if selected and expected:
                        true_positive += 1
                    elif selected:
                        false_positive += 1
                    elif expected:
                        false_negative += 1
                    else:
                        true_negative += 1
    return {
        "sample_count": sample_count,
        "iou": _safe_ratio(
            true_positive,
            true_positive + false_positive + false_negative,
        ),
        "f1": _safe_ratio(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
        "precision": _safe_ratio(
            true_positive,
            true_positive + false_positive,
        ),
        "recall": _safe_ratio(
            true_positive,
            true_positive + false_negative,
        ),
        "pixel_accuracy": _safe_ratio(
            true_positive + true_negative,
            true_positive
            + true_negative
            + false_positive
            + false_negative,
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def _majority_mask(items: Sequence[AuditSample]) -> BinaryMask:
    if not items:
        raise ValueError("majority mask requires at least one sample")
    height = len(items[0].body_mask)
    width = len(items[0].body_mask[0])
    threshold = ceil(len(items) / 2)
    return tuple(
        tuple(
            sum(bool(item.body_mask[y][x]) for item in items) >= threshold
            for x in range(width)
        )
        for y in range(height)
    )


def _mask_bytes(mask: BinaryMask) -> bytes:
    return bytes(value for row in mask for value in row)


def _mask_iou(first: BinaryMask, second: BinaryMask) -> float:
    intersection = 0
    union = 0
    for first_row, second_row in zip(first, second, strict=True):
        for first_value, second_value in zip(
            first_row,
            second_row,
            strict=True,
        ):
            intersection += int(first_value and second_value)
            union += int(first_value or second_value)
    return _safe_ratio(intersection, union)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "maximum": max(values),
    }


def _sample_description(sample: AuditSample) -> dict[str, Any]:
    return {
        "seed": sample.seed,
        "step": sample.step,
        "shoulder_angle": sample.body_state.shoulder_angle,
        "elbow_angle": sample.body_state.elbow_angle,
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0


def _action_index(action: BodyAction) -> int:
    return tuple(BodyAction).index(action)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit exact visual-action history identifiability."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/V2-identifiability-summary.json"),
    )
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--steps-per-seed", type=int, default=128)
    parser.add_argument(
        "--history-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_HISTORY_LENGTHS),
    )
    arguments = parser.parse_args(argv)
    if arguments.seed_count <= 0:
        parser.error("--seed-count must be positive")
    result = run_identifiability_audit(
        output_path=arguments.output,
        seeds=tuple(range(arguments.seed_count)),
        steps_per_seed=arguments.steps_per_seed,
        history_lengths=tuple(arguments.history_lengths),
    )
    for policy, histories in result["policies"].items():
        for length, metrics in histories.items():
            print(
                f"{policy} K={length}: "
                f"ambiguity={metrics['ambiguity_sample_rate']:.4f}; "
                f"collision Bayes IoU="
                f"{metrics['majority_ceiling_collisions']['iou']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
