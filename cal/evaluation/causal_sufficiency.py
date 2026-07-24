"""Audit how much body evidence exact visual interventions can provide."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from cal.env.body import BodyAction, BodyConfig, BodyState
from cal.env.world import BodyDiscoveryWorld, WorldConfig
from cal.evaluation.metrics import segmentation_metrics
from cal.infra.provenance import capture_provenance

DEFAULT_HISTORY_LENGTHS = (1, 2, 4, 8, 16, 32)
GEODESIC_DEPTHS = (0, 1, 2, 4, 8, 16)


@dataclass(frozen=True, slots=True)
class CausalAuditStep:
    seed: int
    step: int
    state: BodyState
    vision: Tensor
    body_mask: Tensor
    external_mask: Tensor
    action_effects: Tensor
    exhaustive_envelope: Tensor


def causal_action_effects(world: BodyDiscoveryWorld) -> Tensor:
    """Return exact action-vs-NOOP visual effects for every action."""

    noop = torch.tensor(
        deepcopy(world).step(BodyAction.NOOP).observation.vision,
        dtype=torch.float32,
    )
    effects = []
    for action in world.actions:
        branch = torch.tensor(
            deepcopy(world).step(action).observation.vision,
            dtype=torch.float32,
        )
        effects.append((branch - noop).abs())
    return torch.stack(effects)


def effect_envelope(
    action_effects: Tensor,
    current_vision: Tensor,
    *,
    action_indexes: Sequence[int] | None = None,
) -> Tensor:
    """Dilate selected action effects and constrain them to current occupancy."""

    if action_effects.ndim != 3:
        raise ValueError("action effects must be [action, height, width]")
    selected = (
        action_effects
        if action_indexes is None
        else action_effects[list(action_indexes)]
    )
    if selected.shape[0] == 0:
        raise ValueError("at least one action effect is required")
    union = selected.amax(dim=0).unsqueeze(0).unsqueeze(0)
    dilated = torch.nn.functional.max_pool2d(
        union,
        kernel_size=3,
        stride=1,
        padding=1,
    ).squeeze(0).squeeze(0)
    return dilated * current_vision


def geodesic_propagation(
    seeds: Tensor,
    occupancy: Tensor,
    *,
    steps: int,
) -> Tensor:
    """Propagate causal seeds only through currently occupied pixels."""

    if steps < 0:
        raise ValueError("geodesic steps cannot be negative")
    if seeds.shape != occupancy.shape or seeds.ndim != 2:
        raise ValueError("geodesic inputs must be equal 2D tensors")
    current = (seeds > 0.5).float() * (occupancy > 0.5).float()
    support = (occupancy > 0.5).float()
    for _ in range(steps):
        current = torch.nn.functional.max_pool2d(
            current.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(0).squeeze(0)
        current = current * support
    return current


def collect_causal_audit_steps(
    world_config: WorldConfig,
    body_config: BodyConfig,
    seeds: Sequence[int],
    *,
    steps_per_seed: int,
) -> tuple[tuple[CausalAuditStep, ...], ...]:
    """Collect exact intervention evidence and evaluation-only labels."""

    if steps_per_seed <= 0:
        raise ValueError("steps_per_seed must be positive")
    trajectories = []
    for seed in seeds:
        resolved_seed = int(seed)
        world = BodyDiscoveryWorld(
            replace(world_config, seed=resolved_seed),
            body_config,
        )
        current = world.reset(resolved_seed)
        steps = []
        for step_index in range(steps_per_seed):
            snapshot = world.evaluation_snapshot()
            vision = torch.tensor(current.vision, dtype=torch.float32)
            effects = causal_action_effects(world)
            steps.append(
                CausalAuditStep(
                    seed=resolved_seed,
                    step=step_index,
                    state=snapshot.body_state,
                    vision=vision,
                    body_mask=torch.tensor(
                        snapshot.masks.body,
                        dtype=torch.float32,
                    ),
                    external_mask=torch.tensor(
                        snapshot.masks.objects,
                        dtype=torch.float32,
                    ),
                    action_effects=effects,
                    exhaustive_envelope=effect_envelope(effects, vision),
                )
            )
            current = world.step(world.sample_action()).observation
        trajectories.append(tuple(steps))
    if not trajectories:
        raise ValueError("at least one causal audit seed is required")
    return tuple(trajectories)


def collect_pose_grid_steps(
    world_config: WorldConfig,
    body_config: BodyConfig,
    seeds: Sequence[int],
) -> tuple[CausalAuditStep, ...]:
    """Evaluate exact limits, near-limits, and interior joint poses."""

    shoulder_values = _pose_axis(
        body_config.shoulder_limits,
        body_config.angle_step,
    )
    elbow_values = _pose_axis(
        body_config.elbow_limits,
        body_config.angle_step,
    )
    steps = []
    for seed in seeds:
        resolved_seed = int(seed)
        world = BodyDiscoveryWorld(
            replace(world_config, seed=resolved_seed),
            body_config,
        )
        world.reset(resolved_seed)
        pose_index = 0
        for shoulder in shoulder_values:
            for elbow in elbow_values:
                world.body.reset(
                    BodyState(
                        shoulder_angle=shoulder,
                        elbow_angle=elbow,
                    )
                )
                current = world.observe()
                snapshot = world.evaluation_snapshot()
                vision = torch.tensor(current.vision, dtype=torch.float32)
                effects = causal_action_effects(world)
                steps.append(
                    CausalAuditStep(
                        seed=resolved_seed,
                        step=pose_index,
                        state=snapshot.body_state,
                        vision=vision,
                        body_mask=torch.tensor(
                            snapshot.masks.body,
                            dtype=torch.float32,
                        ),
                        external_mask=torch.tensor(
                            snapshot.masks.objects,
                            dtype=torch.float32,
                        ),
                        action_effects=effects,
                        exhaustive_envelope=effect_envelope(effects, vision),
                    )
                )
                pose_index += 1
    return tuple(steps)


def run_causal_sufficiency_audit(
    *,
    output_path: str | Path,
    seeds: Sequence[int] = tuple(range(400, 416)),
    steps_per_seed: int = 128,
    history_lengths: Sequence[int] = DEFAULT_HISTORY_LENGTHS,
    geodesic_depths: Sequence[int] = GEODESIC_DEPTHS,
    world_config: WorldConfig | None = None,
    body_config: BodyConfig | None = None,
) -> dict[str, Any]:
    """Run coverage, pose, history, geometry, and action-cost audits."""

    resolved_world = world_config or WorldConfig(
        image_size=(16, 16),
        object_count=0,
        distractor_body_count=2,
        distractor_body_motion_probability=1.0,
    )
    resolved_body = body_config or BodyConfig()
    if max(history_lengths) > steps_per_seed:
        raise ValueError("history length cannot exceed trajectory length")
    started = time.perf_counter()
    trajectories = collect_causal_audit_steps(
        resolved_world,
        resolved_body,
        seeds,
        steps_per_seed=steps_per_seed,
    )
    flat_steps = tuple(step for trajectory in trajectories for step in trajectory)
    pose_grid_steps = collect_pose_grid_steps(
        resolved_world,
        resolved_body,
        seeds,
    )
    targets = torch.stack([step.body_mask for step in flat_steps]).unsqueeze(1)
    exhaustive = torch.stack(
        [step.exhaustive_envelope for step in flat_steps]
    ).unsqueeze(1)
    action_results = {}
    for action_index, action in enumerate(tuple(BodyAction)):
        prediction = torch.stack(
            [
                effect_envelope(
                    step.action_effects,
                    step.vision,
                    action_indexes=(action_index,),
                )
                for step in flat_steps
            ]
        ).unsqueeze(1)
        action_results[action.value] = _binary_metrics(prediction, targets)
    active_indexes = [
        _largest_visual_effect_index(step.action_effects)
        for step in flat_steps
    ]
    active_prediction = torch.stack(
        [
            effect_envelope(
                step.action_effects,
                step.vision,
                action_indexes=(action_index,),
            )
            for step, action_index in zip(
                flat_steps,
                active_indexes,
                strict=True,
            )
        ]
    ).unsqueeze(1)
    action_comparison = {
        "fixed_single_actions": action_results,
        "oracle_best_single_action": {
            **_binary_metrics(active_prediction, targets),
            "mean_branch_queries_per_state": float(len(tuple(BodyAction))),
            "formal_policy": False,
            "note": "diagnostic upper bound selected after all branches were observed",
        },
        "exhaustive_all_actions": {
            **_binary_metrics(exhaustive, targets),
            "mean_branch_queries_per_state": float(len(tuple(BodyAction))),
        },
    }
    history_results = {}
    for length in history_lengths:
        predictions = []
        history_targets = []
        for trajectory in trajectories:
            for end in range(int(length) - 1, len(trajectory)):
                start = end - int(length) + 1
                current_vision = trajectory[end].vision
                historical_union = torch.stack(
                    [
                        item.exhaustive_envelope
                        for item in trajectory[start : end + 1]
                    ]
                ).amax(dim=0)
                predictions.append(historical_union * current_vision)
                history_targets.append(trajectory[end].body_mask)
        history_results[str(length)] = _binary_metrics(
            torch.stack(predictions).unsqueeze(1),
            torch.stack(history_targets).unsqueeze(1),
        )
    geodesic_results = {}
    for depth in geodesic_depths:
        prediction = torch.stack(
            [
                geodesic_propagation(
                    step.exhaustive_envelope,
                    step.vision,
                    steps=int(depth),
                )
                for step in flat_steps
            ]
        ).unsqueeze(1)
        geodesic_results[str(depth)] = _binary_metrics(prediction, targets)
    pose_buckets = {
        "random_trajectories": _pose_bucket_metrics(
            flat_steps,
            resolved_body,
        ),
        "analytic_pose_grid": _pose_bucket_metrics(
            pose_grid_steps,
            resolved_body,
        ),
    }
    per_step = []
    for step in (*flat_steps, *pose_grid_steps):
        metrics = _binary_metrics(
            step.exhaustive_envelope.unsqueeze(0).unsqueeze(0),
            step.body_mask.unsqueeze(0).unsqueeze(0),
        )
        overlap = _overlap_fraction(step.body_mask, step.external_mask)
        per_step.append(
            {
                "seed": step.seed,
                "step": step.step,
                "shoulder_angle": step.state.shoulder_angle,
                "elbow_angle": step.state.elbow_angle,
                "external_overlap_fraction": overlap,
                "iou": metrics["iou"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "body_pixels": int(step.body_mask.sum()),
                "causal_pixels": int(step.exhaustive_envelope.sum()),
            }
        )
    worst = sorted(
        per_step,
        key=lambda item: (item["iou"], item["recall"], item["precision"]),
    )[:20]
    exhaustive_metrics = _binary_metrics(exhaustive, targets)
    pose_grid_metrics = _binary_metrics(
        torch.stack(
            [step.exhaustive_envelope for step in pose_grid_steps]
        ).unsqueeze(1),
        torch.stack(
            [step.body_mask for step in pose_grid_steps]
        ).unsqueeze(1),
    )
    best_geodesic_depth, best_geodesic = max(
        geodesic_results.items(),
        key=lambda item: item[1]["iou"],
    )
    best_history_length, best_history = max(
        history_results.items(),
        key=lambda item: item[1]["iou"],
    )
    summary = {
        "result_schema_version": 1,
        "audit": "v2_causal_sufficiency",
        "diagnostic_only": True,
        "learnable_parameter_count": 0,
        "world_config": asdict(resolved_world),
        "body_config": asdict(resolved_body),
        "seeds": [int(seed) for seed in seeds],
        "steps_per_seed": steps_per_seed,
        "exhaustive_current": exhaustive_metrics,
        "analytic_pose_grid": {
            "sample_count": len(pose_grid_steps),
            "metrics": pose_grid_metrics,
        },
        "action_comparison": action_comparison,
        "history_union": history_results,
        "geodesic_propagation": geodesic_results,
        "best_deterministic_geometry": {
            "depth": int(best_geodesic_depth),
            "metrics": best_geodesic,
        },
        "best_history_union": {
            "length": int(best_history_length),
            "metrics": best_history,
        },
        "pose_buckets": pose_buckets,
        "worst_poses": worst,
        "resource_gates": {
            "parameter_limit": 100_000,
            "parameter_passed": True,
            "cpu_only": True,
            "duration_limit_seconds": 7_200.0,
            "duration_passed": (time.perf_counter() - started) <= 7_200.0,
        },
        "duration_seconds": time.perf_counter() - started,
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _pose_bucket_metrics(
    steps: Sequence[CausalAuditStep],
    body_config: BodyConfig,
) -> dict[str, Any]:
    buckets: dict[str, list[CausalAuditStep]] = {
        "shoulder_near_limit": [],
        "shoulder_interior": [],
        "elbow_near_limit": [],
        "elbow_interior": [],
        "overlap_none": [],
        "overlap_low": [],
        "overlap_medium": [],
        "overlap_high": [],
    }
    for step in steps:
        shoulder_near = _near_limit(
            step.state.shoulder_angle,
            body_config.shoulder_limits,
            body_config.angle_step,
        )
        elbow_near = _near_limit(
            step.state.elbow_angle,
            body_config.elbow_limits,
            body_config.angle_step,
        )
        buckets[
            "shoulder_near_limit" if shoulder_near else "shoulder_interior"
        ].append(step)
        buckets[
            "elbow_near_limit" if elbow_near else "elbow_interior"
        ].append(step)
        overlap = _overlap_fraction(step.body_mask, step.external_mask)
        if overlap == 0.0:
            name = "overlap_none"
        elif overlap < 0.25:
            name = "overlap_low"
        elif overlap < 0.5:
            name = "overlap_medium"
        else:
            name = "overlap_high"
        buckets[name].append(step)
    result = {}
    for name, items in buckets.items():
        if not items:
            result[name] = {"sample_count": 0, "metrics": None}
            continue
        prediction = torch.stack(
            [item.exhaustive_envelope for item in items]
        ).unsqueeze(1)
        target = torch.stack([item.body_mask for item in items]).unsqueeze(1)
        result[name] = {
            "sample_count": len(items),
            "metrics": _binary_metrics(prediction, target),
        }
    return result


def _binary_metrics(prediction: Tensor, target: Tensor) -> dict[str, Any]:
    logits = torch.logit(prediction.clamp(1e-5, 1.0 - 1e-5))
    return asdict(segmentation_metrics(logits, target))


def _largest_visual_effect_index(action_effects: Tensor) -> int:
    areas = action_effects.flatten(start_dim=1).sum(dim=1)
    return int(torch.argmax(areas))


def _near_limit(
    value: float,
    limits: tuple[float, float],
    margin: float,
) -> bool:
    return value - limits[0] <= margin or limits[1] - value <= margin


def _pose_axis(
    limits: tuple[float, float],
    step: float,
) -> tuple[float, ...]:
    low, high = limits
    return (low, low + step, (low + high) / 2.0, high - step, high)


def _overlap_fraction(body: Tensor, external: Tensor) -> float:
    body_pixels = float(body.sum())
    if body_pixels == 0.0:
        return 0.0
    return float(((body > 0.5) & (external > 0.5)).sum()) / body_pixels


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the sufficiency of exact causal visual evidence."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/V2-causal-sufficiency-summary.json"),
    )
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--steps-per-seed", type=int, default=128)
    arguments = parser.parse_args(argv)
    if arguments.seed_count <= 0:
        parser.error("--seed-count must be positive")
    result = run_causal_sufficiency_audit(
        output_path=arguments.output,
        seeds=tuple(range(400, 400 + arguments.seed_count)),
        steps_per_seed=arguments.steps_per_seed,
    )
    current = result["exhaustive_current"]
    geometry = result["best_deterministic_geometry"]
    print(
        f"exhaustive IoU={current['iou']:.4f}; "
        f"recall={current['recall']:.4f}; "
        f"precision={current['precision']:.4f}"
    )
    print(
        f"best geometry depth={geometry['depth']}; "
        f"IoU={geometry['metrics']['iou']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
