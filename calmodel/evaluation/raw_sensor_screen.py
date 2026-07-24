"""Environment-only body-leak screen using fixed and raw-sensor baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from calmodel.env.body import BodyConfig
from calmodel.env.world import WorldConfig
from calmodel.evaluation.body_probe import (
    ProbeConfig,
    apply_probe_projection,
    evaluate_fixed_position_baseline,
    evaluate_probe,
    evaluate_visual_threshold_baseline,
    extract_raw_sensor_probe_data,
    fit_pca_probe_projection,
    train_linear_probe,
)
from calmodel.infra.provenance import capture_provenance
from calmodel.learning.dataset import SeedSplit, collect_trajectories


def run_raw_sensor_screen(
    config_path: str | Path,
    *,
    output_path: str | Path,
    raw_sensor_max_iou: float | None = None,
    visual_threshold_min_iou: float | None = None,
    visual_threshold_max_iou: float | None = None,
    reference_raw_sensor_iou: float | None = None,
    reference_visual_threshold_iou: float | None = None,
    minimum_iou_reduction: float = 0.0,
) -> dict[str, Any]:
    """Fit only label-side baselines; no representation model is required."""

    source = Path(config_path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration must be a mapping")
    environment = _mapping(payload, "environment")
    probe = _mapping(payload, "probe")
    image_size = tuple(int(value) for value in environment["image_size"])
    world = WorldConfig(
        image_size=image_size,  # type: ignore[arg-type]
        object_count=int(environment.get("object_count", 3)),
        object_radius=float(environment.get("object_radius", 0.045)),
        body_visual_value=float(environment.get("body_visual_value", 1.0)),
        object_visual_value=float(environment.get("object_visual_value", 1.0)),
        external_object_motion_probability=float(
            probe.get(
                "evaluation_external_motion_probability",
                environment.get("external_object_motion_probability", 0.0),
            )
        ),
        external_object_motion_distance=float(
            environment.get("external_object_motion_distance", 0.08)
        ),
        distractor_body_count=int(
            environment.get("distractor_body_count", 0)
        ),
        distractor_body_motion_probability=float(
            environment.get("distractor_body_motion_probability", 0.0)
        ),
        seed=int(payload.get("seed", 0)),
    )
    split = SeedSplit(
        train=tuple(int(value) for value in probe["train_seeds"]),
        validation=tuple(int(value) for value in probe["validation_seeds"]),
        test=tuple(int(value) for value in probe["test_seeds"]),
    )
    steps = int(probe.get("steps_per_seed", 128))
    body = BodyConfig()
    train_trajectories = collect_trajectories(
        world,
        body,
        split.train,
        steps_per_seed=steps,
    )
    validation_trajectories = collect_trajectories(
        world,
        body,
        split.validation,
        steps_per_seed=steps,
    )
    test_trajectories = collect_trajectories(
        world,
        body,
        split.test,
        steps_per_seed=steps,
    )
    modality_options = {
        "include_vision": bool(environment.get("include_vision", True)),
        "include_proprioception": bool(
            environment.get("include_proprioception", True)
        ),
        "include_touch": bool(environment.get("include_touch", True)),
        "include_action": bool(environment.get("include_action", True)),
    }
    train = extract_raw_sensor_probe_data(
        train_trajectories,
        **modality_options,
    )
    validation = extract_raw_sensor_probe_data(
        validation_trajectories,
        **modality_options,
    )
    test = extract_raw_sensor_probe_data(
        test_trajectories,
        **modality_options,
    )
    projection_size = int(probe.get("projection_size", 128))
    projection = fit_pca_probe_projection(train, projection_size)
    projected_train = apply_probe_projection(train, projection)
    projected_validation = apply_probe_projection(validation, projection)
    projected_test = apply_probe_projection(test, projection)
    config = ProbeConfig(
        epochs=int(probe.get("epochs", 100)),
        batch_size=int(probe.get("batch_size", 64)),
        learning_rate=float(probe.get("learning_rate", 0.01)),
        seed=int(payload.get("seed", 0)),
        device=str(probe.get("device", "cpu")),
    )
    trained = train_linear_probe(
        projected_train,
        projected_validation,
        config=config,
    )
    raw_metrics = evaluate_probe(
        trained.probe,
        projected_test,
        batch_size=config.batch_size,
        device=config.device,
    )
    visual_metrics = evaluate_visual_threshold_baseline(test)
    gates: dict[str, bool] = {}
    if raw_sensor_max_iou is not None:
        gates["raw_sensor_iou_at_most_threshold"] = (
            raw_metrics.iou <= raw_sensor_max_iou
        )
    if visual_threshold_max_iou is not None:
        gates["visual_threshold_iou_at_most_threshold"] = (
            visual_metrics.iou <= visual_threshold_max_iou
        )
    if visual_threshold_min_iou is not None:
        gates["visual_threshold_iou_at_least_threshold"] = (
            visual_metrics.iou >= visual_threshold_min_iou
        )
    if reference_raw_sensor_iou is not None:
        gates["raw_sensor_reduction_at_least_minimum"] = (
            reference_raw_sensor_iou - raw_metrics.iou
            >= minimum_iou_reduction
        )
    if reference_visual_threshold_iou is not None:
        gates["visual_threshold_reduction_at_least_minimum"] = (
            reference_visual_threshold_iou - visual_metrics.iou
            >= minimum_iou_reduction
        )
    passed = all(gates.values()) if gates else None
    summary = {
        "result_schema_version": 1,
        "screen": "raw-sensor-environment-leak",
        "config": str(source),
        "seed_split": asdict(split),
        "projection_size": projection_size,
        "available_modalities": modality_options,
        "raw_sensor_probe": asdict(raw_metrics),
        "visual_threshold_baseline": asdict(visual_metrics),
        "fixed_position_baseline": asdict(
            evaluate_fixed_position_baseline(train, test)
        ),
        "gates": gates,
        "passed": passed,
        "decision": (
            "run_model_screen"
            if passed is True
            else (
                "stop_before_model_training"
                if passed is False
                else "not_evaluated"
            )
        ),
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure raw single-frame body-label leakage"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-sensor-max-iou", type=float)
    parser.add_argument("--visual-threshold-min-iou", type=float)
    parser.add_argument("--visual-threshold-max-iou", type=float)
    parser.add_argument("--reference-raw-sensor-iou", type=float)
    parser.add_argument("--reference-visual-threshold-iou", type=float)
    parser.add_argument("--minimum-iou-reduction", type=float, default=0.0)
    arguments = parser.parse_args()
    summary = run_raw_sensor_screen(
        arguments.config,
        output_path=arguments.output,
        raw_sensor_max_iou=arguments.raw_sensor_max_iou,
        visual_threshold_min_iou=arguments.visual_threshold_min_iou,
        visual_threshold_max_iou=arguments.visual_threshold_max_iou,
        reference_raw_sensor_iou=arguments.reference_raw_sensor_iou,
        reference_visual_threshold_iou=(
            arguments.reference_visual_threshold_iou
        ),
        minimum_iou_reduction=arguments.minimum_iou_reduction,
    )
    print(
        "raw_sensor_iou="
        f"{summary['raw_sensor_probe']['iou']:.4f}; "
        "visual_threshold_iou="
        f"{summary['visual_threshold_baseline']['iou']:.4f}"
    )


if __name__ == "__main__":
    main()
