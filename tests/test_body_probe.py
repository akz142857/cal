"""Tests for frozen state extraction and linear body probing."""

import torch

from cal.env.body import BodyConfig
from cal.env.world import WorldConfig
from cal.evaluation.body_probe import (
    ProbeConfig,
    ProbeData,
    apply_probe_projection,
    evaluate_fixed_position_baseline,
    evaluate_visual_threshold_baseline,
    evaluate_probe,
    extract_probe_data,
    extract_proprioception_probe_data,
    extract_raw_sensor_probe_data,
    fit_pca_probe_projection,
    train_linear_probe,
    unseen_pose_subset,
    write_probe_visualization,
)
from cal.learning.dataset import collect_trajectories
from cal.model.predictors import PredictorConfig, SensorimotorPredictor


def test_probe_extraction_is_detached_and_matches_trajectory_steps() -> None:
    trajectories = collect_trajectories(
        WorldConfig(),
        BodyConfig(),
        (10, 11),
        steps_per_seed=8,
    )
    model = SensorimotorPredictor(PredictorConfig(hidden_size=12))

    data = extract_probe_data(model, trajectories)

    assert data.representations.shape == (16, 12)
    assert data.body_masks.shape == (16, 1, 16, 16)
    assert data.visions.shape == (16, 1, 16, 16)
    assert data.poses.shape == (16, 2)
    assert not data.representations.requires_grad
    assert data.body_masks.sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())

    proprioception_data = extract_proprioception_probe_data(trajectories)
    assert proprioception_data.representations.shape == (16, 4)
    assert torch.equal(proprioception_data.body_masks, data.body_masks)
    sensor_data = extract_raw_sensor_probe_data(trajectories)
    assert sensor_data.representations.shape == (16, 267)
    assert torch.equal(sensor_data.body_masks, data.body_masks)


def test_raw_sensor_probe_respects_available_modalities() -> None:
    trajectories = collect_trajectories(
        WorldConfig(),
        BodyConfig(),
        (5,),
        steps_per_seed=4,
    )

    sensor_data = extract_raw_sensor_probe_data(
        trajectories,
        include_proprioception=False,
        include_touch=False,
    )

    assert sensor_data.representations.shape == (4, 16 * 16 + 5)


def test_probe_can_select_control_state() -> None:
    trajectories = collect_trajectories(
        WorldConfig(),
        BodyConfig(),
        (10,),
        steps_per_seed=4,
    )
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=12,
            control_state_size=5,
            use_control_vision_delta=True,
        )
    )

    data = extract_probe_data(
        model,
        trajectories,
        representation_source="control_state",
    )

    assert data.representations.shape == (4, 5)


def test_linear_probe_learns_a_linearly_decodable_mask() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]] * 64,
        dtype=torch.float32,
    )
    first_mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    second_mask = 1.0 - first_mask
    masks = torch.stack([first_mask, second_mask] * 64)
    visions = masks.clone()
    poses = features.clone()
    train = ProbeData(features[:96], masks[:96], visions[:96], poses[:96])
    validation = ProbeData(
        features[96:112],
        masks[96:112],
        visions[96:112],
        poses[96:112],
    )
    test = ProbeData(features[112:], masks[112:], visions[112:], poses[112:])

    result = train_linear_probe(
        train,
        validation,
        config=ProbeConfig(
            epochs=30,
            batch_size=16,
            learning_rate=0.05,
            seed=4,
        ),
    )
    metrics = evaluate_probe(result.probe, test)

    assert metrics.iou > 0.99
    assert metrics.f1 > 0.99


def test_pca_probe_projection_uses_only_unlabeled_representations() -> None:
    features = torch.randn(20, 6)
    masks = torch.zeros(20, 1, 2, 2)
    data = ProbeData(
        features,
        masks,
        masks.clone(),
        torch.zeros(20, 2),
    )
    altered_labels = ProbeData(
        features,
        1.0 - masks,
        masks.clone(),
        torch.ones(20, 2),
    )

    projection = fit_pca_probe_projection(data, 3)
    altered_projection = fit_pca_probe_projection(altered_labels, 3)
    projected = apply_probe_projection(data, projection)

    assert projected.representations.shape == (20, 3)
    assert torch.equal(projection.mean, altered_projection.mean)
    assert torch.equal(
        projection.components.abs(),
        altered_projection.components.abs(),
    )


def test_fixed_position_baseline_ignores_representations() -> None:
    features = torch.randn(4, 3)
    masks = torch.zeros(4, 1, 2, 2)
    masks[:, :, 0, 0] = 1.0
    visions = masks.clone()
    poses = torch.zeros(4, 2)
    train = ProbeData(features, masks, visions, poses)
    test = ProbeData(features * 100.0, masks, visions, poses)

    metrics = evaluate_fixed_position_baseline(train, test)

    assert metrics.iou == 1.0


def test_visual_threshold_exposes_label_leakage() -> None:
    features = torch.randn(2, 3)
    masks = torch.zeros(2, 1, 2, 2)
    masks[:, :, 0, 0] = 1.0
    visions = masks.clone()
    visions[:, :, 1, 1] = 0.25
    data = ProbeData(features, masks, visions, torch.zeros(2, 2))

    metrics = evaluate_visual_threshold_baseline(data)

    assert metrics.iou == 1.0


def test_unseen_pose_subset_excludes_reference_poses() -> None:
    features = torch.randn(3, 4)
    masks = torch.zeros(3, 1, 2, 2)
    visions = masks.clone()
    reference = ProbeData(
        features[:2],
        masks[:2],
        visions[:2],
        torch.tensor([[0.0, 0.0], [0.1, 0.0]]),
    )
    candidate = ProbeData(
        features,
        masks,
        visions,
        torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.3, 0.1]]),
    )

    unseen = unseen_pose_subset(reference, candidate)

    assert len(unseen) == 2
    assert torch.equal(
        unseen.poses,
        torch.tensor([[0.2, 0.0], [0.3, 0.1]]),
    )


def test_probe_visualization_writes_svg(tmp_path: object) -> None:
    features = torch.tensor([[1.0, 0.0]])
    masks = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    data = ProbeData(features, masks, masks.clone(), torch.zeros(1, 2))
    result = train_linear_probe(
        data,
        data,
        config=ProbeConfig(epochs=2, batch_size=1, learning_rate=0.1),
    )
    destination = tmp_path / "probe.svg"  # type: ignore[operator]

    write_probe_visualization(result.probe, data, destination)

    assert "<svg" in destination.read_text(encoding="utf-8")
