"""Tests for training, baselines, and checkpoint reproducibility."""

from math import isfinite

import torch

from calmodel.env.body import BodyConfig
from calmodel.env.world import WorldConfig
from calmodel.learning.dataset import (
    CounterfactualTrajectorySequenceDataset,
    PairedTrajectorySequenceDataset,
    TrajectorySequenceDataset,
    collect_trajectories,
    collect_paired_trajectories,
)
from calmodel.learning.trainer import (
    LossConfig,
    TrainingConfig,
    control_vision_delta_loss,
    evaluate_copy_baseline,
    evaluate_mean_baseline,
    fit_mean_baseline,
    fit_model,
    load_checkpoint,
    make_data_loader,
)
from calmodel.model.predictors import PredictorConfig, SensorimotorPredictor


def _dataset(seed: int) -> TrajectorySequenceDataset:
    trajectories = collect_trajectories(
        WorldConfig(image_size=(16, 16)),
        BodyConfig(),
        (seed,),
        steps_per_seed=20,
    )
    return TrajectorySequenceDataset(
        trajectories,
        sequence_length=5,
        stride=5,
    )


def test_baselines_produce_finite_metrics() -> None:
    train_loader = make_data_loader(
        _dataset(1), batch_size=2, shuffle=False, seed=0
    )
    validation_loader = make_data_loader(
        _dataset(100), batch_size=2, shuffle=False, seed=0
    )

    copy_metrics = evaluate_copy_baseline(validation_loader)
    mean = fit_mean_baseline(train_loader)
    mean_metrics = evaluate_mean_baseline(mean, validation_loader)

    assert isfinite(copy_metrics.total)
    assert isfinite(mean_metrics.total)
    assert copy_metrics.samples == len(validation_loader.dataset)


def test_tiny_training_and_checkpoint_round_trip(tmp_path: object) -> None:
    train_loader = make_data_loader(
        _dataset(2), batch_size=2, shuffle=True, seed=3
    )
    validation_loader = make_data_loader(
        _dataset(101), batch_size=2, shuffle=False, seed=3
    )
    config = PredictorConfig(hidden_size=24)
    torch.manual_seed(3)
    model = SensorimotorPredictor(config)
    checkpoint = tmp_path / "checkpoint.pt"  # type: ignore[operator]

    result = fit_model(
        model,
        train_loader,
        validation_loader,
        training_config=TrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            seed=3,
        ),
        checkpoint_path=checkpoint,
    )
    restored = load_checkpoint(checkpoint)

    assert len(result.history) == 2
    assert result.best_epoch in {1, 2}
    assert isfinite(result.best_validation_loss)
    assert restored.config == config


def test_final_checkpoint_selection_uses_last_epoch(tmp_path: object) -> None:
    loader = make_data_loader(
        _dataset(6), batch_size=2, shuffle=False, seed=0
    )
    model = SensorimotorPredictor(PredictorConfig(hidden_size=12))

    result = fit_model(
        model,
        loader,
        loader,
        training_config=TrainingConfig(
            epochs=2,
            batch_size=2,
            checkpoint_selection="final",
        ),
        checkpoint_path=tmp_path / "final.pt",  # type: ignore[operator]
    )

    assert result.best_epoch == 2


def test_action_swap_and_inverse_losses_train_without_labels(tmp_path: object) -> None:
    train_loader = make_data_loader(
        _dataset(12), batch_size=2, shuffle=True, seed=5
    )
    validation_loader = make_data_loader(
        _dataset(112), batch_size=2, shuffle=False, seed=5
    )
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=16,
            use_motion=True,
            use_inverse_dynamics=True,
        )
    )

    result = fit_model(
        model,
        train_loader,
        validation_loader,
        training_config=TrainingConfig(epochs=1, batch_size=2, seed=5),
        loss_config=LossConfig(
            action_swap_weight=0.1,
            inverse_dynamics_weight=0.1,
        ),
        checkpoint_path=tmp_path / "m1b-checkpoint.pt",  # type: ignore[operator]
    )

    assert isfinite(result.best_validation_loss)


def test_control_vision_delta_loss_is_finite() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=16,
            control_state_size=8,
            use_control_vision_delta=True,
        )
    )
    batch = next(
        iter(
            make_data_loader(
                _dataset(18),
                batch_size=2,
                shuffle=False,
                seed=0,
            )
        )
    )
    prediction = model(
        vision=batch["vision"],
        proprioception=batch["proprioception"],
        touch=batch["touch"],
        actions=batch["actions"],
    )

    loss = control_vision_delta_loss(
        prediction,
        batch,
        change_weight=9.0,
    )

    assert isfinite(float(loss.detach()))


def test_paired_control_consistency_trains_without_labels(
    tmp_path: object,
) -> None:
    pairs = collect_paired_trajectories(
        WorldConfig(),
        BodyConfig(),
        (20, 21),
        steps_per_seed=10,
        action_repeat_probability=0.75,
    )
    dataset = PairedTrajectorySequenceDataset(
        pairs,
        sequence_length=5,
        stride=5,
    )
    loader = make_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        seed=0,
    )
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=16,
            control_state_size=8,
            use_control_vision_delta=True,
        )
    )

    result = fit_model(
        model,
        loader,
        loader,
        training_config=TrainingConfig(epochs=1, batch_size=2),
        loss_config=LossConfig(
            control_vision_delta_weight=1.0,
            paired_control_consistency_weight=1.0,
        ),
        checkpoint_path=tmp_path / "paired.pt",  # type: ignore[operator]
    )

    assert isfinite(result.best_validation_loss)


def test_counterfactual_action_effect_trains_without_body_labels(
    tmp_path: object,
) -> None:
    trajectories = collect_trajectories(
        WorldConfig(
            object_count=0,
            distractor_body_count=1,
            distractor_body_motion_probability=1.0,
        ),
        BodyConfig(),
        (25,),
        steps_per_seed=10,
    )
    dataset = CounterfactualTrajectorySequenceDataset(
        trajectories,
        sequence_length=5,
        stride=5,
    )
    loader = make_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        seed=0,
    )
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=16,
            action_effect_size=4,
            ownership_state_size=6,
        )
    )

    result = fit_model(
        model,
        loader,
        loader,
        training_config=TrainingConfig(epochs=1, batch_size=2),
        loss_config=LossConfig(
            action_effect_weight=1.0,
            ownership_weight=1.0,
            ownership_dice_weight=1.0,
        ),
        checkpoint_path=tmp_path / "action-effect.pt",  # type: ignore[operator]
    )

    assert isfinite(result.best_validation_loss)


def test_part_slot_mask_trains_on_counterfactual_occupancy(
    tmp_path: object,
) -> None:
    trajectories = collect_trajectories(
        WorldConfig(
            object_count=0,
            distractor_body_count=1,
            distractor_body_motion_probability=1.0,
        ),
        BodyConfig(),
        (26,),
        steps_per_seed=10,
    )
    dataset = CounterfactualTrajectorySequenceDataset(
        trajectories,
        sequence_length=5,
        stride=5,
    )
    loader = make_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        seed=0,
    )
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=16,
            action_effect_size=4,
            part_slot_count=3,
            part_slot_size=5,
        )
    )

    result = fit_model(
        model,
        loader,
        loader,
        training_config=TrainingConfig(epochs=1, batch_size=2),
        loss_config=LossConfig(
            action_effect_weight=1.0,
            part_slot_weight=1.0,
            part_slot_target="action_envelope",
        ),
        checkpoint_path=tmp_path / "part-slots.pt",  # type: ignore[operator]
    )

    assert isfinite(result.best_validation_loss)


def test_spatial_ownership_trains_on_action_envelope(
    tmp_path: object,
) -> None:
    trajectories = collect_trajectories(
        WorldConfig(
            object_count=0,
            distractor_body_count=1,
            distractor_body_motion_probability=1.0,
        ),
        BodyConfig(),
        (28,),
        steps_per_seed=10,
    )
    dataset = CounterfactualTrajectorySequenceDataset(
        trajectories,
        sequence_length=5,
        stride=5,
    )
    loader = make_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        seed=0,
    )
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=16,
            spatial_ownership_channels=3,
        )
    )
    result = fit_model(
        model,
        loader,
        loader,
        training_config=TrainingConfig(epochs=1, batch_size=2),
        loss_config=LossConfig(
            spatial_ownership_weight=1.0,
            spatial_ownership_dice_weight=1.0,
        ),
        checkpoint_path=tmp_path / "spatial-ownership.pt",  # type: ignore[operator]
    )

    assert isfinite(result.best_validation_loss)


def test_global_ownership_query_trains_on_action_envelope(
    tmp_path: object,
) -> None:
    trajectories = collect_trajectories(
        WorldConfig(object_count=0, distractor_body_count=1),
        BodyConfig(),
        (29,),
        steps_per_seed=10,
    )
    dataset = CounterfactualTrajectorySequenceDataset(
        trajectories,
        sequence_length=5,
        stride=5,
    )
    loader = make_data_loader(
        dataset, batch_size=2, shuffle=False, seed=0
    )
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=16,
            global_ownership_query_size=8,
            global_ownership_token_size=8,
        )
    )
    result = fit_model(
        model,
        loader,
        loader,
        training_config=TrainingConfig(epochs=1, batch_size=2),
        loss_config=LossConfig(
            global_ownership_weight=1.0,
            global_ownership_dice_weight=1.0,
        ),
        checkpoint_path=tmp_path / "global-ownership.pt",  # type: ignore[operator]
    )

    assert isfinite(result.best_validation_loss)


def test_competitive_object_slots_train_on_action_envelope(
    tmp_path: object,
) -> None:
    trajectories = collect_trajectories(
        WorldConfig(
            object_count=0,
            distractor_body_count=2,
            distractor_body_motion_probability=1.0,
        ),
        BodyConfig(),
        (29,),
        steps_per_seed=10,
    )
    dataset = CounterfactualTrajectorySequenceDataset(
        trajectories,
        sequence_length=5,
        stride=5,
    )
    loader = make_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        seed=0,
    )
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=16,
            object_slot_count=3,
            object_slot_size=8,
            object_slot_ownership_size=5,
        )
    )

    result = fit_model(
        model,
        loader,
        loader,
        training_config=TrainingConfig(epochs=1, batch_size=2),
        loss_config=LossConfig(
            object_slot_weight=1.0,
            object_slot_dice_weight=1.0,
        ),
        checkpoint_path=tmp_path / "object-slots.pt",  # type: ignore[operator]
    )

    assert isfinite(result.best_validation_loss)


def test_causal_action_basis_trains_on_all_counterfactual_actions(
    tmp_path: object,
) -> None:
    trajectories = collect_trajectories(
        WorldConfig(
            object_count=0,
            distractor_body_count=2,
            distractor_body_motion_probability=1.0,
        ),
        BodyConfig(),
        (30,),
        steps_per_seed=10,
    )
    dataset = CounterfactualTrajectorySequenceDataset(
        trajectories,
        sequence_length=5,
        stride=5,
    )
    loader = make_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        seed=0,
    )
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=16,
            causal_effect_channels=4,
        )
    )

    result = fit_model(
        model,
        loader,
        loader,
        training_config=TrainingConfig(epochs=1, batch_size=2),
        loss_config=LossConfig(
            causal_effect_weight=1.0,
            causal_effect_dice_weight=1.0,
            causal_envelope_weight=1.0,
            causal_envelope_dice_weight=1.0,
        ),
        checkpoint_path=tmp_path / "causal-action-basis.pt",  # type: ignore[operator]
    )

    assert isfinite(result.best_validation_loss)
