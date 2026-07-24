"""Tests for changed-body evaluation and finite-experience adaptation."""

from calmodel.env.world import WorldConfig
from calmodel.evaluation.adaptation import (
    AdaptationConfig,
    evaluate_variant_adaptation,
    standard_variants,
)
from calmodel.model.predictors import PredictorConfig, SensorimotorPredictor


def test_standard_variants_cover_required_body_and_sensor_changes() -> None:
    variants = standard_variants(WorldConfig())
    names = {variant.name for variant in variants}

    assert {
        "long_upper_arm",
        "long_forearm",
        "larger_action_step",
        "elbow_disabled",
        "touch_dropout_50",
        "proprioception_noise",
    } <= names


def test_tiny_adaptation_curve_is_runnable() -> None:
    base_world = WorldConfig(object_count=1)
    variant = standard_variants(base_world)[0]
    model = SensorimotorPredictor(PredictorConfig(hidden_size=12))
    config = AdaptationConfig(
        train_seeds=(10,),
        test_seeds=(20,),
        original_test_seeds=(30,),
        experience_budgets=(0, 8),
        test_steps_per_seed=8,
        sequence_length=4,
        stride=4,
        optimization_epochs=1,
        batch_size=2,
    )

    result = evaluate_variant_adaptation(
        model,
        base_world,
        variant,
        config=config,
    )

    assert len(result.curve) == 2
    assert result.curve[0].actual_experience_steps == 0
    assert result.curve[1].actual_experience_steps == 8
