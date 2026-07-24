"""Tests for action-swap future prediction evaluation."""

from calmodel.env.body import BodyConfig
from calmodel.env.world import WorldConfig
from calmodel.evaluation.counterfactual import evaluate_counterfactual
from calmodel.learning.dataset import TrajectorySequenceDataset, collect_trajectories
from calmodel.model.predictors import PredictorConfig, SensorimotorPredictor


def test_counterfactual_evaluation_returns_finite_metrics() -> None:
    trajectories = collect_trajectories(
        WorldConfig(object_count=1),
        BodyConfig(),
        (19,),
        steps_per_seed=12,
    )
    dataset = TrajectorySequenceDataset(
        trajectories,
        sequence_length=4,
        stride=4,
    )
    metrics = evaluate_counterfactual(
        SensorimotorPredictor(PredictorConfig(hidden_size=16)),
        dataset,
    )

    assert metrics.samples == len(dataset)
    assert metrics.correct_action_loss >= 0.0
    assert 0.0 <= metrics.correct_preference_rate <= 1.0
