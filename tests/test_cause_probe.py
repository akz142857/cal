"""Tests for evaluation-only self/external cause probes."""

import torch

from cal.env.body import BodyConfig
from cal.env.world import WorldConfig
from cal.evaluation.cause_probe import (
    balance_cause_data,
    classification_metrics,
    extract_cause_data,
    train_linear_classifier,
)
from cal.learning.dataset import collect_trajectories
from cal.model.predictors import PredictorConfig, SensorimotorPredictor


def test_extract_cause_data_aligns_balanced_event_labels() -> None:
    trajectories = collect_trajectories(
        WorldConfig(
            object_count=2,
            external_object_motion_probability=0.5,
        ),
        BodyConfig(),
        (10, 11),
        steps_per_seed=32,
    )
    model = SensorimotorPredictor(PredictorConfig(hidden_size=16))

    extracted = extract_cause_data(model, trajectories)
    balanced = balance_cause_data(extracted, seed=3)

    assert len(extracted) == 62
    assert len(balanced) > 0
    assert int(balanced.labels.sum()) * 2 == len(balanced)
    assert balanced.pre_representations.shape == balanced.post_representations.shape


def test_linear_classifier_learns_separable_features() -> None:
    negative = torch.zeros((20, 2))
    positive = torch.ones((20, 2))
    features = torch.cat((negative, positive))
    labels = torch.cat((torch.zeros(20), torch.ones(20)))

    result = train_linear_classifier(
        features,
        labels,
        features,
        labels,
        features,
        labels,
    )

    assert result.test.accuracy > 0.95


def test_classification_metrics_are_balanced() -> None:
    metrics = classification_metrics(
        torch.tensor((-2.0, 2.0, -2.0, 2.0)),
        torch.tensor((0.0, 1.0, 1.0, 0.0)),
    )

    assert metrics.accuracy == 0.5
    assert metrics.balanced_accuracy == 0.5
