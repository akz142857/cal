"""Tests for deterministic architecture cost proxies."""

from cal.evaluation.efficiency import (
    active_parameter_proxy,
    estimate_macs_per_step,
    persistent_state_io_bytes_per_step,
)
from cal.model.predictors import PredictorConfig, SensorimotorPredictor


def test_gru_cost_and_state_exceed_feedforward() -> None:
    gru = SensorimotorPredictor(
        PredictorConfig(hidden_size=32, core_type="gru")
    )
    feedforward = SensorimotorPredictor(
        PredictorConfig(hidden_size=32, core_type="feedforward")
    )

    assert estimate_macs_per_step(gru) > estimate_macs_per_step(feedforward)
    assert persistent_state_io_bytes_per_step(gru) == (128, 128)
    assert persistent_state_io_bytes_per_step(feedforward) == (0, 0)


def test_active_parameter_proxy_counts_one_embedding_row() -> None:
    model = SensorimotorPredictor(PredictorConfig(hidden_size=16))
    action = model.encoder.action
    assert action is not None
    inactive = (
        action.embedding.weight.numel() - action.embedding.embedding_dim
    )

    assert active_parameter_proxy(model) == (
        sum(parameter.numel() for parameter in model.parameters()) - inactive
    )
