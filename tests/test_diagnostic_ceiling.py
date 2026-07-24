"""Tests for isolated V2 supervised information diagnostics."""

from cal.env.body import BodyConfig
from cal.env.world import WorldConfig
from cal.evaluation.diagnostic_ceiling import (
    DIAGNOSTIC_MODES,
    DiagnosticHistoryDataset,
    SmallMaskDiagnostic,
    collect_diagnostic_sequences,
    diagnostic_mac_per_step,
    diagnostic_parameter_count,
    run_diagnostic_ceiling,
)


def _sequences() -> object:
    return collect_diagnostic_sequences(
        WorldConfig(
            image_size=(8, 8),
            object_count=0,
            distractor_body_count=1,
            distractor_body_motion_probability=1.0,
        ),
        BodyConfig(),
        (0,),
        steps_per_seed=4,
    )


def test_diagnostic_modes_have_expected_fixed_channels() -> None:
    sequences = _sequences()
    expected = {
        "frame": 1,
        "video": 3,
        "video_action": 13,
        "causal_evidence": 15,
    }

    for mode in DIAGNOSTIC_MODES:
        dataset = DiagnosticHistoryDataset(
            sequences,  # type: ignore[arg-type]
            history_length=3,
            mode=mode,
        )
        sample = dataset[0]
        assert dataset.input_channels == expected[mode]
        assert sample["inputs"].shape == (expected[mode], 8, 8)
        assert sample["target"].shape == (1, 8, 8)


def test_small_diagnostics_fit_resource_limits() -> None:
    model = SmallMaskDiagnostic(91)

    assert diagnostic_parameter_count(model) < 100_000
    assert diagnostic_mac_per_step(model, (16, 16)) < 5_000_000


def test_tiny_diagnostic_ceiling_runs_without_saving_models(
    tmp_path: object,
) -> None:
    output = tmp_path / "diagnostics.json"  # type: ignore[operator]
    result = run_diagnostic_ceiling(
        output_path=output,
        train_seeds=(0,),
        validation_seeds=(100,),
        test_seeds=(200,),
        steps_per_seed=4,
        history_length=2,
        epochs=1,
        batch_size=2,
        hidden_channels=4,
        world_config=WorldConfig(
            image_size=(8, 8),
            object_count=0,
            distractor_body_count=1,
            distractor_body_motion_probability=1.0,
        ),
    )

    assert output.exists()
    assert set(result["diagnostics"]) == set(DIAGNOSTIC_MODES)
    assert all(
        item["diagnostic_only"]
        for item in result["diagnostics"].values()
    )
    assert not list(tmp_path.glob("*.pt"))  # type: ignore[union-attr]
