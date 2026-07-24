"""Smoke tests for the initial package scaffold."""

import importlib


def test_expected_modules_are_importable() -> None:
    modules = [
        "cal.env.world",
        "cal.env.body",
        "cal.env.sensors",
        "cal.model.encoders",
        "cal.model.recurrent_core",
        "cal.model.predictors",
        "cal.learning.trainer",
        "cal.learning.replay",
        "cal.evaluation.body_probe",
        "cal.evaluation.metrics",
    ]

    for module in modules:
        importlib.import_module(module)


def test_legacy_package_name_is_not_importable() -> None:
    assert importlib.util.find_spec("calmodel") is None
