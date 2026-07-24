"""Smoke tests for the initial package scaffold."""

import importlib


def test_expected_modules_are_importable() -> None:
    modules = [
        "calmodel.env.world",
        "calmodel.env.body",
        "calmodel.env.sensors",
        "calmodel.model.encoders",
        "calmodel.model.recurrent_core",
        "calmodel.model.predictors",
        "calmodel.learning.trainer",
        "calmodel.learning.replay",
        "calmodel.evaluation.body_probe",
        "calmodel.evaluation.metrics",
    ]

    for module in modules:
        importlib.import_module(module)
