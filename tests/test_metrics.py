"""Tests for aggregated body segmentation metrics."""

import torch

from cal.evaluation.metrics import segmentation_metrics


def test_perfect_segmentation_has_unit_scores() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = torch.where(targets > 0.5, 20.0, -20.0)

    metrics = segmentation_metrics(logits, targets)

    assert metrics.iou == 1.0
    assert metrics.f1 == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.pixel_accuracy == 1.0


def test_segmentation_rejects_mismatched_shapes() -> None:
    targets = torch.zeros(1, 1, 2, 2)
    logits = torch.zeros(1, 1, 3, 3)

    try:
        segmentation_metrics(logits, targets)
    except ValueError as error:
        assert "identical" in str(error)
    else:
        raise AssertionError("shape mismatch should fail")
