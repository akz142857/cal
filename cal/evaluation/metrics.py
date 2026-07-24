"""Prediction, body-discovery, sparsity, and adaptation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class SegmentationMetrics:
    """Aggregated binary body-mask metrics."""

    bce: float
    iou: float
    f1: float
    precision: float
    recall: float
    pixel_accuracy: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    sample_count: int


@torch.no_grad()
def segmentation_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
) -> SegmentationMetrics:
    """Measure body segmentation without modifying the representation model."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("segmentation tensors must be [samples, 1, height, width]")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")

    expected = targets >= 0.5
    predicted = torch.sigmoid(logits) >= threshold
    true_positive = int((predicted & expected).sum())
    false_positive = int((predicted & ~expected).sum())
    false_negative = int((~predicted & expected).sum())
    true_negative = int((~predicted & ~expected).sum())

    iou = _safe_ratio(
        true_positive,
        true_positive + false_positive + false_negative,
    )
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    accuracy = _safe_ratio(
        true_positive + true_negative,
        true_positive + true_negative + false_positive + false_negative,
    )
    return SegmentationMetrics(
        bce=float(F.binary_cross_entropy_with_logits(logits, targets)),
        iou=iou,
        f1=f1,
        precision=precision,
        recall=recall,
        pixel_accuracy=accuracy,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        sample_count=targets.shape[0],
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0
