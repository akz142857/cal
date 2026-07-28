"""Small non-object-centric GRU baseline for the item-3 forward benchmark.

Given the recent sensed-patch + action-copy history, predict the occupancy of
currently-hidden distractor cells.  This is the "no object structure" neural
control: it must learn permanence from sequence statistics alone, with no
explicit entity/belief representation.  The model is deliberately small
(~4.4e4 parameters, within the V2 100k learnable-parameter budget).

Not a gated experiment: trained fresh on the supplied non-reserved train seeds
each call, deterministic under a fixed torch seed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cal.evaluation.v2_i1_integration import ARENA_HIGH, ARENA_LOW

if TYPE_CHECKING:
    from cal.evaluation.permanence_forward_benchmark import _Sample

_SIDE = ARENA_HIGH - ARENA_LOW + 1  # 11
_CELLS = _SIDE * _SIDE  # 121
_ACTIONS = 5
_SEQUENCE = 16
_HIDDEN = 64


def _local_index(cell: tuple[int, int]) -> int:
    return (cell[1] - ARENA_LOW) * _SIDE + (cell[0] - ARENA_LOW)


def _sequence_tensor(sample: "_Sample") -> np.ndarray:
    frames = []
    history = list(zip(sample.sensed_history, sample.action_history))
    for sensed, action in history[-_SEQUENCE:]:
        action_one_hot = np.zeros(_ACTIONS, dtype=np.float32)
        action_one_hot[int(action)] = 1.0
        frames.append(np.concatenate((sensed.ravel(), action_one_hot)))
    array = np.stack(frames).astype(np.float32)
    if array.shape[0] < _SEQUENCE:
        pad = np.zeros(
            (_SEQUENCE - array.shape[0], array.shape[1]), dtype=np.float32
        )
        array = np.concatenate((pad, array), axis=0)
    return array


class _GRUOccupancy(nn.Module):
    def __init__(self, hidden: int = _HIDDEN) -> None:
        super().__init__()
        self.gru = nn.GRU(_CELLS + _ACTIONS, hidden, batch_first=True)
        self.head = nn.Linear(hidden, _CELLS)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return self.head(hidden[-1])


def _occupancy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    samples: list["_Sample"],
) -> torch.Tensor:
    """Combine full-map calibration with the benchmark's queried-cell task."""

    global_loss = F.binary_cross_entropy_with_logits(logits, targets)
    rows = torch.arange(logits.shape[0], device=logits.device)
    positive = torch.as_tensor(
        [_local_index(sample.positive) for sample in samples],
        dtype=torch.long,
        device=logits.device,
    )
    negative = torch.as_tensor(
        [_local_index(sample.negative) for sample in samples],
        dtype=torch.long,
        device=logits.device,
    )
    pair_logits = torch.stack(
        (logits[rows, positive], logits[rows, negative]), dim=1
    )
    pair_targets = torch.tensor(
        (1.0, 0.0), dtype=logits.dtype, device=logits.device
    ).expand_as(pair_logits)
    query_loss = F.binary_cross_entropy_with_logits(pair_logits, pair_targets)
    return global_loss + query_loss


def gru_predictor_maps(
    train: list["_Sample"],
    evaluation: list["_Sample"],
    *,
    epochs: int = 25,
    hidden: int = _HIDDEN,
    seed: int = 0,
) -> np.ndarray:
    """Train a GRU occupancy predictor and return per-eval-sample 121-cell maps."""

    if not train:
        raise ValueError("GRU training samples must not be empty")
    if not evaluation:
        raise ValueError("GRU evaluation samples must not be empty")
    torch.manual_seed(seed)
    train_x = torch.from_numpy(np.stack([_sequence_tensor(s) for s in train]))
    train_y = torch.from_numpy(
        np.stack([s.hidden_occupancy.ravel() for s in train])
    )
    model = _GRUOccupancy(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = _occupancy_loss(model(train_x), train_y, train)
        loss.backward()
        optimizer.step()
    model.eval()
    eval_x = torch.from_numpy(np.stack([_sequence_tensor(s) for s in evaluation]))
    with torch.no_grad():
        return torch.sigmoid(model(eval_x)).numpy()


def gru_predictor_pairs(
    train: list["_Sample"],
    evaluation: list["_Sample"],
    *,
    epochs: int = 25,
    seed: int = 0,
) -> list[tuple[float, float]]:
    maps = gru_predictor_maps(train, evaluation, epochs=epochs, seed=seed)
    return [
        (
            float(row[_local_index(sample.positive)]),
            float(row[_local_index(sample.negative)]),
        )
        for sample, row in zip(evaluation, maps, strict=True)
    ]


def parameter_count(hidden: int = _HIDDEN) -> int:
    return sum(p.numel() for p in _GRUOccupancy(hidden=hidden).parameters())
