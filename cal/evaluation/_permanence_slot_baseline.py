"""SlotSSM-style object-centric baseline for the item-3 forward benchmark.

An Embodied-SlotSSM-style neural predictor: a per-frame spatial slot-attention
encoder binds the sensed patch into K object slots, the slots persist and update
recurrently across the sequence (a per-slot state-space / GRU recurrence
conditioned on the action copy), and a spatial-broadcast decoder renders the
predicted hidden-object occupancy map.

This carries the *object-centric* inductive bias the plain GRU baseline lacks.
It typically exceeds the V2 100k learnable-parameter budget; per V2 research
plan section 4.3 it runs as a **baseline** (never the formal agent) under a
documented budget exception, to answer whether object-centric structure lets a
neural net approach the belief filter's calibrated permanence.

Not a gated experiment: trained fresh on the supplied non-reserved seeds,
deterministic under a fixed torch seed.
"""

from __future__ import annotations

from math import sqrt
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
_DIM = 32
_SLOTS = 4
_ITERS = 2
_EPS = 1e-8


def _local_index(cell: tuple[int, int]) -> int:
    return (cell[1] - ARENA_LOW) * _SIDE + (cell[0] - ARENA_LOW)


def _coordinate_grid() -> torch.Tensor:
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, _SIDE), np.linspace(0.0, 1.0, _SIDE), indexing="ij"
    )
    grid = np.stack((xs.ravel(), ys.ravel()), axis=-1).astype(np.float32)
    return torch.from_numpy(grid)  # (121, 2)


def _frames(sample: "_Sample") -> np.ndarray:
    frames = []
    history = list(zip(sample.sensed_history, sample.action_history))
    for sensed, action in history[-_SEQUENCE:]:
        action_one_hot = np.zeros(_ACTIONS, dtype=np.float32)
        action_one_hot[int(action)] = 1.0
        frames.append(np.concatenate((sensed.ravel(), action_one_hot)))
    array = np.stack(frames).astype(np.float32)
    if array.shape[0] < _SEQUENCE:
        pad = np.zeros((_SEQUENCE - array.shape[0], array.shape[1]), dtype=np.float32)
        array = np.concatenate((pad, array), axis=0)
    return array  # (T, 126)


class _SlotSSM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, _DIM)
        self.register_buffer("coords", _coordinate_grid())
        self.pos = nn.Linear(2, _DIM)
        self.action = nn.Linear(_ACTIONS, _DIM)
        self.slot_init = nn.Parameter(torch.randn(1, _SLOTS, _DIM) * 0.1)
        self.norm_slots = nn.LayerNorm(_DIM)
        self.norm_inputs = nn.LayerNorm(_DIM)
        self.to_q = nn.Linear(_DIM, _DIM, bias=False)
        self.to_k = nn.Linear(_DIM, _DIM, bias=False)
        self.to_v = nn.Linear(_DIM, _DIM, bias=False)
        self.gru = nn.GRUCell(_DIM, _DIM)
        self.decoder = nn.Sequential(
            nn.Linear(_DIM + 2, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def _encode(self, occupancy: torch.Tensor) -> torch.Tensor:
        # occupancy: (B, 121) -> per-cell features (B, 121, DIM)
        features = self.encoder(occupancy.unsqueeze(-1))
        return features + self.pos(self.coords).unsqueeze(0)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch = sequence.shape[0]
        slots = self.slot_init.expand(batch, -1, -1).contiguous()
        for step in range(sequence.shape[1]):
            occupancy = sequence[:, step, :_CELLS]
            action_code = sequence[:, step, _CELLS:]
            inputs = self.norm_inputs(self._encode(occupancy))
            key = self.to_k(inputs)
            value = self.to_v(inputs)
            action_emb = self.action(action_code).unsqueeze(1)
            for _ in range(_ITERS):
                query = self.to_q(self.norm_slots(slots))
                logits = torch.einsum("bnd,bkd->bnk", key, query) / sqrt(_DIM)
                attn = torch.softmax(logits, dim=-1)
                weights = attn / (attn.sum(dim=1, keepdim=True) + _EPS)
                updates = torch.einsum("bnk,bnd->bkd", weights, value)
                gru_in = (updates + action_emb).reshape(batch * _SLOTS, _DIM)
                slots = self.gru(gru_in, slots.reshape(batch * _SLOTS, _DIM))
                slots = slots.reshape(batch, _SLOTS, _DIM)
        return self._decode(slots)

    def _decode(self, slots: torch.Tensor) -> torch.Tensor:
        batch = slots.shape[0]
        broadcast = slots.unsqueeze(2).expand(-1, -1, _CELLS, -1)
        coords = self.coords.unsqueeze(0).unsqueeze(0).expand(batch, _SLOTS, -1, -1)
        decoded = self.decoder(torch.cat((broadcast, coords), dim=-1))
        logit = decoded[..., 0]
        alpha = torch.softmax(decoded[..., 1], dim=1)
        return (alpha * logit).sum(dim=1)  # (B, 121) occupancy logits


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


def slot_predictor_maps(
    train: list["_Sample"],
    evaluation: list["_Sample"],
    *,
    epochs: int = 40,
    seed: int = 0,
) -> np.ndarray:
    if not train:
        raise ValueError("slot training samples must not be empty")
    if not evaluation:
        raise ValueError("slot evaluation samples must not be empty")
    torch.manual_seed(seed)
    train_x = torch.from_numpy(np.stack([_frames(s) for s in train]))
    train_y = torch.from_numpy(np.stack([s.hidden_occupancy.ravel() for s in train]))
    model = _SlotSSM()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = _occupancy_loss(model(train_x), train_y, train)
        loss.backward()
        optimizer.step()
    model.eval()
    eval_x = torch.from_numpy(np.stack([_frames(s) for s in evaluation]))
    with torch.no_grad():
        return torch.sigmoid(model(eval_x)).numpy()


def parameter_count() -> int:
    return sum(p.numel() for p in _SlotSSM().parameters())
