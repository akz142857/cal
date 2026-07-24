"""Persistent and non-recurrent cores for controlled comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class CoreOutput:
    """Per-step representation and final recurrent state."""

    sequence: Tensor
    final_state: Tensor


class GRUCore(nn.Module):
    """Persistent state baseline used as the first temporal measurement tool."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        layer_count: int = 1,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or layer_count <= 0:
            raise ValueError("GRU dimensions must be positive")
        self.hidden_size = hidden_size
        self.layer_count = layer_count
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layer_count,
            batch_first=True,
        )

    def forward(
        self,
        inputs: Tensor,
        initial_state: Tensor | None = None,
    ) -> CoreOutput:
        if inputs.ndim != 3:
            raise ValueError("core inputs must have shape [batch, time, features]")
        sequence, final_state = self.gru(inputs, initial_state)
        return CoreOutput(sequence=sequence, final_state=final_state)


class FeedForwardCore(nn.Module):
    """A no-memory baseline that processes each time step independently."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("feed-forward dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )

    def forward(
        self,
        inputs: Tensor,
        initial_state: Tensor | None = None,
    ) -> CoreOutput:
        if initial_state is not None:
            raise ValueError("feed-forward core does not accept recurrent state")
        if inputs.ndim != 3:
            raise ValueError("core inputs must have shape [batch, time, features]")
        sequence = self.network(inputs)
        return CoreOutput(
            sequence=sequence,
            final_state=sequence[:, -1].unsqueeze(0),
        )
