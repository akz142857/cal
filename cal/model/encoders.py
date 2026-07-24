"""Independent encoders for visual, proprioceptive, touch, and action input."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class VisionEncoder(nn.Module):
    """Small convolutional encoder for low-resolution occupancy frames."""

    def __init__(self, output_size: int) -> None:
        super().__init__()
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        self.network = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, output_size),
            nn.LayerNorm(output_size),
            nn.Tanh(),
        )

    def forward(self, vision: Tensor) -> Tensor:
        if vision.ndim != 5 or vision.shape[2] != 1:
            raise ValueError("vision must have shape [batch, time, 1, height, width]")
        batch, time = vision.shape[:2]
        encoded = self.network(vision.reshape(batch * time, *vision.shape[2:]))
        return encoded.reshape(batch, time, -1)


class VectorEncoder(nn.Module):
    """Two-layer encoder for low-dimensional continuous modalities."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        if input_size <= 0 or output_size <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.LayerNorm(output_size),
            nn.Tanh(),
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.input_size:
            raise ValueError(
                f"values must have shape [batch, time, {self.input_size}]"
            )
        return self.network(values)


class ActionEncoder(nn.Module):
    """Learned embedding for discrete body actions."""

    def __init__(self, action_count: int, output_size: int) -> None:
        super().__init__()
        if action_count <= 0 or output_size <= 0:
            raise ValueError("action encoder dimensions must be positive")
        self.action_count = action_count
        self.embedding = nn.Embedding(action_count, output_size)

    def forward(self, actions: Tensor) -> Tensor:
        if actions.ndim != 2:
            raise ValueError("actions must have shape [batch, time]")
        if actions.numel() and (
            int(actions.min()) < 0 or int(actions.max()) >= self.action_count
        ):
            raise ValueError("action index outside configured vocabulary")
        return self.embedding(actions)


class MotionEncoder(VectorEncoder):
    """Encode unlabeled interoceptive and visual change summaries."""

    def __init__(self, output_size: int) -> None:
        # Four visual statistics, four proprioceptive deltas and two touch
        # deltas; this remains a deliberately compact auxiliary pathway.
        super().__init__(10, output_size)


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Dimensions and ablation switches for multimodal encoding."""

    action_count: int
    vision_size: int = 64
    proprioception_size: int = 16
    touch_size: int = 8
    action_size: int = 8
    use_vision: bool = True
    use_proprioception: bool = True
    use_touch: bool = True
    use_action: bool = True
    use_motion: bool = False
    motion_size: int = 8

    def __post_init__(self) -> None:
        if self.action_count <= 0:
            raise ValueError("action_count must be positive")
        if not any(
            (
                self.use_vision,
                self.use_proprioception,
                self.use_touch,
                self.use_action,
                self.use_motion,
            )
        ):
            raise ValueError("at least one modality must be enabled")
        if self.motion_size <= 0:
            raise ValueError("motion_size must be positive")

    @property
    def output_size(self) -> int:
        return sum(
            size
            for enabled, size in (
                (self.use_vision, self.vision_size),
                (self.use_proprioception, self.proprioception_size),
                (self.use_touch, self.touch_size),
                (self.use_action, self.action_size),
                (self.use_motion, self.motion_size),
            )
            if enabled
        )


class MultimodalEncoder(nn.Module):
    """Encode enabled modalities and concatenate them at each time step."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.vision = (
            VisionEncoder(config.vision_size) if config.use_vision else None
        )
        self.proprioception = (
            VectorEncoder(4, config.proprioception_size)
            if config.use_proprioception
            else None
        )
        self.touch = (
            VectorEncoder(2, config.touch_size) if config.use_touch else None
        )
        self.action = (
            ActionEncoder(config.action_count, config.action_size)
            if config.use_action
            else None
        )
        self.motion = (
            MotionEncoder(config.motion_size) if config.use_motion else None
        )

    def forward(
        self,
        *,
        vision: Tensor,
        proprioception: Tensor,
        touch: Tensor,
        actions: Tensor,
        motion: Tensor | None = None,
    ) -> Tensor:
        encoded: list[Tensor] = []
        if self.vision is not None:
            encoded.append(self.vision(vision))
        if self.proprioception is not None:
            encoded.append(self.proprioception(proprioception))
        if self.touch is not None:
            encoded.append(self.touch(touch))
        if self.action is not None:
            encoded.append(self.action(actions))
        if self.motion is not None:
            if motion is None:
                raise ValueError("motion features are required when enabled")
            encoded.append(self.motion(motion))

        batch_time = {(item.shape[0], item.shape[1]) for item in encoded}
        if len(batch_time) != 1:
            raise ValueError("all enabled modalities must share batch and time")
        return torch.cat(encoded, dim=-1)
