"""Action-conditioned multimodal next-observation predictor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from cal.learning.dataset import ACTION_VOCABULARY
from cal.model.encoders import EncoderConfig, MultimodalEncoder
from cal.model.recurrent_core import FeedForwardCore, GRUCore


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    """Serializable architecture configuration for a prediction baseline."""

    image_size: tuple[int, int] = (16, 16)
    hidden_size: int = 128
    core_type: str = "gru"
    vision_latent_size: int = 64
    proprioception_latent_size: int = 16
    touch_latent_size: int = 8
    action_latent_size: int = 8
    residual_prediction: bool = True
    copy_logit_strength: float = 3.0
    use_vision: bool = True
    use_proprioception: bool = True
    use_touch: bool = True
    use_action: bool = True
    use_motion: bool = False
    motion_size: int = 8
    motion_passthrough: bool = False
    use_inverse_dynamics: bool = False
    control_state_size: int = 0
    use_control_vision_delta: bool = False
    action_effect_size: int = 0
    ownership_state_size: int = 0
    ownership_recurrent: bool = True
    ownership_vision_support: bool = False
    ownership_copy_logit_strength: float = 3.0
    ownership_effect_attention: bool = False
    ownership_effect_attention_strength: float = 2.0
    ownership_prediction_effect_attention: bool = False
    part_slot_count: int = 0
    part_slot_size: int = 0
    part_slot_recurrent: bool = True
    spatial_ownership_channels: int = 0
    spatial_ownership_action_size: int = 4
    spatial_ownership_recurrent: bool = True
    spatial_ownership_vision_support: bool = True
    spatial_ownership_hard_support: bool = False
    global_ownership_query_size: int = 0
    global_ownership_token_size: int = 16
    global_ownership_action_size: int = 8
    global_ownership_recurrent: bool = True
    global_ownership_hard_support: bool = True
    global_ownership_self_attention: bool = False
    object_slot_count: int = 0
    object_slot_size: int = 16
    object_slot_iterations: int = 2
    object_slot_action_size: int = 4
    object_slot_ownership_size: int = 8
    object_slot_recurrent: bool = True
    object_slot_hard_support: bool = True
    object_slot_exclusive_ownership: bool = False
    causal_effect_channels: int = 0
    causal_effect_action_size: int = 4
    causal_effect_recurrent: bool = True
    causal_effect_hard_support: bool = True

    def __post_init__(self) -> None:
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image dimensions must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.core_type not in {"gru", "feedforward"}:
            raise ValueError("core_type must be 'gru' or 'feedforward'")
        if self.copy_logit_strength <= 0.0:
            raise ValueError("copy_logit_strength must be positive")
        if self.motion_size <= 0:
            raise ValueError("motion_size must be positive")
        if self.control_state_size < 0:
            raise ValueError("control_state_size cannot be negative")
        if self.action_effect_size < 0:
            raise ValueError("action_effect_size cannot be negative")
        if self.ownership_state_size < 0:
            raise ValueError("ownership_state_size cannot be negative")
        if self.ownership_state_size > 0 and self.action_effect_size == 0:
            raise ValueError(
                "ownership state requires a positive action effect size"
            )
        if self.ownership_copy_logit_strength <= 0.0:
            raise ValueError(
                "ownership_copy_logit_strength must be positive"
            )
        if self.ownership_effect_attention_strength <= 0.0:
            raise ValueError(
                "ownership_effect_attention_strength must be positive"
            )
        if self.ownership_effect_attention and self.action_effect_size == 0:
            raise ValueError(
                "ownership effect attention requires an action effect head"
            )
        if (
            self.ownership_effect_attention
            and self.ownership_prediction_effect_attention
        ):
            raise ValueError("ownership effect attention sources are exclusive")
        if (self.part_slot_count == 0) != (self.part_slot_size == 0):
            raise ValueError(
                "part slot count and size must both be zero or positive"
            )
        if self.part_slot_count < 0 or self.part_slot_size < 0:
            raise ValueError("part slot dimensions cannot be negative")
        if self.part_slot_count > 0 and self.action_effect_size == 0:
            raise ValueError(
                "part slots require a positive action effect size"
            )
        if self.spatial_ownership_channels < 0:
            raise ValueError("spatial ownership channels cannot be negative")
        if self.spatial_ownership_action_size <= 0:
            raise ValueError(
                "spatial ownership action size must be positive"
            )
        if self.global_ownership_query_size < 0:
            raise ValueError("global ownership query size cannot be negative")
        if self.global_ownership_token_size <= 0:
            raise ValueError("global ownership token size must be positive")
        if self.global_ownership_action_size <= 0:
            raise ValueError("global ownership action size must be positive")
        if self.object_slot_count < 0:
            raise ValueError("object slot count cannot be negative")
        if self.object_slot_size <= 0:
            raise ValueError("object slot size must be positive")
        if self.object_slot_iterations <= 0:
            raise ValueError("object slot iterations must be positive")
        if self.object_slot_action_size <= 0:
            raise ValueError("object slot action size must be positive")
        if self.object_slot_ownership_size <= 0:
            raise ValueError("object slot ownership size must be positive")
        if self.causal_effect_channels < 0:
            raise ValueError("causal effect channels cannot be negative")
        if self.causal_effect_action_size <= 0:
            raise ValueError("causal effect action size must be positive")
        if self.use_control_vision_delta and self.control_state_size == 0:
            raise ValueError(
                "control vision delta requires a positive control state size"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SensorimotorPrediction:
    """Raw predictions and the representation used to produce them."""

    vision_logits: Tensor
    proprioception: Tensor
    touch_logits: Tensor
    representation: Tensor
    final_state: Tensor
    inverse_action_logits: Tensor | None = None
    control_state: Tensor | None = None
    control_vision_delta: Tensor | None = None
    action_effect_state: Tensor | None = None
    action_effect_logits: Tensor | None = None
    ownership_state: Tensor | None = None
    ownership_logits: Tensor | None = None
    part_slot_state: Tensor | None = None
    part_slot_mask: Tensor | None = None
    part_slot_logits: Tensor | None = None
    spatial_ownership_state: Tensor | None = None
    spatial_ownership_mask: Tensor | None = None
    spatial_ownership_logits: Tensor | None = None
    global_ownership_query: Tensor | None = None
    global_ownership_mask: Tensor | None = None
    global_ownership_logits: Tensor | None = None
    object_slot_state: Tensor | None = None
    object_slot_assignment: Tensor | None = None
    object_slot_ownership_state: Tensor | None = None
    object_slot_mask: Tensor | None = None
    object_slot_logits: Tensor | None = None
    causal_effect_state: Tensor | None = None
    causal_action_effect_logits: Tensor | None = None
    causal_envelope_mask: Tensor | None = None
    causal_envelope_logits: Tensor | None = None


class SensorimotorPredictor(nn.Module):
    """Predict all next sensory modalities from current sensation and action."""

    def __init__(self, config: PredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or PredictorConfig()
        encoder_config = EncoderConfig(
            action_count=len(ACTION_VOCABULARY),
            vision_size=self.config.vision_latent_size,
            proprioception_size=self.config.proprioception_latent_size,
            touch_size=self.config.touch_latent_size,
            action_size=self.config.action_latent_size,
            use_vision=self.config.use_vision,
            use_proprioception=self.config.use_proprioception,
            use_touch=self.config.use_touch,
            use_action=self.config.use_action,
            use_motion=self.config.use_motion,
            motion_size=self.config.motion_size,
        )
        self.encoder = MultimodalEncoder(encoder_config)
        self.representation_size = self.config.hidden_size + (
            10 if self.config.use_motion and self.config.motion_passthrough else 0
        )
        if self.config.core_type == "gru":
            self.core: nn.Module = GRUCore(
                encoder_config.output_size,
                self.config.hidden_size,
            )
        else:
            self.core = FeedForwardCore(
                encoder_config.output_size,
                self.config.hidden_size,
            )

        width, height = self.config.image_size
        self.vision_head = nn.Linear(self.representation_size, width * height)
        self.proprioception_head = nn.Linear(self.representation_size, 4)
        self.touch_head = nn.Linear(self.representation_size, 2)
        self.inverse_action_head = (
            nn.Linear(self.representation_size * 2, len(ACTION_VOCABULARY))
            if self.config.use_inverse_dynamics
            else None
        )
        self.control_projection = (
            nn.Sequential(
                nn.Linear(
                    self.representation_size,
                    self.config.control_state_size,
                ),
                nn.Tanh(),
            )
            if self.config.control_state_size > 0
            else None
        )
        self.control_vision_delta_head = (
            nn.Linear(self.config.control_state_size, width * height)
            if self.config.use_control_vision_delta
            else None
        )
        self.action_effect_projection = (
            nn.Sequential(
                nn.Linear(
                    self.representation_size,
                    self.config.action_effect_size,
                ),
                nn.Tanh(),
            )
            if self.config.action_effect_size > 0
            else None
        )
        self.action_effect_head = (
            nn.Linear(self.config.action_effect_size, width * height)
            if self.config.action_effect_size > 0
            else None
        )
        self.ownership_core = (
            nn.GRU(
                self.config.action_effect_size,
                self.config.ownership_state_size,
                batch_first=True,
            )
            if self.config.ownership_state_size > 0
            else None
        )
        self.ownership_head = (
            nn.Linear(self.config.ownership_state_size, width * height)
            if self.config.ownership_state_size > 0
            else None
        )
        if self.config.part_slot_count > 0:
            self.part_slot_initial = nn.Parameter(
                torch.empty(
                    self.config.part_slot_count,
                    self.config.part_slot_size,
                )
            )
            nn.init.normal_(self.part_slot_initial, std=0.1)
            self.part_slot_cell = nn.GRUCell(
                self.config.action_effect_size,
                self.config.part_slot_size,
            )
            self.part_slot_decoder = nn.Linear(
                self.config.part_slot_size,
                width * height,
            )
        else:
            self.register_parameter("part_slot_initial", None)
            self.part_slot_cell = None
            self.part_slot_decoder = None
        if self.config.spatial_ownership_channels > 0:
            spatial_action_channels = (
                self.config.spatial_ownership_action_size
                if self.config.use_action
                else 0
            )
            spatial_input_channels = 3 + spatial_action_channels
            spatial_hidden = self.config.spatial_ownership_channels
            self.spatial_action_embedding = (
                nn.Embedding(
                    len(ACTION_VOCABULARY),
                    spatial_action_channels,
                )
                if spatial_action_channels > 0
                else None
            )
            self.spatial_ownership_gates = nn.Conv2d(
                spatial_input_channels + spatial_hidden,
                spatial_hidden * 2,
                kernel_size=3,
                padding=1,
            )
            self.spatial_ownership_candidate = nn.Conv2d(
                spatial_input_channels + spatial_hidden,
                spatial_hidden,
                kernel_size=3,
                padding=1,
            )
            self.spatial_ownership_head = nn.Conv2d(
                spatial_hidden,
                1,
                kernel_size=1,
            )
        else:
            self.spatial_action_embedding = None
            self.spatial_ownership_gates = None
            self.spatial_ownership_candidate = None
            self.spatial_ownership_head = None
        if self.config.global_ownership_query_size > 0:
            token_size = self.config.global_ownership_token_size
            query_size = self.config.global_ownership_query_size
            self.global_ownership_token_encoder = nn.Conv2d(
                3,
                token_size,
                kernel_size=3,
                padding=1,
            )
            self.global_ownership_action_embedding = (
                nn.Embedding(
                    len(ACTION_VOCABULARY),
                    self.config.global_ownership_action_size,
                )
                if self.config.use_action
                else None
            )
            global_action_size = (
                self.config.global_ownership_action_size
                if self.config.use_action
                else 0
            )
            self.global_ownership_query_cell = nn.GRUCell(
                token_size + global_action_size,
                query_size,
            )
            self.global_ownership_key = nn.Conv2d(
                token_size,
                query_size,
                kernel_size=1,
            )
            self.global_ownership_bias = nn.Parameter(torch.zeros(1))
            self.global_ownership_token_attention = (
                nn.MultiheadAttention(
                    token_size,
                    num_heads=2,
                    batch_first=True,
                )
                if self.config.global_ownership_self_attention
                else None
            )
        else:
            self.global_ownership_token_encoder = None
            self.global_ownership_action_embedding = None
            self.global_ownership_query_cell = None
            self.global_ownership_key = None
            self.global_ownership_token_attention = None
            self.register_parameter("global_ownership_bias", None)
        if self.config.object_slot_count > 0:
            slot_size = self.config.object_slot_size
            action_size = (
                self.config.object_slot_action_size
                if self.config.use_action
                else 0
            )
            self.object_slot_initial = nn.Parameter(
                torch.empty(self.config.object_slot_count, slot_size)
            )
            nn.init.normal_(self.object_slot_initial, std=0.1)
            self.object_slot_token_encoder = nn.Conv2d(
                5,
                slot_size,
                kernel_size=3,
                padding=1,
            )
            self.object_slot_key = nn.Linear(slot_size, slot_size, bias=False)
            self.object_slot_value = nn.Linear(
                slot_size,
                slot_size,
                bias=False,
            )
            self.object_slot_query = nn.Linear(
                slot_size,
                slot_size,
                bias=False,
            )
            self.object_slot_norm = nn.LayerNorm(slot_size)
            self.object_slot_update = nn.GRUCell(slot_size, slot_size)
            self.object_slot_action_embedding = (
                nn.Embedding(len(ACTION_VOCABULARY), action_size)
                if action_size > 0
                else None
            )
            self.object_slot_ownership_update = nn.GRUCell(
                slot_size + action_size,
                self.config.object_slot_ownership_size,
            )
            self.object_slot_ownership_head = nn.Linear(
                self.config.object_slot_ownership_size,
                1,
            )
        else:
            self.register_parameter("object_slot_initial", None)
            self.object_slot_token_encoder = None
            self.object_slot_key = None
            self.object_slot_value = None
            self.object_slot_query = None
            self.object_slot_norm = None
            self.object_slot_update = None
            self.object_slot_action_embedding = None
            self.object_slot_ownership_update = None
            self.object_slot_ownership_head = None
        if self.config.causal_effect_channels > 0:
            causal_action_channels = (
                self.config.causal_effect_action_size
                if self.config.use_action
                else 0
            )
            causal_input_channels = 3 + causal_action_channels
            causal_hidden = self.config.causal_effect_channels
            self.causal_effect_action_embedding = (
                nn.Embedding(
                    len(ACTION_VOCABULARY),
                    causal_action_channels,
                )
                if causal_action_channels > 0
                else None
            )
            self.causal_effect_gates = nn.Conv2d(
                causal_input_channels + causal_hidden,
                causal_hidden * 2,
                kernel_size=3,
                padding=1,
            )
            self.causal_effect_candidate = nn.Conv2d(
                causal_input_channels + causal_hidden,
                causal_hidden,
                kernel_size=3,
                padding=1,
            )
            self.causal_effect_head = nn.Conv2d(
                causal_hidden,
                len(ACTION_VOCABULARY),
                kernel_size=1,
            )
        else:
            self.causal_effect_action_embedding = None
            self.causal_effect_gates = None
            self.causal_effect_candidate = None
            self.causal_effect_head = None

    def forward(
        self,
        *,
        vision: Tensor,
        proprioception: Tensor,
        touch: Tensor,
        actions: Tensor,
        initial_state: Tensor | None = None,
    ) -> SensorimotorPrediction:
        motion = (
            _motion_features(vision, proprioception, touch)
            if self.config.use_motion
            else None
        )
        encoded = self.encoder(
            vision=vision,
            proprioception=proprioception,
            touch=touch,
            actions=actions,
            motion=motion,
        )
        if self.config.core_type == "feedforward" and initial_state is not None:
            raise ValueError("feed-forward predictor does not accept initial state")
        core_output = self.core(encoded, initial_state)
        representation = core_output.sequence
        if self.config.motion_passthrough:
            if motion is None:
                raise ValueError("motion passthrough requires motion features")
            representation = torch.cat((representation, motion), dim=-1)
        batch, time = representation.shape[:2]
        width, height = self.config.image_size
        vision_residual = self.vision_head(representation).reshape(
            batch,
            time,
            1,
            height,
            width,
        )
        proprioception_prediction = self.proprioception_head(representation)
        touch_logits = self.touch_head(representation)
        if self.config.residual_prediction:
            if self.config.use_vision:
                vision_residual = vision_residual + (
                    vision * 2.0 - 1.0
                ) * self.config.copy_logit_strength
            if self.config.use_proprioception:
                proprioception_prediction = (
                    proprioception + proprioception_prediction
                )
            if self.config.use_touch:
                touch_logits = touch_logits + (
                    touch * 2.0 - 1.0
                ) * self.config.copy_logit_strength
        inverse_action_logits = None
        if self.inverse_action_head is not None:
            inverse_action_logits = self.inverse_action_head(
                torch.cat((representation[:, :-1], representation[:, 1:]), dim=-1)
            )
        control_state = (
            self.control_projection(representation)
            if self.control_projection is not None
            else None
        )
        control_vision_delta = None
        if self.control_vision_delta_head is not None:
            assert control_state is not None
            control_vision_delta = self.control_vision_delta_head(
                control_state
            ).reshape(batch, time, 1, height, width)
        action_effect_state = (
            self.action_effect_projection(representation)
            if self.action_effect_projection is not None
            else None
        )
        action_effect_logits = None
        if self.action_effect_head is not None:
            assert action_effect_state is not None
            action_effect_logits = self.action_effect_head(
                action_effect_state
            ).reshape(batch, time, 1, height, width)
        ownership_state = None
        ownership_logits = None
        if self.ownership_core is not None:
            assert action_effect_state is not None
            if self.config.ownership_recurrent:
                ownership_state, _ = self.ownership_core(
                    action_effect_state
                )
            else:
                independent = action_effect_state.reshape(
                    batch * time,
                    1,
                    self.config.action_effect_size,
                )
                ownership_state, _ = self.ownership_core(independent)
                ownership_state = ownership_state.reshape(
                    batch,
                    time,
                    self.config.ownership_state_size,
                )
            assert self.ownership_head is not None
            ownership_logits = self.ownership_head(
                ownership_state
            ).reshape(batch, time, 1, height, width)
            if self.config.ownership_vision_support:
                visible = (
                    vision.flatten(start_dim=2).sum(dim=-1) > 0.0
                ).reshape(batch, time, 1, 1, 1)
                support = (
                    (vision * 2.0 - 1.0)
                    * self.config.ownership_copy_logit_strength
                )
                ownership_logits = ownership_logits + torch.where(
                    visible,
                    support,
                    torch.zeros_like(support),
                )
            if self.config.ownership_effect_attention:
                assert action_effect_logits is not None
                effect_probability = torch.sigmoid(
                    action_effect_logits
                ).reshape(batch * time, 1, height, width)
                effect_attention = torch.nn.functional.max_pool2d(
                    effect_probability,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ).reshape(batch, time, 1, height, width)
                signed_attention = effect_attention * 2.0 - 1.0
                ownership_logits = ownership_logits + (
                    self.config.ownership_effect_attention_strength
                    * signed_attention
                )
            if self.config.ownership_prediction_effect_attention:
                predicted_next = torch.sigmoid(vision_residual)
                predicted_effect = (predicted_next - vision).abs().reshape(
                    batch * time,
                    1,
                    height,
                    width,
                )
                effect_attention = torch.nn.functional.max_pool2d(
                    predicted_effect,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ).reshape(batch, time, 1, height, width)
                ownership_logits = ownership_logits + (
                    self.config.ownership_effect_attention_strength
                    * (effect_attention * 2.0 - 1.0)
                )
        part_slot_state = None
        part_slot_mask = None
        part_slot_logits = None
        if self.part_slot_cell is not None:
            assert action_effect_state is not None
            assert self.part_slot_initial is not None
            assert self.part_slot_decoder is not None
            slots = self.part_slot_initial.unsqueeze(0).expand(
                batch,
                -1,
                -1,
            )
            slot_history = []
            mask_history = []
            logit_history = []
            for step in range(time):
                if not self.config.part_slot_recurrent:
                    slots = self.part_slot_initial.unsqueeze(0).expand(
                        batch,
                        -1,
                        -1,
                    )
                effect = action_effect_state[:, step].unsqueeze(1).expand(
                    -1,
                    self.config.part_slot_count,
                    -1,
                )
                slots = self.part_slot_cell(
                    effect.reshape(
                        batch * self.config.part_slot_count,
                        self.config.action_effect_size,
                    ),
                    slots.reshape(
                        batch * self.config.part_slot_count,
                        self.config.part_slot_size,
                    ),
                ).reshape(
                    batch,
                    self.config.part_slot_count,
                    self.config.part_slot_size,
                )
                component_logits = self.part_slot_decoder(slots).reshape(
                    batch,
                    self.config.part_slot_count,
                    1,
                    height,
                    width,
                )
                component_probability = torch.sigmoid(component_logits)
                union_probability = 1.0 - torch.prod(
                    1.0 - component_probability,
                    dim=1,
                )
                union_logits = torch.logit(
                    union_probability.clamp(1e-5, 1.0 - 1e-5)
                )
                slot_history.append(slots)
                mask_history.append(union_probability)
                logit_history.append(union_logits)
            part_slot_state = torch.stack(slot_history, dim=1)
            part_slot_mask = torch.stack(mask_history, dim=1)
            part_slot_logits = torch.stack(logit_history, dim=1)
        spatial_ownership_state = None
        spatial_ownership_mask = None
        spatial_ownership_logits = None
        if self.spatial_ownership_gates is not None:
            assert self.spatial_ownership_candidate is not None
            assert self.spatial_ownership_head is not None
            previous_vision = torch.cat(
                (vision[:, :1], vision[:, :-1]),
                dim=1,
            )
            signed_delta = vision - previous_vision
            absolute_delta = signed_delta.abs()
            spatial_inputs = [vision, signed_delta, absolute_delta]
            if self.spatial_action_embedding is not None:
                action_features = self.spatial_action_embedding(actions)
                action_features = action_features.unsqueeze(-1).unsqueeze(-1)
                spatial_inputs.append(
                    action_features.expand(-1, -1, -1, height, width)
                )
            spatial_input = torch.cat(spatial_inputs, dim=2)
            hidden = torch.zeros(
                batch,
                self.config.spatial_ownership_channels,
                height,
                width,
                device=vision.device,
                dtype=vision.dtype,
            )
            state_history = []
            mask_history = []
            logit_history = []
            for step in range(time):
                if not self.config.spatial_ownership_recurrent:
                    hidden = torch.zeros_like(hidden)
                current_input = spatial_input[:, step]
                gates = torch.sigmoid(
                    self.spatial_ownership_gates(
                        torch.cat((current_input, hidden), dim=1)
                    )
                )
                reset, update = gates.chunk(2, dim=1)
                candidate = torch.tanh(
                    self.spatial_ownership_candidate(
                        torch.cat(
                            (current_input, reset * hidden),
                            dim=1,
                        )
                    )
                )
                hidden = (1.0 - update) * hidden + update * candidate
                logits = self.spatial_ownership_head(hidden)
                if self.config.spatial_ownership_vision_support:
                    visible = (
                        vision[:, step].flatten(start_dim=1).sum(dim=1)
                        > 0.0
                    ).reshape(batch, 1, 1, 1)
                    support = (vision[:, step] * 2.0 - 1.0) * 3.0
                    logits = logits + torch.where(
                        visible,
                        support,
                        torch.zeros_like(support),
                    )
                if self.config.spatial_ownership_hard_support:
                    visible = (
                        vision[:, step].flatten(start_dim=1).sum(dim=1)
                        > 0.0
                    ).reshape(batch, 1, 1, 1)
                    constrained = torch.where(
                        vision[:, step] > 0.5,
                        logits,
                        torch.full_like(logits, -10.0),
                    )
                    logits = torch.where(visible, constrained, logits)
                state_history.append(hidden)
                logit_history.append(logits)
                mask_history.append(torch.sigmoid(logits))
            spatial_ownership_state = torch.stack(state_history, dim=1)
            spatial_ownership_logits = torch.stack(logit_history, dim=1)
            spatial_ownership_mask = torch.stack(mask_history, dim=1)
        global_ownership_query = None
        global_ownership_mask = None
        global_ownership_logits = None
        if self.global_ownership_token_encoder is not None:
            assert self.global_ownership_query_cell is not None
            assert self.global_ownership_key is not None
            assert self.global_ownership_bias is not None
            previous_vision = torch.cat(
                (vision[:, :1], vision[:, :-1]),
                dim=1,
            )
            signed_delta = vision - previous_vision
            absolute_delta = signed_delta.abs()
            query = torch.zeros(
                batch,
                self.config.global_ownership_query_size,
                device=vision.device,
                dtype=vision.dtype,
            )
            query_history = []
            logit_history = []
            mask_history = []
            scale = self.config.global_ownership_query_size ** -0.5
            for step in range(time):
                token_input = torch.cat(
                    (
                        vision[:, step],
                        signed_delta[:, step],
                        absolute_delta[:, step],
                    ),
                    dim=1,
                )
                tokens = torch.tanh(
                    self.global_ownership_token_encoder(token_input)
                )
                if self.global_ownership_token_attention is not None:
                    token_sequence = tokens.flatten(start_dim=2).transpose(
                        1,
                        2,
                    )
                    attended, _ = self.global_ownership_token_attention(
                        token_sequence,
                        token_sequence,
                        token_sequence,
                        need_weights=False,
                    )
                    tokens = (
                        token_sequence + attended
                    ).transpose(1, 2).reshape(
                        batch,
                        self.config.global_ownership_token_size,
                        height,
                        width,
                    )
                pooled = tokens.mean(dim=(2, 3))
                query_input = [pooled]
                if self.global_ownership_action_embedding is not None:
                    query_input.append(
                        self.global_ownership_action_embedding(
                            actions[:, step]
                        )
                    )
                if not self.config.global_ownership_recurrent:
                    query = torch.zeros_like(query)
                query = self.global_ownership_query_cell(
                    torch.cat(query_input, dim=-1),
                    query,
                )
                keys = self.global_ownership_key(tokens)
                logits = (
                    keys
                    * query.unsqueeze(-1).unsqueeze(-1)
                ).sum(dim=1, keepdim=True) * scale
                logits = logits + self.global_ownership_bias
                if self.config.global_ownership_hard_support:
                    visible = (
                        vision[:, step].flatten(start_dim=1).sum(dim=1)
                        > 0.0
                    ).reshape(batch, 1, 1, 1)
                    constrained = torch.where(
                        vision[:, step] > 0.5,
                        logits,
                        torch.full_like(logits, -10.0),
                    )
                    logits = torch.where(visible, constrained, logits)
                query_history.append(query)
                logit_history.append(logits)
                mask_history.append(torch.sigmoid(logits))
            global_ownership_query = torch.stack(query_history, dim=1)
            global_ownership_logits = torch.stack(logit_history, dim=1)
            global_ownership_mask = torch.stack(mask_history, dim=1)
        object_slot_state = None
        object_slot_assignment = None
        object_slot_ownership_state = None
        object_slot_mask = None
        object_slot_logits = None
        if self.object_slot_token_encoder is not None:
            assert self.object_slot_initial is not None
            assert self.object_slot_key is not None
            assert self.object_slot_value is not None
            assert self.object_slot_query is not None
            assert self.object_slot_norm is not None
            assert self.object_slot_update is not None
            assert self.object_slot_ownership_update is not None
            assert self.object_slot_ownership_head is not None
            previous_vision = torch.cat(
                (vision[:, :1], vision[:, :-1]),
                dim=1,
            )
            signed_delta = vision - previous_vision
            absolute_delta = signed_delta.abs()
            x_coordinate = torch.linspace(
                -1.0,
                1.0,
                width,
                device=vision.device,
                dtype=vision.dtype,
            ).reshape(1, 1, 1, width).expand(batch, 1, height, width)
            y_coordinate = torch.linspace(
                -1.0,
                1.0,
                height,
                device=vision.device,
                dtype=vision.dtype,
            ).reshape(1, 1, height, 1).expand(batch, 1, height, width)
            slots = self.object_slot_initial.unsqueeze(0).expand(
                batch,
                -1,
                -1,
            )
            ownership = torch.zeros(
                batch,
                self.config.object_slot_count,
                self.config.object_slot_ownership_size,
                device=vision.device,
                dtype=vision.dtype,
            )
            state_history = []
            assignment_history = []
            ownership_history = []
            mask_history = []
            logit_history = []
            scale = self.config.object_slot_size ** -0.5
            for step in range(time):
                if not self.config.object_slot_recurrent:
                    slots = self.object_slot_initial.unsqueeze(0).expand(
                        batch,
                        -1,
                        -1,
                    )
                    ownership = torch.zeros_like(ownership)
                token_input = torch.cat(
                    (
                        vision[:, step],
                        signed_delta[:, step],
                        absolute_delta[:, step],
                        x_coordinate,
                        y_coordinate,
                    ),
                    dim=1,
                )
                tokens = torch.tanh(
                    self.object_slot_token_encoder(token_input)
                ).flatten(start_dim=2).transpose(1, 2)
                keys = self.object_slot_key(tokens)
                values = self.object_slot_value(tokens)
                occupancy = vision[:, step].flatten(start_dim=2)
                for _ in range(self.config.object_slot_iterations):
                    queries = self.object_slot_query(
                        self.object_slot_norm(slots)
                    )
                    attention_logits = torch.einsum(
                        "bsd,bnd->bsn",
                        queries,
                        keys,
                    ) * scale
                    competition = torch.softmax(attention_logits, dim=1)
                    update_weights = competition * occupancy
                    update_weights = update_weights / (
                        update_weights.sum(dim=-1, keepdim=True) + 1e-6
                    )
                    updates = torch.einsum(
                        "bsn,bnd->bsd",
                        update_weights,
                        values,
                    )
                    slots = self.object_slot_update(
                        updates.reshape(
                            batch * self.config.object_slot_count,
                            self.config.object_slot_size,
                        ),
                        slots.reshape(
                            batch * self.config.object_slot_count,
                            self.config.object_slot_size,
                        ),
                    ).reshape(
                        batch,
                        self.config.object_slot_count,
                        self.config.object_slot_size,
                    )
                queries = self.object_slot_query(
                    self.object_slot_norm(slots)
                )
                assignment = torch.softmax(
                    torch.einsum("bsd,bnd->bsn", queries, keys) * scale,
                    dim=1,
                )
                ownership_inputs = [slots]
                if self.object_slot_action_embedding is not None:
                    action_features = self.object_slot_action_embedding(
                        actions[:, step]
                    ).unsqueeze(1).expand(
                        -1,
                        self.config.object_slot_count,
                        -1,
                    )
                    ownership_inputs.append(action_features)
                ownership = self.object_slot_ownership_update(
                    torch.cat(ownership_inputs, dim=-1).reshape(
                        batch * self.config.object_slot_count,
                        -1,
                    ),
                    ownership.reshape(
                        batch * self.config.object_slot_count,
                        self.config.object_slot_ownership_size,
                    ),
                ).reshape(
                    batch,
                    self.config.object_slot_count,
                    self.config.object_slot_ownership_size,
                )
                owner_scores = self.object_slot_ownership_head(ownership)
                owner_probability = (
                    torch.softmax(owner_scores, dim=1)
                    if self.config.object_slot_exclusive_ownership
                    else torch.sigmoid(owner_scores)
                )
                probability = (
                    assignment * owner_probability
                ).sum(dim=1).reshape(batch, 1, height, width)
                logits = torch.logit(
                    probability.clamp(1e-5, 1.0 - 1e-5)
                )
                if self.config.object_slot_hard_support:
                    visible = (
                        vision[:, step].flatten(start_dim=1).sum(dim=1)
                        > 0.0
                    ).reshape(batch, 1, 1, 1)
                    constrained = torch.where(
                        vision[:, step] > 0.5,
                        logits,
                        torch.full_like(logits, -10.0),
                    )
                    logits = torch.where(visible, constrained, logits)
                    probability = torch.sigmoid(logits)
                state_history.append(slots)
                assignment_history.append(
                    assignment.reshape(
                        batch,
                        self.config.object_slot_count,
                        height,
                        width,
                    )
                )
                ownership_history.append(ownership)
                mask_history.append(probability)
                logit_history.append(logits)
            object_slot_state = torch.stack(state_history, dim=1)
            object_slot_assignment = torch.stack(
                assignment_history,
                dim=1,
            )
            object_slot_ownership_state = torch.stack(
                ownership_history,
                dim=1,
            )
            object_slot_mask = torch.stack(mask_history, dim=1)
            object_slot_logits = torch.stack(logit_history, dim=1)
        causal_effect_state = None
        causal_action_effect_logits = None
        causal_envelope_mask = None
        causal_envelope_logits = None
        if self.causal_effect_gates is not None:
            assert self.causal_effect_candidate is not None
            assert self.causal_effect_head is not None
            previous_vision = torch.cat(
                (vision[:, :1], vision[:, :-1]),
                dim=1,
            )
            signed_delta = vision - previous_vision
            absolute_delta = signed_delta.abs()
            causal_inputs = [vision, signed_delta, absolute_delta]
            if self.causal_effect_action_embedding is not None:
                action_features = self.causal_effect_action_embedding(actions)
                action_features = action_features.unsqueeze(-1).unsqueeze(-1)
                causal_inputs.append(
                    action_features.expand(-1, -1, -1, height, width)
                )
            causal_input = torch.cat(causal_inputs, dim=2)
            hidden = torch.zeros(
                batch,
                self.config.causal_effect_channels,
                height,
                width,
                device=vision.device,
                dtype=vision.dtype,
            )
            state_history = []
            action_effect_history = []
            mask_history = []
            logit_history = []
            for step in range(time):
                if not self.config.causal_effect_recurrent:
                    hidden = torch.zeros_like(hidden)
                current_input = causal_input[:, step]
                gates = torch.sigmoid(
                    self.causal_effect_gates(
                        torch.cat((current_input, hidden), dim=1)
                    )
                )
                reset, update = gates.chunk(2, dim=1)
                candidate = torch.tanh(
                    self.causal_effect_candidate(
                        torch.cat(
                            (current_input, reset * hidden),
                            dim=1,
                        )
                    )
                )
                hidden = (1.0 - update) * hidden + update * candidate
                action_effect_logits = self.causal_effect_head(
                    hidden
                ).unsqueeze(2)
                envelope_probability = torch.sigmoid(
                    action_effect_logits
                ).amax(dim=1)
                envelope_probability = torch.nn.functional.max_pool2d(
                    envelope_probability,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                envelope_logits = torch.logit(
                    envelope_probability.clamp(1e-5, 1.0 - 1e-5)
                )
                if self.config.causal_effect_hard_support:
                    visible = (
                        vision[:, step].flatten(start_dim=1).sum(dim=1)
                        > 0.0
                    ).reshape(batch, 1, 1, 1)
                    constrained = torch.where(
                        vision[:, step] > 0.5,
                        envelope_logits,
                        torch.full_like(envelope_logits, -10.0),
                    )
                    envelope_logits = torch.where(
                        visible,
                        constrained,
                        envelope_logits,
                    )
                    envelope_probability = torch.sigmoid(envelope_logits)
                state_history.append(hidden)
                action_effect_history.append(action_effect_logits)
                mask_history.append(envelope_probability)
                logit_history.append(envelope_logits)
            causal_effect_state = torch.stack(state_history, dim=1)
            causal_action_effect_logits = torch.stack(
                action_effect_history,
                dim=1,
            )
            causal_envelope_mask = torch.stack(mask_history, dim=1)
            causal_envelope_logits = torch.stack(logit_history, dim=1)
        return SensorimotorPrediction(
            vision_logits=vision_residual,
            proprioception=proprioception_prediction,
            touch_logits=touch_logits,
            representation=representation,
            final_state=core_output.final_state,
            inverse_action_logits=inverse_action_logits,
            control_state=control_state,
            control_vision_delta=control_vision_delta,
            action_effect_state=action_effect_state,
            action_effect_logits=action_effect_logits,
            ownership_state=ownership_state,
            ownership_logits=ownership_logits,
            part_slot_state=part_slot_state,
            part_slot_mask=part_slot_mask,
            part_slot_logits=part_slot_logits,
            spatial_ownership_state=spatial_ownership_state,
            spatial_ownership_mask=spatial_ownership_mask,
            spatial_ownership_logits=spatial_ownership_logits,
            global_ownership_query=global_ownership_query,
            global_ownership_mask=global_ownership_mask,
            global_ownership_logits=global_ownership_logits,
            object_slot_state=object_slot_state,
            object_slot_assignment=object_slot_assignment,
            object_slot_ownership_state=object_slot_ownership_state,
            object_slot_mask=object_slot_mask,
            object_slot_logits=object_slot_logits,
            causal_effect_state=causal_effect_state,
            causal_action_effect_logits=causal_action_effect_logits,
            causal_envelope_mask=causal_envelope_mask,
            causal_envelope_logits=causal_envelope_logits,
        )


def _motion_features(
    vision: Tensor,
    proprioception: Tensor,
    touch: Tensor,
) -> Tensor:
    """Build compact unlabeled change features from adjacent observations."""

    previous_vision = torch.cat((vision[:, :1] * 0.0, vision[:, :-1]), dim=1)
    previous_proprioception = torch.cat(
        (proprioception[:, :1] * 0.0, proprioception[:, :-1]),
        dim=1,
    )
    previous_touch = torch.cat((touch[:, :1] * 0.0, touch[:, :-1]), dim=1)
    delta_vision = vision - previous_vision
    absolute_vision = delta_vision.abs()
    vision_statistics = torch.stack(
        (
            absolute_vision.flatten(start_dim=2).mean(dim=-1),
            delta_vision.flatten(start_dim=2).mean(dim=-1),
            absolute_vision.flatten(start_dim=2).amax(dim=-1),
            (absolute_vision > 0.05).float().flatten(start_dim=2).mean(dim=-1),
        ),
        dim=-1,
    )
    return torch.cat(
        (
            vision_statistics,
            proprioception - previous_proprioception,
            touch - previous_touch,
        ),
        dim=-1,
    )
