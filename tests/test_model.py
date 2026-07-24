"""Tests for multimodal predictor shapes, gradients, and ablations."""

import torch

from cal.model.predictors import PredictorConfig, SensorimotorPredictor


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "vision": torch.zeros(2, 5, 1, 16, 16),
        "proprioception": torch.zeros(2, 5, 4),
        "touch": torch.zeros(2, 5, 2),
        "actions": torch.zeros(2, 5, dtype=torch.long),
    }


def test_gru_predictor_outputs_all_modalities_and_representation() -> None:
    model = SensorimotorPredictor(PredictorConfig(hidden_size=32))

    output = model(**_inputs())

    assert output.vision_logits.shape == (2, 5, 1, 16, 16)
    assert output.proprioception.shape == (2, 5, 4)
    assert output.touch_logits.shape == (2, 5, 2)
    assert output.representation.shape == (2, 5, 32)
    assert output.final_state.shape == (1, 2, 32)


def test_predictor_backpropagates_through_all_heads() -> None:
    model = SensorimotorPredictor(PredictorConfig(hidden_size=32))
    output = model(**_inputs())
    loss = (
        output.vision_logits.mean()
        + output.proprioception.mean()
        + output.touch_logits.mean()
    )

    loss.backward()

    assert model.vision_head.weight.grad is not None
    assert model.proprioception_head.weight.grad is not None
    assert model.touch_head.weight.grad is not None


def test_feedforward_and_modality_ablation_are_runnable() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            core_type="feedforward",
            use_touch=False,
            use_action=False,
        )
    )

    output = model(**_inputs())

    assert output.representation.shape == (2, 5, 24)
    assert output.final_state.shape == (1, 2, 24)


def test_zero_residual_heads_implement_soft_copy_prior() -> None:
    config = PredictorConfig(hidden_size=16, copy_logit_strength=2.5)
    model = SensorimotorPredictor(config)
    for head in (
        model.vision_head,
        model.proprioception_head,
        model.touch_head,
    ):
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    inputs = _inputs()
    inputs["vision"][:, :, :, 3, 4] = 1.0
    inputs["proprioception"][:, :, 0] = 0.7
    inputs["touch"][:, :, 1] = 1.0

    output = model(**inputs)

    assert torch.allclose(
        output.proprioception,
        inputs["proprioception"],
    )
    assert torch.all(output.vision_logits[:, :, :, 3, 4] == 2.5)
    assert torch.all(output.touch_logits[:, :, 1] == 2.5)


def test_motion_and_inverse_dynamics_paths_are_trainable() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            use_motion=True,
            use_inverse_dynamics=True,
        )
    )

    output = model(**_inputs())

    assert output.inverse_action_logits is not None
    assert output.inverse_action_logits.shape == (2, 4, 5)
    loss = output.inverse_action_logits.mean() + output.representation.mean()
    loss.backward()
    assert model.encoder.motion is not None
    assert model.encoder.motion.network[0].weight.grad is not None
    assert model.inverse_action_head is not None
    assert model.inverse_action_head.weight.grad is not None


def test_control_state_predicts_visual_change() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            control_state_size=8,
            use_control_vision_delta=True,
        )
    )

    output = model(**_inputs())
    assert output.control_vision_delta is not None
    loss = output.control_vision_delta.square().mean()
    loss.backward()

    assert output.control_state is not None
    assert output.control_state.shape == (2, 5, 8)
    assert output.control_vision_delta.shape == (2, 5, 1, 16, 16)
    assert model.control_projection is not None
    assert model.control_projection[0].weight.grad is not None


def test_low_rank_action_effect_head_is_trainable() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            action_effect_size=7,
        )
    )

    output = model(**_inputs())
    assert output.action_effect_state is not None
    assert output.action_effect_logits is not None
    assert output.action_effect_state.shape == (2, 5, 7)
    assert output.action_effect_logits.shape == (2, 5, 1, 16, 16)
    output.action_effect_logits.mean().backward()
    assert model.action_effect_projection is not None
    assert model.action_effect_projection[0].weight.grad is not None


def test_ownership_state_can_accumulate_or_reset_with_same_parameters() -> None:
    recurrent = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            action_effect_size=7,
            ownership_state_size=9,
            ownership_recurrent=True,
        )
    )
    reset = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            action_effect_size=7,
            ownership_state_size=9,
            ownership_recurrent=False,
        )
    )
    reset.load_state_dict(recurrent.state_dict())

    recurrent_output = recurrent(**_inputs())
    reset_output = reset(**_inputs())

    assert recurrent_output.ownership_state is not None
    assert recurrent_output.ownership_logits is not None
    assert recurrent_output.ownership_state.shape == (2, 5, 9)
    assert recurrent_output.ownership_logits.shape == (2, 5, 1, 16, 16)
    assert sum(p.numel() for p in recurrent.parameters()) == sum(
        p.numel() for p in reset.parameters()
    )
    assert not torch.equal(
        recurrent_output.ownership_state[:, 1:],
        reset_output.ownership_state[:, 1:],
    )


def test_ownership_visual_support_respects_blackout_state() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            action_effect_size=7,
            ownership_state_size=9,
            ownership_vision_support=True,
        )
    )
    inputs = _inputs()
    inputs["vision"][:, :, :, 3, 4] = 1.0
    visible = model(**inputs)
    inputs["vision"].zero_()
    blackout = model(**inputs)

    assert visible.ownership_logits is not None
    assert blackout.ownership_logits is not None
    assert torch.all(
        visible.ownership_logits[:, :, :, 3, 4]
        > visible.ownership_logits[:, :, :, 0, 0]
    )
    assert torch.isfinite(blackout.ownership_logits).all()


def test_ownership_effect_attention_preserves_pixel_correspondence() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            action_effect_size=7,
            ownership_state_size=9,
            ownership_effect_attention=True,
        )
    )
    output = model(**_inputs())

    assert output.action_effect_logits is not None
    assert output.ownership_logits is not None
    output.ownership_logits.mean().backward()
    assert model.action_effect_head is not None
    assert model.action_effect_head.weight.grad is not None


def test_prediction_effect_attention_uses_main_vision_head() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            action_effect_size=7,
            ownership_state_size=9,
            ownership_prediction_effect_attention=True,
        )
    )
    output = model(**_inputs())
    assert output.ownership_logits is not None
    output.ownership_logits.mean().backward()
    assert model.vision_head.weight.grad is not None


def test_spatial_ownership_convgru_preserves_grid_and_gradients() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            spatial_ownership_channels=4,
        )
    )
    output = model(**_inputs())

    assert output.spatial_ownership_state is not None
    assert output.spatial_ownership_logits is not None
    assert output.spatial_ownership_mask is not None
    assert output.spatial_ownership_state.shape == (2, 5, 4, 16, 16)
    assert output.spatial_ownership_logits.shape == (2, 5, 1, 16, 16)
    output.spatial_ownership_logits.mean().backward()
    assert model.spatial_ownership_gates is not None
    assert model.spatial_ownership_gates.weight.grad is not None


def test_spatial_hard_support_rejects_background_without_forcing_foreground() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            spatial_ownership_channels=4,
            spatial_ownership_vision_support=False,
            spatial_ownership_hard_support=True,
        )
    )
    inputs = _inputs()
    inputs["vision"][:, :, :, 3, 4] = 1.0
    output = model(**inputs)

    assert output.spatial_ownership_logits is not None
    assert torch.all(output.spatial_ownership_logits[:, :, :, 0, 0] == -10.0)
    assert torch.all(output.spatial_ownership_logits[:, :, :, 3, 4] != -10.0)


def test_global_ownership_query_scores_all_tokens_and_backpropagates() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            global_ownership_query_size=12,
            global_ownership_token_size=10,
        )
    )
    output = model(**_inputs())

    assert output.global_ownership_query is not None
    assert output.global_ownership_logits is not None
    assert output.global_ownership_mask is not None
    assert output.global_ownership_query.shape == (2, 5, 12)
    assert output.global_ownership_logits.shape == (2, 5, 1, 16, 16)
    output.global_ownership_logits.mean().backward()
    assert model.global_ownership_token_encoder is not None
    assert model.global_ownership_token_encoder.weight.grad is not None


def test_global_ownership_self_attention_is_trainable() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            global_ownership_query_size=12,
            global_ownership_token_size=16,
            global_ownership_self_attention=True,
        )
    )
    output = model(**_inputs())
    assert output.global_ownership_logits is not None
    output.global_ownership_logits.mean().backward()
    assert model.global_ownership_token_attention is not None
    assert (
        model.global_ownership_token_attention.in_proj_weight.grad
        is not None
    )


def test_shared_part_slots_track_or_reset_with_same_parameters() -> None:
    recurrent = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            action_effect_size=7,
            part_slot_count=3,
            part_slot_size=8,
            part_slot_recurrent=True,
        )
    )
    reset = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            action_effect_size=7,
            part_slot_count=3,
            part_slot_size=8,
            part_slot_recurrent=False,
        )
    )
    reset.load_state_dict(recurrent.state_dict())

    recurrent_output = recurrent(**_inputs())
    reset_output = reset(**_inputs())

    assert recurrent_output.part_slot_state is not None
    assert recurrent_output.part_slot_mask is not None
    assert recurrent_output.part_slot_logits is not None
    assert recurrent_output.part_slot_state.shape == (2, 5, 3, 8)
    assert recurrent_output.part_slot_mask.shape == (2, 5, 1, 16, 16)
    assert sum(p.numel() for p in recurrent.parameters()) == sum(
        p.numel() for p in reset.parameters()
    )
    assert not torch.equal(
        recurrent_output.part_slot_state[:, 1:],
        reset_output.part_slot_state[:, 1:],
    )


def test_competitive_object_slots_partition_pixels_and_backpropagate() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            object_slot_count=3,
            object_slot_size=8,
            object_slot_ownership_size=5,
        )
    )
    inputs = _inputs()
    inputs["vision"][:, :, :, 3, 4] = 1.0

    output = model(**inputs)

    assert output.object_slot_state is not None
    assert output.object_slot_assignment is not None
    assert output.object_slot_ownership_state is not None
    assert output.object_slot_mask is not None
    assert output.object_slot_logits is not None
    assert output.object_slot_state.shape == (2, 5, 3, 8)
    assert output.object_slot_assignment.shape == (2, 5, 3, 16, 16)
    assert output.object_slot_ownership_state.shape == (2, 5, 3, 5)
    assert output.object_slot_mask.shape == (2, 5, 1, 16, 16)
    assert torch.allclose(
        output.object_slot_assignment.sum(dim=2),
        torch.ones(2, 5, 16, 16),
    )
    assert torch.all(output.object_slot_logits[:, :, :, 0, 0] == -10.0)
    output.object_slot_logits.mean().backward()
    assert model.object_slot_token_encoder is not None
    assert model.object_slot_token_encoder.weight.grad is not None


def test_object_slot_recurrence_can_be_deleted_with_same_parameters() -> None:
    recurrent = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            object_slot_count=3,
            object_slot_size=8,
            object_slot_ownership_size=5,
            object_slot_recurrent=True,
        )
    )
    reset = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            object_slot_count=3,
            object_slot_size=8,
            object_slot_ownership_size=5,
            object_slot_recurrent=False,
        )
    )
    reset.load_state_dict(recurrent.state_dict())

    recurrent_output = recurrent(**_inputs())
    reset_output = reset(**_inputs())

    assert recurrent_output.object_slot_ownership_state is not None
    assert reset_output.object_slot_ownership_state is not None
    assert sum(p.numel() for p in recurrent.parameters()) == sum(
        p.numel() for p in reset.parameters()
    )
    assert not torch.equal(
        recurrent_output.object_slot_ownership_state[:, 1:],
        reset_output.object_slot_ownership_state[:, 1:],
    )


def test_exclusive_object_slot_selection_changes_no_parameter_count() -> None:
    independent = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            object_slot_count=3,
            object_slot_size=8,
            object_slot_ownership_size=5,
        )
    )
    exclusive = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            object_slot_count=3,
            object_slot_size=8,
            object_slot_ownership_size=5,
            object_slot_exclusive_ownership=True,
        )
    )
    exclusive.load_state_dict(independent.state_dict())

    output = exclusive(**_inputs())

    assert output.object_slot_mask is not None
    assert sum(p.numel() for p in independent.parameters()) == sum(
        p.numel() for p in exclusive.parameters()
    )
    assert torch.all(output.object_slot_mask >= 0.0)
    assert torch.all(output.object_slot_mask <= 1.0)


def test_causal_action_basis_predicts_every_action_and_envelope() -> None:
    model = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            causal_effect_channels=6,
        )
    )
    inputs = _inputs()
    inputs["vision"][:, :, :, 3, 4] = 1.0

    output = model(**inputs)

    assert output.causal_effect_state is not None
    assert output.causal_action_effect_logits is not None
    assert output.causal_envelope_mask is not None
    assert output.causal_envelope_logits is not None
    assert output.causal_effect_state.shape == (2, 5, 6, 16, 16)
    assert output.causal_action_effect_logits.shape == (
        2,
        5,
        5,
        1,
        16,
        16,
    )
    assert output.causal_envelope_mask.shape == (2, 5, 1, 16, 16)
    assert torch.all(output.causal_envelope_logits[:, :, :, 0, 0] == -10.0)
    output.causal_envelope_logits.mean().backward()
    assert model.causal_effect_gates is not None
    assert model.causal_effect_gates.weight.grad is not None


def test_causal_action_basis_can_reset_with_same_parameters() -> None:
    recurrent = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            causal_effect_channels=6,
            causal_effect_recurrent=True,
        )
    )
    reset = SensorimotorPredictor(
        PredictorConfig(
            hidden_size=24,
            causal_effect_channels=6,
            causal_effect_recurrent=False,
        )
    )
    reset.load_state_dict(recurrent.state_dict())

    recurrent_output = recurrent(**_inputs())
    reset_output = reset(**_inputs())

    assert recurrent_output.causal_effect_state is not None
    assert reset_output.causal_effect_state is not None
    assert sum(p.numel() for p in recurrent.parameters()) == sum(
        p.numel() for p in reset.parameters()
    )
    assert not torch.equal(
        recurrent_output.causal_effect_state[:, 1:],
        reset_output.causal_effect_state[:, 1:],
    )
