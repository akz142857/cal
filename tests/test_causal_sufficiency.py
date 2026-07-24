"""Tests for V2 causal evidence sufficiency audits."""

import torch

from cal.env.body import BodyConfig
from cal.env.world import WorldConfig
from cal.evaluation.causal_sufficiency import (
    collect_pose_grid_steps,
    effect_envelope,
    geodesic_propagation,
    run_causal_sufficiency_audit,
)


def test_effect_envelope_selects_requested_action_and_current_occupancy() -> None:
    effects = torch.zeros(5, 4, 4)
    effects[1, 1, 1] = 1.0
    effects[2, 3, 3] = 1.0
    vision = torch.zeros(4, 4)
    vision[0:3, 0:3] = 1.0

    selected = effect_envelope(effects, vision, action_indexes=(1,))

    assert selected[1, 1] == 1.0
    assert selected[3, 3] == 0.0
    assert torch.all(selected <= vision)


def test_geodesic_propagation_is_monotonic_and_occupancy_bounded() -> None:
    occupancy = torch.zeros(5, 5)
    occupancy[2, :] = 1.0
    seeds = torch.zeros(5, 5)
    seeds[2, 0] = 1.0

    one = geodesic_propagation(seeds, occupancy, steps=1)
    four = geodesic_propagation(seeds, occupancy, steps=4)

    assert torch.all(one <= four)
    assert torch.all(four <= occupancy)
    assert torch.all(four[2] == 1.0)


def test_tiny_causal_sufficiency_audit_is_diagnostic_only(
    tmp_path: object,
) -> None:
    output = tmp_path / "causal.json"  # type: ignore[operator]
    result = run_causal_sufficiency_audit(
        output_path=output,
        seeds=(400,),
        steps_per_seed=4,
        history_lengths=(1, 2),
        geodesic_depths=(0, 1),
        world_config=WorldConfig(
            image_size=(8, 8),
            object_count=0,
            distractor_body_count=1,
            distractor_body_motion_probability=1.0,
        ),
        body_config=BodyConfig(),
    )

    assert output.exists()
    assert result["diagnostic_only"]
    assert result["learnable_parameter_count"] == 0
    assert set(result["history_union"]) == {"1", "2"}
    assert set(result["geodesic_propagation"]) == {"0", "1"}
    assert result["analytic_pose_grid"]["sample_count"] == 25
    assert (
        result["pose_buckets"]["analytic_pose_grid"][
            "shoulder_near_limit"
        ]["sample_count"]
        > 0
    )


def test_pose_grid_includes_joint_limits() -> None:
    body = BodyConfig()
    steps = collect_pose_grid_steps(
        WorldConfig(
            image_size=(8, 8),
            object_count=0,
            distractor_body_count=1,
        ),
        body,
        (7,),
    )
    states = {item.state for item in steps}

    assert len(steps) == 25
    assert any(
        state.shoulder_angle == body.shoulder_limits[0]
        and state.elbow_angle == body.elbow_limits[0]
        for state in states
    )
    assert any(
        state.shoulder_angle == body.shoulder_limits[1]
        and state.elbow_angle == body.elbow_limits[1]
        for state in states
    )
