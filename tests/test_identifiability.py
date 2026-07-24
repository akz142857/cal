"""Tests for exact visual-action identifiability audits."""

from cal.env.body import BodyAction, BodyConfig, BodyState
from cal.env.sensors import BinaryMask
from cal.env.world import WorldConfig
from cal.evaluation.identifiability import (
    AuditSample,
    analyze_equivalence_classes,
    collect_audit_sequences,
    history_signature,
    permutation_symmetry_audit,
    run_identifiability_audit,
)


def _mask(*rows: str) -> BinaryMask:
    return tuple(tuple(value == "1" for value in row) for row in rows)


def test_history_signature_includes_intervening_actions() -> None:
    frame = ((0.0, 1.0), (1.0, 0.0))
    visions = (frame, frame)

    first = history_signature(
        visions,
        (BodyAction.NOOP, BodyAction.NOOP),
        end=1,
        length=2,
    )
    second = history_signature(
        visions,
        (BodyAction.ELBOW_INCREASE, BodyAction.NOOP),
        end=1,
        length=2,
    )

    assert first != second


def test_equivalence_audit_detects_ambiguous_cross_seed_masks() -> None:
    first = AuditSample(
        signature="same",
        seed=1,
        step=3,
        body_mask=_mask("10", "00"),
        body_state=BodyState(0.0, 0.0),
    )
    second = AuditSample(
        signature="same",
        seed=2,
        step=4,
        body_mask=_mask("01", "00"),
        body_state=BodyState(0.2, 0.0),
    )

    result = analyze_equivalence_classes((first, second))

    assert result["collision_class_count"] == 1
    assert result["cross_seed_ambiguous_class_count"] == 1
    assert result["ambiguity_sample_rate"] == 1.0
    assert result["majority_ceiling_ambiguous"]["iou"] == 0.5
    assert len(result["ambiguous_examples"]) == 1


def test_random_and_active_sequences_are_deterministic() -> None:
    world = WorldConfig(
        image_size=(8, 8),
        object_count=0,
        distractor_body_count=1,
        distractor_body_motion_probability=1.0,
    )
    first = collect_audit_sequences(
        world,
        BodyConfig(),
        (3,),
        steps_per_seed=6,
        policy="active_visual",
    )
    second = collect_audit_sequences(
        world,
        BodyConfig(),
        (3,),
        steps_per_seed=6,
        policy="active_visual",
    )

    assert first == second


def test_tiny_identifiability_audit_writes_both_policies(
    tmp_path: object,
) -> None:
    output = tmp_path / "audit.json"  # type: ignore[operator]
    result = run_identifiability_audit(
        output_path=output,
        seeds=(0, 1),
        steps_per_seed=4,
        history_lengths=(1, 2),
        world_config=WorldConfig(
            image_size=(8, 8),
            object_count=0,
            distractor_body_count=1,
            distractor_body_motion_probability=1.0,
        ),
    )

    assert output.exists()
    assert set(result["policies"]) == {"random", "active_visual"}
    assert result["policies"]["random"]["1"]["sample_count"] == 8
    assert result["policies"]["active_visual"]["2"]["sample_count"] == 6
    assert "permutation_symmetry" in result


def test_permutation_symmetry_proves_cross_seed_ambiguity() -> None:
    result = permutation_symmetry_audit(
        BodyConfig(),
        image_size=(16, 16),
        history_lengths=(1, 4),
        case_count=8,
    )

    assert result["exact_history_pairs"] == {"1": 8, "4": 8}
    assert (
        result["histories"]["4"]["cross_seed_ambiguous_class_count"] > 0
    )
    assert (
        result["histories"]["4"]["majority_ceiling_ambiguous"]["iou"] < 1.0
    )
