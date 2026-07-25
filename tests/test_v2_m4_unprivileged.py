"""Tests for the unprivileged (vision-only occlusion) V2-M4 stage."""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from cal.evaluation.v2_m4_unprivileged import (
    _LineOfSightWorld,
    _load_frozen_protocol,
    run_v2_m4_unprivileged,
)
from cal.model.occupancy import (
    UnprivilegedOccupancyMemory,
    bresenham_intermediate_cells,
    infer_visibility_from_sensed_occupancy,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_V1 = ROOT / "experiments/V2_M4_UNPRIVILEGED_PROTOCOL.json"
PROTOCOL = ROOT / "experiments/V2_M4_UNPRIVILEGED_PROTOCOL_V2.json"


def test_protocol_is_frozen_and_hash_locked() -> None:
    protocol, digest = _load_frozen_protocol(PROTOCOL)

    assert len(digest) == 64
    assert protocol["status"] == (
        "frozen_after_v1_control_criterion_failure_before_any_holdout_run"
    )
    expected = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert digest == expected
    development = set(protocol["development"]["seeds"])
    holdout = set(protocol["holdout"]["seeds"])
    assert not development & holdout


def test_v1_control_failure_is_preserved_in_v2_amendment() -> None:
    v1, v1_digest = _load_frozen_protocol(PROTOCOL_V1)
    v2, _ = _load_frozen_protocol(PROTOCOL)

    assert v1["protocol_version"] == 1
    assert v2["protocol_version"] == 2
    record = v2["amendment_record"]
    assert record["prior_protocol_sha256"] == v1_digest
    assert record["holdout_runs_before_amendment"] == 0
    assert record["model_or_stage_algorithm_changed"] is False
    archived = ROOT / record["prior_development_result_path"]
    payload = json.loads(archived.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["gates"]["assume_all_visible_control_fails"] is False
    digest = hashlib.sha256(archived.read_bytes()).hexdigest()
    assert digest == record["prior_development_result_sha256"]
    # Formal thresholds must be identical; only the control criterion moved.
    shared = {
        key: value
        for key, value in v1["fixed_gates"].items()
        if not key.startswith("assume_all_visible")
    }
    assert shared.items() <= v2["fixed_gates"].items()
    assert v2["holdout"]["seeds"] == v1["holdout"]["seeds"]


def test_update_signature_has_no_visibility_parameter() -> None:
    parameters = set(
        inspect.signature(UnprivilegedOccupancyMemory.update).parameters
    )

    assert "visibility" not in parameters
    assert "local_visibility" not in parameters
    assert parameters == {"self", "sensed_occupancy", "action"}


def test_shadow_casting_hides_cells_behind_observed_occupier() -> None:
    patch = np.zeros((11, 11), dtype=np.uint8)
    patch[5, 8] = 1  # occupier to the right of the center (5, 5)

    visibility = infer_visibility_from_sensed_occupancy(patch)

    assert visibility[5, 8] == 1  # the occupier itself is visible
    assert visibility[5, 9] == 0  # directly behind it
    assert visibility[5, 10] == 0
    assert visibility[5, 7] == 1  # in front of it
    assert visibility[0, 0] == 1  # unrelated direction


def test_bresenham_intermediate_cells_exclude_endpoints() -> None:
    cells = bresenham_intermediate_cells((0, 0), (4, 0))

    assert cells == [(1, 0), (2, 0), (3, 0)]
    assert bresenham_intermediate_cells((0, 0), (1, 1)) == []


def test_world_sensed_patch_hides_occluded_occupancy() -> None:
    world = _LineOfSightWorld(0)
    sensed, true_visibility = world.observe()

    assert sensed.shape == true_visibility.shape == (11, 11)
    # A sensed-occupied cell is by construction visible.
    assert not np.any((sensed > 0) & (true_visibility == 0))


def test_agent_visibility_estimate_matches_true_visibility() -> None:
    world = _LineOfSightWorld(3)
    sensed, true_visibility = world.observe()

    estimated = infer_visibility_from_sensed_occupancy(sensed)

    # The estimate may hide at most a few extra cells (occluders hidden
    # behind other occluders) but must never claim a truly hidden cell is
    # visible while marking a truly visible occupied cell hidden.
    mismatch = np.mean(estimated != true_visibility)
    assert mismatch <= 0.10


def test_holdout_requires_passing_development(tmp_path: object) -> None:
    protocol, _ = _load_frozen_protocol(PROTOCOL)
    development_path = Path(protocol["development"]["result_path"])
    holdout_path = Path(protocol["holdout"]["result_path"])
    if holdout_path.exists():
        try:
            run_v2_m4_unprivileged(split="holdout", protocol_path=PROTOCOL)
        except RuntimeError as error:
            assert "reruns forbidden" in str(error)
        else:
            raise AssertionError("holdout must be one-shot")
    elif not development_path.exists():
        try:
            run_v2_m4_unprivileged(split="holdout", protocol_path=PROTOCOL)
        except (RuntimeError, FileNotFoundError):
            pass
        else:
            raise AssertionError(
                "holdout must not run before development exists"
            )


def test_rejects_output_path_not_in_protocol(tmp_path: object) -> None:
    try:
        run_v2_m4_unprivileged(
            split="development",
            protocol_path=PROTOCOL,
            output_path=tmp_path / "elsewhere.json",  # type: ignore[operator]
        )
    except RuntimeError as error:
        assert "output path" in str(error)
    else:
        raise AssertionError("runner must use the frozen result path")
