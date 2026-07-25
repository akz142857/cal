"""Tests for the fresh M1-M3 integrated confirmation protocol."""

import json
from pathlib import Path

from cal.evaluation.v2_m1_m3_confirmation import (
    _load_protocol,
    _verify_locked_sources,
    run_v2_m1_m3_confirmation,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_V1 = (
    ROOT / "experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL.json"
)
PROTOCOL_V2 = (
    ROOT / "experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL_V2.json"
)
PROTOCOL = (
    ROOT / "experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL_V3.json"
)
DEVELOPMENT_RESULT = (
    ROOT / "results/V2-M1-M3-integrated-v2-development-summary.json"
)
CONFIRMATION_RESULT = (
    ROOT / "results/V2-M1-M3-integrated-v2-confirmation-summary.json"
)


def test_integrated_protocol_and_locked_sources_are_immutable() -> None:
    protocol, digest = _load_protocol(PROTOCOL)

    assert len(digest) == 64
    assert protocol["historical_holdout_policy"][
        "old_holdout_results_are_archival_only"
    ]
    assert protocol["historical_holdout_policy"][
        "runner_must_not_read_old_holdout_artifacts"
    ]
    assert _verify_locked_sources(protocol) == protocol[
        "locked_source_sha256"
    ]


def test_v1_failure_is_preserved_in_v2_amendment_record() -> None:
    v1, v1_digest = _load_protocol(PROTOCOL_V1)
    v2, _ = _load_protocol(PROTOCOL_V2)

    assert v1["protocol_version"] == 1
    assert v2["protocol_version"] == 2
    assert v2["amendment_record"]["prior_protocol_sha256"] == v1_digest
    assert v2["amendment_record"]["confirmation_runs_before_amendment"] == 0
    assert not v2["amendment_record"]["model_or_stage_algorithm_changed"]


def test_v2_bugfix_is_preserved_in_v3_amendment_record() -> None:
    v2, v2_digest = _load_protocol(PROTOCOL_V2)
    v3, _ = _load_protocol(PROTOCOL)

    assert v2["protocol_version"] == 2
    assert v3["protocol_version"] == 3
    assert v3["amendment_record"]["prior_protocol_sha256"] == v2_digest
    assert v3["amendment_record"]["confirmation_runs_before_amendment"] == 1
    assert not v3["amendment_record"]["model_or_stage_algorithm_changed"]
    assert v3["amendment_record"]["gate_computation_changed"]
    assert v3["fixed_gates"] == v2["fixed_gates"]
    assert v3["confirmation"]["result_path"] == v2["confirmation"]["result_path"]


def test_integrated_protocol_uses_only_fresh_seeds() -> None:
    protocol, _ = _load_protocol(PROTOCOL)
    old = set(
        protocol["historical_holdout_policy"]["old_m2_holdout_seeds"]
        + protocol["historical_holdout_policy"]["old_m3_holdout_seeds"]
    )
    fresh = {
        int(seed)
        for split in ("development", "confirmation")
        for stage in ("m1", "m2", "m3")
        for seed in protocol[split][f"{stage}_seeds"]
    }

    assert not old.intersection(fresh)
    assert len(fresh) == 72


def test_integrated_runner_rejects_unfrozen_output_path(
    tmp_path: object,
) -> None:
    try:
        run_v2_m1_m3_confirmation(
            split="development",
            output_path=tmp_path / "result.json",  # type: ignore[operator]
            protocol_path=PROTOCOL,
        )
    except RuntimeError as error:
        assert "output path" in str(error)
    else:
        raise AssertionError("runner must use the frozen result path")


def test_v2_development_passes_without_old_holdout_reads() -> None:
    result = json.loads(DEVELOPMENT_RESULT.read_text(encoding="utf-8"))

    assert result["passed"]
    assert all(result["gates"].values())
    assert result["historical_holdout_artifacts_read"] == []
    assert result["historical_holdout_seeds_used"] == []
    assert result["evaluation_labels_used_for_learning"] is False


def test_fresh_confirmation_passes_exactly_once() -> None:
    result = json.loads(CONFIRMATION_RESULT.read_text(encoding="utf-8"))

    assert result["review_split"] == "confirmation"
    assert result["confirmation_run_count"] == 1
    assert result["passed"]
    assert all(result["gates"].values())
    assert result["historical_holdout_artifacts_read"] == []
    assert result["historical_holdout_seeds_used"] == []

    try:
        run_v2_m1_m3_confirmation(
            split="confirmation",
            output_path=CONFIRMATION_RESULT.relative_to(ROOT),
            protocol_path=PROTOCOL,
        )
    except RuntimeError as error:
        assert "reruns forbidden" in str(error)
    else:
        raise AssertionError("fresh confirmation must be one-shot")
