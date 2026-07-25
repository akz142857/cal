"""Tests for the V2-I1 system-integration probe (first shared-world attempt)."""

import hashlib
from pathlib import Path

from cal.evaluation.v2_i1_integration import _load_frozen_protocol, run_v2_i1


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/V2_I1_INTEGRATION_PROTOCOL.json"


def test_protocol_is_frozen_and_hash_locked() -> None:
    protocol, digest = _load_frozen_protocol(PROTOCOL)

    assert len(digest) == 64
    assert protocol["status"] == "frozen_before_integration_probe_implementation"
    expected = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert digest == expected
    development = set(protocol["development"]["seeds"])
    holdout = set(protocol["holdout"]["seeds"])
    assert not development & holdout


def test_rejects_output_path_not_in_protocol(tmp_path: object) -> None:
    try:
        run_v2_i1(
            split="development",
            protocol_path=PROTOCOL,
            output_path=tmp_path / "elsewhere.json",  # type: ignore[operator]
        )
    except RuntimeError as error:
        assert "output path" in str(error)
    else:
        raise AssertionError("runner must use the frozen result path")


def test_holdout_blocked_until_development_passes() -> None:
    protocol, _ = _load_frozen_protocol(PROTOCOL)
    development_path = Path(protocol["development"]["result_path"])
    holdout_path = Path(protocol["holdout"]["result_path"])
    assert not holdout_path.exists(), (
        "the one-shot integration holdout must remain unconsumed: "
        "development has not passed"
    )
    if development_path.exists():
        import json

        development = json.loads(development_path.read_text(encoding="utf-8"))
        if development.get("passed") is not True:
            try:
                run_v2_i1(split="holdout", protocol_path=PROTOCOL)
            except RuntimeError as error:
                assert "must pass" in str(error)
            else:
                raise AssertionError(
                    "holdout must not run before development passes"
                )
