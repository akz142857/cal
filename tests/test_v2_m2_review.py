"""Tests for the frozen V2-M2 review aggregation."""

from calmodel.evaluation.v2_m2 import run_v2_m2
from calmodel.evaluation.v2_m2_review import build_v2_m2_review


def test_frozen_review_passes_and_records_source_hashes(tmp_path: object) -> None:
    output = tmp_path / "review.json"  # type: ignore[operator]
    report = tmp_path / "review.md"  # type: ignore[operator]
    result = build_v2_m2_review(output_path=output, report_path=report)

    assert result["passed"]
    assert result["decision"] == "authorize_v2_m3"
    assert all(result["gates"].values())
    assert len(result["source_sha256"]) == 5
    assert "只运行一次" in report.read_text(encoding="utf-8")


def test_frozen_holdout_cannot_be_rerun() -> None:
    try:
        run_v2_m2(
            output_path="results/V2-M2-probabilistic-holdout-summary.json",
            protocol_path=(
                "experiments/V2_M2_PROBABILISTIC_ASSOCIATION_PROTOCOL.json"
            ),
            split="holdout",
        )
    except RuntimeError as error:
        assert "forbids reruns" in str(error)
    else:
        raise AssertionError("the frozen holdout must be one-shot")
