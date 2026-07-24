"""Tests for frozen V2-M3 review aggregation."""

from calmodel.evaluation.v2_m3_hypotheses import run_v2_m3_hypothesis_review
from calmodel.evaluation.v2_m3_review import build_v2_m3_review


def test_frozen_m3_review_passes_and_records_scope(tmp_path: object) -> None:
    output = tmp_path / "review.json"  # type: ignore[operator]
    report = tmp_path / "review.md"  # type: ignore[operator]

    result = build_v2_m3_review(output_path=output, report_path=report)

    assert result["passed"]
    assert result["decision"] == "authorize_v2_m4"
    assert all(result["gates"].values())
    assert len(result["source_sha256"]) == 4
    rendered = report.read_text(encoding="utf-8")
    assert "只运行一次" in rendered
    assert "并不声称已经" in rendered


def test_frozen_m3_holdout_cannot_be_rerun() -> None:
    try:
        run_v2_m3_hypothesis_review(
            output_path="results/V2-M3-body-graph-holdout-summary.json",
            split="holdout",
        )
    except RuntimeError as error:
        assert "reruns forbidden" in str(error)
    else:
        raise AssertionError("the frozen M3 holdout must be one-shot")
