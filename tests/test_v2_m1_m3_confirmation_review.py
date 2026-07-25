"""Tests for the audited M1-M3 fresh confirmation report."""

from cal.evaluation.v2_m1_m3_confirmation_review import (
    build_v2_m1_m3_confirmation_review,
)


def test_confirmation_review_passes_and_preserves_amendment(
    tmp_path: object,
) -> None:
    output = tmp_path / "review.json"  # type: ignore[operator]
    report = tmp_path / "review.md"  # type: ignore[operator]

    result = build_v2_m1_m3_confirmation_review(
        output_path=output,
        report_path=report,
    )

    assert result["passed"]
    assert result["decision"] == (
        "confirm_m1_m3_and_authorize_unprivileged_v2_m4_design"
    )
    assert all(result["gates"].values())
    rendered = report.read_text(encoding="utf-8")
    assert "旧留出种子使用列表均为空" in rendered
    assert "行为组合确认" in rendered
    assert "reacquisition_window" in rendered
    assert result["protocol_v4_sha256"] in rendered
