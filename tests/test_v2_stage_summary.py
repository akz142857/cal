"""Tests for the final V2 chain validation."""

from cal.evaluation.v2_stage_summary import build_v2_stage_summary


def test_v2_stage_summary_validates_complete_chain(tmp_path: object) -> None:
    output = tmp_path / "summary.json"  # type: ignore[operator]
    report = tmp_path / "report.md"  # type: ignore[operator]
    result = build_v2_stage_summary(output_path=output, report_path=report)

    assert result["chain_validation"]["m2_authorized_m3"] is True
    assert result["chain_validation"]["m3_authorized_m4"] is True
    assert result["chain_validation"][
        "m1_m3_fresh_confirmation_passed"
    ] is True
    rendered = report.read_text(encoding="utf-8")
    assert "不能外推" in rendered
    assert output.exists()
    unprivileged = "V2-M4-unprivileged-holdout-summary.json"
    if unprivileged in result["sources"]:
        # The one-shot unprivileged M4 holdout passed, so the chain now
        # authorizes the reconnection design review.
        assert result["passed"] is True
        assert result["decision"] == "authorize_reconnection_design_review"
        assert result["chain_validation"]["m4_authorized_reconnection"] is True
        assert "重连设计评审" in rendered
    else:
        assert result["passed"] is False
        assert result["decision"] == "retain_v2_stage_stop"
        assert "停在 V2-M4" in rendered
