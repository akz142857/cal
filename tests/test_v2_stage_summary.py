"""Tests for the final V2 chain validation."""

from calmodel.evaluation.v2_stage_summary import build_v2_stage_summary


def test_v2_stage_summary_validates_complete_chain(tmp_path: object) -> None:
    output = tmp_path / "summary.json"  # type: ignore[operator]
    report = tmp_path / "report.md"  # type: ignore[operator]
    result = build_v2_stage_summary(output_path=output, report_path=report)

    assert result["passed"] is False
    assert result["chain_validation"]["m2_authorized_m3"] is True
    assert result["chain_validation"]["m3_authorized_m4"] is True
    assert result["chain_validation"][
        "m1_m3_fresh_confirmation_passed"
    ] is True
    assert result["decision"] == "retain_v2_stage_stop"
    assert output.exists()
    assert "不能外推" in report.read_text(encoding="utf-8")
    assert "停在 V2-M4" in report.read_text(encoding="utf-8")
