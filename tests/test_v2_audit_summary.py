"""Tests for the combined V2-A-C entry decision."""

import json

from calmodel.evaluation.v2_audit_summary import build_v2_audit_summary


def test_real_v2_audits_build_a_report_when_available(
    tmp_path: object,
) -> None:
    output = tmp_path / "summary.json"  # type: ignore[operator]
    report = tmp_path / "report.md"  # type: ignore[operator]

    result = build_v2_audit_summary(
        "results/V2-identifiability-summary.json",
        "results/V2-diagnostic-ceiling-summary.json",
        "results/V2-causal-sufficiency-summary.json",
        output_path=output,
        report_path=report,
    )

    assert output.exists()
    assert report.exists()
    assert result["decision"] in {
        "authorize_v2_m1",
        "stop_before_v2_m1_and_revise_observability",
    }
    assert set(result["gates"]) == {
        "identifiability_classes_and_ceiling_quantified",
        "intervention_information_advantage_confirmed",
        "formal_metrics_adjusted_to_identifiability",
        "all_audits_within_resource_budget",
        "evaluation_labels_isolated",
    }
    assert json.loads(output.read_text())["passed"] == result["passed"]
