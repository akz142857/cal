"""Tests for the read-only Cal dashboard data contract."""

from cal.dashboard.data import load_dashboard_snapshot


def test_dashboard_snapshot_reflects_current_stage_and_confirmation() -> None:
    snapshot = load_dashboard_snapshot()

    assert snapshot["current_decision"] in {
        "retain_v2_stage_stop",
        "authorize_reconnection_design_review",
    }
    assert snapshot["confirmation"]["confirmation_run_count"] == 1
    assert snapshot["confirmation"]["historical_holdout_artifacts_read"] == []
    assert len(snapshot["stage_rows"]) == 6
    assert len(snapshot["resource_rows"]) == 3
    assert set(snapshot["episodes"]) == {
        "M1 正式",
        "M2 交叉正式",
        "M3 正式",
        "M3 删除因果似然",
    }


def test_dashboard_comparisons_preserve_required_controls() -> None:
    snapshot = load_dashboard_snapshot()
    rows = snapshot["comparison_rows"]

    assert len(rows) == 6
    formal_m2 = next(
        row
        for row in rows
        if row["机制"] == "M2 交叉身份" and row["版本"] == "正式"
    )
    nearest_m2 = next(
        row
        for row in rows
        if row["机制"] == "M2 交叉身份" and row["版本"] == "最近邻"
    )
    assert formal_m2["指标"] > nearest_m2["指标"]
