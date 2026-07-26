"""Tests for the read-only Cal dashboard data contract."""

from pathlib import Path

from cal.dashboard.data import PROJECT_ROOT, load_dashboard_snapshot


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

    # The V2-M4 row's wording must track which artifact actually backs the
    # displayed number - a stale hardcoded "still privileged" description
    # would misinform a viewer once the unprivileged holdout has passed.
    m4_row = next(
        row for row in snapshot["stage_rows"] if row["阶段"] == "V2-M4"
    )
    m4_unprivileged_exists = (
        Path(PROJECT_ROOT) / "results" / "V2-M4-unprivileged-holdout-summary.json"
    ).exists()
    if m4_unprivileged_exists:
        assert "无特权" in m4_row["核心证据"]
        assert "重连设计评审" == m4_row["下一门"]
        assert "已授权重连设计评审" in snapshot["current_blocker"]
    else:
        assert "特权可见性" in m4_row["核心证据"]
        assert "删除模拟器可见性掩码" == m4_row["下一门"]


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
