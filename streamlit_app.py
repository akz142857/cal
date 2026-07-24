"""Read-only Streamlit dashboard for Cal persisted research artifacts."""

from __future__ import annotations

import streamlit as st

from calmodel.dashboard.data import DashboardDataError, load_dashboard_snapshot


st.set_page_config(
    page_title="Cal 研究状态",
    page_icon="🧭",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _snapshot() -> dict:
    return load_dashboard_snapshot()


def _overview(data: dict) -> None:
    st.subheader("当前阶段")
    columns = st.columns(4)
    confirmed = data["review"]["confirmation"]
    columns[0].metric("M1 新确认 F1", f"{confirmed['m1']['mean_f1']:.4f}")
    columns[1].metric(
        "M2 交叉身份",
        f"{confirmed['m2']['crossing_identity_retention']:.4f}",
    )
    columns[2].metric(
        "M3 真实图概率",
        f"{confirmed['m3']['broken_true_probability_mean']:.6f}",
    )
    columns[3].metric(
        "组合端点状态",
        f"{confirmed['resources']['stage_resources']['m3_composed']['active_state_bytes'] / 1024:.1f} KiB",
    )
    st.warning(
        f"当前正式停止点：{data['current_blocker']}。"
        "M1–M3 已确认，但尚未授权重新连接原研究计划。"
    )
    st.dataframe(
        data["stage_rows"],
        width="stretch",
        hide_index=True,
        column_config={
            "阶段": st.column_config.TextColumn(width="medium"),
            "状态": st.column_config.TextColumn(width="small"),
            "核心证据": st.column_config.TextColumn(width="large"),
            "下一门": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "“通过”只表示当前预注册合成任务；不代表原始像素视觉或 FSD 等级能力。"
    )


def _comparisons(data: dict) -> None:
    st.subheader("正式机制与消融")
    st.bar_chart(
        data["comparison_rows"],
        x="机制",
        y="指标",
        color="版本",
        stack=False,
        width="stretch",
        height=420,
    )
    st.dataframe(
        data["comparison_rows"],
        width="stretch",
        hide_index=True,
        column_config={
            "指标": st.column_config.NumberColumn(format="%.6f"),
        },
    )
    st.info(
        "M1 删除动作、M2 最近邻、M3 删除因果似然分别检验动作因果、"
        "概率身份关联和破缺证据是否必要。"
    )


def _audit(data: dict) -> None:
    st.subheader("协议与一次性确认审计")
    confirmation = data["confirmation"]
    columns = st.columns(3)
    columns[0].metric(
        "确认运行次数",
        confirmation["confirmation_run_count"],
    )
    columns[1].metric(
        "旧留出读取",
        len(confirmation["historical_holdout_artifacts_read"]),
    )
    columns[2].metric(
        "旧留出种子使用",
        len(confirmation["historical_holdout_seeds_used"]),
    )
    st.dataframe(
        data["gate_rows"],
        width="stretch",
        hide_index=True,
    )
    with st.expander("协议与源码哈希"):
        st.code(
            f"协议: {confirmation['protocol_sha256']}\n"
            + "\n".join(
                f"{path}: {digest}"
                for path, digest in confirmation[
                    "locked_source_sha256"
                ].items()
            ),
            language="text",
        )
    with st.expander("v1 → v2 修订边界"):
        st.write(
            "v1 开发发现 0.5/0.5 平坦后验的硬 MAP 平局拓扑门无效。"
            "在确认运行次数仍为 0 时发布 v2，没有修改 M1–M3 算法，"
            "并换用另一批确认种子。"
        )


def _resources(data: dict) -> None:
    st.subheader("资源预算")
    st.dataframe(
        data["resource_rows"],
        width="stretch",
        hide_index=True,
        column_config={
            "参数": st.column_config.NumberColumn(format="%d"),
            "活动状态(B)": st.column_config.NumberColumn(format="%d"),
            "MAC/步": st.column_config.NumberColumn(format="%d"),
            "步/种子": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.bar_chart(
        data["resource_rows"],
        x="阶段",
        y=["参数", "MAC/步"],
        stack=False,
        width="stretch",
        height=360,
    )
    st.caption(
        "M1 是阶段机制确认代理；组合端点是 M2 实体图 + M3 身体图滤波器，"
        "不会同时驻留 M1 状态。"
    )


def _episodes(data: dict) -> None:
    st.subheader("确认回合浏览")
    family = st.selectbox("结果族", list(data["episodes"]))
    episode_list = data["episodes"][family]
    seeds = [int(item["seed"]) for item in episode_list]
    seed = st.selectbox("确认种子", seeds)
    episode = next(item for item in episode_list if int(item["seed"]) == seed)
    scenario = episode.get("scenario")
    if scenario:
        st.markdown("**场景参数**")
        st.json(scenario, expanded=False)
    metrics = {
        key: value
        for key, value in episode.items()
        if key not in {"seed", "scenario"}
        and isinstance(value, (bool, int, float, str))
    }
    st.markdown("**回合指标**")
    st.dataframe(
        [{"指标": key, "值": value} for key, value in metrics.items()],
        width="stretch",
        hide_index=True,
    )


st.title("Cal 研究状态")
st.caption("只读取已固化 JSON 结果；不会运行实验或改写冻结协议。")

try:
    snapshot = _snapshot()
except DashboardDataError as error:
    st.error(str(error))
    st.stop()

overview_tab, comparison_tab, audit_tab, resource_tab, episode_tab = st.tabs(
    ["总览", "机制对照", "协议审计", "资源", "回合"]
)
with overview_tab:
    _overview(snapshot)
with comparison_tab:
    _comparisons(snapshot)
with audit_tab:
    _audit(snapshot)
with resource_tab:
    _resources(snapshot)
with episode_tab:
    _episodes(snapshot)
