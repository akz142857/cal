"""Load and normalize persisted experiment results for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DashboardDataError(RuntimeError):
    """Raised when a required persisted dashboard artifact is unavailable."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DashboardDataError(f"missing dashboard artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DashboardDataError(f"invalid dashboard artifact: {path}") from error
    if not isinstance(payload, dict):
        raise DashboardDataError(f"dashboard artifact is not an object: {path}")
    return payload


def load_dashboard_snapshot(
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root)
    results = root / "results"
    stage = _read_json(results / "V2-stage-summary.json")
    confirmation = _read_json(
        results / "V2-M1-M3-integrated-v2-confirmation-summary.json"
    )
    review = _read_json(
        results / "V2-M1-M3-integrated-confirmation-review-summary.json"
    )
    index = _read_json(results / "INDEX.json")
    chain = stage["chain_validation"]
    key = stage["key_results"]
    confirmed = review["confirmation"]
    stages = [
        {
            "阶段": "V2-A–C",
            "状态": "通过" if chain["audit_authorized_m1"] else "停止",
            "核心证据": (
                f"镜像历史 Bayes IoU {key['mirrored_32_frame_majority_iou']:.3f}"
            ),
            "下一门": "M1",
        },
        {
            "阶段": "V2-M1",
            "状态": "通过" if chain["m1_authorized_m2"] else "停止",
            "核心证据": f"新确认 F1 {confirmed['m1']['mean_f1']:.4f}",
            "下一门": "M2",
        },
        {
            "阶段": "V2-M2",
            "状态": "通过" if chain["m2_authorized_m3"] else "停止",
            "核心证据": (
                "新确认交叉身份 "
                f"{confirmed['m2']['crossing_identity_retention']:.4f}"
            ),
            "下一门": "M3",
        },
        {
            "阶段": "V2-M3",
            "状态": "通过" if chain["m3_authorized_m4"] else "停止",
            "核心证据": (
                "新确认真实图概率 "
                f"{confirmed['m3']['broken_true_probability_mean']:.6f}"
            ),
            "下一门": "M4",
        },
        {
            "阶段": "M1–M3 综合确认",
            "状态": (
                "通过"
                if chain["m1_m3_fresh_confirmation_passed"]
                else "停止"
            ),
            "核心证据": "旧留出零读取；一次性新种子确认",
            "下一门": "无特权 M4 设计",
        },
        {
            "阶段": "V2-M4",
            "状态": (
                "通过"
                if chain["m4_authorized_reconnection"]
                else "诊断/停止"
            ),
            "核心证据": (
                f"特权可见性占据 IoU {key['m4_occupancy_iou']:.3f}"
            ),
            "下一门": "删除模拟器可见性掩码",
        },
    ]
    comparison = [
        {
            "机制": "M1 控制发现",
            "版本": "正式",
            "指标": confirmed["m1"]["mean_f1"],
        },
        {
            "机制": "M1 控制发现",
            "版本": "删除动作",
            "指标": max(
                0.0,
                confirmed["m1"]["mean_f1"]
                - confirmed["m1"]["no_action_f1_drop"],
            ),
        },
        {
            "机制": "M2 交叉身份",
            "版本": "正式",
            "指标": confirmed["m2"]["crossing_identity_retention"],
        },
        {
            "机制": "M2 交叉身份",
            "版本": "最近邻",
            "指标": confirmed["m2"][
                "nearest_crossing_identity_retention"
            ],
        },
        {
            "机制": "M3 真实图后验",
            "版本": "正式",
            "指标": confirmed["m3"]["broken_true_probability_mean"],
        },
        {
            "机制": "M3 真实图后验",
            "版本": "删除因果似然",
            "指标": confirmed["m3"]["no_likelihood_true_probability"],
        },
    ]
    gate_rows = [
        {
            "门": name,
            "状态": "通过" if passed else "失败",
        }
        for name, passed in review["gates"].items()
    ]
    resource_rows = [
        {
            "阶段": name,
            "参数": values["learnable_parameter_count"],
            "活动状态(B)": values["active_state_bytes"],
            "MAC/步": values["estimated_mac_per_step"],
            "步/种子": values["steps_per_seed"],
        }
        for name, values in confirmation["resources"][
            "stage_resources"
        ].items()
    ]
    episodes = {
        "M1 正式": confirmation["m1"]["variants"]["formal_active"][
            "episodes"
        ],
        "M2 交叉正式": confirmation["m2"]["formal"][
            "crossing_episodes"
        ],
        "M3 正式": confirmation["m3"]["formal"]["episodes"],
        "M3 删除因果似然": confirmation["m3"][
            "no_causal_prediction_likelihood"
        ]["episodes"],
    }
    return {
        "project_root": str(root),
        "stage": stage,
        "confirmation": confirmation,
        "review": review,
        "index": index,
        "stage_rows": stages,
        "comparison_rows": comparison,
        "gate_rows": gate_rows,
        "resource_rows": resource_rows,
        "episodes": episodes,
        "current_decision": stage["decision"],
        "current_blocker": "V2-M4 仍使用模拟器生成的可见性掩码",
    }
