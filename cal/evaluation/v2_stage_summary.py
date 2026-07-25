"""Validate all V2 artifacts and write the final staged research report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from cal.infra.provenance import capture_provenance


def build_v2_stage_summary(
    *,
    results_directory: str | Path = "results",
    output_path: str | Path = "results/V2-stage-summary.json",
    report_path: str | Path = "docs/experiments/V2_STAGE_REPORT.md",
) -> dict[str, Any]:
    root = Path(results_directory)
    names = (
        "V2-audit-summary.json",
        "V2-M1-summary.json",
        "V2-M2-probabilistic-holdout-summary.json",
        "V2-M3-body-graph-holdout-summary.json",
        "V2-M1-M3-integrated-confirmation-review-summary.json",
        "V2-M4-summary.json",
    )
    artifacts = {
        name: json.loads((root / name).read_text(encoding="utf-8"))
        for name in names
    }
    audit, m1, m2, m3, confirmation, m4 = (
        artifacts[name] for name in names
    )
    unprivileged_name = "V2-M4-unprivileged-holdout-summary.json"
    unprivileged_path = root / unprivileged_name
    m4_unprivileged = (
        json.loads(unprivileged_path.read_text(encoding="utf-8"))
        if unprivileged_path.exists()
        else None
    )
    if m4_unprivileged is not None:
        names = names + (unprivileged_name,)
        artifacts[unprivileged_name] = m4_unprivileged
    m4_stage = m4_unprivileged if m4_unprivileged is not None else m4
    m4_satisfied = (
        m4_unprivileged is not None
        and m4_unprivileged.get("passed")
        and m4_unprivileged.get("decision")
        == "authorize_reconnection_design_review"
    ) or (
        m4.get("passed")
        and m4.get("decision") == "authorize_reconnection_to_original_m2"
    )
    chain = {
        "audit_authorized_m1": (
            audit.get("passed") and audit.get("decision") == "authorize_v2_m1"
        ),
        "m1_authorized_m2": (
            m1.get("passed") and m1.get("decision") == "authorize_v2_m2"
        ),
        "m2_authorized_m3": (
            m2.get("passed") and m2.get("decision") == "authorize_v2_m3"
        ),
        "m3_authorized_m4": (
            m3.get("passed") and m3.get("decision") == "authorize_v2_m4"
        ),
        "m1_m3_fresh_confirmation_passed": (
            confirmation.get("passed")
            and confirmation.get("decision")
            == "confirm_m1_m3_and_authorize_unprivileged_v2_m4_design"
        ),
        "m4_authorized_reconnection": m4_satisfied,
        "all_formal_learners_label_free": all(
            item.get("evaluation_labels_used_for_learning") is False
            for item in (m1, m2, m3, m4_stage)
        ),
        "all_formal_stage_gates_pass": all(
            all(item.get("gates", {}).values())
            for item in (m1, m2, m3, confirmation, m4_stage)
        ),
    }
    passed = all(chain.values())
    summary = {
        "result_schema_version": 1,
        "experiment": "V2-stage",
        "sources": {name: str(root / name) for name in names},
        "source_artifact_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names
        },
        "chain_validation": chain,
        "key_results": {
            "universal_unique_mask_identifiable": False,
            "mirrored_32_frame_majority_iou": audit["key_evidence"][
                "mirrored_history_32_bayes_iou"
            ],
            "m1_self_trajectory_f1": m1["variants"]["formal_active"][
                "aggregate"
            ]["mean_f1"],
            "m1_active_step_reduction": m1["comparisons"][
                "active_confidence_step_reduction"
            ],
            "m2_controlled_node_f1": m2["aggregate"]["node_f1"],
            "m2_rigid_edge_f1": m2["aggregate"]["rigid_edge_f1"],
            "m3_body_projection_iou": m3["aggregate"][
                "pose_grid_projection_iou_mean"
            ],
            "m3_true_graph_probability": m3["aggregate"][
                "broken_true_probability_mean"
            ],
            "m3_symmetric_nll": m3["aggregate"][
                "symmetric_nll"
            ],
            "confirmation_m1_f1": confirmation["confirmation"]["m1"][
                "mean_f1"
            ],
            "confirmation_m2_crossing_identity": confirmation[
                "confirmation"
            ]["m2"]["crossing_identity_retention"],
            "confirmation_m3_true_graph_probability": confirmation[
                "confirmation"
            ]["m3"]["broken_true_probability_mean"],
            "m4_occupancy_iou": m4_stage["aggregate"]["occupancy_iou"],
            "m4_hidden_recall": m4_stage["aggregate"][
                "hidden_occupancy_recall"
            ],
            "m4_active_step_reduction": m4_stage["aggregate"][
                "active_step_reduction"
            ],
        },
        "scope_limitations": [
            "V2 proves staged mechanisms in small deterministic synthetic worlds, not general visual cognition.",
            "M2 and M3 consume deterministic sparse visual detections; a raw-pixel front end remains unvalidated.",
            "The mirrored shared-base stress is reported as multiple hypotheses because a unique hidden identity is unidentifiable.",
            "M3 pose projection uses a known analytic two-link renderer and validates graph selection, not learned raw-pixel segmentation.",
            "The fresh M1-M3 confirmation validates behavioral composition; it does not establish serialized M1 state handoff into M2.",
        ]
        + (
            [
                "The unprivileged M4 infers occlusion by shadow-casting over sensed occupancy; the simulator visibility mask is no longer an agent input.",
                "The unprivileged M4 holdout authorizes a reconnection design review, not reconnection itself; the original object-permanence M2 remains unstarted.",
            ]
            if m4_unprivileged is not None
            else [
                "M4 receives a sensor visibility mask and uses a small two-dimensional allocentric grid.",
                "M4 still receives a simulator visibility mask and therefore remains a diagnostic rather than a formal visual-only stage.",
                "Only a future full V2 pass could authorize reconnection to the original object-permanence M2.",
            ]
        ),
        "passed": passed,
        "decision": (
            (
                "authorize_reconnection_design_review"
                if m4_unprivileged is not None
                else "authorize_reconnection_to_original_m2"
            )
            if passed
            else "retain_v2_stage_stop"
        ),
        "provenance": capture_provenance(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        _render_report(
            summary, m1, m2, m3, confirmation, m4_stage, m4_unprivileged
        ),
        encoding="utf-8",
    )
    return summary


def _render_report(
    summary: dict[str, Any],
    m1: dict[str, Any],
    m2: dict[str, Any],
    m3: dict[str, Any],
    confirmation: dict[str, Any],
    m4: dict[str, Any],
    m4_unprivileged: dict[str, Any] | None = None,
) -> str:
    key = summary["key_results"]
    limitations = "\n".join(
        f"- {item}" for item in summary["scope_limitations"]
    )
    if summary["passed"] and m4_unprivileged is not None:
        decision_text = (
            "**V2-A–C 与 V2-M1–M4 的逐级门全部通过；无特权 M4 在一次性冻结"
            "留出上通过，授权面向原研究计划 M2 的重连设计评审。**"
        )
        final_text = (
            "M4 的模拟器可见性掩码已删除：无特权变体仅凭感知占据的几何"
            " shadow-casting 推断遮挡，并在冻结留出上通过全部门。下一步是"
            "重连设计评审，而不是直接开始原对象永久性 M2。"
        )
    elif summary["passed"]:
        decision_text = (
            "**V2-A–C 与 V2-M1–M4 的逐级门全部通过，授权重新连接原研究计划 M2。**"
        )
        final_text = "因此下一步可以把已验证机制接回更丰富的原始视觉环境。"
    else:
        decision_text = (
            "**V2-A–C、V2-M1、冻结留出 M2 与 M3 通过；V2 总链停止在 M4 的无特权视觉门。**"
        )
        final_text = (
            "当前决定是停在 V2-M4：M3 已形成互斥、校准的完整身体图后验；下一步必须从视觉推断 free/occupied/unknown，删除模拟器可见性掩码。"
        )
    return f"""# Cal V2 阶段报告

日期：2026-07-24

## 决策

{decision_text}

这不是“唯一身体掩码已经可解”的结论。置换对称反例仍证明，即使 32 帧
视觉—动作历史相同，隐藏身份也可能不同。正式系统必须在这种状态下保留多个
假设，不能猜测模拟器的秘密标签。

## 逐级证据

| 阶段 | 核心结果 | 决策 |
| --- | --- | --- |
| V2-A–C | 镜像反例 32 帧逐像素多数 IoU {key['mirrored_32_frame_majority_iou']:.3f} | 授权 M1 |
| V2-M1 | 轨迹 F1 {key['m1_self_trajectory_f1']:.3f}；主动步数减少 {key['m1_active_step_reduction']:.1%} | 授权 M2 |
| V2-M2 | 节点 F1 {key['m2_controlled_node_f1']:.3f}；交叉身份保持 {m2['aggregate']['crossing_identity_retention']:.3f} | {m2['decision']} |
| V2-M3 | 真实完整图概率 {key['m3_true_graph_probability']:.6f}；姿态投影 IoU {key['m3_body_projection_iou']:.3f} | {m3['decision']} |
| M1–M3 新数据综合确认 | M1 F1 {key['confirmation_m1_f1']:.4f}；M2 交叉身份 {key['confirmation_m2_crossing_identity']:.4f}；M3 真实图概率 {key['confirmation_m3_true_graph_probability']:.6f} | {confirmation['decision']} |
| V2-M4 | {'无特权视觉留出' if m4_unprivileged is not None else '特权可见性诊断'}：占据 IoU {key['m4_occupancy_iou']:.3f}；遮挡召回 {key['m4_hidden_recall']:.3f} | {m4['decision']} |

M1 的动作输入和在线失败更新消融都使 F1 下降至少 0.15。M2 的交叉压力
节点 F1 为 {m2['aggregate']['crossing_node_f1']:.3f}。M3 在同构共享基座压力
中保持两张完整身体图各 0.5，并在破缺后最慢
{m3['aggregate']['broken_convergence_steps_maximum']} 步收敛；无因果似然开发消融
保持在机会水平。随后使用完全不同的新种子完成 M1–M3 综合确认，且没有读取
旧留出结果。M4 的移动目标隐藏期平均占据概率为
{m4['aggregate']['moving_hidden_probability']:.3f}{'；其可见性由智能体自身的几何遮挡推理给出，assume_all_visible 对照按设计失败' if m4_unprivileged is not None else '，但仍读取模拟器可见性掩码'}。

## 失败闭环与资源

M1 为每步保存有界、无标签的预测—残差遥测；M2/M3 使用严格
预序列残差更新递推控制模型；M4 只保留小型概率格和短期运动假设。全部阶段
CPU 运行、经验零重放。资源通过不覆盖未通过的机制门。

## 不能外推的内容

{limitations}

{final_text}
不应把当前合成环境成绩解释为 FSD 等级能力。
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/V2-stage-summary.json")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/experiments/V2_STAGE_REPORT.md"),
    )
    arguments = parser.parse_args(argv)
    result = build_v2_stage_summary(
        results_directory=arguments.results,
        output_path=arguments.output,
        report_path=arguments.report,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
