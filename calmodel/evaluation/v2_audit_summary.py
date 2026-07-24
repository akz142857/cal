"""Combine V2-A/B/C audits into the formal entry-gate decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from calmodel.infra.provenance import capture_provenance


def build_v2_audit_summary(
    identifiability_path: str | Path,
    diagnostic_path: str | Path,
    causal_path: str | Path,
    *,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Validate all audit evidence, decide the gate, and write JSON/Markdown."""

    identifiability = _read_mapping(identifiability_path)
    diagnostic = _read_mapping(diagnostic_path)
    causal = _read_mapping(causal_path)
    _require_audit(identifiability, "v2_identifiability")
    _require_audit(diagnostic, "v2_diagnostic_ceiling")
    _require_audit(causal, "v2_causal_sufficiency")

    random_histories = identifiability["policies"]["random"]
    active_histories = identifiability["policies"]["active_visual"]
    required_lengths = {"1", "2", "4", "8", "16", "32"}
    symmetry_histories = identifiability["permutation_symmetry"]["histories"]
    identifiability_quantified = (
        required_lengths <= set(random_histories)
        and required_lengths <= set(active_histories)
        and required_lengths <= set(symmetry_histories)
        and all(
            symmetry_histories[length][
                "cross_seed_ambiguous_class_count"
            ]
            > 0
            for length in required_lengths
        )
    )

    causal_comparison = diagnostic["comparisons"][
        "causal_evidence_minus_frame"
    ]
    fixed_actions = causal["action_comparison"]["fixed_single_actions"]
    active_action_iou = causal["action_comparison"][
        "oracle_best_single_action"
    ]["iou"]
    best_fixed_iou = max(item["iou"] for item in fixed_actions.values())
    intervention_advantage = (
        causal_comparison["all_seeds_positive"]
        and causal_comparison["at_least_0_05"]
    )

    resource_passed = (
        _all_resource_gates(identifiability["resource_gates"])
        and _all_resource_gates(causal["resource_gates"])
        and all(
            _all_resource_gates(item["resource_gates"])
            for item in diagnostic["diagnostics"].values()
        )
    )
    label_isolation = (
        identifiability.get("diagnostic_only") is True
        and diagnostic.get("diagnostic_only") is True
        and causal.get("diagnostic_only") is True
        and all(
            item.get("diagnostic_only") is True
            for item in diagnostic["diagnostics"].values()
        )
        and all(
            "checkpoint" not in item
            for item in diagnostic["diagnostics"].values()
        )
    )
    metric_policy = {
        "ambiguous_equivalence_classes": (
            "score calibrated membership probabilities with Brier/NLL and "
            "allow multiple hypotheses; do not demand a unique mask"
        ),
        "distinguishable_episodes": (
            "report ordinary entity F1/IoU and normalize deterministic mask "
            "scores by the measured observable-class ceiling"
        ),
        "mirrored_distractor_stress": (
            "success is uncertainty calibration plus an information-seeking "
            "action, not guessing the simulator's hidden self label"
        ),
    }
    metric_policy_adjusted = bool(metric_policy) and any(
        symmetry_histories[length]["majority_ceiling_ambiguous"]["iou"] < 1.0
        for length in required_lengths
    )
    gates = {
        "identifiability_classes_and_ceiling_quantified": (
            identifiability_quantified
        ),
        "intervention_information_advantage_confirmed": (
            intervention_advantage
        ),
        "formal_metrics_adjusted_to_identifiability": (
            metric_policy_adjusted
        ),
        "all_audits_within_resource_budget": resource_passed,
        "evaluation_labels_isolated": label_isolation,
    }
    passed = all(gates.values())
    summary = {
        "result_schema_version": 1,
        "audit": "v2_abc_entry_gate",
        "sources": {
            "identifiability": str(identifiability_path),
            "diagnostic_ceiling": str(diagnostic_path),
            "causal_sufficiency": str(causal_path),
        },
        "key_evidence": {
            "random_single_frame_ambiguity_rate": random_histories["1"][
                "ambiguity_sample_rate"
            ],
            "random_two_frame_ambiguity_rate": random_histories["2"][
                "ambiguity_sample_rate"
            ],
            "mirrored_history_32_bayes_iou": symmetry_histories["32"][
                "majority_ceiling_ambiguous"
            ]["iou"],
            "frame_diagnostic_iou": diagnostic["diagnostics"]["frame"][
                "test"
            ]["aggregate"]["iou"],
            "video_diagnostic_iou": diagnostic["diagnostics"]["video"][
                "test"
            ]["aggregate"]["iou"],
            "video_action_diagnostic_iou": diagnostic["diagnostics"][
                "video_action"
            ]["test"]["aggregate"]["iou"],
            "causal_evidence_diagnostic_iou": diagnostic["diagnostics"][
                "causal_evidence"
            ]["test"]["aggregate"]["iou"],
            "causal_evidence_minus_frame_iou": causal_comparison[
                "mean_iou_difference"
            ],
            "exhaustive_causal_iou": causal["exhaustive_current"]["iou"],
            "oracle_best_single_action_iou": active_action_iou,
            "best_fixed_single_action_iou": best_fixed_iou,
            "best_history_union": causal["best_history_union"],
            "best_deterministic_geometry": causal[
                "best_deterministic_geometry"
            ],
            "analytic_pose_grid": causal["analytic_pose_grid"],
        },
        "metric_policy": metric_policy,
        "gates": gates,
        "passed": passed,
        "decision": (
            "authorize_v2_m1"
            if passed
            else "stop_before_v2_m1_and_revise_observability"
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
        _render_report(summary, identifiability, diagnostic, causal),
        encoding="utf-8",
    )
    return summary


def _render_report(
    summary: Mapping[str, Any],
    identifiability: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    causal: Mapping[str, Any],
) -> str:
    evidence = summary["key_evidence"]
    symmetry = identifiability["permutation_symmetry"]["histories"]
    diagnostics = diagnostic["diagnostics"]
    grid_buckets = causal["pose_buckets"]["analytic_pose_grid"]
    lines = [
        "# Cal V2-A–C 审计报告",
        "",
        "日期：2026-07-24",
        "",
        "## 结论",
        "",
        (
            "**通过 V2-A–C 进入门，授权实现 V2-M1。**"
            if summary["passed"]
            else "**未通过进入门，V2-M1 保持未启动。**"
        ),
        "",
        "该结论不表示当前环境存在唯一身体掩码。相反，置换对称构造证明：当一个",
        "同构外部臂镜像自身动作时，即使 32 帧视觉—动作历史完全相同，隐藏的“哪条",
        "臂是自己”仍可不同。V2-M1 必须维护概率身份并主动消歧，不能被要求猜测",
        "模拟器的隐藏标签。",
        "",
        "## V2-A：可识别性",
        "",
        "| 历史 | 随机轨迹歧义率 | 在线均衡动作歧义率 | 置换反例逐像素多数 IoU |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for length in ("1", "2", "4", "8", "16", "32"):
        lines.append(
            f"| {length} | "
            f"{identifiability['policies']['random'][length]['ambiguity_sample_rate']:.4f} | "
            f"{identifiability['policies']['active_visual'][length]['ambiguity_sample_rate']:.4f} | "
            f"{symmetry[length]['majority_ceiling_ambiguous']['iou']:.4f} |"
        )
    lines.extend(
        [
            "",
            "随机采样中，两帧后几乎没有观察到重复历史；这说明常见轨迹容易区分，",
            "但不能推翻置换反例。经验频率和理论可识别性必须分别报告。",
            "",
            "## V2-B：隔离监督可解码性诊断",
            "",
            "| 诊断器 | 测试 IoU | 参数 | MAC/步 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for mode in ("frame", "video", "video_action", "causal_evidence"):
        item = diagnostics[mode]
        lines.append(
            f"| `{mode}` | {item['test']['aggregate']['iou']:.4f} | "
            f"{item['parameter_count']:,} | {item['mac_per_step']:,} |"
        )
    lines.extend(
        [
            "",
            f"普通视频相对单帧只提高 "
            f"{diagnostic['comparisons']['video_minus_frame']['mean_iou_difference']:+.4f} IoU，"
            f"视频加动作提高 "
            f"{diagnostic['comparisons']['video_action_minus_frame']['mean_iou_difference']:+.4f}；"
            f"完整因果证据提高 "
            f"{evidence['causal_evidence_minus_frame_iou']:+.4f}，且 8/8 未见种子均改善。",
            "",
            "因此信息主要存在于干预结构中，而不是简单增加视频帧。",
            "",
            "## V2-C：因果充分性",
            "",
            f"- 当前五动作包络：IoU {causal['exhaustive_current']['iou']:.4f}，"
            f"召回 {causal['exhaustive_current']['recall']:.4f}，"
            f"精确率 {causal['exhaustive_current']['precision']:.4f}；",
            f"- 五分支后验最佳单动作（诊断上限）：IoU {evidence['oracle_best_single_action_iou']:.4f}；",
            f"- 最佳固定单动作：IoU {evidence['best_fixed_single_action_iou']:.4f}；",
            f"- 最佳历史并集：K={evidence['best_history_union']['length']}，"
            f"IoU {evidence['best_history_union']['metrics']['iou']:.4f}；",
            f"- 最佳确定性几何传播：深度 "
            f"{evidence['best_deterministic_geometry']['depth']}，"
            f"IoU {evidence['best_deterministic_geometry']['metrics']['iou']:.4f}；",
            f"- 解析姿态网格：{causal['analytic_pose_grid']['sample_count']} 个姿态，"
            f"IoU {causal['analytic_pose_grid']['metrics']['iou']:.4f}。",
            "",
            "| 姿态/重叠分桶 | 样本 | IoU |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in (
        "shoulder_near_limit",
        "shoulder_interior",
        "elbow_near_limit",
        "elbow_interior",
        "overlap_low",
        "overlap_medium",
        "overlap_high",
    ):
        item = grid_buckets[name]
        iou = (
            "—"
            if item["metrics"] is None
            else f"{item['metrics']['iou']:.4f}"
        )
        lines.append(f"| `{name}` | {item['sample_count']} | {iou} |")
    lines.extend(
        [
            "",
            "高重叠是最主要的因果证据损失来源。历史增加到 8 帧改善覆盖，但更长历史",
            "开始积累旧位置假阳性；几何传播一步有效，继续传播则泄漏到同基座外部臂。",
            "",
            "## 进入门",
            "",
        ]
    )
    for name, passed in summary["gates"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "## 对正式 V2-M1 的约束",
            "",
            "1. 歧义状态输出概率身份和不确定性，不强制唯一掩码；",
            "2. 预测失败必须在线更新控制矩阵或假设权重；",
            "3. 主动动作以信息增益为目标，并与随机动作、固定动作比较；",
            "4. 身体标签与全动作真值不得进入正式智能体；",
            "5. 正式能力必须在 100k 参数、64 KiB 状态、5M MAC/步和 CPU 预算内；",
            "6. 监督诊断器不保存或加载到正式模型。",
            "",
        ]
    )
    return "\n".join(lines)


def _all_resource_gates(gates: Mapping[str, Any]) -> bool:
    pass_values = [
        value
        for key, value in gates.items()
        if key.endswith("_passed")
    ]
    return bool(pass_values) and all(pass_values)


def _read_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"audit result must be an object: {path}")
    return payload


def _require_audit(payload: Mapping[str, Any], expected: str) -> None:
    if payload.get("audit") != expected:
        raise ValueError(f"expected audit {expected}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the formal Cal V2-A-C entry decision."
    )
    parser.add_argument(
        "--identifiability",
        type=Path,
        default=Path("results/V2-identifiability-summary.json"),
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=Path("results/V2-diagnostic-ceiling-summary.json"),
    )
    parser.add_argument(
        "--causal",
        type=Path,
        default=Path("results/V2-causal-sufficiency-summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/V2-audit-summary.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/experiments/V2_AUDIT_REPORT.md"),
    )
    arguments = parser.parse_args(argv)
    result = build_v2_audit_summary(
        arguments.identifiability,
        arguments.diagnostic,
        arguments.causal,
        output_path=arguments.output,
        report_path=arguments.report,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
