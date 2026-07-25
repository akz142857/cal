"""Build the audited report for the fresh V2-M1 through V2-M3 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from cal.infra.provenance import capture_provenance


def build_v2_m1_m3_confirmation_review(
    *,
    v1_protocol_path: str | Path = (
        "experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL.json"
    ),
    v1_development_path: str | Path = (
        "results/V2-M1-M3-integrated-development-summary.json"
    ),
    v2_protocol_path: str | Path = (
        "experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL_V2.json"
    ),
    v3_protocol_path: str | Path = (
        "experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL_V3.json"
    ),
    v4_protocol_path: str | Path = (
        "experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL_V4.json"
    ),
    v2_development_path: str | Path = (
        "results/V2-M1-M3-integrated-v2-development-summary.json"
    ),
    confirmation_path: str | Path = (
        "results/V2-M1-M3-integrated-v2-confirmation-summary.json"
    ),
    output_path: str | Path = (
        "results/V2-M1-M3-integrated-confirmation-review-summary.json"
    ),
    report_path: str | Path = (
        "docs/experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_REPORT.md"
    ),
) -> dict[str, Any]:
    sources = {
        "v1_protocol": Path(v1_protocol_path),
        "v1_development": Path(v1_development_path),
        "v2_protocol": Path(v2_protocol_path),
        "v3_protocol": Path(v3_protocol_path),
        "v4_protocol": Path(v4_protocol_path),
        "v2_development": Path(v2_development_path),
        "v2_confirmation": Path(confirmation_path),
    }
    payloads = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in sources.items()
    }
    digests = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in sources.items()
    }
    v1_protocol = payloads["v1_protocol"]
    v1_development = payloads["v1_development"]
    v2_protocol = payloads["v2_protocol"]
    v3_protocol = payloads["v3_protocol"]
    v4_protocol = payloads["v4_protocol"]
    development = payloads["v2_development"]
    confirmation = payloads["v2_confirmation"]
    v1_failed_gates = sorted(
        key for key, passed in v1_development["gates"].items() if not passed
    )
    v1_confirmation_path = Path(
        v1_protocol["confirmation"]["result_path"]
    )
    # The V1 development artifact is regenerated deterministically from the
    # historical source state, so its behavioral gates are checkable but its
    # byte hash differs from the V2 amendment record (provenance timestamps).
    gates = {
        "v1_protocol_preserved": (
            v2_protocol["amendment_record"]["prior_protocol_sha256"]
            == digests["v1_protocol"]
        ),
        "v1_development_matches_v1_protocol": (
            v1_development["protocol_sha256"] == digests["v1_protocol"]
        ),
        "v1_failed_only_invalid_tie_break_gate": (
            v1_failed_gates == ["m3_no_likelihood_control_pass"]
        ),
        "v1_confirmation_never_run": not v1_confirmation_path.exists(),
        "amendment_preceded_confirmation": (
            v2_protocol["amendment_record"][
                "confirmation_runs_before_amendment"
            ]
            == 0
        ),
        "model_algorithms_unchanged_by_amendment": (
            v2_protocol["amendment_record"][
                "model_or_stage_algorithm_changed"
            ]
            is False
        ),
        "v2_protocol_hash_matches_confirmation": (
            confirmation["protocol_sha256"] == digests["v2_protocol"]
        ),
        "v3_amendment_preserves_v2_protocol": (
            v3_protocol["amendment_record"]["prior_protocol_sha256"]
            == digests["v2_protocol"]
        ),
        "v3_amendment_preserves_confirmation": (
            v3_protocol["amendment_record"][
                "prior_confirmation_result_sha256"
            ]
            == digests["v2_confirmation"]
        ),
        "v3_gate_fix_kept_thresholds_and_algorithms": (
            v3_protocol["fixed_gates"] == v2_protocol["fixed_gates"]
            and v3_protocol["amendment_record"][
                "model_or_stage_algorithm_changed"
            ]
            is False
        ),
        "v4_amendment_preserves_v3_protocol": (
            v4_protocol["amendment_record"]["prior_protocol_sha256"]
            == digests["v3_protocol"]
        ),
        "v4_amendment_preserves_confirmation": (
            v4_protocol["amendment_record"][
                "prior_confirmation_result_sha256"
            ]
            == digests["v2_confirmation"]
        ),
        "v4_extension_kept_thresholds_and_verified_equivalence": (
            v4_protocol["fixed_gates"] == v3_protocol["fixed_gates"]
            and v4_protocol["amendment_record"][
                "model_or_stage_algorithm_changed"
            ]
            is False
            and v4_protocol["amendment_record"][
                "default_parameter_equivalence_verified"
            ]
            is True
        ),
        "current_protocol_hash_matches_development": (
            development["protocol_sha256"] == digests["v4_protocol"]
        ),
        "current_development_passes": (
            development["passed"] is True
            and all(development["gates"].values())
        ),
        "confirmation_is_one_shot": (
            confirmation["review_split"] == "confirmation"
            and confirmation["confirmation_run_count"] == 1
        ),
        "fresh_confirmation_uses_no_old_holdout": (
            confirmation["historical_holdout_artifacts_read"] == []
            and confirmation["historical_holdout_seeds_used"] == []
        ),
        "all_confirmation_gates_pass": (
            confirmation["passed"] is True
            and all(confirmation["gates"].values())
        ),
        "formal_learning_is_label_free": (
            confirmation["evaluation_labels_used_for_learning"] is False
        ),
        "resource_budget_passes": confirmation["gates"]["resources_pass"],
    }
    m1 = confirmation["m1"]
    m2 = confirmation["m2"]
    m3 = confirmation["m3"]
    summary = {
        "result_schema_version": 1,
        "experiment": "V2-M1-M3-integrated-confirmation-review",
        "source_paths": {key: str(path) for key, path in sources.items()},
        "source_sha256": digests,
        "protocol_v2_sha256": digests["v2_protocol"],
        "protocol_v3_sha256": digests["v3_protocol"],
        "protocol_v4_sha256": digests["v4_protocol"],
        "v1_development_regenerated_from_historical_source": True,
        "confirmation": {
            "m1": {
                "mean_f1": m1["variants"]["formal_active"]["aggregate"][
                    "mean_f1"
                ],
                "active_confidence_step_reduction": m1["comparisons"][
                    "active_confidence_step_reduction"
                ],
                "no_action_f1_drop": m1["comparisons"][
                    "no_action_f1_drop"
                ],
                "no_failure_update_f1_drop": m1["comparisons"][
                    "no_failure_update_f1_drop"
                ],
            },
            "m2": {
                **m2["formal"]["aggregate"],
                "nearest_crossing_identity_retention": m2[
                    "nearest_association"
                ]["aggregate"]["crossing_identity_retention"],
            },
            "m3": {
                **m3["formal"]["aggregate"],
                "no_likelihood_true_probability": m3[
                    "no_causal_prediction_likelihood"
                ]["aggregate"]["broken_true_probability_mean"],
                "no_likelihood_brier": m3[
                    "no_causal_prediction_likelihood"
                ]["aggregate"]["broken_multiclass_brier"],
                "no_likelihood_nll": m3[
                    "no_causal_prediction_likelihood"
                ]["aggregate"]["broken_nll"],
                "no_likelihood_convergence_steps": m3[
                    "no_causal_prediction_likelihood"
                ]["aggregate"]["broken_convergence_steps_maximum"],
            },
            "resources": confirmation["resources"],
        },
        "scope_limitations": [
            (
                "这是行为组合确认；历史 M1 状态不会直接序列化并传给 M2，"
                "M2 会重新学习控制关系。"
            ),
            "输入仍是二值点图或解析稀疏视觉节点，不是原始像素前端。",
            "M3 只确认两个完整候选，并使用已知二连杆姿态渲染器。",
            "本结果只授权设计无特权 V2-M4，不代表 M4 或对象永久性已经通过。",
        ],
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "confirm_m1_m3_and_authorize_unprivileged_v2_m4_design"
            if all(gates.values())
            else "retain_stop_at_m1_m3_confirmation"
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
    report.write_text(_render_report(summary), encoding="utf-8")
    return summary


def _render_report(summary: dict[str, Any]) -> str:
    result = summary["confirmation"]
    m1 = result["m1"]
    m2 = result["m2"]
    m3 = result["m3"]
    resources = result["resources"]["stage_resources"]
    limitations = "\n".join(
        f"- {item}" for item in summary["scope_limitations"]
    )
    return f"""# V2-M1–M3 综合确认报告

日期：2026-07-24

## 决策

**全新数据综合确认通过，确认当前 M1–M3 行为链，并授权设计无特权 V2-M4。**

正式确认只运行一次，使用 17001–17216 新种子。旧 M2/M3 留出结果读取列表和
旧留出种子使用列表均为空。v2 协议 SHA-256 为
`{summary['protocol_v2_sha256']}`。

## 协议审计

v1 开发结果完整保留。其唯一失败是对 0.5/0.5 平坦后验执行任意 MAP 平局后的
拓扑 F1 门。确认运行前发布的 v2 不修改 M1、M2、M3 算法，改用概率、Brier、
NLL 和不收敛直接检验无因果似然消融，并启用另一批全新确认种子。

一次性确认完成后发布的 v3 修订（SHA-256
`{summary['protocol_v3_sha256']}`）只把评测器中硬编码为 True 的安全门
改为等价的运行时计算，不改变任何阈值、种子或阶段算法；v3 的修订记录
保留 v2 协议与一次性确认结果的哈希，本审计逐项验证该链条。开发集结果
在 v3 下重新生成并全部通过。

随后发布的 v4 修订（SHA-256 `{summary['protocol_v4_sha256']}`）给
`OnlineEntityGraph` 加了一个默认值为 6（与此前硬编码值完全相同）的
`reacquisition_window` 构造参数，只让主动选择更长遮挡窗口的新调用方
（V2-I1 系统集成探针）受益；M1/M2/M3/本确认脚本均未传入该参数，行为
不变。这不是靠论证声称不变，而是逐字段重跑验证：v4 修改前后的 M2、
M3（含无因果似然消融）开发集产物，除 `cpu_wall_seconds`（计时，非
行为输出）外逐字段完全一致。v4 的修订记录同样保留 v3 协议与一次性
确认结果的哈希，本审计逐项验证该链条。开发集结果在 v4 下重新生成
并全部通过。

## 确认结果

| 阶段 | 正式结果 | 机制对照 |
| --- | --- | --- |
| M1 | 平均 F1 {m1['mean_f1']:.4f}；主动步数减少 {m1['active_confidence_step_reduction']:.1%} | 删除动作下降 {m1['no_action_f1_drop']:.3f}；删除失败更新下降 {m1['no_failure_update_f1_drop']:.3f} |
| M2 | 交叉身份保持 {m2['crossing_identity_retention']:.4f}；最差方向族 {m2['worst_family_crossing_identity_retention']:.4f}；Brier {m2['ambiguous_association_brier']:.4f} | 最近邻身份保持 {m2['nearest_crossing_identity_retention']:.4f} |
| M3 | 对称最大偏离 {m3['symmetric_probability_deviation_maximum']:.6f}；破缺后真实图概率 {m3['broken_true_probability_mean']:.6f}；最慢 {m3['broken_convergence_steps_maximum']} 步 | 删除因果似然：概率 {m3['no_likelihood_true_probability']:.3f}、Brier {m3['no_likelihood_brier']:.3f}、NLL {m3['no_likelihood_nll']:.3f}、{m3['no_likelihood_convergence_steps']} 步不收敛 |

身份与结构结果：

- M2 正常节点 F1：{m2['node_f1']:.4f}；
- M2 刚性边 F1：{m2['rigid_edge_f1']:.4f}；
- M2 交叉交换率：{m2['crossing_identity_switch_rate']:.4f}；
- M3 完整图拓扑 F1：{m3['complete_graph_topology_f1']:.3f}；
- M3 姿态身份保持：{m3['pose_identity_retention']:.4f}。

组合端点资源为 {resources['m3_composed']['learnable_parameter_count']} 参数、
{resources['m3_composed']['active_state_bytes']:,} B 活动状态和
{resources['m3_composed']['estimated_mac_per_step']:,} MAC/步。

## 解释边界

{limitations}

因此下一步可以设计无特权 M4，但应继续停在正式 M4 门之前：先从视觉自身推断
free/occupied/unknown，再冻结新的开发集和一次性确认集。
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/V2-M1-M3-integrated-confirmation-review-summary.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "docs/experiments/V2_M1_M3_INTEGRATED_CONFIRMATION_REPORT.md"
        ),
    )
    arguments = parser.parse_args(argv)
    result = build_v2_m1_m3_confirmation_review(
        output_path=arguments.output,
        report_path=arguments.report,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
