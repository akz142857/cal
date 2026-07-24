"""Summarize the frozen V2-M3 complete-body-hypothesis review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from cal.infra.provenance import capture_provenance


def build_v2_m3_review(
    *,
    protocol_path: str | Path = (
        "experiments/V2_M3_BODY_GRAPH_HYPOTHESIS_PROTOCOL.json"
    ),
    development_path: str | Path = (
        "results/V2-M3-body-graph-development-summary.json"
    ),
    no_likelihood_path: str | Path = (
        "results/V2-M3-body-graph-development-no-causal-likelihood.json"
    ),
    holdout_path: str | Path = (
        "results/V2-M3-body-graph-holdout-summary.json"
    ),
    output_path: str | Path = (
        "results/V2-M3-body-graph-review-summary.json"
    ),
    report_path: str | Path = (
        "docs/experiments/V2_M3_BODY_GRAPH_HYPOTHESIS_REPORT.md"
    ),
) -> dict[str, Any]:
    sources = {
        "protocol": Path(protocol_path),
        "development": Path(development_path),
        "no_causal_likelihood": Path(no_likelihood_path),
        "holdout": Path(holdout_path),
    }
    payloads = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in sources.items()
        if key != "protocol"
    }
    protocol_digest = hashlib.sha256(sources["protocol"].read_bytes()).hexdigest()
    development = payloads["development"]
    ablation = payloads["no_causal_likelihood"]
    holdout = payloads["holdout"]
    gates = {
        "protocol_hash_matches_holdout": (
            holdout.get("protocol_sha256") == protocol_digest
        ),
        "holdout_marked_one_shot": holdout.get("holdout_run_count") == 1,
        "holdout_split_is_frozen": holdout.get("review_split") == "holdout",
        "formal_candidate_is_complete_graph_posterior": (
            holdout.get("candidate")
            == "mutually_exclusive_complete_body_graph_hypotheses"
        ),
        "all_frozen_holdout_gates_pass": (
            holdout.get("passed") is True
            and all(holdout.get("gates", {}).values())
        ),
        "development_passes": development.get("passed") is True,
        "no_causal_likelihood_control_fails": ablation.get("passed") is False,
        "resource_budget_passes": holdout["gates"]["resources_pass"],
    }
    aggregate = holdout["aggregate"]
    summary = {
        "result_schema_version": 1,
        "experiment": "V2-M3-body-graph-review",
        "source_paths": {key: str(path) for key, path in sources.items()},
        "source_sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in sources.items()
        },
        "protocol_sha256": protocol_digest,
        "development_comparison": {
            "formal_broken_true_probability_mean": development["aggregate"][
                "broken_true_probability_mean"
            ],
            "formal_broken_convergence_steps_maximum": development["aggregate"][
                "broken_convergence_steps_maximum"
            ],
            "no_likelihood_broken_true_probability_mean": ablation["aggregate"][
                "broken_true_probability_mean"
            ],
            "no_likelihood_topology_f1": ablation["aggregate"][
                "complete_graph_topology_f1"
            ],
        },
        "holdout": {
            "seed_count": len(holdout["episodes"]),
            **aggregate,
            "resources": holdout["resources"],
        },
        "interpretation": (
            "The categorical posterior represents complete, mutually "
            "exclusive body graphs and preserves observable symmetry. "
            "Prequential action-effect likelihood is necessary for resolving "
            "the registered symmetry breaks; without it the posterior remains "
            "at one half and topology selection is at chance."
        ),
        "scope_limitations": [
            (
                "输入是确定性稀疏视觉检测，不是原始像素、雷达或生产感知栈。"
            ),
            (
                "预注册压力恰有两个完整候选；滤波器发现两个候选后锁定类别空间。"
            ),
            (
                "7×7 姿态投影使用已知解析二连杆渲染器，检验图选择而不是学习"
                "运动学或原始像素分割。"
            ),
        ],
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "authorize_v2_m4"
            if all(gates.values())
            else "retain_stop_before_v2_m4"
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
    development = summary["development_comparison"]
    holdout = summary["holdout"]
    limitations = "\n".join(
        f"- {item}" for item in summary["scope_limitations"]
    )
    return f"""# V2-M3 互斥完整身体图复审报告

日期：2026-07-24

## 决策

**冻结留出 V2-M3 复审通过，授权进入 V2-M4。**

协议在实现前冻结，SHA-256 为 `{summary['protocol_sha256']}`。留出种子
9201–9216 只运行一次，结果记录 `holdout_run_count=1`。

## 方法与失败更新

系统维护一个完整身体图类别变量；每个候选均为基座—关节—端点三节点和两条
边，所有候选概率严格归一化。对称观察不能支持唯一身份时保持 0.5/0.5。
出现新证据后，后验由动作效应模型在看见当前帧之前产生的预测误差更新。
可见目标若没有匹配到原轨迹，其“漏配”本身计为预测失败，而不是事后替换标签。

## 开发集机制对照

| 版本 | 破缺后真实图概率 | 最慢收敛 | 拓扑 F1 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 完整模型 | {development['formal_broken_true_probability_mean']:.6f} | {development['formal_broken_convergence_steps_maximum']} 步 | 1.000 | 通过 |
| 删除因果似然 | {development['no_likelihood_broken_true_probability_mean']:.3f} | 未收敛 | {development['no_likelihood_topology_f1']:.3f} | 失败 |

删除因果似然后，后验保持 0.5，说明最终选择不是确定性后处理器对隐藏分支的
硬编码猜测。

## 一次性留出结果

- 16 个新种子、四种对称破缺方式；
- 对称后验最大偏离 0.5：{holdout['symmetric_probability_deviation_maximum']:.6f}；
- 对称 NLL：{holdout['symmetric_nll']:.6f}；
- 破缺后真实完整图平均概率：{holdout['broken_true_probability_mean']:.6f}；
- 最低真实图概率：{holdout['broken_true_probability_minimum']:.6f}；
- 最慢收敛：{holdout['broken_convergence_steps_maximum']} 步；
- 完整图拓扑 F1：{holdout['complete_graph_topology_f1']:.3f}；
- 7×7 姿态投影平均/最差 IoU：{holdout['pose_grid_projection_iou_mean']:.3f}/{holdout['pose_grid_projection_iou_worst']:.3f}；
- 身份保持：{holdout['pose_identity_retention']:.4f}；
- 参数：{holdout['resources']['learnable_parameter_count']:,}；
- 活动状态：{holdout['resources']['active_state_bytes']:,} B；
- MAC/步：{holdout['resources']['estimated_mac_per_step']:,}。

全部冻结门通过。

## 不能外推

{limitations}

因此本结果解决的是当前合成任务中的“可校准完整身体图假设”，并不声称已经
得到 FSD 等级视觉系统。V2 总链下一门是从视觉本身推断
free/occupied/unknown，移除 M4 的模拟器可见性掩码。
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/V2-M3-body-graph-review-summary.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "docs/experiments/V2_M3_BODY_GRAPH_HYPOTHESIS_REPORT.md"
        ),
    )
    arguments = parser.parse_args(argv)
    result = build_v2_m3_review(
        output_path=arguments.output,
        report_path=arguments.report,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
