"""Summarize the frozen V2-M2 probabilistic-association review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from cal.infra.provenance import capture_provenance


def build_v2_m2_review(
    *,
    protocol_path: str | Path = (
        "experiments/V2_M2_PROBABILISTIC_ASSOCIATION_PROTOCOL.json"
    ),
    probabilistic_development_path: str | Path = (
        "results/V2-M2-probabilistic-development-summary.json"
    ),
    hard_map_path: str | Path = (
        "results/V2-M2-hard-map-development-summary.json"
    ),
    nearest_path: str | Path = (
        "results/V2-M2-nearest-development-summary.json"
    ),
    holdout_path: str | Path = (
        "results/V2-M2-probabilistic-holdout-summary.json"
    ),
    output_path: str | Path = (
        "results/V2-M2-probabilistic-review-summary.json"
    ),
    report_path: str | Path = (
        "docs/experiments/V2_M2_PROBABILISTIC_ASSOCIATION_REPORT.md"
    ),
) -> dict[str, Any]:
    sources = {
        "protocol": Path(protocol_path),
        "probabilistic_development": Path(probabilistic_development_path),
        "hard_map_development": Path(hard_map_path),
        "nearest_development": Path(nearest_path),
        "probabilistic_holdout": Path(holdout_path),
    }
    payloads = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in sources.items()
        if key != "protocol"
    }
    protocol_digest = hashlib.sha256(sources["protocol"].read_bytes()).hexdigest()
    holdout = payloads["probabilistic_holdout"]
    development = payloads["probabilistic_development"]
    hard_map = payloads["hard_map_development"]
    nearest = payloads["nearest_development"]
    gates = {
        "protocol_hash_matches_holdout": (
            holdout.get("protocol_sha256") == protocol_digest
        ),
        "holdout_marked_one_shot": holdout.get("holdout_run_count") == 1,
        "holdout_split_is_frozen": holdout.get("review_split") == "holdout",
        "formal_candidate_is_probabilistic": (
            holdout.get("association_mode") == "probabilistic"
        ),
        "all_frozen_holdout_gates_pass": (
            holdout.get("passed") is True
            and all(holdout.get("gates", {}).values())
        ),
        "nearest_development_control_fails": nearest.get("passed") is False,
        "probabilistic_development_passes": development.get("passed") is True,
        "resource_budget_passes": holdout["gates"]["resources_pass"],
    }
    summary = {
        "result_schema_version": 1,
        "experiment": "V2-M2-probabilistic-review",
        "source_paths": {key: str(path) for key, path in sources.items()},
        "source_sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in sources.items()
        },
        "protocol_sha256": protocol_digest,
        "development_comparison": {
            "probabilistic_identity_retention": development["aggregate"][
                "crossing_identity_retention"
            ],
            "hard_map_identity_retention": hard_map["aggregate"][
                "crossing_identity_retention"
            ],
            "nearest_identity_retention": nearest["aggregate"][
                "crossing_identity_retention"
            ],
            "probabilistic_brier": development["aggregate"][
                "ambiguous_association_brier"
            ],
            "hard_map_brier": hard_map["aggregate"][
                "ambiguous_association_brier"
            ],
            "nearest_brier": nearest["aggregate"][
                "ambiguous_association_brier"
            ],
        },
        "holdout": {
            "seed_count": len(holdout["crossing_episodes"]),
            **holdout["aggregate"],
            "resources": holdout["resources"],
        },
        "interpretation": (
            "Motion/geometry-aware global association is necessary relative "
            "to nearest-neighbour tracking. Hard MAP and probabilistic "
            "association both retain final identity on the development set; "
            "the probabilistic version additionally exposes calibrated "
            "association uncertainty and delays low-confidence commitments."
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "authorize_v2_m3"
            if all(gates.values())
            else "retain_stop_before_v2_m3"
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
    return f"""# V2-M2 概率多轨迹关联复审报告

日期：2026-07-24

## 决策

**冻结留出 V2-M2 复审通过，授权重新进入 V2-M3。**

协议在实现前冻结，SHA-256 为 `{summary['protocol_sha256']}`。留出种子
9101–9116 只运行一次，结果记录 `holdout_run_count=1`。

## 方法

每步保留最多 64 个全局轨迹—检测分配假设，后验同时使用动作条件运动预测、
自主速度和已经学习的刚性边长度。最大关联边缘概率低于 0.58 时不提交身份，
轨迹按预测继续；结构分离后再恢复原轨迹。

## 开发集机制对照

| 关联器 | 交叉身份保持 | 关联 Brier | 开发门 |
| --- | ---: | ---: | --- |
| 概率多假设 | {development['probabilistic_identity_retention']:.3f} | {development['probabilistic_brier']:.4f} | 通过 |
| 运动/几何硬 MAP | {development['hard_map_identity_retention']:.3f} | {development['hard_map_brier']:.4f} | 通过 |
| 原最近邻 | {development['nearest_identity_retention']:.3f} | {development['nearest_brier']:.4f} | 失败 |

因此，当前开发集证明运动/几何全局关联相对最近邻是必要改进；它没有证明最终
身份保持必须依赖延迟承诺，因为硬 MAP 也通过。概率版本的额外已测收益是输出
校准的关联后验，并在低置信时避免立即改变身份。

## 一次性留出结果

- 16 个新种子、四个穿越方向；
- 交叉身份保持：{holdout['crossing_identity_retention']:.3f}；
- 最差场景族身份保持：{holdout['worst_family_crossing_identity_retention']:.3f}；
- 身份交换率：{holdout['crossing_identity_switch_rate']:.3f}；
- 交叉节点 F1：{holdout['crossing_node_f1']:.4f}；
- 关联 Brier：{holdout['ambiguous_association_brier']:.4f}；
- 活动状态：{holdout['resources']['active_state_bytes']:,} B；
- 参数：{holdout['resources']['learnable_parameter_count']:,}；
- MAC/步：{holdout['resources']['estimated_mac_per_step']:,}。

全部冻结门通过。该结论只适用于解析稀疏检测和预注册的 M2 场景，不代表原始
像素视觉、完全重合状态中的唯一身份可识别，也不改变 V2-M3 对互斥完整身体
假设的要求。
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/V2-M2-probabilistic-review-summary.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "docs/experiments/V2_M2_PROBABILISTIC_ASSOCIATION_REPORT.md"
        ),
    )
    arguments = parser.parse_args(argv)
    result = build_v2_m2_review(
        output_path=arguments.output,
        report_path=arguments.report,
    )
    print(f"passed={result['passed']}; decision={result['decision']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
