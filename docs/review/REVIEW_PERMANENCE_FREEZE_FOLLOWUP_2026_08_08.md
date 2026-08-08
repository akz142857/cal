# 补充评审：F4 / F8 修复与 F5 顺序决定 2026-08-08

上游：[`REVIEW_PERMANENCE_FREEZE_2026_08_08.md`](REVIEW_PERMANENCE_FREEZE_2026_08_08.md)
（判定 `block`）。该报告按方案 §9 为不可在位修改的历史产物，故本次修复与
决定以补充文档形式追加。

本轮处理范围仅限**与 F1 无关、无需研究判断**的验证层缺陷。**总判定仍为
`block`——F1/F2/F3 三条 P0 未动，不得冻结协议、不得授权消费留出。**

---

## 已修复

### F4（P1）验证器现在从原始数据重算门布尔值

`cal/evaluation/stochastic_permanence_artifacts.py` 新增
`_verify_reference_health_recomputation`，在
`_validate_reference_health_power_artifact` 中调用：用产物自带的
`per_seed_per_bin` 重跑 `build_reference_health`，逐门比对
`mean_at_least_development_floor` 与 `one_sided_99_lower_positive`，不符即拒。

修复前，验证器只校验结构与 `passed == all(details)`，记录的布尔值是**不可证伪
的自我声明**。

**双向验证**

| 方向 | 结果 |
| --- | --- |
| committed V10 仍通过 | ✅ digest 不变 `ede55bb803a2f637` |
| 红队攻击 A7 的原样篡改（`per_seed_per_bin` 改成 oracle 全 0.0 / geometric 全 1.0） | ✅ 被拒：`reference-health gate does not match recomputed evidence: overall/top1_accuracy` |

**附带产出——V10 的追溯审计通过**。重算七个门的置信下界与记录完全一致，
且全部为正、有余量：

| 门 | 记录 | 重算 | 单侧 99% 下界 |
| --- | :---: | :---: | ---: |
| overall/top1_accuracy | True | True | +0.06490 |
| overall/categorical_nll | True | True | +1.78357 |
| overall/brier | True | True | +0.00911 |
| 4-5/top1_accuracy | True | True | +0.06630 |
| 6+/top1_accuracy | True | True | +0.09612 |
| 6+/categorical_nll | True | True | +1.15628 |
| 6+/brier | True | True | +0.00320 |

即：**已提交的开发产物是诚实的**。此前无人能证明这一点，现在可以。

### F8（P1）source lock 接上线

`source_lock` / `verify_source_lock` 本就实现正确，问题是**从未对活文件调用过**
（除测试夹具外无调用方）。但 `verify_source_lock` 一有差异就抛异常，无法用于
巡检已提交产物——开发产物的 lock 是**关于过去的声明**，后续修复导致的漂移是
预期的，不是错误。

因此新增 `audit_artifact_source_lock(path, *, root)`：返回逐文件判定
（`matched` / `drifted` / `missing`）而非抛异常，使调用方可以要求漂移被
**声明**，而不是要求漂移不存在。

配套测试 `test_current_development_artifact_source_drift_is_acknowledged`：
当前世代产物（V10、PHASE_R V3）的漂移必须落在显式声明的
`ACKNOWLEDGED_SOURCE_DRIFT` 集合内，否则失败并提示"regenerate the artifact
or acknowledge the drift"。**反向验证**：向 `benchmark.py` 追加一行注释即触发
失败并指名该文件，复原后恢复通过。

首次巡检结果（本身就是一张有用的锁状态表）：

| 产物 | 文件数 | 漂移 |
| --- | ---: | --- |
| PHASE0 **V10**（当前） | 20 | 仅 `stochastic_permanence_artifacts.py` |
| PHASE_R **V3**（当前） | 19 | 仅 `stochastic_permanence_artifacts.py` |
| PHASE0 V1 / V9（已取代） | 20 | 5–6 个文件 |
| PHASE_R V1 / V2（已取代） | 17 | 3–5 个文件 |

当前世代产物的唯一漂移就是本次 F4 修复本身，已在测试中声明并注明原因；
被取代产物漂移更多，符合修订链预期。

**注意**：`ACKNOWLEDGED_SOURCE_DRIFT` 是临时项，产物在 V11 重新生成后必须清空。

---

## F5 推迟：一个顺序决定，不是放弃

F5（`mean_at_least_development_floor` 恒真）**本轮未修**，理由是成本顺序而非
难度：

1. `floor` 字段有**两个用途**。phase-0 的自比较是 bug；但确认路径用**开发集**
   floor 去卡**确认集**均值（`benchmark.py:795,1041-1044`）是独立且正确的，
   不能动。故只能删掉 phase-0 那个恒真分量。
2. 删除会改变产物 schema 的 key 集合，而验证器对 schema 版本是**单值强校验**
   （`artifact_schema_version != _expected_schema_version` 即拒），因此必须
   2→3 并**重新生成 phase-0 产物**。
3. V10 的生成参数是 `--simulation-trials 1024 --power-bootstrap-samples 2000`，
   覆盖 104 个 development seed，重跑成本可观。
4. **F1 的门重设计必然要重新生成同一个产物**。现在单独为 F5 跑一遍等于白跑。

**决定**：F5 并入 F1 的 V11 修订一次性完成。届时 V10 作为被取代的历史链接
保留（沿用 V9→V10 既有约定），`ACKNOWLEDGED_SOURCE_DRIFT` 同时清空。

风险评估：F5 在被修之前**不产生错误结论**——它只是提供零保护，而同一门中
`one_sided_99_lower_positive` 分量是实质的，且经 F4 后已被独立重算校验。

---

## 改动清单与回归

| 文件 | 改动 |
| --- | --- |
| `cal/evaluation/stochastic_permanence_artifacts.py` | 新增 `_verify_reference_health_recomputation`（F4）、`audit_artifact_source_lock`（F8） |
| `tests/test_stochastic_permanence_artifacts.py` | 新增漂移声明测试与审计器单测（F8） |
| `CLAUDE.md` | 当前生效确认协议 V3 → V7；V1/V2 superseded → V1–V6 |

回归：`uv run pytest` **396 passed**（修复前 393，新增 3）。

**副作用**：本次改动**不改变任何已发表数字**。F4 只增加校验、不改变任何写入
值；已提交产物的 digest 保持不变（V10 仍为 `ede55bb803a2f637`）。唯一的实质
副作用是当前世代产物的 source_lock 失配，已如上声明。

---

## 冻结前仍未完成

F1、F2、F3（P0）与 F5、F6、F7（P1）未动，详见上游报告的"冻结前必须完成"。
其中 **F9（HMAC 种子派生）有硬顺序约束：必须在生成任何留出 seed 之前完成**，
否则反演缺口将固化进那批留出，事后改代码无法补救。
