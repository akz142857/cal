# 评审报告：随机化永久性专案冻结前评审 2026-08-08

范围：`main` @ `4e0c7c2`。被评审对象为 2026-07-29～07-31 落地、尚未进入任何
source lock 的随机化永久性栈（14 个新模块，`cal/model/stochastic_motion_filter.py`
+ `cal/evaluation/stochastic_permanence_*.py` / `permanence_*.py` /
`randomized_occlusion_world.py`），及其预注册草案
`docs/experiments/V2_L0_PERMANENCE_DIAGNOSIS_AND_PREREGISTRATION_DRAFT.md`。

方法：按 [`REVIEW_PLAN.md`](REVIEW_PLAN.md) v1.1 轨道 A 执行。R2/R3 清单评审与
红队攻击**并行且互不通气**（§3.1）；每条候选发现派**不知情验证者**独立复核
（§2.4），只保留 CONFIRMED / PLAUSIBLE。共 5 个独立评审会话 + 6 个盲验证会话，
全部仅使用 development seed（62003–62150 train / 62153–62378 eval），
**未触碰任何 validation / holdout seed**。

**判定：`block`**

阻断依据（§8）：红队攻击 4 成功——一个不实现目标机制的候选通过了全部冻结门。
按放行规则，红队任一攻击成功等价于 P0。另有两条独立 P0（草案与实现的门定义
不一致）。

---

## R0 准入

| 项 | 结果 |
| --- | --- |
| `uv run pytest` | **393 passed**（106 s），含 84 个永久性相关测试 |
| 工作树 | 干净（除本轮评审自身产出） |
| 冻结五要素（假设/指标/对照/split/停止规则） | **不齐**——草案 C.8 自述四项冻结阻断项未解决 |

R0 本可据此直接退回，但为给出完整的红队证据与冻结前修复清单，本轮继续执行。

## R1 锁状态表

```
31 份边车 -> 全部一致
V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL.json:     MISSING × 6
V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL_V2.json:  MISSING × 7
V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL_V3–V6:    DRIFT × 2 each
(未列出者 = 无漂移)
```

| 现象 | 涉及协议 | 归类 |
| --- | --- | --- |
| MISSING `calmodel/...` | 确认协议 V1、V2 | **已修订**：重命名前的历史链接，按 `CLAUDE.md` 不得改写 |
| DRIFT `entity_graph.py` / `v2_m1_m3_confirmation.py` | 确认协议 V3–V6 | **已修订**：被 V7 取代的修订链 |
| 无漂移 | 确认协议 **V7**（当前生效） | 锁健康 |
| 边车一致 | 全部 31 份 | 健康 |

**关键量化**：全仓库含 `locked_source_sha256` 的协议共 **7 份，全部是 M1–M3
确认协议**；永久性专案的协议（`V2_I1_P1_*.json`、`V2_P1_*.json`）中含源码锁的
数量为 **0**。G7 坐实。

本轮已修复 `CLAUDE.md` 中"当前生效协议为 V3"的过期口径（实为 V7）。

---

## 红队攻击台账

§2.2 八条攻击清单的执行结果。**攻击失败同样记录——它是本轮最有价值的正面证据。**

| # | 攻击 | 关键数字 | 成/败 |
| --- | --- | --- | --- |
| A1 | 跨 seed 不变量 | 104/104 布局唯一；单元格最大静态频率 0.279；`(k, seed 特征)` 无动力学预测 top1 **0.0137**（低于随机 0.038）；delta 查表 0.2488 < geometric 0.284 | **FAILED** ✅ |
| A1′ | 种子反演（子攻击） | 首观测指纹在 380 seed 中**完全唯一**；replay 精确复现 200 步隐藏轨迹 10/10 | **SUCCESS** → F9 |
| A2 | 纯位置先验 | query one-hot **0.542**、坐标 0.500、全几何 0.542、raw-sensor 探针 **0.545**（V8 固定世界为 0.875） | **FAILED** ✅ |
| A3 | 负样本几何 | 负样本 99.95% 在曼哈顿距离 1；纯奇偶规则平衡准确率 **0.99973** | **SUCCESS（非门控）** → F18 |
| A4 | 平凡基线 / 无信念候选 | GRU 基线 top1 0.048 vs 随机 0.041 → 失败；**无信念候选 18 门全过 `passed=True`** | **SUCCESS** → **F1** |
| A5 | 对照组不可构造 | V8 五个对照在门控管线中不存在；从未在随机化世界中构造过一次 | **SUCCESS** → F6 |
| A6 | 门名实不符 | `mean_at_least_development_floor` 恒真（`floor = 0.25 × 被检样本自身均值`） | **SUCCESS** → F5 |
| A7 | 锁绕过 | 篡改 `per_seed_per_bin`（oracle 全 0.0 / geometric 全 1.0）后产物**仍验证通过** | **SUCCESS** → F4 |
| A8 | 资源门规避 | 无超出既有记账的发现（125 参数 vs 限额 100 000） | **FAILED** ✅ |

### 正面证据（值得单独记录）

A1 + A2 联合证明：**V8 式几何捷径确实被消除了**。V8 时 raw-sensor 在
permanence 上 0.875 反超正式模型 0.802；随机化世界中同类探针塌到 0.50–0.55，
位置先验在场任务上贴着随机线（0.036–0.049 vs 随机 0.039），GRU 对照也在随机
线。随机化的设计目标达成了——这是作者本轮工作的实质成果，不因下述问题而
被否定。

---

## 发现台账

判定列：CONFIRMED / PLAUSIBLE 由不知情验证者独立给出；REJECTED 项见"争议与裁决"。

### P0（阻断，不修不得冻结）

| # | 判定 | 描述 | 证据 |
| --- | --- | --- | --- |
| **F1** | CONFIRMED | **无信念候选通过全部 18 个冻结确认门。** 常速外推 + 125 条目误差概率表（无转移核、无逐步后验传播、无"持续不可见"条件化），dev-train 40 seed 拟合、dev-eval 64 seed 评估：overall top1 **0.3903**（geometric 0.3382 / oracle 0.4217），6+ top1 **0.1575**（geometric 0.0748），`decision="pass_all_confirmatory_gates"`。验证者独立重建候选并复现，且其 geometric 表与冻结产物比对 **0 处不符** | `stochastic_permanence_benchmark.py:722`；闭合阈值 `:100-194` |
| **F2** | CONFIRMED | **草案 C.4 门表与实际实现的门系统交集为空集。** 草案 5 个门名（`formal_beats_raw_permanence` 等）在 `cal/` 中零出现；实现的 12 个确认门 + 6 个资源门在草案中零出现。照草案冻结 = 预注册一个从未实现的实验 | 草案 C.4 vs `benchmark.py:1047-1091`、`:92-99` |
| **F3** | CONFIRMED | 草案无任何数值阈值、无 ε/K、无多指标决策规则，草案 C.8 自述"仍然阻止正式冻结的事项"四项未解决 | 草案 C.4「阈值（评审定数）」、C.8 |

### P1（阻断，除非书面接受并写入"已知限制"）

| # | 判定 | 描述 | 证据 |
| --- | --- | --- | --- |
| **F4** | CONFIRMED（3/3 腿） | **phase-0 产物验证器不从原始数据重算门布尔值**，只校验结构与 `passed == all(details)`。实测：把 `per_seed_per_bin` 改成 oracle 全 0.0 / geometric 全 1.0，产物仍验证通过。且无任何提交基线哈希覆盖该栈——`source_lock` 是写入时从活文件重算并嵌入的，`verify_source_lock` 仅被测试用例调用 | `stochastic_permanence_artifacts.py:838-848,1658-1688`；`phase0.py:187-220` |
| **F5** | CONFIRMED | **`mean_at_least_development_floor` 恒真**：floor = 0.25 × 被检样本自身均值，化简为 `mean >= 0`；`mean <= 0` 会在 `:3422→1439` 先抛异常，故可达值集为 `{True}`。同款反模式的历史先例是 `v1_development_matches_v1_protocol`。**确认路径不共享此缺陷**（floor 取自开发集产物，独立） | `benchmark.py:624,76,3446-3448`；确认路径 `:795,1041-1044` |
| **F6** | CONFIRMED | **V8 死因所在的 raw-sensor 对照从门控管线中消失**。草案 C.3 承诺保留 `raw_sensor`/`assume_all_visible`/`time_shuffled`/`identity_scrambled`/`random_labels`，实现的比较集是 `("candidate","oracle","geometric","uniform","old_i1")`。精化：raw-sensor 仍存在于**非门控**诊断 `permanence_geometry_diagnostic.py:385-395`，故准确表述是"降级为非门控且未在草案中说明"，非"完全删除" | `benchmark.py:69` |
| **F7** | CONFIRMED | 草案绑定**已被取代的 V1 注册表**（56 seed=40+16、扫到 62193、digest `e31929f7`），而冻结 Phase-0 V10 与代码默认绑定 V2 注册表（104 seed=40+64、扫到 62378、digest `cd868d16`）；草案引用无版本号路径，未声明何者权威 | 草案 C.2/C.8 vs `V10.json`、`phase0.py:45-47` |
| **F8** | CONFIRMED | G7：14 个新模块不在任何 `locked_source_sha256` 中；含源码锁的 7 份协议全是 M1–M3 确认协议 | R1 实测 |

### P2（可 `pass_with_conditions`，须有责任人与期限）

| # | 判定 | 描述 |
| --- | --- | --- |
| F9 | CONFIRMED | **种子反演**：隐藏机动 RNG 为 `default_rng(seed + 90_000)`，布局是同一 seed 的确定函数 → 观测布局可反推 seed 并精确重放隐藏轨迹。custody 只做承诺保密，**不要求 seed 不可枚举**，开发注册表用 62000–70000 连续小整数。不破坏当前任何冻结门，是未来留出的威胁模型缺口。缓解：改用 `HMAC(custodian_salt, seed)` 派生隐藏流 |
| F10 | CONFIRMED | `run_benchmark` / `gru_capacity_sweep` / `run_diagnostic` **不校验 train/eval seed 是否重叠**（守卫只在 `run_candidate_lifecycle` 等上层）。`--train-seeds 101` 即可让两者交叠；实测污染后位置先验 top1 从 0.044 跳到 0.243 |
| F11 | CONFIRMED | CLI 可用 `--turn-probability` / `--steps` / `--warmup` 覆盖注册表绑定参数，**但产物仍盖注册表 digest provenance**；且该 CLI 跳过 phase0/scan 强制的覆盖契约与 digest 复现检查 |
| F12 | CONFIRMED | **27 份已提交结果 JSON 的 provenance 为 `git_dirty=true` 或无 commit**，含两份已消费 holdout 摘要与多份 `authorize_*` 授权产物。关键终局证据（V8、I1 v4 holdout）干净。历史产物不可重跑，只能记为已知限制 |
| F13 | CONFIRMED | 干净检出上运行 README 推荐的 `uv run cal-index --results results`，会把已提交的 `INDEX.json` 从 **680 条截到 22 条**——committed INDEX 引用大量未入库的本地产物。新环境执行该命令会静默摧毁历史索引（本轮已复原） |
| F14 | CONFIRMED | **整个永久性专案在 README / RESEARCH_STATUS 中引用数为 0**（三份文档均未链接，两个新 console script 未进命令序列）；RESEARCH_STATUS 停在 2026-07-28 而代码提交至 07-31。`CLAUDE.md` 称 README "kept in sync with what has actually been run"，对该栈已不成立 |
| F15 | CONFIRMED | V8 结果用 `result_schema_version=2`，而 `require_authorization` 硬要求 `== 1`。当前无下游消费方，非活跃 bug，但校验器无法校验 schema-2 产物 |
| F16 | CONFIRMED | `autonomous_successors` 在**非二值** static 概率下不是精确的拓扑混合（数值探针 L1 偏差 **0.0231**）。当前调用方全喂二值网格故无害，但模块已预留学习型 `static_probability` 槽位，接入时"精确推断"假设会静默失效 |
| F17 | CONFIRMED | `GridSpec` 把 `grid_size=25, arena_low=7, arena_high=17` 作为**默认值**复制自评估世界且无运行时交叉校验；上游常量改动后，滤波器会静默把界外单元当作确定墙体 |

### P3（记录，不阻断）

| # | 描述 |
| --- | --- |
| F18 | **pairwise 负样本奇偶可分（0.99973）**。经验证：`.negative` 字段**不进入任何冻结门**（门控指标全部走整个隐藏场 `candidate_cells`），故仅为非门控诊断问题。**附禁令：pairwise 正/负构造永不得重新进入任何门** |
| F19 | 多个恒真门：`fully_detached_safe`（`s_max >= H*E*K` 而 `s_max` 定义即 `H*E*K`）、`shared_expansion_workspace_safe`（`12*k_max` 与自身比较）、`formal_research_budget_declared`、`branch_evidence_accounting`（残差恒为 0.0） |
| F20 | 容量验证器与 runner 容差不对称（`np.isclose` rtol 1e-5 vs `math.isclose` abs_tol 1e-12），导致**诚实的 no-go 产物无法序列化**——`--turn-probability 0.45` 会以"registry provenance mismatch"崩溃而非写出 no-go |
| F21 | 零质量状态保留在后验中（`packed_retained_support` 与 `reference_support` 语义不可比）；`maximum_step_pruned_mass` 只写不读；`bayesian_no_detection_update` 无生产调用方；`replace_factor_atomic` 不校验 code 唯一性；`maximum_tv_checkpoint` 记录最后而非最先达到最大值的步 |
| F22 | `permanence_turn_probability_scan.run_scan` 硬编码 `steps=200, warmup=12` 而非读 `registry["coverage_contract"]`；两者今天一致，未来契约变更会静默失步 |
| F23 | RNG 流间距 40 000 的假设无断言：`layout=seed`、`hidden=seed+90_000`、`action=seed+50_000`，相距 40 000 的两个世界会出现流重合。当前范围安全 |

---

## 争议与裁决

| 事项 | 红队主张 | 验证者主张 | 裁决 |
| --- | --- | --- | --- |
| 容量产物验证器把 `shared_expansion_workspace_safe` / `atomic_overflow_safe` / `registry_turn_probability` 硬编码为 `True` 是绕过路径 | 可绕过，P1 | **REFUTED**：三个字面 `True` 是同一验证器内更严格跨字段检查的必然推论——workspace 用**相等性**（严于门的 `>=`）、turn probability 用 **1e-12 绝对容差**（严于 runner 的 rtol 1e-5）、overflow 探针字段即门的底层数据。篡改实测：每一种可表示的违规都被拒绝 | **采纳验证者**。P1 主张驳回，保留 P3 残余（F20 容差不对称）。R3 管线扫描**独立收敛**到同一判断，无需裁决人介入 |

这是本轮唯一的红队—验证者分歧，且在证据层面解决，未触发 §8 的"相持不下判 block"。

---

## 副作用

本轮评审**未改动任何机制代码**，因此无机制副作用、无需重跑任何已发表数字。
唯一的仓库改动是文档口径更正：

| 文件 | 改动 | 性质 |
| --- | --- | --- |
| `CLAUDE.md` | 当前生效确认协议 V3 → **V7**；V1/V2 superseded → **V1–V6** | 文档更正，不影响任何结果 |

评审过程中运行过 `uv run cal-index`（产生 F13），已用 `git checkout` 复原。

---

## 结论与放行条件

**判定：`block`。不得冻结协议，不得授权消费任何留出。**

本轮最重要的结论不是"代码有 bug"——恰恰相反，两份 R3 扫描**均未发现 P0/P1
代码缺陷**，历史三大 bug（LRU 淘汰、复合键、构造函数完整性）的修复完好，
指标实现、遮挡区间记账、38 400 例核对齐门全部经独立核验正确。

真正的问题在**协议层**：随机化世界成功消灭了 V8 的点质量几何捷径（A1/A2 给出
了扎实的正面证据），但**门系统检测不到"平滑化"的无信念捷径**。根因比"阈值
定低了"更深——`_rank` 把每张图投影到隐藏候选场并按场内质量重新归一化
（`permanence_forward_benchmark.py:544-633`），也就是说**"只在仍然隐藏的格子上
保留质量"这一信念滤波最主要的行为特征，是由评分框架代劳的，不是候选做的**；
同时点质量的 geometric 基线因场投影落空而被严重惩罚（NLL 13.8、位置误差 20），
反过来夸大了 candidate-vs-geometric 的对比。

具体讽刺之处：随机化遮挡世界的引入理由（`randomized_occlusion_world.py:1-30`）
正是"旧世界让无信念外推器通过"。随机化打掉了点质量捷径，却没有打掉平滑捷径。

### 冻结前必须完成（P0/P1）

1. **重新设计确认门，使其测量目标机制本身**——例如对长空观测段后的后验重归一化、
   对"持续未检测"的条件化响应设门；或大幅提高闭合阈值至逼近滤波器独有的增益。
   最低要求：把本报告的无信念候选作为**新的强制对照基线**加入门系统，
   `formal_beats_belief_free_smearing` 必须成为门（F1）。
2. 重写草案的预注册半部，使其与实际实现的 `stochastic_permanence` 契约一致，
   或明确声明由 `V2_I1_STOCHASTIC_PERMANENCE_PLAN.md` 取代（F2、F3）。
3. 让 phase-0 验证器**从 `per_seed_per_bin` 重算门布尔值**；为永久性栈建立
   提交基线哈希并在运行时校验（F4、F8）。
4. 修复 `mean_at_least_development_floor`：floor 必须来自独立样本或固定常量（F5）。
5. 决定 raw-sensor / assume-all-visible 对照的去向——恢复为门，或在草案中
   写明降级理由；并在 development split 上**实跑一遍全部对照的构造代码**证明
   可构造（F6）。
6. 统一注册表版本绑定，声明权威文件（F7）。

### 建议同时处理（P2）

F9 的 HMAC 派生（在留出 seed 生成前改成本最低）、F10 的 seed 重叠断言、
F14 的文档同步。F12/F13 属历史遗留与工具缺陷，建议记录后择期。

### 未修复项去向

F16/F17 留待学习型 `static_probability` 接入时处理；F18–F23 记录在案不阻断，
其中 F18 附带一条永久禁令。V2-I1 第五层摩擦仍是独立的机制设计问题，与本轮
无关（见 `docs/experiments/V2_I1_INTEGRATION_REPORT.md`）。

---

本报告按 [`REVIEW_PLAN.md`](REVIEW_PLAN.md) §9 为**不可在位修改**的历史产物。
后续更正以新增"补充评审"文档的形式追加，并在两份文档之间互相引用。
