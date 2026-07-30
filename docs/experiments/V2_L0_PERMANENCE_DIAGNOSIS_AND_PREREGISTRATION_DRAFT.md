# V2-L0 永久性失败根因诊断 + 新留出预注册（草案）

> **状态：非冻结草案，供评审。** 本文档不含任何 SHA-256 source lock，不构成
> 已冻结协议；它只固化根因诊断，并提出新环境设计与门定义，等待评审（含潜在
> 第三方评审）后再由 runner 生成冻结 `*.json` + `.sha256` + `locked_source_sha256`。
> 在冻结之前，本文档中的一切阈值、seed 段和门都可修改。

日期：2026-07-28
上游证据：`results/V2-L0-language-readout-holdout-v8.json`
（`decision=stop_and_report`，SHA-256
`ae5ad9d4ef457d22680dc30048bf8e0421f5e708c724351b8751f8061a2d9d04`）
关联：[research status](../../RESEARCH_STATUS.md)、
[L0 报告](V2_L0_LANGUAGE_READOUT.md)、[里程碑评估 1.1.0](../MilestoneSummary-1.1.0.md)

---

## A. 根因诊断（一手代码 + 冻结结果）

### A.1 现象

V8 唯一留出上，正式实体图在 permanence 上是**唯一被平凡基线反超**的组别。
逐对照组平衡准确率（直接取自冻结 JSON 的八个条件）：

| 条件 | self | spatial | identity | **permanence** | macro |
|---|---:|---:|---:|---:|---:|
| formal_entity_graph（正式模型） | 0.999 | 0.875 | 0.885 | **0.802** | 0.890 |
| raw_sensor（裸传感器） | 0.500 | 0.787 | 0.500 | **0.875** | 0.665 |
| no_action（删动作） | 0.862 | 0.820 | 0.944 | **0.885** | 0.878 |
| assume_all_visible（假设全可见） | 0.960 | 0.845 | 0.996 | **0.854** | 0.914 |
| time_shuffled | 0.501 | 0.644 | 0.633 | 0.635 | 0.603 |
| identity_scrambled | 0.999 | 0.875 | 0.162 | 0.802 | 0.710 |
| referent_swapped | 0.999 | 0.661 | 0.115 | 0.146 | 0.480 |
| random_labels | 0.506 | 0.485 | 0.455 | 0.479 | 0.481 |

三个失败门：`formal_beats_raw_permanence=false`、
`all_visible_permanence_fails=false`、`identity_scramble_integrity_pass=false`。

### A.2 根因：固定几何 + 确定性隐藏动力学 = 跨 seed 稳定的几何捷径

证据在 `cal/evaluation/v2_i1_integration.py:38-63`：

```python
# 遮挡屏固定在 x=10 列、rows 9-15、row 12 为门洞；跨所有 seed 仅 1 格有 ±1 jitter
self.static = { *((10, y) for y in range(9, 16) if y != 12), (15, 9 + jitter), (14, 16) }
# distractor A 恒定速度沿 row 12 穿门进入"阴影带"，方向仅由 seed 奇偶决定
self.velocity_a = np.asarray((1 if seed % 2 == 0 else -1, 0))
```

因此隐藏物体 A 被遮 `k` 步后所在格子，是 `(hidden_steps, seed 奇偶)` 的
**确定函数**——世界几乎没有真实的永久性不确定性，只是一次线性外推。

- **探针是 train/eval seed 分离的**（`v2_l0_language_readout.py:3393` 的
  `train_seeds` vs `evaluation_seeds`），但**遮挡几何跨 seed 不变**，所以在
  train seed 上学到的几何/位置先验能完美迁移到 holdout seed。
- **raw_sensor 特征**（`_raw_features`，`:1125`）包含整张占据栅格 + 查询格
  one-hot 位置。虽然被遮格的 `sensed` 值对正负样本都为 0，但**查询 one-hot 直接
  编码"问的是哪个格子"**，配合固定遮挡布局，线性探针可学到"这个门洞几何 + 这个
  查询格 ⇒ 隐藏物体在此"。这是记忆固定世界的空间先验，不是永久性推理。
- **permanence 负样本**（`:1500`）取"离正样本最近的、不可见且无真值的非静态格"，
  与正样本仅相差很小的曼哈顿距离——查询 one-hot 足以按位置先验把二者分开。

### A.3 为什么正式模型反而更差

正式实体图特征（`_graph_features`，`:1098`）读的是**智能体自己的概率占据信念**。
在一个本可由固定几何外推干净解决的任务上，这层信念是带噪的估计，于是：

- **formal(0.802) < raw(0.875)**：正式模型用带噪信念替换了干净的几何外推 → 减分；
- **no_action(0.885) ≥ formal**：去掉动作使信念更平滑，更贴近几何先验 → 反而更高；
- **all_visible(0.854)、`all_visible_permanence_fails=false`**：即使假设"从不遮挡"
  仍有 0.85——**本该证明"你需要遮挡推理"的对照，反而证明了"你不需要"**。

### A.4 结论与可证伪断言

> permanence 门失败**主要是环境/任务设计缺陷**，不（仅）是模型缺陷：这套探针
> 无法区分"真·物体永久性推理"和"跨 seed 不变的固定几何外推"。

**可证伪断言：** 构造一个只吃"遮挡布局 + 末见位置 + 速度 + hidden_steps、不含
任何信念"的**纯几何外推预测器**。若它在 permanence 上达到 ≈0.85（≈ 与 formal/raw
同级）或更高，则几何捷径被经验坐实；若它显著低于 formal，则本诊断被推翻。

### A.5 确认实验结果（已运行，仅未消费 seed 60000+）

脚本：`cal/evaluation/permanence_geometry_diagnostic.py`（非门控分析，无冻结协议、
无 source lock、不产出证据、绝不触碰任何已消费 holdout）。它在**当前**环境上重建
与 L0 readout 完全一致的 world、`hidden_steps` 规则和 permanence 正负样本构造，
用全新种子 train 60000–60039 / eval 60100–60119。复现命令：

```bash
uv run python -m cal.evaluation.permanence_geometry_diagnostic
```

| 预测器 | permanence 平衡准确率 |
|---|---:|
| **纯几何解析外推（无训练、无信念）** | **1.0000** |
| 隐藏位置精确重建率 | **1.0000** |
| 几何-only 学习探针（无信念、无占据栅格） | 0.7870 |
| raw-sensor 探针 | 0.7717 |
| 参照：V8 holdout 上 formal 实体图 permanence | 0.8021 |

**断言被坐实，且强于预期。** 一个零信念、只做确定性物理外推的预测器拿到 **1.000**，
而正式实体图在 V8 holdout 上只有 **0.802**；隐藏位置可由可观测的
"末见位置 + 观测速度 + 固定遮挡布局"精确重建。**该任务根本不测物体永久性，它测
运动学外推，且信念层反而比平凡物理更差。** 连一个简单的无信念线性探针（0.787）
都追平了正式模型（0.802），在全新种子上复现了"formal 打不过平凡基线"。

这直接支撑第 B 节的环境改造：只有引入**随机几何 + 随机隐藏动力学**（让隐藏位置
不再能被确定性外推命中），permanence 才会成为对"维持信念"的真实测试。

---

## B. 共享新环境设计（item 2 与 item 3 共用）

**目标：打破 A 节的几何捷径，让 permanence 成为对"维持信念"的真实测试。**
采用最彻底方案：**随机几何 + 随机隐藏动力学**。

**实现约束（重要）：** 不修改锁定文件 `cal/evaluation/v2_i1_integration.py`
（它被 I1 冻结协议的 `locked_source_sha256` 钉住）。新建旁路环境类，例如
`cal/evaluation/randomized_occlusion_world.py::RandomizedOcclusionWorld`。

> **实现状态（已完成，未冻结）：** `RandomizedOcclusionWorld` 已实现为
> `_IntegratedWorld` 的同接口 drop-in（`grid_size`/`static`/`self_position`/
> `distractor_a,b`/`velocity_a,b`/`step`/`observe`/`truth`），单元测试
> `tests/test_randomized_occlusion_world.py`（9 项）覆盖接口对齐、按 seed
> 确定性、跨 seed 几何随机化、遮挡事件发生，以及**捷径被打破**的对比断言。
> 用 A.5 的同一诊断脚本（`--world randomized`）验证：
>
> | 指标 | 固定环境 | 随机环境 |
> |---|---:|---:|
> | 恒速重建隐藏位置 | 1.0000 | **0.2212** |
> | 解析外推 permanence 平衡准确率 | 1.0000 | **0.6029** |
> | 几何-only 探针 | 0.7870 | **0.4925** |
> | raw-sensor 探针 | 0.7717 | **0.5021** |
>
> 随机环境下两个学习型无信念探针约为随机（0.5）；解析恒速外推仍有 0.603，说明
> 早期二元真格/最近诱饵诊断中仍留有残余短程运动学信号，不能声称所有捷径已完全消除。
> 该诊断的二元平衡准确率也不是 D 节 hidden-field-conditioned top-1，二者的 chance
> 口径不可直接比较。最终 P1 难度选择应以 C.6 的长遮挡 field 指标为准。

### B.1 每 episode 随机化遮挡几何

- 遮挡屏的列位置、长度、门洞行位置，以及 1–2 个额外阻挡格，均按 episode seed
  重新采样；
- 保证连通性与可达性（自体能移动、隐藏带存在），拒绝退化布局；
- 关键性质：**几何不再跨 seed 不变**，train seed 上的固定空间先验无法迁移到
  holdout seed。

### B.2 随机化隐藏动力学

- 隐藏物体速度大小/方向按 episode 重采样；
- 允许"遮挡中变向"或在 ≥2 条候选隐藏轨迹间随机分支；
- 关键性质：**隐藏位置不再是 `(hidden_steps, seed)` 的确定函数**——真·永久性
  才能维持一个可读的信念，线性外推无法命中。

### B.3 几何对称负样本

- 正式 localization 已改为全 hidden field 排序，主指标不再选择单个负 query，因而没有
  “最近诱饵决定评分”的偏差。
- GRU/Slot 的 selected-positive vs 最近 hidden decoy 辅助 loss 已移除；训练改为整个 hidden
  field 上的 binary BCE，加上与 evaluator 完全同构的“sigmoid occupancy 所有正格质量相对
  整个 field 质量”的 categorical NLL。更换 selected decoy 或改动 field 外 logits 都不再
  影响训练 loss，并有显式回归测试锁定。

### B.4 不变的公平性约束（继承 V2 §4.1）

- 仅向智能体提供**可见占据 + 已执行动作副本**；被遮占据与空格观察上不可区分；
- 不提供任何可见性通道或坐标/身份标签；
- 评测真值只在智能体更新后、评测端读取，永不进入 learner；
- 给定 seed 完全确定可复现。

---

## C. 永久性新留出预注册（草案）

### C.1 假设

> 在随机几何 + 随机隐藏动力学环境下，正式实体图的永久性信念**优于纯几何外推
> 基线**，且 `assume_all_visible` 对照在 permanence 上坍缩到接近
> 随机——即永久性能力来自维持的信念，而非固定几何。

### C.2 模型无关 development seed registry

不再把公开连续整数段称为“密封 holdout”。development registry 由
`cal.evaluation.permanence_seed_registry` 按固定规则生成：从 62000 开始升序扫描，
**不读取任何模型输出**，仅检查 layout 与事件覆盖；选择前 56 个合格 seed，前 40 个用于
train、后 16 个用于 evaluation，不允许人工替换。

紧凑、可重生成的 registry artifact：
`experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY.json`。artifact 内的 selection
digest 绑定 coverage contract、扫描范围、split、accepted audit 与 rejected prefix；不得只把
seed 列表当成完整选择证据。以 artifact 自带的 `reproduction_command` 重生成并逐字段比较。
接受条件在五个候选 `turn_probability={0.15,0.25,0.35,0.45,0.55}` 上同时成立：

- 单隐藏对象合格样本至少 12；
- `2–3`、`4–5`、`6+` 每档至少 2 个不同 focus-episode 组；
- focus 与所有隐藏正格都有已知 track；
- 碰撞和重复正格事件过滤；
- layout 连通、可达、actor 初始可移动且存在 visible-hidden boundary。

共扫描到 62193，拒绝 138 个不满足条件的候选后得到 40+16 registry。validation/holdout
seed **尚未生成，也不得公开写入本文**；未来必须由独立 custodian 保管秘密 seed stream，
在候选定稿前公布该 stream/manifest 的密码学承诺，并以已锁定算法生成 split。只有候选源码
锁定且执行获授权后，custodian 才可运行相应 split；公开承诺不等于公开 seed。

### C.3 对照条件

沿用 V8 的负对照体系，**新增 `geometric_extrapolation`**：

- `formal_entity_graph`：正式模型；
- `raw_sensor`：裸传感器（保留，但降级为次要下限）；
- `geometric_extrapolation`（**新，主要下限**）：只含遮挡布局 + 末见位置 + 速度 +
  hidden_steps 的确定性外推特征，无任何信念；
- `assume_all_visible`、`time_shuffled`、`random_labels`、
  `referent_swapped`、`identity_scramble`：语义同 V8。

`no_action` 只作为非门控耦合诊断保留，不进入 permanence 成败判定。

### C.4 门（相对 V8 的关键重设计）

> **评审更正（重要）：相机固定、干扰物动力学与隐藏转向都与智能体动作无关**，
> 因此任何"删动作必降低 permanence"的门都**没有因果依据**——原
> `no_action_permanence_degrades` 已删除。动作类门只用于 self / 身份归因任务，
> **不用于 permanence**。若删动作反而降低了 permanence，那是模型发生了不该有的
> 动作耦合，属于需要排查的信号，而非通过条件。

| 门 | 判据 | 相对 V8 |
|---|---|---|
| `formal_beats_geometric_extrapolation_permanence` | formal permanence 显著 > geometric_extrapolation，**seed-level 配对**，报告效应量与置信区间 | 新增主门，取代过弱的 raw 基线 |
| `formal_beats_raw_permanence` | formal > raw（配对 + CI） | 保留 |
| `all_visible_permanence_fails` | assume_all_visible 在遮挡长度 ≥K 的 hidden-field-conditioned 指标上落入 runner 逐事件计算的 chance+ε 容差 | 保留，但不再假定 chance=0.5 |
| `shortcuts_fail_by_occlusion_length` | geometric / 位置先验 / all-visible 在 ≥K 分箱上均落入 runner 报告的 exact chance+ε | **新增，取代笼统的 `geometry_near_chance`** |
| `formal_permanence_pass` | formal permanence ≥ 阈值（评审定数） | 保留 |
| 其余身份/自我/结构/资源门 | 同 V8 | 保留 |

**关键定义（评审前必须敲定，草案不写死具体数）：**

- **最小遮挡长度 K**：短遮挡下恒速外推本就可能有效，笼统说"几何已接近随机"是错的。
  permanence 门只在遮挡长度 **≥K** 的分箱上评估。
- **精确 chance 与容差 ε**：any-positive top-1 chance 由 runner 按事件计算
  `mean(|positives| / |field|)`；候选数和正格数变化时不得写死 0.5 或 0.026。
  ε 仍须在冻结前给出具体数值。
- **配对统计**：所有"A 优于 B"的门一律用 **seed-level paired bootstrap / 置信区间**，
  不用汇总均值直接比较。
- **每 seed 有效事件的数量与覆盖**：门需保证各 seed 在 ≥K 分箱内有足够的独立
  episode-bin 组，并报告 field 大小与正格数分布；这里不是旧二元正/负 query 任务。

### C.5 托管（评审冻结后才实施）

沿用 I1/L0 既有模式：实现前冻结协议 JSON + `.sha256` sidecar +
`locked_source_sha256`（覆盖新环境与新探针源码）；holdout 用带 nonce 的
`--force-with-lease` CAS 一次性消费 tag + immutable Git blob evidence。
**本草案阶段不创建任何一次性 tag，不消费任何 seed。**

validation 也不是可反复调参的 development split：候选与契约锁定后只执行一次；若任何
模型、阈值、指标或 evaluator 发生实质修改，该候选/协议版本的 validation 结论即终止，必须
建立新版本及新的预承诺 validation stream，原 holdout 保持未消费。正式协议还必须预先定义
基础设施失败与部分输出的重跑规则，避免观察结果后决定是否把一次运行算作“有效消费”。

### C.6 环境标定必须模型无关（评审更正）

标定 `turn_probability` 及其它难度参数时，**不得**采用"geometric / entity_graph 明显
高于神经基线"这类**依赖当前模型表现**的标准——那是把当前模型表现反向用于调环境，
构成另一种测试设计过拟合。标定采用**模型无关**判据：

1. 已知核 oracle（belief）在规定遮挡长度上稳定有效（提供可达上界）；
2. 恒速外推、位置先验、all-visible 等捷径在遮挡长度 ≥K 上**失败**（落入 runner
   报告的 exact chance+ε）；
3. 每 seed 在三档中有足够的独立 episode-bin 覆盖，并报告 field/positive-count 分布；
4. 难度随参数**平滑**变化（无突变/退化区）；
5. **不**根据当前 entity graph 是否占优来选参数。

### C.7 两阶段冻结（评审更正）

A（改进 I1）与 B（冻结契约）的正确关系是**两阶段冻结**，而非提前锁死 A 的实现：

1. **A 之前**：冻结**评测契约**——环境、seed 段、指标、阈值、决策规则；
2. **A 之后**（仅在 train/dev seed 上开发完成）：追加**只增加最终候选源码哈希**的
   exact-source-lock amendment，然后运行 validation；**amendment 不得借机改门槛**；
3. validation 全门通过 + 另行授权后，才一次性消费 holdout。

A 的正确形态是**新一代候选 + 新协议版本**，不改写任何已消费协议；修改锁定文件
**不会**使历史结果失效（旧结果由旧 commit/hash 永久绑定）。

### C.8 冻结前阻塞项（评审，未解决即不可冻结）

下列 development 层问题已实现、确定性复跑，并经三路独立 review；review 发现的 Brier、
分箱、artifact provenance 与 neural objective 问题均已修复：

1. **field 口径**：所有图先投影到 hidden field；field 外无质量时明确记 miss；top-1 chance
   逐事件计算 `|positives|/|field|`；最大分数并列时距离取并列集合平均，不受 candidate
   顺序影响；
2. **事件资格**：缺失 hidden track、actor collision、重复正格显式报告并过滤；时长统计
   只使用单隐藏对象事件；
3. **模型无关 registry**：相同 seeds 在全部五个候选难度上覆盖三档时长，每档至少两个
   episode-bin 组；不读取模型分数，不允许人工替换；当前 selection digest 为
   `e31929f7563ded1163979abaa353c2c26771912328e319175615dbb90ae4d9e4`；
4. **统计单位**：主指标固定为 `time → episode/bin → seed/bin → 三档等权 → 完整 seed 等权`，
   当前最长的 71 步 focus episode 不再获得 71 倍权重；
5. **泄漏分解**：同时报告 uniform-field、中心距离、绝对坐标、field-relative geometry、
   duration-conditioned coordinate 五类先验；
6. **统计**：10,000 次确定性 seed-level paired bootstrap 已接入，正向 advantage 统一表示
   第一模型更好；
7. **模型无关难度选择**：按预声明条件扫描五个候选，0.35 是第一个通过者；完整 artifact
   为 `experiments/V2_P1_PERMANENCE_TURN_PROBABILITY_DEVELOPMENT_SCAN.json`，scan digest
   为 `b30b41cdff80bd949e5f593e1120bf108318f98e4c810a5b05c023063e8cb42e`；
8. **D.5/D.6 已按新口径重生成**：完整摘要见
   `experiments/V2_P1_PERMANENCE_DEVELOPMENT_AUDIT.json`。

仍然阻止正式冻结的事项：

1. 成败门的最小实质效应量、绝对阈值与多指标决策规则仍未冻结；
2. 尚无正式协议 JSON、sidecar、transitive source/dependency lock 或 clean reviewed commit；
3. validation/holdout secret custody、承诺与一次性 runner 尚未建立；
4. 新一代 I1 候选尚未开发；当前 D.6 只诊断旧 I1 的 OOD 表现。

**可复现 development 审计（2026-07-28，非门控）：** registry 的 16/16 evaluation seeds
均有样本且三档齐全；共 1367 个事件，其中 1162 个单隐藏事件进入 411 个 episode-bin 组。
等权 chance top-1 为 **0.03702**。position-prior 分解为：center 0.04308、global coordinate
0.04405、field geometry 0.04195、duration-conditioned coordinate 0.02292。相对 uniform，最高的
global-coordinate lift 在 registry 与 episode 等权后为绝对 **+0.00702**。这些已测试先验
远低于旧 I1 的 0.25960，但没有为“先验相对 chance”预注册配对 CI 或成败门，因此这里只能
说未发现足以复现模型分数的已测试位置捷径，不能据此证明不存在其它泄漏。

---

## D. item 3：实体图动作条件化前向模型 + 四基线（设计）

### D.1 任务

在 B 节随机环境上，给定 `t` 时刻状态 + 动作，预测 `t+1`（并多步递推穿过遮挡）
的占据 / 实体状态，**重点评测经历非确定性遮挡运动的物体的再现位置**——正是
几何捷径失效、必须靠维持信念的地方。

### D.2 四个对照（按用户选定，全上）

1. **实体图前向模型（本体，核心）**：在 I1 实体信念图上做动作条件化下一状态预测；
2. **几何外推基线**：末见位置 + 速度线性外推（与 C.3 的下限一致）；
3. **神经对象中心基线**：小型 slot-based 前向模型（对标 Embodied-SlotSSM）。
   **预算提示**：可能超出 V2 §4.3 的 100k 参数 / CPU 预算——作为**基线**（非正式
   智能体）需按 §4.3 单列预算说明；
4. **神经非对象中心基线**：复用 `cal.model.recurrent_core` 的 GRU/ConvGRU，作为
   "无对象结构"的神经对照。

### D.3 指标

- 遮挡后**再现位置误差**（主指标）；
- 多步占据预测 NLL / IoU（随占据步数增长）；
- 校准（Brier、可靠性曲线）——尤其遮挡期间的熵变化；
- 资源（每步 MAC、活动状态、参数计数）逐条对预算。

以上是正式协议目标。当前 D.5/D.6 development runner 只完成 hidden-field top-1、categorical
NLL、hidden-field per-cell proper Brier 与并列平均位置误差；尚未实现 IoU、可靠性曲线、
熵轨迹和资源审计，
因此不能把当前 audit 当成 D.3 的完整验收。

### D.4 成功含义

> 正式实体图前向模型在再现位置误差上**同时优于几何外推与两个神经基线**，
> 且在随机动力学下差距随遮挡时长扩大——这才证明"显式实体信念"相对"无结构神经
> 潜空间"和"平凡外推"有必要增益（对应 V2 §12 决策规则：解析结构通过后，神经网络
> 必须证明相对它的必要增益）。

### D.5 development 重生成（已运行、已复跑，未冻结）

使用模型无关 registry、`turn_probability=0.35`、hidden-field-conditioned localization，
以及 `time→episode/bin→seed/bin→三档等权→完整 seed 等权`。16/16 evaluation seeds 三档齐全；
1162 个单隐藏事件聚合为 411 个 episode-bin 组。精确等权 chance top-1 为 0.0370。

runner 原始摘要复现命令：

```bash
uv run python -m cal.evaluation.permanence_forward_benchmark \
  --seed-registry experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY.json \
  --gru --slot --entity-graph --paired-ci --audit-summary
```

命令向 stdout 输出完整 audit summary；
`experiments/V2_P1_PERMANENCE_DEVELOPMENT_AUDIT.json` 是对该输出的 reviewed projection，
不是 runner 直接写入的原始文件。其 `source` 字段声明生成关系与本轮逐字段复核状态。
GRU/Slot 已按 B.3 使用相同的 field-wide 训练口径；但协议与模型初始化策略尚未冻结，当前
排名仍只作 development 诊断。

| 预测器 | episode-bin top-1 | categorical NLL | Brier | 并列平均位置误差 |
|---|---:|---:|---:|---:|
| belief（已知核 oracle） | **0.420** | **1.434** | **0.025** | **1.380** |
| geometric | 0.322 | 9.371 | 0.037 | 7.842 |
| GRU | 0.042 | 3.625 | 0.037 | 6.061 |
| Slot | 0.020 | 3.323 | 0.036 | 6.589 |

belief 相对 geometric 的 10,000 次 paired-seed bootstrap 全部支持 oracle 更优：top-1
advantage `+0.0979`（95% CI `[+0.0722,+0.1266]`），categorical-NLL advantage
`+7.937`（`[+7.312,+8.524]`），Brier advantage `+0.0122`
（`[+0.0103,+0.0141]`），位置误差 advantage `+6.461`（`[+5.178,+7.868]`）。

独立的模型无关 `turn_probability` scan 在 `6+` 分箱给出 chance `0.03704`、geometric
`0.03767`、belief `0.19985`，支持“长遮挡时几何捷径失败而已知核仍有 headroom”；完整
条件和全部候选行以 turn-scan artifact 为准，不能由上表总体均值单独推出该结论。
GRU/Slot 的 top-1 接近 chance，NLL/Brier 也接近 uniform-field 基线；这与较分散、不过度
集中的预测相容，但不是学到正确动力学的证据。仍只有单一神经初始化，不能推出模型族能力
上限。

### D.6 真实 I1 实体图作为"学习型信念"预测器（已运行，未冻结）

新口径结果：

| 预测器 | episode-bin top-1 | categorical NLL | Brier | 并列平均位置误差 |
|---|---:|---:|---:|---:|
| belief oracle | 0.420 | 1.434 | 0.025 | 1.380 |
| geometric | 0.322 | 9.371 | 0.037 | 7.842 |
| **entity_graph（旧 I1）** | **0.260** | **7.583** | **0.077** | **3.207** |
| GRU | 0.042 | 3.625 | 0.037 | 6.061 |
| Slot | 0.020 | 3.323 | 0.036 | 6.589 |

paired-seed 结论不是单一“谁赢”：

1. 相对 geometric，旧 I1 的 top-1 development 95% bootstrap CI 不含 0：advantage `−0.0621`，CI
   `[−0.1270,−0.0063]`；但 categorical NLL 和位置误差更好，CI 分别为
   `[+0.9637,+2.5732]` 与 `[+3.0967,+6.1253]`。它比点质量几何外推更分散、更少灾难性
   失配，但最高峰更不常落在真格。
   Brier advantage 为 `−0.0405`，CI `[−0.0527,−0.0289]`，说明它的 per-cell proper
   Brier 明确差于 geometric。
2. 相对 GRU/Slot，旧 I1 的定位 top-1 CI 均为正（分别
   `[+0.1708,+0.2613]`、`[+0.2082,+0.2696]`），但 categorical NLL CI 均为负
   （CI 分别 `[−4.7901,−3.2070]`、`[−5.0658,−3.5384]`）。神经基线接近 uniform，
   因而定位差但 categorical NLL 较低；旧 I1 有尖峰定位信号，但该 proper score 较差。
   Brier advantage 也都为负，分别为 `−0.0406`（CI `[−0.0535,−0.0286]`）和
   `−0.0418`（CI `[−0.0549,−0.0299]`），说明旧 I1 的 per-cell proper Brier 明确更差。
3. 旧 I1 相对 exact-kernel oracle 的四项点估计均明显更差；当前 runner 尚未报告这对模型
   的 paired CI，因此这里只能据此否定“stochastic permanence 已解决”，不能给出冻结门
   结论。新候选必须同时改善定位和 proper scoring，不能只优化 top-1 或只把分布抹平。

**诚实的 OOD 说明：** I1 实体图是在固定几何、确定性动力学上设计并冻结的；这里是分布外
诊断，不改写历史 I1 结论。下一代机制目标是表征并传播机动不确定性，同时改善定位与 proper
score，而不是坍缩成恒速尖峰或近 uniform 分布。

---

## E. 排序与执行（评审修正版）

1. **已完成当前 development evaluator 实现与确定性复跑**：model-blind cross-`turn_p` registry、layout/
   leakage/edge 审计、episode/bin/seed 等权、paired bootstrap、模型无关难度扫描和 D.5/D.6
   重生成；未生成或运行任何 validation/holdout seed；
2. **冻结前 review gate**：2026-07-28 已完成环境/runner、统计、文档、registry 及生成 artifact
   的三路独立 review，并修复所有已发现问题；后续如有实质修改必须重新打开相应 review；
3. 当前 review 清零后，另行决定并**冻结评测契约**：环境、秘密 seed 生成算法、
   指标、阈值、决策规则（这是"B = 先冻契约"，不是提前锁死 A 的实现）；
4. **再做 A**：新一代 I1 候选，仅用 `*_train`/`*_dev` seed 开发；
5. 候选确定后，追加**只增加最终候选源码哈希**的 exact-source-lock amendment，运行一次
   validation（amendment 不得改门槛）；
6. validation 全门通过 **且另行授权**后，才一次性消费一次 `permanence_holdout`，产出
   immutable 证据；随后回写 `RESEARCH_STATUS.md` 与里程碑评估。

**硬约束：A 不得早于第 3 步的契约冻结；第 3 步前不冻结任何协议、不消费任何 holdout
seed；标定 `turn_probability` 只用模型无关判据（C.6），不看当前模型是否占优。**
