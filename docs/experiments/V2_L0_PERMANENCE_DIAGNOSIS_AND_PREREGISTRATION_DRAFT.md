# V2-L0 永久性失败根因诊断 + 新留出预注册（草案）

> **状态：非冻结草案，供评审。** 本文档不含任何 SHA-256 source lock，不构成
> 已冻结协议；它只固化根因诊断，并提出新环境设计与门定义，等待评审（含潜在
> 第三方评审）后再由 runner 生成冻结 `*.json` + `.sha256` + `locked_source_sha256`。
> 在冻结之前，本文档中的一切阈值、seed 段和门都可修改。

日期：2026-07-28（A/B 节诊断）；**C 节于 2026-08-08 整体重写**
上游证据：`results/V2-L0-language-readout-holdout-v8.json`
（`decision=stop_and_report`，SHA-256
`ae5ad9d4ef457d22680dc30048bf8e0421f5e708c724351b8751f8061a2d9d04`），
终局证据 tag `calmodel-l0-v8-holdout-terminal-evidence`，
干净 commit `e26c613e4648528f38f7125b662c6daf89448983`
关联：[research status](../../RESEARCH_STATUS.md)、
[L0 报告](V2_L0_LANGUAGE_READOUT.md)、[里程碑评估 1.1.0](../MilestoneSummary-1.1.0.md)、
[2026-08-08 评审报告](../review/REVIEW_PERMANENCE_FREEZE_2026_08_08.md)（判定 `block`）、
[门重设计草案](V2_P1_PERMANENCE_GATE_REDESIGN_DRAFT.md)

> **A/B 节（根因诊断）经 2026-08-08 评审逐条核实无误**，包括代码引用、
> 八条件表与全部 digest；C 节则因门名与实现零重合等四条阻断项被整体重写。

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

## C. 永久性预注册（2026-08-08 重写版）

> **本节已整体重写。** 2026-08-08 评审（判定 `block`）发现旧 C 节的门表与实际
> 实现的门系统**交集为空集**——照旧稿冻结等于预注册一个从未实现的实验，而真正
> 实现的管线反而没有被任何文档预注册（评审 F2）；且旧稿不含任何数值阈值与决策
> 规则（F3）、承诺保留的 raw-sensor 对照在门控管线中不存在（F6）、绑定的是已被
> 取代的 V1 注册表（F7）。旧稿备份于评审报告的修订链引用中，不再作为预注册依据。
>
> 本节的每一条都与 `cal/evaluation/stochastic_permanence_benchmark.py` 的实现
> 逐一对应。**仍为非冻结草案**：待 Phase-0 V11 产出后补齐 C.9 的实测前提，再送评审。

### C.1 假设与失败语义

**假设**：在随机化遮挡几何 + HMAC 密钥化随机隐藏动力学的环境中，一个维持
显式信念的候选，其被遮挡物体的位置预测**优于任何不做信念滤波的方案**。

**判定为失败的情形**（写明失败，而不只写成功）：

- 候选未能在 6+ 遮挡区间闭合 `oracle` 与 `belief_free` 之间差距的 40%；
- 候选在任一资源门上越界；
- 参照健康门不通过——此时不是候选失败，而是**该数据规模不足以测量任何东西**，
  必须停止并扩充规模，不得据此宣称任何结论；
- 一次性留出消费后任一冻结门失败 → `stop_and_report`，且**不得重试**。

### C.2 权威注册表与切分

**权威文件唯一**：`experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY_V4.json`
（`selection_digest_sha256` = `8bbed3df12c08a2e…`）。任何引用无版本号路径的
表述作废。

| 项 | 值 |
| --- | --- |
| train | 40 seed |
| development evaluation | **150 seed**（由 64 修订，见 C.8） |
| 候选扫描区间 | 62000–70000，实际扫至 62722 |
| 拒绝数 | 533 |
| 转向概率 | **0.45**，由 `permanence_turn_probability_scan` 按契约自动选出 |
| 转向概率扫描产物 | `..._TURN_PROBABILITY_DEVELOPMENT_SCAN_V4.json` |

转向概率 0.45 在 64-seed（V3）与 150-seed（V4）两个独立注册表上**复现同一
选择**，故非小样本噪声。p=0.15/0.25/0.35 均因 `geometric_6plus_near_chance`
失败而被拒——即在那些转向概率下，无信念外推没有被压到随机线附近。

validation / holdout seed **不在本仓库出现**，由保管人按 `holdout_policy` 用
密盐生成并承诺（见 C.5）。

### C.3 对照条件（实现中的预测器集合）

`PREDICTORS = ("candidate", "oracle", "geometric", "belief_free", "uniform", "old_i1")`

| 名称 | 含义 | 角色 |
| --- | --- | --- |
| `oracle` | 精确信念滤波，读真值静态拓扑 | **上限**，闭合门的分母 |
| `belief_free` | 反射外推 + 拟合误差表，**不做任何信念滤波** | **下限**，top1 闭合门的分子基准 |
| `geometric` | 反射外推点质量 | 参考，不再作为 top1 闭合下限 |
| `uniform` | 场内均匀 | NLL / Brier 的下限 |
| `old_i1` | 既有 I1 实体图 | 非劣性对照 |

**关于 raw_sensor 等 V8 对照的去向（评审 F6）**：V8 的
`raw_sensor` / `assume_all_visible` / `time_shuffled` / `identity_scrambled` /
`random_labels` 属于**线性读出探针**范式，而本阶段已转为**前向预测评分**范式，
两者的样本与指标不可通约。`raw_sensor` 对照保留在**非门控诊断**
`permanence_geometry_diagnostic.py` 中。**这是一次范式变更，不是静默删除**；
其代价是本阶段不再直接检验"正式模型是否跑赢裸传感器"，该检验须在恢复读出
范式时另行预注册。

**为什么必须有 `belief_free`**：V8 的教训是"跑赢点质量外推"不构成永久性证据。
2026-08-08 红队构造了一个仅靠反射外推 + 125 条目误差表的候选，通过了当时全部
18 个门。把它固化为参照下限，候选就必须显示**它在平滑之上多做了什么**。

### C.4 门（与实现逐一对应）

#### C.4.1 参照健康门（前置条件，候选无关）

7 对 `(scope, metric)`，每对要求 `oracle` 与其参照之间差距的**单侧 99% 下界为正**：

| scope | metric | 参照 |
| --- | --- | --- |
| overall | top1_accuracy | `oracle − belief_free` |
| overall | categorical_nll | `uniform − oracle` |
| overall | brier | `uniform − oracle` |
| **4-5** | top1_accuracy | `oracle − belief_free` |
| 6+ | top1_accuracy | `oracle − belief_free` |
| 6+ | categorical_nll | `uniform − oracle` |
| 6+ | brier | `uniform − oracle` |

它是**设门的前置条件**：若某对没有可测差距，则针对它的门无意义。
恒真的 `mean_at_least_development_floor` 分量已删除（评审 F5）。

#### C.4.2 确认门（12 项）

`reference_health`、`overall_top1_superiority`、`overall_nll`、`overall_brier`、
`short_top1_protection`、`medium_top1`、`long_top1`、`long_nll`、`long_brier`、
`position_error_superiority`、`position_error_noninferiority`、`coverage_stability`

闭合阈值（先验固定，沿用既有预注册值，本轮未新造）：

| 闭合门 | 阈值 | 下限参照 |
| --- | ---: | --- |
| overall/top1 | 0.30 | **belief_free** |
| overall/nll | 0.20 | uniform |
| overall/brier | 0.15 | uniform |
| 4-5/top1 | 0.20 | **belief_free** |
| 6+/top1 | **0.40** | **belief_free** |
| 6+/nll | 0.20 | uniform |
| 6+/brier | 0.15 | uniform |

非劣性边界：`-0.03`（top1）、`-0.02`（old_i1 对比）、`-0.25`（位置误差）。

#### C.4.3 资源门（6 项）

`parameters_within_budget`、`persistent_active_state_within_budget`、
`mac_per_step_within_budget`、`step_budget_respected`、`replay_budget_respected`、
`runtime_budget_respected`。限额：100 000 参数、65 536 活跃字节、
5 000 000 MAC/步（沿用 Phase-R 容量修订 V3）。

#### C.4.4 诊断（**不是门**）

`wasted_field_mass`（放在候选场之外的质量占比）作为诊断记录。**刻意不设为门**：
候选可读到 `visible` 掩码，把可见格清零即可把该指标伪造成 0 而无需任何信念，
设为门会给出虚假的保障感。

### C.5 托管与一次性语义

隐藏机动流由 `HMAC(custodian_salt, seed)` 派生
（`randomized_occlusion_world.hidden_stream_key`）。开发使用已公布的默认盐；
**保管人持有一次性切分的密盐**。

这一条是必要的：可见布局是 episode seed 的确定函数，若隐藏流由 `seed + 常数`
派生，观察者可由布局反推 seed 并精确重放隐藏轨迹——一个零信念的永久性 oracle。
2026-08-08 评审实测：首观测指纹在 380 个候选 seed 中**完全唯一**，重放 200 步
精确复现。密盐切断了这条链路。

**硬顺序约束：密盐机制必须在生成任何留出 seed 之前就位**，事后更换代码无法
补救已生成的留出。

其余一次性语义（CAS 消费、终态不可重试、基础设施失败与部分输出的处理）由
`stochastic_permanence_custody.py` 的状态机实现，`VALIDATION_CONSUMED_FAIL` /
`NO_DECISION` 为终态。

### C.6 环境标定必须模型无关

转向概率选择、seed 合格性审计均只读世界与解析基线，不读任何候选模型指标
（注册表 `model_metrics_read: false`，Phase-0 `candidate_maps_read: false`）。

### C.7 两阶段冻结

第一阶段冻结环境与注册表（已完成：V4 + 扫描 V4）；第二阶段在候选实现完成后
冻结门与 `locked_source_sha256`。评审报告的 commit 必须早于冻结 commit。

### C.8 相对旧稿的实质变更（全部为修订，须随本稿一并评审）

| # | 变更 | 动因 |
| --- | --- | --- |
| 1 | top1 闭合下限 `geometric` → `belief_free` | 无信念候选通过了全部旧门（评审 F1） |
| 2 | 删除恒真门 `mean_at_least_development_floor` | 它恒等于 `mean ≥ 0.25·mean`（F5） |
| 3 | 隐藏流改 HMAC 派生 | 种子反演可得零信念 oracle（F9） |
| 4 | 开发 evaluation seed **64 → 150** | 换强参照后效应量变小；4-5 区间解析需求 ~75，取 2.0 安全系数 |
| 5 | 转向概率 0.35 → **0.45** | 选择契约在新世界自动改选，两个规模复现 |
| 6 | 注册表 V2 → **V4**，扫描 → **V4** | 上述 3 与 4 的连带 |
| 7 | 产物 schema 2 → **3** | 变更 2 改变了门结构 |
| 8 | 拓扑分层门（曾提议）**不采纳** | 纠正径向中心后分层失去区分力，详见门重设计草案 D4′ |

第 4 项修改了预注册的**锁定常量**，第 1、2 项修改了预注册的**门定义**。按
`docs/review/REVIEW_PLAN.md` §8，这些须经评审 `pass` 且由人类 Gatekeeper 签署
后方可冻结。

### C.9 冻结前仍未解决（待补）

- [ ] **Phase-0 V11 尚未产出**：150 seed 下 7 项参照健康门的实测结果未知，
      尤其 4-5/top1 在 64 seed 时下界为 −0.0048（均值 +0.0605，为功效不足而非
      效应缺失）。**V11 通过前，本稿不得冻结。**
- [ ] 尚无候选实现，故 12 项确认门从未在真实候选上运行过。
- [ ] `docs/experiments/V2_I1_STOCHASTIC_PERMANENCE_PLAN.md` 与本稿的
      关系需明确：二者对同一门系统的描述必须合并或声明其一取代另一。

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
