# I1-P1 Stochastic Permanence Candidate v1：架构与预注册计划

> **状态：非冻结 development 计划。**
>
> 本文只定义下一代 permanence 候选的技术路线、权限边界、资源预研、评测门和冻结顺序。
> 它不是正式协议，不包含 validation/holdout seed，也不授权消费任何新 split。
> 历史 I1 V3/V4 及其已消费证据保持不变；本候选不得改写或追溯解释历史结论。
> 2026-07-29 grid v2 独立评审发现：旧 Phase 0 的 64-trial/16-source-seed 结果只能解释为
> conditional development simulation，artifact validator 也没有真正锁住 exact grid 与底层 evidence。
> simulation design v3 的 fail-closed 重跑进一步证明：直接指定 correlation/variance inflation 的
> grid 在 outer-resampled empirical population 上经常没有物理解。v6 的 endpoint-fraction 轴又把
> 部分 metric 的方差放大到 reference scale 的数十至数百倍，不能代表可解释的 sensitivity。
> simulation design v8 因此改用 reference-relative variance envelope、该 envelope 下的
> feasibility-native covariance 坐标，并把 null 独立为四格 outer-robust grid。
> v8 的锁定重跑得到有效 `phase0_no_go`：16-source outer population 下 binding power
> 不足，并有少量 component-null outer recalibration 不可实现。v9 在不读取 candidate 的前提下，
> 将 development source 扩大为首批 64 个模型盲合格 seeds，允许 reference gap 在逐 seed 上有符号，
> 并把 power 明确定义为条件于预注册唯一 `model_seed=17011`；任意多初始化主张必须另建协议。
> v9 的锁定重跑进一步把唯一 blocker 定位为窄可行 covariance 区间中的数值精度：4,096 个
> outer-null trials 中 10 个 trial 的绝对 cross-moment 误差仅约
> `2.8e-11–4.1e-11`，但除以约 `1.6e-5–2.6e-5` 的区间宽度后超过 `1e-6` fraction tolerance。
> v10 在查看任何 candidate 前仅把内层 bounded-mean bisection 从 42 步预注册提高到 56 步；
> grid、效应、RNG、trial 数、family 和接受门均保持不变。
> 旧 `Phase0 Go / 774+774` 已撤回，不是 seed commitment；v3 No-Go 也不得被解释为候选模型失败。
> base lock 与 secret manifest commitment 尚未完成，因此本文仍是非冻结计划，不能生成或消费
> validation/holdout。

### 0.1 Phase 0 / Phase R development 实现状态

- Phase 0 v8：`V2_I1_P1_PHASE0_REFERENCE_HEALTH_POWER_DEVELOPMENT.json`；其 canonical
  decision 是 `phase0_no_go`，只保留为不可改写的历史 development evidence。
- Phase 0 v9：使用
  `V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY_V2.json` 的 64 个完整 development source seeds。
  registry 从同一连续候选流按原模型盲 coverage contract 选择首批合格 seeds，train 仍为 40；
  `turn_p=0.35` 由绑定新 registry 的 scan V2 重新选择。simulation design v9 使用 outer seed
  bootstrap、逐 trial
  feasibility-native moment recalibration、独立四格物理 component-null grid 和 exact simultaneous
  Monte-Carlo bounds；任何 outer population 不可实现、Type-I null 不可实现或 simultaneous evidence
  不足均直接 No-Go。artifact 内的 canonical decision 是当前唯一结果来源；只有全部 gate 通过时
  validation/holdout recommendation 才能非 `null`。旧 774/774 只是在固定经验分布 `F16` 下得到的
  历史 development 数值，已撤销推荐资格；仓库没有生成或保存秘密 seed。canonical digest 只记录在
  同名 `.sha256` sidecar，避免把 digest 写回被 source lock 覆盖的文档形成自引用。
- Phase 0 v9 锁定结果：canonical digest
  `eebe8a9770ad6a31e71aca9890fa9046a45a715b09446adec550a912d3e4237a`，decision 为
  `phase0_no_go`。四个 alternative cells 的 power、108-cell coverage、reference health 和
  physical-domain gates 均通过；已成功构造的 Type-I cells 也低于门槛。唯一失败是
  `overall/candidate_vs_old_i1/position_error` 的 10 个 outer-null 数值校准，不能删去失败 trial
  或用 4,086 个成功 trial 代替预注册分母。
- Phase 0 v10：保留 v9 全部统计设计，只将 bounded conditional-mean 内层二分固定为 56 步，
  以使窄可行区间上的实际 achieved covariance fraction 也满足原 `1e-6` tolerance。v9 artifact
  保留为不可改写证据；v10 必须生成新 artifact/sidecar，重新执行全部 4,096 个 Type-I trials，
  不得复用 v9 的通过计数。v10 通过前，Phase R/base freeze/validation/holdout 仍暂停。
- Phase R 本轮只读内存重跑：`H=5`、`E=11`、`K=48`、`S=2640`，221 个完整遮挡
  episode/1602 个 checkpoint 无错误；38,400 个 known-topology transition cases 与真实 environment
  kernel 完全对齐；最大累计裁剪质量约 `2.11e-15`，最大 conditional spatial TV 约 `6.09e-8`，
  branch-evidence accounting residual 约 `8.88e-16`；声明 active-state 52,145 B、实测 deep-size
  46,709 B、16 个参数、1,013,760 MAC/step，修订后的 development capacity gate 全部通过。
- 修订后的 artifact schema v2 使用 canonical JSON + 严格 filename-bound SHA-256 sidecar，绑定显式
  transitive source lock；validator 精确锁定 alternative/null grid rows、构造坐标、物理域、gate population、
  Monte-Carlo counts/bounds、recommendation 和 decision，并从保存的 counts/calibrations/observed ranges
  重算。相同输入连续运行的 canonical bytes 必须完全一致，wall-clock timing 不进入 artifact。两份
  JSON 都只是 development snapshot，不等于 frozen protocol、final candidate capacity artifact 或
  validation/holdout evidence。

## 1. 结论与适用范围

下一代 I1 的正确方向不是延长 track 生命周期，而是把单点运动假设升级为可校准的随机运动
信念，并在 sensor-only 条件下同时保持身份、存在性和位置不确定性。

该方向适合当前系统，但只有在以下前置条件全部解决后才可进入正式候选实现：

1. posterior 容量、存储和最坏资源契约能够满足 100k 参数、64 KiB 活动状态和
   5M MAC/step；
2. privileged oracle、sensor-only candidate 和 evaluator reference 的权限严格分层；
3. train → frozen kernel artifact → per-episode reset 生命周期完全确定；
4. candidate runner、逐分箱 paired CI、机器可执行门、power 和 custody 先完成并锁定；
5. existence、association 和 spatial posterior 使用一致的概率更新，而不是只重归一化位置。

当前总体可行性评为**中等**。参数预算风险低，active-state 与端点核识别风险高。

## 2. 当前 development 证据

以下均为固定 development registry 上的点估计，不代表逐分箱显著性结论：

| Top-1 | 2–3 步 | 4–5 步 | 6+ 步 |
|---|---:|---:|---:|
| belief oracle | 0.716 | 0.343 | 0.200 |
| geometric | 0.680 | 0.248 | 0.038 |
| 旧 I1 | 0.464 | 0.175 | 0.140 |

总体结果：

| 预测器 | Top-1 | categorical NLL | Brier | 位置误差 |
|---|---:|---:|---:|---:|
| belief oracle | 0.420 | 1.434 | 0.025 | 1.380 |
| geometric | 0.322 | 9.371 | 0.037 | 7.842 |
| 旧 I1 | 0.260 | 7.583 | 0.077 | 3.207 |
| uniform-field | 0.037 | 3.324 | 0.036 | 5.371 |

可以据此作出的有限结论：

- 旧 I1 在长遮挡上有高于 geometric 点估计的 permanence 信号，但逐分箱 paired CI
  尚未实现，因此不能称为统计显著优势；
- 旧 I1 总体位置误差优于 geometric，但 Top-1 更低，说明它经常把质量放在真格附近却没有
  把最高质量稳定放在真格；
- 旧 I1 的 categorical NLL 和 Brier 明显差于 oracle。结合当前点状态和高 existence
  渲染机制，错误尖峰、stale track 和身份碎片是待验证的主要解释，而不是已证明的唯一根因；
- 单纯增加 persistence 很可能延长错误状态寿命，不能解决校准问题。

权威 development 摘要见
[V2_P1_PERMANENCE_DEVELOPMENT_AUDIT.json](../../experiments/V2_P1_PERMANENCE_DEVELOPMENT_AUDIT.json)。

## 3. 当前机制缺口

当前 [EntityBelief](../../cal/model/entity_belief_graph.py) 在每个全局分支中只保存：

- 一个 position；
- 一个 velocity；
- 一个 existence；
- action/motion 位移计数。

当前 missed-state 展开只保留每个实体预测分布的前三个位置，再由最多五个全局 hypothesis
保留联合历史。随机隐藏路径因此会快速坍缩成少量点状态。

需要精确区分：

- 旧 detection matcher 已对**完整的一步 prediction distribution**计算检测 likelihood；
- top-3 截断发生在 missed-state branch expansion；
- 新候选的修复要求是：检测关联对所有保留的 `(position, velocity)` 质量边缘化，并让
  pruning 丢失质量进入 association evidence，而不是把旧 matcher 错误描述为只比较三个点。

旧模型只把连续可见的一步位移计入 motion counts。遮挡后重现的整段位移不会被错误当成一步
样本，但系统也没有其它端点似然机制学习隐藏机动核。

## 4. 权限分层与不可泄漏边界

### 4.1 正式 sensor-only candidate

正式候选及其训练管线只能接收：

- sensed occupancy；
- 已执行 action copy；
- 从自身历史内部推断出的 visibility、static/unknown-static belief 和 tracks；
- 仅由 40 个 train seeds 的完整 sensor/action streams 拟合并冻结的 kernel artifact。

禁止输入：

- world truth；
- evaluator visibility 或真实 static layout；
- hidden tracks、hidden_steps、正格或 candidate field；
- environment seed；
- evaluation/validation/holdout 标签或端点真值配对；
- oracle identity。

正式公共接口必须继续兼容：

- `update(sensed_occupancy, action)`；
- `probability()`；
- `track_positions()`；
- `self_track_identity()`；
- learnable parameters、active-state bytes、MAC/step 三项资源属性。

### 4.2 Privileged diagnostics

以下条件只用于机制归因，必须使用独立 runner/class，并输出
`privileged_diagnostic_only=true`：

1. 真实 topology + known kernel 的纯 transition/filter conformance；
2. known kernel + oracle identity；
3. learned kernel + oracle identity；
4. 真实 static/visibility 下的 unpruned posterior reference；
5. sensor-only runtime input + environment-known kernel parameter 的 Phase A ceiling。

privileged diagnostic 产生的参数必须在运行结束后丢弃，不得：

- 初始化正式 candidate；
- 写入 frozen kernel artifact；
- 参与候选选择；
- 进入 validation/holdout 决策。

### 4.3 Evaluator-only references

belief oracle、geometric 和 uniform-field 都拥有正式 candidate 没有的 evaluator 信息：

- oracle 使用精确 kernel、known hidden tracks、真实 static/visibility；
- geometric 使用 known hidden tracks；
- uniform-field 使用 evaluator 的 hidden-field mask。

它们可以用于 split-relative normalization 和强参考基线，但不能称为正式 agent 可无条件达到
的上界，也不能把 candidate–oracle 全部差距归因为 transition learning。

## 5. 修订架构

```text
SensorOnlyFrontEnd
  ├─ inferred visibility
  ├─ static / unknown-static belief
  └─ dynamic detections

FrozenKernelModel
  ├─ autonomous transition parameters
  └─ action-conditioned self transition parameters

GlobalAssociationBank
  └─ hypothesis weight w_h
       └─ EntityFactor
            ├─ existence e_hi
            ├─ normalized q_hi(position, velocity | exists)
            ├─ branch-local categorical self probability π_hi
            └─ age / last-seen / missed

Dynamic occupancy(cell)
  = Σ_h w_h · [1 - Π_i(1 - e_hi · q_hi(cell))]
```

### 5.1 Sparse kinematic posterior

每个需要遮挡传播的实体维护归一化的
`q(position, velocity | entity exists, hypothesis)`，支持：

- 当前速度保持；
- 隐藏转向；
- static/unknown-static 下的阻挡与反弹；
- inferred-visible empty observation conditioning；
- 检测后的 emission update；
- deterministic packed pruning。

posterior 不得使用 Python dict/list 作为满载正式存储。正式表示必须是固定容量 packed
NumPy SoA 或等价紧凑数组。

### 5.2 Unknown-static 概率模型

正式 candidate 不能把未观察格直接当作 free，也不能读取真实 static layout。front-end 必须维护
共享的有界栅格概率 `m_t(c)=P(static_c | sensor history)`；越界格按 `m_t(c)=1` 处理。第一版固定
使用逐格 Bernoulli 近似，相关 map-hypothesis 模型不属于本版本。

对 inferred-visible observation `o_t(c)`：

```text
m_t(c) ∝ P(o_t(c) | static_c) · m_{t-1}(c)
```

`P(o|static)`、prior、clamp 和数值精度必须写入最终 candidate config，只能从 train stream
确定。若某条 branch 给出 bounce evidence，则用 global-hypothesis/self-association posterior 加权的
likelihood 更新对应 cell；不得硬选一个 oracle identity。

对状态 `(x,v)`，令 forward cell `f=x+v`、reflected cell `b=x-v`。在逐格独立近似下，未转向
的 bounce transition 至少包含：

```text
P((f,  v) | x,v,m) += 1 - m(f)
P((b, -v) | x,v,m) += m(f) · (1 - m(b))
P((x, -v) | x,v,m) += m(f) · m(b)
```

hidden-turn 先按 inferred-hidden probability 混合 turn/no-turn kernel；每个候选方向的权重再乘以
该方向在 `m_t` 下可移动的概率并归一化，零可移动质量时回退原速度。第一 hidden transition
不得转向，必须与 environment 时序一致。该近似、likelihood 表和回退规则属于 candidate lock，
不得在 validation 后调整。

### 5.3 Existence、检测与多目标 branch 更新

删除不可能空间状态后不能只重归一化 `q`。对一次 no-detection observation，至少必须满足：

```text
L_no = Σ_s q_pred(s) · P(no_detection | s)

q_post(s)
  = q_pred(s) · P(no_detection | s) / L_no

e_post
  = e_pred · L_no
    / ((1 - e_pred) + e_pred · L_no)

branch_log_weight
  += log((1 - e_pred) + e_pred · L_no)
```

对 matched detection，检测证据必须使用：

```text
e_pred · Σ_s q_pred(s) · P(detection | s)
```

完整更新固定为以下生成语义：

- prediction：`e_pred = p_survive · e_previous`，`q_pred` 由 transition kernel 传播；
- matched detection：
  `Z_match=e_pred·Σ_s q_pred(s)P_D(s)P(d|s)`，并按相同 integrand 归一化 `q_post`；在该
  matched-assignment branch 内 `e_post=1`；
- unmatched track：
  `Z_miss=(1-e_pred)+e_pred·L_no`，并使用上式更新 `e_post/q_post`；
- unmatched detection 必须同时展开 frozen `λ_birth·b(d)` 与 `λ_clutter·c(d)` alternatives；
- 在 birth-assignment branch 内新 identity 的 `e_birth=1`；birth intensity 已由 branch prior
  承担，不能再用额外 existence boost 重复计权；
- 每条 global branch 的 detection assignment 必须 one-to-one，branch log weight 只增加各个
  normalizer/assignment prior 一次；
- birth prior、`P_D`、emission、clutter/birth intensity、survival/death hazard、retirement threshold
  和容量淘汰规则全部进入最终 canonical candidate config；
- hard retirement 只是有界存储操作；其被删除概率必须计入 approximation/overflow audit，不能
  伪装成观测到的 death evidence。

association uncertainty 只由全局 hypothesis weight `w_h` 表示，不再添加一个重复计权的独立
association probability 标量。任何 heuristic birth penalty 或 existence decay 若保留，都必须能
映射到上述 frozen likelihood/prior，不能作为未记录的额外分数。

### 5.4 Action conditioning 与 branch-local self identity

每个 global hypothesis `h` 必须维护一个 categorical self label：

```text
π_h0 + Σ_i π_hi = 1

π_h0 = P(self=null | h, observations)
π_hi = P(self identity=i | h, observations)
```

joint transition 按“至多一个 self”边缘化：

```text
P({s'_i} | h)
  = π_h0 · Π_i P_autonomous(s'_i | s_i)
    + Σ_j π_hj · P_action(s'_j | s_j, action)
                  · Π_{i≠j} P_autonomous(s'_i | s_i)
```

单实体 marginal 因而等于
`π_hi·P_action + (1-π_hi)·P_autonomous`，但实现不能把跨 hypothesis 聚合后的 self probability
重新灌回每个 branch，也不能让多个独立 Bernoulli self 同时响应 action。self label 可以在 episode
内由 sensor/action evidence 更新；frozen action/autonomous kernel 参数不得在 evaluation 内更新。

### 5.5 Occupancy 合成

第 5 节图中的 branch-local Bernoulli union 明确采用“给定 `h` 后 entity existence/spatial factors
条件独立”的有界近似。最终输出同时包含 static 与 dynamic：

```text
P_occ(c) = 1 - (1 - m_t(c)) · (1 - P_dynamic(c))
```

point merge/collision 仍由 global hypotheses 表示；若实现改用 joint spatial factor，必须作为新的
candidate configuration 计入尝试预算。多峰 `track_positions()` 固定输出存在概率过阈值实体的
position-marginal MAP，平局使用 canonical cell order。

### 5.6 现有能力不得丢失

新候选必须保留或明确替代：

- DynamicCellFrontEnd 与 inferred visibility；
- static/unknown-static map；
- global association hypothesis bank；
- birth、death 和 stale-track retirement；
- point merge/collision；
- self posterior 与显式 null class；
- 历史 I1 的 action-alignment 负对照。

## 6. Phase R：posterior 容量与资源预研

当前历史 I1 的正式资源为：

| 资源 | 当前值 | 上限 | 剩余 |
|---|---:|---:|---:|
| learnable parameters | 2,605 | 100,000 | 97,395 |
| active state | 56,527 B | 65,536 B | 9,009 B |
| MAC/step | 3,997,392 | 5,000,000 | 1,002,608 |

2026-07-29 评审对固定 development registry 的只读 posterior 测量：

| 指标 | 中位数 | P90 | 最大 |
|---|---:|---:|---:|
| joint support | 16 | 31 | 41 |
| 保留 99% 质量所需 K | 16 | 30 | 40 |

观察到：

- K=16 的 P90 pruning loss 约 26%；
- K=32 的最坏 pruning loss 仍约 7.6%；
- 不能把一个无界 sparse posterior 复制到 `5 hypotheses × 11 entities`；
- 每步丢失 1% 在 72 步遮挡下最坏只保留 `0.99^72 ≈ 0.485`，因此逐步均值不是充分门。

### 6.1 必须先冻结的容量契约

正式 candidate 实现前必须完成并 review 一份 capacity artifact，固定：

- 全局 packed posterior state pool 容量 `S_max`；
- 单 entity/hypothesis 最大 support `K_max`；
- 槽位索引、概率精度和数组布局；
- posterior 是否跨 association hypotheses copy-on-write/shared；
- pool overflow 与 empty-posterior policy；
- pruning tie-break；
- 被裁质量如何进入 branch likelihood；
- exact active-state 与 MAC 最坏公式。

capacity artifact 必须使用当前独立的 capacity schema v2 canonical JSON，并包含生成命令、platform/dependency、
registry/reference source hashes、`H_max/E_max/K_max/S_max`、layout、measured/declaration 双重资源
记录和 SHA-256 sidecar。validator 对缺字段、非 canonical serialization 或 digest mismatch 必须
fail closed；还必须从 evidence 重算每个 resource gate、严格 AND 与 decision label，不能只验证 outer
fields。sidecar 固定绑定 artifact filename；wall-clock timing 不属于 canonical payload。

Phase R artifact 只证明候选接口与 layout 的可行性。最终 selected candidate 若与 Phase R prototype
任一 source/config hash 不同，必须在 locked harness/reference 下重新生成 final capacity/conformance
artifact；最终 amendment 绑定后者。不得把旧 prototype 的资源结论移植到新实现。

### 6.2 Pruning 与 TV 的机器定义

在每个 transition/observation checkpoint，对一个 normalized pre-pruning joint posterior
`r_t(position,velocity)`，令 deterministic retained set 为 `A_t`：

```text
ρ_t = Σ_{s∈A_t} r_t(s)
δ_t = 1 - ρ_t
q_t(s) = r_t(s) / ρ_t,  s∈A_t
branch_log_weight += log(ρ_t)
```

`ρ_t≤0` 立即失败。裁掉的概率不得进入 absent/death likelihood；它通过 `log(ρ_t)` 留在 branch
evidence，并进入 approximation audit。对任一完整 entity/branch lineage：

```text
cumulative_pruned_mass(T) = 1 - Π_{t≤T} ρ_t
```

正式门使用所有完整遮挡 episode、所有 lineage 的最大值，不使用平均值。若 association branch
自身被裁，其 branch probability 同样计入该 lineage 的累计裁剪质量。

与 privileged unpruned reference 的比较在相同 transition/observation checkpoint、相同观测与
确定性 branch ID 上进行。conditional spatial TV 固定为 position marginal：

```text
TV_t(h,i)
  = 0.5 · Σ_position
      |Σ_velocity q_packed(h,i) - Σ_velocity q_reference(h,i)|
```

branch 无法匹配时，其 reference branch mass 计入 cumulative pruned mass；不得从 TV population
静默删除。正式报告同时保存每 step/factor 的 `ρ_t`、最大 TV 所在 checkpoint 和 rendered dynamic
occupancy 的 L1 error，保证结果可独立重算。

`S_max` 必须覆盖 fully detached worst case：
`S_max ≥ H_max·E_max·K_max`。共享/COW 只能节省实测空间，不能成为安全性证明。primary pool、
propagation double buffer、refcount/free-list 和 association expansion workspace 都必须有固定上限；
overflow 必须原子失败，不能部分提交 graph update，也不能临时增加容量。

Phase R prototype 固定使用一个跨 factor 共享的 `12·K_max` expansion workspace、覆盖全部
`H_max·E_max·K_max` 的 full scratch pool，以及按合法 kinematic state 编号的 direct-index
accumulator。factor 顺序更新到 scratch，全部检查通过后才交换 primary/scratch；任何 overflow 在交换前
失败。不得使用逐 successor 线性扫描 accumulator 后仍按 O(`H·E·K·12`) 申报 MAC。

### 6.3 Capacity Go/No-Go

在 privileged unpruned reference 上，packed 方案必须同时满足：

- 每个完整遮挡 episode 的累计 pruned mass ≤ 0.01；
- pruned 与 unpruned conditional spatial posterior 的最大 total variation ≤ 0.01；
- 无 NaN、负概率或空 posterior；
- 完整 candidate 的 persistent active-state deep-size ≤ 65,536 B；
- learnable parameters ≤ 100k；
- estimated MAC/step ≤ 5M；
- 100k steps/seed、replay ≤4、CPU 总时长 ≤2h 的正式研究预算。

64 KiB 统计对象不是 posterior bank，而是 update 边界之间仍存活的完整 candidate object graph，
至少包括 front-end、static/unknown-static map、全部 hypotheses/entities、primary/scratch pool（若预分配
并持久存在）、COW metadata、frozen kernel、RNG、configuration 和容器/header guard。training mode
还必须把跨 episode 存活的 learner state 计入同一门。纯函数调用内的瞬时 workspace 另报 peak
working memory，不与 persistent active-state 混写；任何被 cache/preallocate 到 candidate 上的
workspace 都按 persistent 计。

历史 I1 的 56,527 B 只作为参考，不代表新架构只能追加 9,009 B；若新 candidate 替换旧 graph，
必须从零核算整个 candidate，不能把旧声明与新 posterior 简单相加或只测新 bank。

若不能同时满足，不得降低误差门来迁就实现；必须减少重复状态、采用共享状态池/更紧凑近似，
或终止该架构路线。

## 7. Kernel 学习生命周期

### 7.1 Train stream

kernel learner 必须读取 40 个 registry train seeds 的**完整 sensor/action episode**，不能读取
由 evaluator truth/track filters 选出的 `_Sample` 列表。

固定生命周期：

```text
fit_kernel(train_sensor_streams_in_ascending_registry_order)
  -> immutable FrozenKernelParameters

new_episode(FrozenKernelParameters, fixed_model_seed)
  -> fresh SensorOnlyFrontEnd
  -> fresh GlobalAssociationBank
  -> fresh episode-local tracks
```

规则：

- 每个 train episode 开始时重置 tracker/front-end；
- kernel learner state 可以按固定 seed 顺序跨 train episodes 累积；
- 每个 train episode 只允许预注册次数的 replay；
- fit 完成后 kernel 参数不可变；
- 每个 development/evaluation/validation/holdout seed 从同一 frozen kernel snapshot 开始；
- evaluation episode 内和 episode 间都不得继续更新 kernel；
- episode 结果不得依赖 evaluation seed 执行顺序；
- model RNG 不得由 world seed 派生；第一版固定单一 `PCG64(seed=17011)`，不做多初始化选择；
  改变初始化数量或聚合方式必须建立新协议版本。

### 7.2 Frozen kernel artifact

artifact 至少绑定：

- registry path 与 selection digest；
- 40 个 train seeds 及固定顺序；
- 完整 train-stream collector hash；
- learner/source/dependency hashes；
- kernel 表示、prior、超参数和数值精度；
- replay 次数、model RNG policy；
- fit diagnostics 与失败状态；
- artifact schema version 和 SHA-256。

### 7.3 Kernel 表示必须预先选择

正式计划冻结前必须在以下一种有界方法中选定并锁定：

- 小型固定 θ-grid，每个 θ 维护 forward likelihood；
- `q, ∂q/∂θ` forward sensitivity + 有界在线梯度；
- 有界 online EM。

还必须定义：

- 多实体/多遮挡事件如何共享更新；
- association uncertainty 如何对 endpoint likelihood 加权；
- 先验、数值稳定、不可识别和零 likelihood 的失败规则。

正式 learner 不能硬选 oracle identity 端点配对。

## 8. Evaluator、统计与 power 必须先完成

候选实现前，runner 必须支持并经过独立 review：

- predictor/candidate factory；
- fit-kernel 与 per-episode reset；
- candidate vs geometric、uniform、旧 I1、oracle 的 paired seed bootstrap；
- 总体和每个遮挡分箱的相同 complete-seed population；
- oracle-gap closure；
- reliability、conditional occupancy entropy、existence entropy；
- pruned mass/TV；
- 参数、active-state、MAC、runtime 和 replay 审计；
- 一条机器可执行、任一门失败即停止的 decision function。

当前 `permanence_forward_benchmark.py` 仍是 NON-GATED development 工具：它的现有
`ci95_low` 是双侧 95% interval 的 2.5% quantile，不得重命名或直接解释为本文的单侧 99%
lower bound。Phase 0 必须新增 formal statistics API 和独立 decision artifact，并用 golden
per-seed fixtures 证明第 9 节所有符号方向、分箱 population 与 fail-closed 路径。

Reliability 与 entropy 在第一版只作预注册诊断：

- reliability 使用固定 edges `0.0, 0.1, ..., 1.0`（最后一档含右端点），每档报告 mean
  probability、empirical frequency 与权重，并按主 reducer 聚合 ECE；
- conditional occupancy entropy 由 evaluator 在 hidden field 内归一化后计算
  `-Σ p_i log(p_i) / log(|field|)`；
- existence entropy 单独报告；
- 在冻结阈值前不得事后升级为成败门。

### 8.1 Power

10,000 次 bootstrap 只是对有限 seeds 重采样，不增加独立样本量。

在生成秘密 validation/holdout split 前，必须进行 candidate-independent power simulation：

- 使用当前 baseline/oracle seed 间差异；reference gap 允许在逐 seed 上为负，只要求用于 closure
  的 population mean gap 为正。oracle、uniform 或 geometric 不需要在每个 seed 上逐点支配；
  sensitivity grid 的 covariance 轴必须由每个 outer
  empirical population 在固定 reference-relative variance envelope 下的可行域定义；不能把固定
  correlation 当作所有 population 都必须精确达到的 hard target，也不能用物理 endpoint fraction
  直接定义跨 metric 不可比的方差；
- candidate sufficient statistics 只能由有界生成模型合成，但不能强制每个 seed 都位于
  baseline–oracle 线段，因为真实 candidate 可以在个别 seed 上差于 baseline 或优于 oracle。v2
  直接生成 positive-is-better advantage `A`，并约束最终 candidate score 落在真实物理域：Top‑1/
  Brier `[0,1]`、categorical NLL `[0,-log(10^-6)]`、位置误差 `[0,20]`；
- 对每个 metric/bin，clipped-affine conditional mean 精确匹配 target closure；先以
  `κ·e_adjusted·SD(R)` 定义目标 SD，`κ∈{1.0,1.5}`，再按 endpoint fraction `≤0.90` 的物理余量
  clamp 到最近可实现 SD。固定该 SD 后求完整 covariance 可行区间，并按预注册 fraction 取区间内点；
  alternative 与 component null 都使用该完整区间。围绕 conditional mean 的 contracted-endpoint
  two-point distribution 只负责实现已固定的目标 SD，endpoint fraction 是 achieved diagnostic，
  不是 grid 轴。`Corr(A,R)` 与实际 variance scale 同样保存为 achieved diagnostics；这里
  `e_adjusted` 明确包含本 trial 的共享 initialization shift；
  不得用稀有 full-endpoint jump 或无界加性正态后裁剪。前者会造成重尾并使 percentile bootstrap
  欠覆盖，后者会改变目标效应与 coverage truth；
- 每个 binding scenario 固定运行 1,024 trials。正式候选预注册唯一 `model_seed=17011`，因此
  power 条件于该固定算法实例，不再合成共享初始化 Rademacher 轴。未来若要主张初始化鲁棒性，
  必须预注册多个独立 model seeds、训练预算和 reducer，不能把 evaluation seeds 当成初始化样本；
- 每个 trial 先从 64 个 development source seeds 有放回抽取 64-row outer empirical population，
  再在该 population 内重新求全部 bounded moments 并抽取 candidate seeds；coverage truth 使用该
  outer population 上对 seed-level innovation 积分后的条件期望。任何 outer draw 不可实现或
  endpoint variance fraction 超过 0.90 均记为该 scenario No-Go，不能只重复经验行到推荐 N；
- 所有 requested feasibility-native coordinates 必须保存 requested/achieved/feasible interval/
  tolerance；任何 metric/bin/init-sign 的 endpoint variance fraction 必须 `≤0.90`。covariance
  区间退化为单点时使用唯一物理解并把 correlation 标为不可识别诊断；目标均值不可行、完整可行区间
  为空或坐标误差超限均令该场景 No-Go，不得静默替换为较有利的 achieved moments；
- 以第 9 节的联合 closure contrast、superiority 和非劣 margin 为目标；
- superiority 和 non-inferiority power 的通过量不是 point estimate：对四个 binding scenario 使用
  Bonferroni family-wise alpha 0.05 的 one-sided exact Clopper–Pearson lower bound，每个 scenario
  的 lower bound 都必须 `≥0.80`；
- 对 20 个 binding contrasts 分别生成真实物理 metric score：conditional mean 位于该 component
  精确边界，covariance/endpoint variance 使用独立的四格 null grid，最终 score 仍在物理域内。
  component boundary 以绝对 mean advantage 直接校准，不得除以可能为零的 outer reference gap；
  每个 null trial 也必须先 outer-resample 64 个 development seeds，再在该 outer population 上重新
  校准全部 component boundaries；不得把 alternative contrast vector 事后平移到边界。四场景共
  80 cells，每个 cell 的 Bonferroni family-wise exact Type-I upper bound 必须 `≤0.05`；任一物理
  null 或 outer recalibration 不可实现即 No-Go；
- one-sided CI coverage 的 family 同时覆盖四场景的全部 27 contrasts，共 108 cells。每个 cell
  必须同时满足 observed coverage `≥0.93` 且 Bonferroni family-wise exact lower bound `≥0.90`；
  “区间包含 0.95”不再作为通过条件；
- analytic maximum 使用单侧 `alpha=0.01`，再按固定 2.0× safety multiplier 得到唯一 candidate
  seed count；该系数和数量不能由 candidate 指标、某个“好 seed”或 simulation pass/fail 反向选择。
  只有该数量在
  完整 sensitivity grid 同时通过 moment feasibility、power、Type-I 与 coverage 时才可称为
  recommended，并据此冻结 validation/holdout seed 数量和 coverage contract；
- 若 power 不足，只能在承诺前增加秘密 seed 数量，不能观察候选结果后放宽门。

power artifact 必须 canonical-hash 后进入 base protocol lock；冻结后不得替换 variance/covariance
假设或只重算有利场景。

simulation design v10 固定：bounded conditional-mean calibration 使用 56 步二分；power RNG
`PCG64(20260731)`、component-null RNG
`PCG64(20260732)`、每场景 1,024 trials、每 trial 2,000 次 power bootstrap；正式 decision runner
使用单侧 99% lower bound 和 10,000 次 common bootstrap `PCG64(20260729)`。64-source reference
table 给出的 analytic maximum 是 1,315，固定 2.0× 后唯一 candidate seed count 为 2,630。
1,024 trials 下，四格 power family 至少需要 848/1,024 次通过，108-cell coverage family至少需要
953/1,024 次覆盖；80-cell Type-I family 最多允许 29/1,024 次 false rejection，以上阈值均由
Bonferroni-adjusted exact Clopper–Pearson bounds 直接重算。任何较小 smoke-test artifact 只能是
No-Go。

### 8.2 预注册 outer-robust moment/null sensitivity grid v6

令 `R` 为当前 outer empirical population 中逐 seed 的正向 reference gap，`A` 为合成 candidate
advantage。在固定 target mean 和真实 metric bounds 后，runner 先要求
`SD(A)=κ·e_adjusted·SD(R)`；若该值超出 endpoint fraction `≤0.90` 时的物理可行区间，则只允许
确定性地 clamp 到最近边界并记录 floor/ceiling。固定该方差后再求 clipped-affine family 可达到的
cross-moment 范围，并换算为 covariance 区间 `[C_low,C_high]`。预注册坐标
`feasible_covariance_fraction=q` 定义 `C=C_low+q(C_high-C_low)`；若区间退化为单点，任意 q
映射到同一唯一物理解。achieved endpoint fraction、correlation 和 variance scale 都是审计字段。

Alternative 的四个 binding 场景是完全固定的 Cartesian product：

| scenario | covariance interval | feasible fraction q | variance scale κ |
|---|---|---:|---:|
| `low_cov_nominal_variance` | full feasible | 0.25 | 1.0 |
| `low_cov_high_variance` | full feasible | 0.25 | 1.5 |
| `high_cov_nominal_variance` | full feasible | 0.75 | 1.0 |
| `high_cov_high_variance` | full feasible | 0.75 | 1.5 |

Null 不含没有统计意义的 initialization 符号重复，只使用四个唯一 Cartesian cells：

| scenario | covariance interval | feasible fraction q | variance scale κ |
|---|---|---:|---:|
| `low_cov_nominal_variance_null` | full feasible | 0.25 | 1.0 |
| `low_cov_high_variance_null` | full feasible | 0.25 | 1.5 |
| `high_cov_nominal_variance_null` | full feasible | 0.75 | 1.0 |
| `high_cov_high_variance_null` | full feasible | 0.75 | 1.5 |

两个 grid 均在查看 power、coverage 或 Type-I 结果前由常量固定；没有 frontier search、cell replacement
或 development-outcome selection。runner 和 validator 必须锁定完整 rows、构造版本、q/κ achieved
误差、每次 outer recalibration、物理上下界及 achieved endpoint fraction `≤0.90`。任何坐标、
物理上界或 0.90 余量的未来
修改都必须建立新协议版本，不得在 validation 后原地替换。

## 9. 精确预注册门

### 9.1 统一统计定义

对同一 split、同一 complete-seed population：

- `C`：正式 candidate；
- `O`：belief oracle；
- `G`：geometric；
- `U`：uniform-field；
- `I`：旧 I1；
- Top-1 越高越好；
- categorical NLL、Brier、位置误差越低越好。

所有 superiority/non-inferiority CI 使用：

- seed-level paired bootstrap；
- 10,000 次；
- complete seed IDs 升序排列；
- NumPy `PCG64(seed=20260729)`；
- 对同一 split 生成一份 `10,000 × N_seed` common resample-index matrix，并由所有 predictor、
  metric、overall/bin contrast 复用；
- one-sided 99% percentile lower bound `LB(X)=quantile(X_boot, 0.01, method="linear")`；
- 主 reducer：
  `time → episode/bin → seed/bin → bins 等权 → complete seeds 等权`。

正式 primary population 固定为单隐藏对象 accepted events；collision、重复正格、缺失完整 hidden
track 的事件按 evaluator 的预锁定资格规则报告并排除。每个 seed 必须在三档均达到 coverage
contract，且 overall 与逐 bin 使用完全相同的 complete-seed IDs。model RNG 固定为单一
`PCG64(seed=17011)`；本协议第一版不对多个 model initializations 扩充独立样本量。

规则：

- superiority：positive-is-better advantage 的 lower bound 必须严格 `> 0`；
- non-inferiority：lower bound 必须 `≥ -margin`；
- joint closure/effect lower-bound threshold 使用 `≥`；
- missing seed、缺 bin、NaN、空 posterior、资源超限或 artifact mismatch 一律失败；
- 所有 confirmatory gates 严格 AND，不能用一个指标补偿另一个；
- 因决策是预声明 intersection-union，不通过事后选择 baseline/分箱 rescue；
- secondary diagnostics 不参与通过判定。

### 9.2 Split-relative closure 与 reference-health gate

点估计 closure 继续报告，但不单独决定通过：

```text
top1_closure(C; G, O)
  = (Top1_C - Top1_G) / (Top1_O - Top1_G)

nll_closure(C; U, O)
  = (NLL_U - NLL_C) / (NLL_U - NLL_O)

brier_closure(C; U, O)
  = (Brier_U - Brier_C) / (Brier_U - Brier_O)
```

真正的门对每个 complete seed 构造联合 paired contrast：

```text
D_top1(τ)  = (Top1_C - Top1_G) - τ · (Top1_O - Top1_G)
D_nll(τ)   = (NLL_U - NLL_C)   - τ · (NLL_U - NLL_O)
D_brier(τ) = (Brier_U - Brier_C) - τ · (Brier_U - Brier_O)
```

因此 `LB(D_metric(τ)) ≥ 0` 才表示在预注册置信口径下至少关闭 `τ` 的 reference gap；不能用
“优势 CI 刚过零 + closure 点估计过线”替代。

每个 metric/bin 的 reference gap `R` 还必须同时满足：

```text
mean(R_formal) ≥ R_floor
LB(R_formal) > 0
R_floor = 0.25 · mean(R_development)
```

其中 Top-1 的 `R=O-G`，NLL/Brier 的 `R=U-O`；`R_development` 必须来自 Phase 0 在当前固定
development registry 上、完全不读取 candidate maps 的 reference-health artifact。该 artifact 保存
full-precision per-seed/per-bin sufficient statistics，经独立 review 和 canonical hash 后，在 protocol
JSON 中物化不可修改的 metric/bin-specific `R_floor` 常数。若 reference-health 失败，该 secret split
记为**已消费、
protocol no-decision**，不得补 seed 或重跑；只有经独立原因审计、建立新协议版本与新预承诺
secret stream 后才能再评估。

development 的约 `0.351/2.946/0.03396/0.103` 只作为点估计公式示例，不是 formal split 的
固定常数；closure 超过 1 时不得 clip。

### 9.3 Confirmatory gates

| Gate | 机器可执行判据 |
|---|---|
| reference health | 本 gate 使用的每个 metric/bin 均满足 `mean(R)≥R_floor` 且 `LB(R)>0` |
| overall Top-1 superiority | `LB(C−G)>0` 且 `LB(D_top1(0.30))≥0` |
| overall NLL | `LB(G−C)>0`、`LB(U−C)>0`，且 `LB(D_nll(0.20))≥0` |
| overall Brier | `LB(G−C)>0`、`LB(U−C)>0`，且 `LB(D_brier(0.15))≥0` |
| 2–3 Top-1 protection | `LB(C−G)≥-0.03` |
| 4–5 Top-1 | `LB(C−G)>0` 且该分箱 `LB(D_top1(0.20))≥0` |
| 6+ Top-1 | `LB(C−G)>0`、该分箱 `LB(D_top1(0.40))≥0`、点估计 `C≥I`，且 `LB(C−I)≥-0.02` |
| 6+ NLL | `LB(U−C)>0` 且该分箱 `LB(D_nll(0.20))≥0` |
| 6+ Brier | `LB(U−C)>0` 且该分箱 `LB(D_brier(0.15))≥0` |
| position-error superiority | `LB(G_error−C_error)>0` |
| position-error non-inferiority | `LB(I_error−C_error)≥-0.25` |
| coverage/stability | 所有预注册 seeds 三档齐全；无 NaN、空 posterior 或顺序依赖 |
| resources | 参数、state、MAC、steps、replay、runtime 全部门通过 |

当前门是 candidate-independent 的机制要求；一旦 evaluator/contract 锁定，不能因开发候选失败而
降低 closure 或 margin。

## 10. 分阶段实现与诊断

### Phase 0：评测与 custody

在候选实现前完成：

1. runner、逐分箱 pairing、uniform pairing、closure 和 decision function；
2. candidate-independent power 与秘密 split 数量；
3. secret manifest commitment；
4. validation/holdout CAS reservation；
5. 中断、部分输出、基础设施失败和 invalid-split policy；
6. immutable result evidence；
7. 下述候选尝试预算、唯一最终选择规则和 custody 状态机；
8. 定义 world/environment、truth-only evaluator/reducer、registry/secret-stream generator、原始
   sensor/action train collector、formal statistics/decision、custody/CAS 和 runtime/dependency 的
   transitive base-lock coverage；实际 base lock 在 Phase R review 通过后冻结。

正式 sensor-only front-end、packed filter、association/existence、kernel learner、candidate factory
与 candidate-specific shared dependencies **不属于 base lock**；它们全部进入最终 candidate
amendment。privileged unpruned reference/diagnostic 可以进入 base lock，但不得被正式 candidate
import 或读取。

### 10.1 第一版 candidate attempt budget

- base lock 后最多登记 12 个 candidate-informing configurations；
- configuration 是 canonical JSON；任一架构、prior、threshold、layout、precision 或超参数变化都
  产生新 ID 并消耗一次；
- 每个 configuration 只允许固定 `model_seed=17011`，train replay 上限 4；不得把 world seed
  派生为 model RNG，也不得通过多初始化挑最好结果；
- 从消费第一个 train sensor step 起，NaN、OOM、资源失败和代码异常均计一次 attempt；纯基础设施
  失败只有在没有产生/查看任何 model metric 且日志证明 candidate 未开始时才允许同 ID 重试；
- capacity layout prototype 若只读取 train/reference conformance、不查看 candidate development
  metrics，不计入 12 次；任何 privileged/Phase A/B configuration 一旦在 development population
  上产生并查看性能指标，就消耗 12 次预算中的一次，即使它不参与最终排名；privileged 参数仍
  不得直接回流正式 artifact；
- 所有正式配置在相同 development registry 上运行全部 gates。先过滤全门通过者，再最大化
  `min(top1_closure/0.30, nll_closure/0.20, brier_closure/0.15)` 的 overall 点估计；完全相等时依次
  选择较低 persistent bytes、较低 MAC、最后选择 lexicographically smallest configuration ID；
- 无配置全门通过即该 protocol version No-Go，不得增加第 13 次或修改选择规则。

attempt ledger 必须 append-only、canonical-hashed，并记录 configuration、开始/终止原因、源码、
train artifact、完整 development result 与是否计数。

### 10.2 Validation/holdout custody 状态机

```text
DRAFT -> BASE_LOCKED -> CANDIDATE_LOCKED
      -> VALIDATION_RESERVED -> VALIDATION_CONSUMED_PASS
                             -> VALIDATION_CONSUMED_FAIL
                             -> VALIDATION_CONSUMED_NO_DECISION
VALIDATION_CONSUMED_PASS -> HOLDOUT_RESERVED -> HOLDOUT_CONSUMED
```

规则：

- reservation 前，机器验证 train/development/validation/holdout、历史已消费 streams 的 seed sets
  两两不相交，并验证全部 manifest/source/artifact digests；任一冲突 fail closed；
- validation 只执行一次。FAIL 终止该版本；任何 candidate、artifact、门或 evaluator 实质修改也
  终止该版本，必须使用新版本与新预承诺 validation stream；原 holdout 保持未消费；
- 进程启动后 nonce 即被 CAS reservation。若在任何 metric/result bytes 可见前发生可证明的基础
  设施失败，可由 custodian 用同一 source/artifact/nonce 重启；一旦任何结果可见，partial output
  即按已消费处理，不得重跑；
- reference-health、coverage 或其它 invalid-split 条件在运行后失败时进入
  `VALIDATION_CONSUMED_NO_DECISION`，不得补 seed；
- 只有 `VALIDATION_CONSUMED_PASS` 且得到另行人工授权，才能 reservation 一次 holdout；holdout
  无论 PASS/FAIL 都只生成一次 immutable evidence，不能触发候选修改后重测同一 stream。

### Phase R：容量与纯 filter conformance

1. 完成第 6 节 capacity artifact；
2. privileged unpruned reference 必须直接调用 environment 的真实 hidden-motion kernel，不能复用
   candidate transition；另对全部局部 topology、四种速度、turn/no-turn 和边界情形做穷举对齐；
3. conformance 只验证传播公式，不声称 sensor-only integrated candidate 达到 privileged oracle；
4. base lock 只锁 reference、measurement harness/schema 和 evaluator；packed prototype 属于 candidate
   side，不得因此提前锁死正式 candidate；
5. 资源或累计 pruning 门失败则停止；final candidate source/config 有变化时按第 6.1 节重跑。

### Phase A：sensor-only-input / known-kernel privileged ceiling

- 使用正式 sensor/action API；
- 注入 environment 的真实 kernel 参数，但 static/visibility/identity 不可知；
- 必须输出 `privileged_diagnostic_only=true`，不得写入 frozen learner artifact、参与最终候选选择、
  进入 validation/holdout 或被报告成正式 learned candidate；
- 使用与正式候选相同的指标、资源和 gates 只作 Phase B feasibility Go/No-Go；
- 它是“运行输入 sensor-only、参数 privileged”的表示/关联诊断 ceiling，不要求逐位复现真实
  topology oracle。

若 Phase A 不通过，不进入 kernel learning；先修 front-end、unknown-static、association、
existence 或 posterior approximation。

### Phase B：kernel learner diagnostics

依次运行、但严格标记权限：

1. known kernel + oracle identity：privileged diagnostic；
2. learned kernel + oracle identity：privileged diagnostic，参数丢弃；
3. known kernel + 完整 sensor-only association（即 Phase A privileged ceiling）；
4. learned frozen kernel + 完整 sensor-only association。

四个条件使用相同 development seeds 和 reset policy，分别归因：

- transition learning；
- identity association；
- unknown-static uncertainty；
- posterior pruning；
- existence calibration。

### Phase C：完整正式候选

- 只用 40 个 train seeds 的完整 sensor/action streams fit；
- 加载 immutable frozen kernel artifact；
- 在 64 个 development evaluation seeds 上比较；
- 不重新选择 seed、turn_probability 或 evaluator；
- 记录候选/超参数尝试预算和选择日志；
- learned candidate 必须同时通过全部 confirmatory gates。

## 11. Artifacts 与测试

### 11.1 建议模块

- `cal/model/stochastic_motion_filter.py`
- `cal/model/stochastic_entity_belief_graph.py`
- `cal/evaluation/stochastic_permanence_kernel_diagnostic.py`
- `cal/evaluation/stochastic_permanence_benchmark.py`
- learned-kernel artifact schema/validator
- frozen protocol/decision/custody runner

### 11.2 必需 artifacts

- capacity/conformance development artifact；
- candidate-independent reference-health/power artifact；
- learned-kernel development artifact；
- 四象限 privileged diagnostic artifact；
- complete stochastic-I1 development audit；
- append-only attempt/selection ledger；
- frozen protocol JSON + `.sha256`；
- transitive base source/dependency lock；
- final candidate code-and-artifact-only amendment（含 frozen-kernel/config/capacity/ledger digests）；
- 后续 validation/holdout immutable evidence。

### 11.3 必需测试

- transition、第一 hidden-step 时序、turn、概率 unknown-static bounce、visible-empty conditioning；
- probability normalization、survival/existence、matched/missed detection、birth/clutter/death、
  one-to-one branch evidence；
- pruning mass、TV、overflow 和 empty-posterior policy；
- K/global-pool/COW fully-detached/full-candidate deep-size、training/evaluation persistent state、参数和
  MAC 最坏值；
- endpoint learner synthetic recovery；
- full train stream 与 evaluation seed 隔离；
- kernel artifact 不可变性；
- episode reset、seed 顺序不变性；
- oracle identity/true topology 不进入正式 API；
- full-distribution association；
- birth/death、stale retirement、point merge/collision；
- branch-local categorical self/action joint mixture、null class 与跨 hypothesis 隔离；
- candidate runner determinism；
- feasibility-native q/κ Cartesian grid 的 exact rows、构造与 power/coverage 隔离；
- outer development-seed bootstrap、逐 outer population q/κ recalibration 与不可实现 fail-closed；
- 物理 component-null 边界均值、metric bounds、四格 null outer recalibration 和 80-cell exact Type-I
  simultaneous upper bound；
- 108-cell coverage point/lower-bound 门与四场景 simultaneous power lower bound；
- policy/result 协调篡改、truthy 非布尔 gate、空 provenance、伪造 moment/physical evidence、seed
  recommendation 漂移均必须被 artifact validator 拒绝；
- 总体/逐 bin common-resample paired CI、1% lower bound、joint closure contrast、reference-health
  和 decision 重算；
- base/candidate dependency lock 分离、frozen-kernel/config/ledger amendment mismatch；
- seed-set 防碰撞、CAS 并发、validation 状态机、partial output 和 invalid-split policy。

## 12. 冻结与 Go/No-Go 顺序

1. 完成 Phase 0 evaluator、power、custody 和 transitive base-lock coverage；
2. 完成 Phase R capacity/conformance；
3. 独立 review 环境、runner、统计、资源和权限边界；
4. 冻结评测契约、门和 base source/dependency lock；
5. 实现 Phase A sensor-only-input / known-kernel privileged ceiling；
6. 实现 Phase B kernel learner diagnostics；
7. 实现 Phase C learned formal candidate；
8. 只在 train/development registry 上开发并记录候选选择；
9. 独立 review/fix，所有 P0/P1 清零；
10. amendment **只能增加**最终候选及其全部 transitive dependencies 的 source hashes，以及唯一
    frozen-kernel artifact SHA、kernel schema、fit command/train-collector digest、canonical candidate
    config、`model_seed=17011`、capacity/conformance artifact digest 和 attempt/selection ledger digest；
    不得修改 base lock、门、secret manifests 或引用结果；
11. 运行一次 validation；
12. validation 全门通过且另行授权后，才允许消费一次新 holdout。

## 13. 最终判断

该路线适合当前系统的科学目标，因为它直接测试：

- 显式实体信念能否传播随机隐藏动力学；
- learned sensor-only belief 能否同时改善定位和 proper scoring；
- 相对几何捷径、uniform 和旧 I1 是否存在必要增益。

但它不再把“最大风险不是计算量”作为前提。当前最高风险依次是：

1. 64 KiB 内的 posterior 表示；
2. association-weighted endpoint kernel identification；
3. sensor-only unknown-static 信息差；
4. existence/identity/spatial posterior 的一致概率更新；
5. evaluator、power 与一次性 evidence governance。

Phase R development gate 已通过；Phase 0 必须以本节固定的 outer-robust grid v6 重新生成并通过
strict artifact validation，才能形成 seed-count recommendation 和进入 base-lock review。无论
development 决定为何，本计划仍不授权生成或消费 validation/holdout。
