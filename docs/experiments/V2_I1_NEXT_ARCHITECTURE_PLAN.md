# V2-I1 下一代架构开发方案：统一实体信念图

日期：2026-07-27
状态：V4 review 修复完成，等待 clean commit 后的正式校准
对应协议：`experiments/V2_I1_INTEGRATION_PROTOCOL_V4.json`（V2 冻结
完整架构与门；V3 在任何新 development 运行前仅修订不可移植的前置证据
路径；V4 在子 Agent review 后、任何新 validation 或 holdout 前冻结算法、
指标、资源核算和一次性执行修复）

## 1. 决策

旧 I1 停止在开发集，不消费 31000–31015 一次性留出。下一代不再把
`OnlineEntityGraph` 与 `UnprivilegedOccupancyMemory` 两套独立实体跟踪器
拼接，而是新建 **统一实体信念图（Unified Entity Belief Graph, UEBG）**：
一次关联结果同时驱动身份连续性、自我归属和遮挡下对象永久性。

旧 `entity_graph.py`、`occupancy.py`、V1 runner、V1 协议和负结果保持不变。
新实现放在独立文件中；在 scratch/开发验证证明有净收益之前，不再修改
M1–M3 锁定组件。

## 2. 旧架构被证伪的接缝

1. 身份图与占据记忆各自提取检测、各自维护 track，没有共享实体 ID；
2. 一次错误硬关联立即污染唯一的控制估计，后续只能预防，不能回滚；
3. self lock 是独立于关联后验的阈值状态机，oracle 关联下 F1 仍只有
   0.6517；
4. M4 reachable-floor 是 visibility-agnostic，真实可见性 oracle 几乎不
   改善隐藏概率，说明永久性必须改为实体条件的运动后验。

## 3. 正式输入与禁止项

正式 agent 每步唯一输入仍是：

```text
update(sensed_occupancy, action_copy)
```

允许：

- 从 sensed occupancy 自行做 shadow casting；
- 在线维护有限个关联假设、控制转移计数和运动后验；
- 在固定内存内保留未决身份，不重放经验。

禁止：

- 身体、self、物体 ID、可见性真值或位置真值进入学习；
- 从旧留出结果选择参数；
- 修改旧冻结协议或把旧失败改写为通过；
- 以降低现有门替代机制修复。

## 4. 模块设计

### 4.1 `DynamicCellFrontEnd`

维护逐格静态证据。可见且持续占据的格逐步成为背景；可见空格快速撤销
静态判断。检测是 `sensed occupied - learned static cells`，因此运动点
即使贴着墙也不会被墙 blob 吞并。背景学习只使用感知和自行推断的可见性。

### 4.2 `EntityBelief`

每个实体保存：

- 稳定 ID、位置、速度、存在概率、last-seen/missed；
- 动作条件离散位移计数；
- 自主运动位移计数；
- 动作依赖的累计对数证据；
- 当前分支下的 self logit。

动作条件模型不是预装的身体动力学。它从
`P(delta | supplied action)` 相对 `P(delta)` 的在线似然比学习控制关系。

### 4.3 `GlobalHypothesisBank`

每步先预测、后观测。每个旧分支对所有 track/detection 做有限宽度全局
beam assignment，显式允许：

- match；
- visible miss；
- occluded miss；
- birth；
- crossing 时的多个身份排列。

只保留最多 5 个全局假设。错误关联保留为低权重分支，而不是立即污染唯一
状态。分支权重由位置预测、动作条件预测、连续速度、漏检可见性和存在概率
共同更新。

### 4.4 概率 self 归属

每个分支内对 live entities 的 self logits 做 softmax，再按全局分支权重
边缘化。内部不使用永久 hard lock；对外仍输出最大后验 ID，以兼容冻结
`self_f1` 指标。没有足够动作依赖证据时允许输出 `None`。

### 4.5 实体条件对象永久性

遮挡时实体沿同一分支的动作/自主运动后验传播，存在概率仅轻微衰减；
agent 自己判断为可见空时则强烈衰减。全局占据概率由各分支实体位置后验
边缘化得到，不再绘制与实体身份无关的 reachable floor。

## 5. 有界实现

首版固定：

| 项目 | 上限 |
| --- | ---: |
| 全局假设数 | 5 |
| 每分支实体数 | 11 |
| 位移类别 | stay + 4-neighbor + other |
| 重识别/遮挡时限 | 40 步 |
| 静态背景阈值 | 5 次净支持 |

不为每个假设复制 25×25 占据网格；网格只保存共享静态证据，动态占据按需
从实体分支渲染。必须继续满足 100k 参数、64 KiB 活动状态、5M MAC/步。

## 6. 开发与判定顺序

### 阶段 A：单元机制

1. 静态背景能分离贴墙运动点；
2. 动作相关轨迹的 self posterior 高于自主运动轨迹；
3. crossing/短遮挡后稳定 ID 不交换；
4. inferred-occlusion 下隐藏实体保持，all-visible 下同一空观测被清除；
5. 输入签名和资源门通过。

### 阶段 B：校准集

- 30000–30015 已在 V3 开发中全部暴露，V4 将其明确降为可重复校准集；
- 现有全部绝对门和对照门不降低；
- 身份指标以全部可见且可区分的真值机会为分母，漏报计失败；
- 新增 `visible_identity_coverage >= 0.90`，防止“只报一次也得 1.0”。

### 阶段 C：一次性新验证

- 32000–32007 在 V4 冻结前未运行；
- review 修复必须先提交到 clean commit；
- validation reservation 先以共享 origin 上的新 Git tag 原子消费，再写本地
  记录；跨机器并发时只有一个 push 能成功，崩溃也视为已消费；
- 通过结果保存为 Git blob，并以 annotated tag 固化结果 SHA 和授权证书；
- validation 通过后才能进入原留出。

### 阶段 D：一次性留出

只有以下条件同时成立才运行 31000–31015：

1. 阶段 A 测试和全量 pytest 通过；
2. calibration 与 one-shot validation 全部门通过；
3. 协议 SHA sidecar 匹配；
4. 正式路径无标签、无真值可见性；
5. 工作区中影响正式算法的改动已被审查。

留出只运行一次，不因失败调参或重跑。

## 7. Review 要求

开发完成后至少从以下角度做独立子 Agent review：

1. 多目标跟踪/关联正确性；
2. 概率、数值稳定性与校准；
3. 无标签/无特权和协议完整性；
4. 资源核算、测试覆盖与可维护性。

所有发现必须逐项判定、修复或记录不采纳理由，然后复跑 calibration、
协议审计和全量测试。若修复改变正式算法且留出尚未运行，更新协议修订记录；
若留出已经运行，不得再改变算法后重跑同一留出。
