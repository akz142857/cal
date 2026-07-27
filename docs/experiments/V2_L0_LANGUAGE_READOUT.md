# V2-L0：冻结实体世界模型的受控语言可读性验证

日期：2026-07-27

当前状态：V4 修复版 development 全部门通过，三路最终独立 review 均无
P0/P1 阻塞；V5 exact source-lock 已发布。唯一一次 V5 review holdout 已获得
授权并被原子消费，但身份负对照无法在该 holdout 事件集合上构造，运行在产出
指标前终止。协议禁止重试，因此没有 V5 holdout 通过结论。后续 V6 已在实现前
冻结逐行身份反事实，development 全部 24 门通过；新的 holdout 仍未运行，
下一步是独立审查与 V7 exact source-lock。

## 1. 这个实验究竟验证什么

I1 从局部二值占据视觉和动作副本中形成 self、实体、运动、遮挡与持续性表征。
L0 冻结 I1，只训练一个很小的线性读出器，询问：

> I1 的统一实体信念图，是否比学习器当下看到的原始占据栅格，更容易读出
> self、空间关系、遮挡后持续存在和遮挡前后身份匹配等受控语义？

通过只代表“受控语义可读性 + 中文模板渲染”成立。它不代表系统已经学习
自然语言、理解自由文本、生成语言、服从语言指令，或完成像素到语言的端到端
grounding。

## 2. 不可越过的学习边界

I1 每一步仍只收到：

```text
update(local_binary_occupancy, executed_action_copy)
```

运行顺序是：世界产生视觉和动作 → I1 正常更新 → evaluator 才构造查询和标签。
语言标签、世界真值、指称位置从不进入 I1，线性 probe 的梯度也不回传到 I1。
正式特征不包含内部数字实体 ID；运行时会对全部 ID 做任意双射重命名，并要求
特征字节和 probe logits 完全相同。

## 3. 十个命题和真正的成对查询

V4 同时预测十个二值命题：

1. 第一处标记是否指向 self；
2. 第二处标记是否指向 self；
3. 横向运动者是否在 self 左侧；
4. 横向运动者是否在 self 上方；
5. 纵向运动者是否在 self 左侧；
6. 纵向运动者是否在 self 上方；
7. 第一个隐藏位置是否仍有实体；
8. 第二个隐藏位置是否仍有实体；
9. 第一个重现候选是否匹配遮挡前参考；
10. 第二个重现候选是否匹配遮挡前参考。

self 查询每个有效时刻各指向一次 self 和一个干扰者。持续性查询在目标连续隐藏至少两步
后，同时给出一个隐藏真实目标位置和一个隐藏、空且非静态的位置，并轮换正负
顺序。身份查询保存目标遮挡前的 ID-invariant 描述；目标连续隐藏至少两步并
重新出现后，将同一个历史参考与“原目标”和“另一干扰者”两个同帧候选配对，
同样轮换正负顺序。

每个命题必须在 development validation 至少有 20 个正例和 20 个负例。
任何注册列缺少任一类别时，balanced accuracy 不得跳过该列，实验直接失败。

中文句子只是 train、validation、holdout 三套预先冻结的表面模板。probe 预测
命题真假，再把概率填入模板；它没有解析中文。

## 4. 表征和隔离

正式 I1 表征由 15 张 `11×11` ID-invariant 图组成，包括占据概率、推断可见性、
当前感知、静态背景、实体质量/存在/年龄/漏检、运动方向/速度、self 证据/
后验/选择，再加动作 one-hot。

六个查询块分别属于 self 两个候选、持续性两个位置、身份两个候选。每块包含
位置 mask、当前位置描述、保存的历史描述、逐元素乘积与绝对差。V4 的四个线性 head
严格隔离：

- self head 只读候选的 `self_evidence / self_posterior / selected_self`，
  不读绝对位置、base 或一般运动通道；
- spatial head 只读 base；
- permanence head 只读 base 与隐藏位置查询；
- identity head 只读候选/参考描述及乘积、差值，不读绝对位置或 base。

因此 evaluator 为一个概念构造的查询不会泄漏给其他概念。正式特征 2906 维，
raw 对照 876 维；线性 probe 共 13102 个参数，低于冻结上限 50000。

## 5. 负对照

| 条件 | 被破坏的内容 |
|---|---|
| Raw sensor | 只保留当前 learner-facing 占据、动作和同构查询 |
| Time shuffled | episode 内把实体状态错开 17 步，查询保持当前 |
| Referent swapped | 交换训练时的运动者、隐藏位置和身份候选指称 |
| Random labels | 每列训练标签独立随机排列 |
| No action | I1 不利用真实动作，其余条件不变 |
| Assume all visible | 关闭 I1 遮挡推断，检验隐藏状态是否必要 |
| Identity scrambled | 保持正式 probe 不变，仅在 validation 跨 episode 换成相反运动角色的错误参考 |

最后两个对照直接回答“高分是不是由真正需要的机制带来”：持续性必须依赖遮挡
推断；身份匹配必须依赖正确的遮挡前参考。

## 6. V1–V3 为什么不能作为最终证据

V1 的 self 问题是“我是否在固定摄像中心左侧/上方”，位置先验即可回答，导致
no-action 对照不成立。V2 改成 deictic self 查询后 development 通过，但三路
独立 review 发现：

- 两个持续性列没有完整正负类别，旧指标却静默跳过；
- “身份”只是用重现窗口筛选当前空间关系，没有把当前候选与遮挡前参考匹配；
- protocol、输出路径、源码锁和一次性 holdout 防护不够强；
- 缺少逐命题、逐种子证据及运行时完整性审计。

所以 V1/V2 结果只作为负结果和问题发现记录，不能授权 holdout，也不能支持
“客体持续性/身份匹配已验证”的最终主张。

V3 初次修复后又在 development 暴露两个捷径。其一，代码一度把“两个 actor
落在同一可见格、因此不唯一”误算成遮挡；改为严格的 sensor-invisible 后，
raw 当前帧仍可用绝对位置和固定运动角色回答 identity。其二，V3 的 episode 内
reference rotation 在多数 episode 只交换了同一 actor 的不同时间描述，没有
真正破坏 referent。正确的负结果被保存在：

```text
results/V2-L0-language-readout-development-v3.json
```

它的 decision 是 `stop_and_report`。V4 没有降门，而是冻结了三项反捷径修复：
self 只读显式 self 通道；identity 不读绝对位置和 global base；identity 控制
保持正式 probe 不变，在 development validation 中把历史参考换成不同 episode
的相反运动角色描述，并要求实际替换率写入证据。

审查时意外查看了旧计划 holdout `33200–33203` 的事件结构，因此这些 seed 已
永久作废。V3 在修复实现前冻结新的、从未运行的候选 holdout
`33400–33403`；在最终源码锁协议生成前严禁执行。

## 7. 冻结协议链

- V1：`experiments/V2_L0_LANGUAGE_READOUT_PROTOCOL.json`
  (`39026a8e…9493`)
- V2：`experiments/V2_L0_LANGUAGE_READOUT_PROTOCOL_V2.json`
  (`bb70d798…2e3`)
- V3：`experiments/V2_L0_LANGUAGE_READOUT_PROTOCOL_V3.json`
  (`35aa6458…aa9`)
- V4：`experiments/V2_L0_LANGUAGE_READOUT_PROTOCOL_V4.json`
  (`85aee950…f6a9`)
- V5：`experiments/V2_L0_LANGUAGE_READOUT_PROTOCOL_V5.json`
  (`51a4f561…1aae`)
- V6：`experiments/V2_L0_LANGUAGE_READOUT_PROTOCOL_V6.json`
  (`6742a36c…76aa`)

V3 冻结成对命题、严格覆盖、新候选 holdout 和新增门。V4 记录 V3 的正式负
结果，并在跨 episode identity 控制实现前冻结反捷径规则。两者都只授权重复
development，不授权 holdout。

## 8. V4 development 结果

结果：
`results/V2-L0-language-readout-development-v4.json`

当前结果 SHA-256：
`6af1883cbf163f551184ba5c1284c233e13648bde15c57d9cc168943def6ddb8`

| 条件 | Macro | Self | Spatial | Permanence | Identity |
|---|---:|---:|---:|---:|---:|
| Formal entity graph | **0.9715** | **0.9967** | **0.9068** | **0.9825** | **1.0000** |
| Raw sensor | 0.6684 | 0.5000 | 0.8307 | 0.8430 | 0.5000 |
| Time shuffled | 0.6153 | 0.5345 | 0.6411 | 0.6770 | 0.6086 |
| Referent swapped | 0.4324 | 0.9967 | 0.7155 | 0.0175 | 0.0000 |
| Random labels | 0.4737 | 0.5068 | 0.4848 | 0.5000 | 0.4031 |
| No action | 0.9072 | 0.8738 | 0.8319 | 0.9397 | 0.9836 |
| Assume all visible | 0.8986 | 0.9286 | 0.8775 | **0.7882** | 1.0000 |
| Identity reference replaced | 0.7345 | 0.9967 | 0.9068 | 0.9825 | **0.0521** |

正式图比 raw 的 macro 高 `0.3031`，持续性高 `0.1395`，身份高 `0.5000`。
关闭遮挡推断使持续性下降 `0.1943`；保持正式 probe、只替换历史参考使身份从
`1.0000` 降到 `0.0521`。296 个 active reference blocks 全部改变，全部换成
相反运动角色。四个验证 seed 的 formal macro 分别为
`0.9841 / 0.9540 / 0.9819 / 0.9839`，
全部高于逐 seed 下限 `0.60`。十列正负覆盖、输入边界审计、任意 ID 重命名
审计和全部固定门均通过，decision 为：

```text
authorize_review_and_source_lock
```

这仍是可重复 development 证据，不是 holdout 结论。

## 9. V5 exact source-lock 与一次性边界

V5 协议：

```text
experiments/V2_L0_LANGUAGE_READOUT_PROTOCOL_V5.json
SHA-256 51a4f561bceb23de2c9c483895b82e2f5b1cd4168736b22b166e236be6ce1aae
```

V5 锁定最终 evaluator、测试、I1 传递依赖、provenance、环境清单、passing
development-v4 结果以及完整 source digest。协议本身由 clean commit 和 origin
annotated source-lock CAS tag 共同绑定，避免在协议中写入包含自身的 commit
hash 所造成的循环依赖。

未来 holdout 必须依次满足：

1. clean commit 与 origin `calmodel-l0-v5-source-locked` 完全一致；
2. 用户另行授权后，存在 `calmodel-l0-v5-holdout-authorized`；
3. 第一条 episode 前 CAS 创建 `calmodel-l0-v5-holdout-consumed`；
4. 第一条 episode 前原子创建本地 reservation；
5. run start/end commit 与 source digest 完全相同；
6. 结果作为 Git blob 发布到 `calmodel-l0-v5-holdout-evidence`。

consumption 一旦成功，即使运行中断也不会恢复机会。

## 10. 唯一一次 V5 holdout 的最终状态

2026-07-27，源码锁、显式授权与消费按冻结顺序成功发布：

- `calmodel-l0-v5-source-locked`
- `calmodel-l0-v5-holdout-authorized`
- `calmodel-l0-v5-holdout-consumed`

消费凭证和本地 reservation 都绑定到源码锁定提交
`d43651f4e06f513cddbb6dfe354dd9073f46b0a1`、协议 SHA
`51a4f561bceb23de2c9c483895b82e2f5b1cd4168736b22b166e236be6ce1aae`
与 source SHA
`063888a18c098106780b82a141ba1785a2b4e4bb7b6f4785224016c3728d5246`。

运行随后在构造 `identity_scrambled_at_occlusion` 负对照时终止：

```text
ERROR: identity scramble requires both reference roles
```

这表示本次 holdout 收集出的 active identity 事件，不能满足冻结实现所要求的
跨 episode、按运动角色替换参考的结构条件。它发生在完整结果与 gates 生成前，
所以不是“模型通过”或“模型分数不及格”，而是实验设计对该 holdout 样本覆盖的
前置条件没有满足。

最终可审计状态为：

- 本地 reservation：`consumed_before_first_episode`；
- 正式 holdout result：未生成；
- `calmodel-l0-v5-holdout-evidence`：未生成；
- 唯一一次机会：已消费，按协议不可重试；
- 结论：V5 holdout 未验证当前语言可读性主张。

失败记录保存在：

```text
results/V2-L0-language-readout-holdout-v5-failure.json
SHA-256 4b527f911bcdd803e573308ec80f6a1cfa9ade2e43b93679993dd2c22f5f947d

results/V2-L0-language-readout-holdout-v5-reservation.json
SHA-256 bfdefd0185f74f114fe45880208d8c11f21596db4b3a6987144c9f354abbe2d7
```

若要继续验证，必须把这次结果保留为不可变负证据，另行设计并审查新的协议、
新的未见数据和新的独立一次性注册表；不能修改 V5 后复用本次 holdout。

## 11. V6 下一轮验证

V6 在实现前由提交 `91491d044088d733f2cea8e21f10f9469ade29bd`
冻结。它不修改 I1、命题、probe、阈值或 development 数据，只把失败的
跨-episode 身份控制改成逐行反事实：

> 对每个有效身份查询，保持当前两个候选、标签和学习器状态不动，仅把遮挡前
> 参考替换成同一行中另一个可见干扰者的 ID-invariant 描述。

身份查询本身已经要求两个干扰者在当帧唯一可见，因此该反事实对每个 active row
都有定义，不需要从另一个 episode 寻找配对，也不需要预看 holdout 结构。

干净实现提交：
`515a20dbcf7da761468efe16c5a5eacd161f4844`

development 结果：

```text
results/V2-L0-language-readout-development-v6.json
SHA-256 c02b370ad4d3dd712afd929d91d4ad0ee08a37225f2e095f687b018733139b42
source SHA-256 30230659f14e2146c17694db2727e649b96537d5c18d35c08f1f6bded3b55cee
```

结果绑定到上述干净实现提交，`git_dirty=false`。全部 24 个 gates 通过，
decision 为 `authorize_review_and_source_lock`。关键指标：

| 条件 | Macro | Permanence | Identity |
|---|---:|---:|---:|
| Formal entity graph | **0.9715** | **0.9825** | **1.0000** |
| Raw sensor | 0.6684 | 0.8430 | 0.5000 |
| Assume all visible | 0.8986 | 0.7882 | 1.0000 |
| Row-local identity counterfactual | 0.7371 | 0.9825 | **0.0625** |

逐行反事实使 identity 从 `1.0000` 降到 `0.0625`。296/296 个 active
reference blocks 全部改变，反事实逐行匹配率与角色切换率均为 `1.0`；标签、
mask、当前候选和非身份特征保持不变。

V6 在实现前登记了新的候选 holdout `33600–33603`。这些 seed 当前未检查、
未运行，也不因 development 通过而自动获得授权。下一步必须先完成独立审查，
再冻结 V7 exact-source-lock 与全新 origin tags；仍需用户另行授权后才能消费。

## 12. 如何重复已允许的部分

可重复 development：

```bash
uv run cal-v2-l0-language --split development
```

机制与攻击回归测试：

```bash
uv run pytest tests/test_v2_l0_language_readout.py -q
```

V5 holdout 已消费，不能再次运行。V6 当前只允许 development；新的 holdout
必须先通过审查和 V7 source-lock，并使用新的 tag 名称。
