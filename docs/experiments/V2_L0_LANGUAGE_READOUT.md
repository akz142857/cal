# V2-L0：冻结实体世界模型的受控语言可读性验证

日期：2026-07-27

当前状态：V4 修复版 development 全部门通过，三路最终独立 review 均无
P0/P1 阻塞；等待后续 clean commit 与 exact source-lock。新的 review holdout
尚未运行，也尚未授权。

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

V3 冻结成对命题、严格覆盖、新候选 holdout 和新增门。V4 记录 V3 的正式负
结果，并在跨 episode identity 控制实现前冻结反捷径规则。两者都只授权重复
development，不授权 holdout。

## 8. V4 development 结果

结果：
`results/V2-L0-language-readout-development-v4.json`

当前结果 SHA-256：
`e081d84cb3a235aee7c918c040aa9edb65bf38539409a887ee09526763035edd`

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

## 9. 如何运行与下一道门

可重复 development：

```bash
uv run cal-v2-l0-language --split development
```

机制与攻击回归测试：

```bash
uv run pytest tests/test_v2_l0_language_readout.py -q
```

当前命令会拒绝 holdout：

```bash
uv run cal-v2-l0-language --split holdout
```

只有最终 review 无阻塞问题、全部修复完成，并生成锁定 evaluator、测试、I1
传递依赖、development 结果、完整 provenance 和 clean commit 的后续协议后，
才可以另行请求一次性 holdout 授权。正式 holdout 还必须先原子创建本地
reservation 和 origin CAS consumption tag，运行前后源码一致，并把结果发布为
不可变 evidence tag。
