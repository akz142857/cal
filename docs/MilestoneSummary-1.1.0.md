先给结论：以“当前已经完成的系统”评分，而不是按未来设想评分。

- 方向：8.0 / 10
- 当前综合价值：6.0 / 10
- 潜在价值：如果跨过真实视觉、跨场景泛化和闭环行动三道门，可到 8.5 / 10；现在还不能按这个分数宣传。

这里的“价值”指已经形成、能被第三方验证和利用的价值，不是想象中的商业上限。

> 相对 1.0.0 的更新：纳入了 L0 V8 唯一留出的失败结果（含直接从冻结结果 JSON 逐对照组读出的一手证据分析，见第二节末）、当前项目的开发规模事实（约 31k 行、312 个测试、47 次提交集中在 2026-07-24 至 07-28 的四天内），并把“全球同类型”一节更新到 2025–2026 年年中的公开进展（Embodied-SlotSSM、OCWM×MCTS、V-JEPA 2-AC 真实机械臂闭环等）。评分骨架承接 1.0.0，但关键结论已用一手结果数据重新核验。日期：2026-07-28。

## 评分总览

| 维度 | 评分 | 一句话判断 |
|---|---:|---|
| 研究方向 | 8.0 | 站在 object-centric × 具身自我模型 × world model × grounded language 的正确交叉点上 |
| 实验方法/证据治理 | 8.5 | 冻结协议、源码哈希锁、一次性留出、负对照、失败留痕，超过绝大多数个人乃至团队项目 |
| 当前算法证据 | 5.5 | 合成环境内强，但 L0 V8 唯一留出未通过，真实输入与跨环境证据缺失 |
| 工程可复用性 | 6.5 | 代码、测试、确定性回放、证据链可复用；但四天冲刺、单人、无外部复现 |
| 学术新颖性 | 5.0 | 组合独特，但每个核心组件都已有更成熟的公开对应工作 |
| 当前产品价值 | 2.5 | 距 FSD/机器人/真实视觉产品尚有真实视觉、跨场景、闭环三道门 |
| 综合当前价值 | ≈ 6.0 | 有可信内部结果的研究原型，尚未形成外部 benchmark 验证的成果 |

## 一、方向为什么是 8 分

方向是成立的，而且与公开研究前沿基本同向：

> 先通过视觉与行动建立自我、实体、身份和持续存在的内部状态，再让语言读取这些状态，最后才进入预测、规划与行动。

这条路线比“直接接一个大语言模型，让它看起来会说”更有科学意义，因为语言有机会对应一个真实、可追踪、可证伪的内部世界状态。

当前做对了三件比较关键的事：

1. **把 self 建立在行动—感知因果关系上**

   系统不是依靠外观标签认出“我”，而是通过动作与视觉变化的对应关系识别 self。这与机器人领域长期研究的 sensorimotor contingency/self-perception 路线一致。慕尼黑工业大学的机器人自我感知研究同样利用动作造成的感觉变化区分 in-body 与 out-body；Columbia 的视觉自模型也研究如何仅从第一视角视觉学习自身动力学。V8 留出中 self 平衡准确率 0.9993，是全项目最强、也最干净的结果。[TUM sensorimotor self-perception](https://portal.fis.tum.de/de/publications/yielding-self-perception-in-robots-through-sensorimotor-contingen/)、[Columbia egocentric visual self-modeling](https://doi.org/10.1038/s44182-025-00031-6)

2. **使用统一实体状态，而不是互不相连的任务输出**

   self、其他实体、运动、遮挡、持续存在和身份历史都放入统一 Entity Belief Graph。这符合 object-centric learning 的核心思想：世界应被拆成持续存在、能够独立运动的实体，而不是只有一整块图像特征。

3. **语言建立在状态之上，而不是代替状态**

   L0 冻结 I1，再用线性读出器检查 self、空间、持续存在和身份语义是否能被简单读出。这是很干净的因果设计：如果简单读出器已经能读取，说明语义在底层状态中确实存在；如果必须靠大型语言模型“猜”，就很难判断能力来自哪里。

另外，V2 计划把 FSD 的系统原则（视觉优先、视频而非单帧、占据先于语义、不确定下规划、失败闭环、端侧效率）当作设计约束，同时明确拒绝“用车队规模掩盖机制不清”。这个取舍是清醒的。当前项目对协议冻结、负对照、数据污染和 one-shot 证据的重视，高于大多数个人研究项目。这部分本身有研究方法价值。

## 二、为什么不是 9 分或更高

主要限制不是代码质量，而是外部有效性。

当前系统仍然是：

- `11×11` 二值占据栅格；
- 固定摄像机；
- 三个视觉同构实体；
- 离散且直接提供的动作副本；
- 人工构造的遮挡环境；
- 固定的中文命题模板；
- 没有原始 RGB、深度、噪声、复杂背景和真实物体外观；
- 没有利用内部状态进行长期预测、规划和闭环行动。

因此，今天把它称为“通用世界模型”还偏早。更准确的名称是：

> 一个经过机制验证的、动作条件化的实体信念状态学习器。

现代公开研究中，“world model”通常还要求预测未来状态，有些还要用预测结果规划行动。Meta 的 V-JEPA 2、DreamerV3 等都把“根据当前状态和动作预测未来，并用于规划/控制”作为关键能力。[Meta V-JEPA 2](https://arxiv.org/abs/2506.09985)、[V-JEPA 2 官方开源仓库](https://github.com/facebookresearch/vjepa2)、[DreamerV3](https://github.com/danijar/dreamerv3)

当前系统已经有成为世界模型的“状态基础”，但还没有完成完整闭环。

另一个必须扣分的事实是 L0 V8 唯一留出未通过：

- I1 已通过冻结 validation 和 holdout（13 个 gate 全 true）；
- L0 development（V4/V6）通过所有 24 个开发 gate，并通过多轮独立复审；
- V8 唯一 holdout 已于 2026-07-28 运行并消费，最终为 `stop_and_report`：24 个 gate 中 21 个通过，3 个失败。

三个失败的冻结对照是：

- `formal_beats_raw_permanence` 失败：实体图 permanence 平衡准确率 0.8021，未优于 raw sensor 的 0.8750；
- `all_visible_permanence_fails` 失败：assume-all-visible 对照本应失败却达到 0.8542；
- `identity_scramble_integrity` 不成立：反向运动覆盖仅 0.8777，非完整。

因此，现在可以说“部分受控语言语义（self、空间、身份）在 development 和 holdout 中线性可读”，但**不能**说“完整 L0 语言能力通过独立 holdout 验证”，尤其“隐藏物体永久性依赖实体图”这一最有雄心的主张没有被独立留出证据支持。该留出已消费、不可重试——诚实但代价高。这与项目自己的 [I1 报告](experiments/V2_I1_NEXT_ARCHITECTURE_RESULT.md) 和 [L0 报告](experiments/V2_L0_LANGUAGE_READOUT.md)、[research status](../RESEARCH_STATUS.md) 一致。

### L0 V8 留出的逐对照组一手证据

以下不是从摘要转述，而是直接从冻结结果
`results/V2-L0-language-readout-holdout-v8.json` 的八个对照条件读出的平衡准确率：

| 条件 | self | spatial | identity | permanence | macro |
|---|---:|---:|---:|---:|---:|
| formal_entity_graph（真模型） | 0.999 | 0.875 | 0.885 | **0.802** | 0.890 |
| raw_sensor（裸传感器） | 0.500 | 0.787 | 0.500 | **0.875** | 0.665 |
| no_action（删动作） | 0.862 | 0.820 | 0.944 | **0.885** | 0.878 |
| assume_all_visible（假设全可见） | 0.960 | 0.845 | 0.996 | **0.854** | 0.914 |
| time_shuffled（时间打乱） | 0.501 | 0.644 | 0.633 | 0.635 | 0.603 |
| identity_scrambled（身份打乱） | 0.999 | 0.875 | 0.162 | 0.802 | 0.710 |
| referent_swapped（指称交换） | 0.999 | 0.661 | 0.115 | 0.146 | 0.480 |
| random_labels（随机标签） | 0.506 | 0.485 | 0.455 | 0.479 | 0.481 |

从这张表能读出四条一手结论：

1. **self 归属是被对照组干净验证的强正结果。** raw=0.500、time_shuffled=0.501、
   random=0.506 全部塌到随机，只有需要“动作×时间对应”的 formal 到 0.999。这支持
   self 来自 sensorimotor contingency、而非外观捷径。identity 同理（scramble→0.162、
   swap→0.115 干净塌陷）。spatial 是对 raw 的温和提升（0.875 vs 0.787）。

2. **permanence 不是“没通过门槛”，而是一个明确的负结果。** 把 permanence 一列排序：
   no_action(0.885) > raw(0.875) > all_visible(0.854) > **formal(0.802)**。真模型的
   permanence 低于所有非退化对照：删掉动作会更好、假设从不遮挡会更好、裸传感器也更好。
   I1/M4 整条线的招牌能力（遮挡下物体永久性），在留出上恰恰是唯一被平凡基线反超的一项。

3. **这个负结果扛得住“过拟合”反驳。** 探针特征维度 formal=2906，而 permanence 仅 96
   样本、identity 278、隐藏格命题 48，是 p≫n 的重度超参数化，绝对可读性数字本应谨慎看。
   但 formal 探针容量大于 raw（876 维），permanence 却更低；若是容量假象，容量更大的
   formal 不该更低。所以 permanence 缺陷不是容量伪迹，而是机制本身没有贡献可读的永久性
   信息。这让负结果更可信，也说明问题定位在 I1/M4 的遮挡—永久性机制，而非语言读出层。

4. **一个需要复审解释的测量异常。** random_labels 下
   `first_pointed_entity_is_self`=0.973、`second_pointed_entity_is_self`=0.039——标签
   打乱后单命题走极端（近镜像），宏观被平均成 0.48 才让 `random_labels_fail` 门通过。
   这提示 self 探针在小样本命题上可能咬住近常数特征；宏观门能过，但 permanence/identity
   这些小 N 命题的边际是脆弱的。这是后续复审应追问的点。

结论：这套负对照的价值恰恰在于让 permanence 失败被**诊断出来**而不是被掩盖——这也是把
“实验方法/证据治理”评为 8.5 的直接依据。“当前算法证据”的 5.5 因此可以精确辩护，而非
模糊扣分：self/identity 是被对照验证的强正结果，spatial 是温和提升，permanence 是一个
扛得住过拟合反驳的干净负结果。

## 三、当前价值为什么是 6 分

我会这样拆分：

| 价值维度 | 分数 | 判断 |
|---|---:|---|
| 研究方向价值 | 8.0 | 问题分解正确，符合 object-centric、embodied、world-model 路线 |
| 实验方法价值 | 8.5 | 冻结协议、源码哈希锁、一次性证据和失败记录很扎实 |
| 当前算法证据 | 5.5 | 合成环境内强，但真实输入与跨环境证据不足，且 L0 permanence 对照未过 |
| 可复用工程价值 | 6.5 | 代码、测试、证据链和交互回放可复用；但单人四天冲刺、无外部复现 |
| 学术新颖性 | 5.0 | 组合有特色，但各核心组成部分已有相近且更成熟的研究 |
| 当前产品价值 | 2.5 | 尚不能直接用于 FSD、机器人或真实视觉产品 |

综合约为 **6.0/10**。

这不是低分。对一个探索阶段系统来说，6 分意味着：

> 已经不是概念或玩具代码，而是一个有可信内部结果的研究原型；但尚未形成经过外部 benchmark 验证的算法成果，更不是可部署产品。

当前最有价值的资产不是“它已经会语言”，而是：

- 建立了一条可解释的感知—自我—实体—语义链；
- 能用负对照解释能力为什么出现（以及本次为什么在 permanence 上没有出现）；
- 有严格证据治理（`locked_source_sha256` 把六个核心文件哈希钉进协议，改动即在下次确认运行时抛错）；
- 后续真实场景实验可以明确知道是哪一层失败。

需要泼的冷水：整个 `cal/` 约 31k 行、312 个测试、49 个评测脚本，但 47 次提交集中在 2026-07-24 至 07-28 的四天内。这说明系统年轻、单人、尚无外部复现——这正是 ROADMAP 把“获得 I1 结果的独立复现”排在第一条的原因。证据治理做得再好，也替代不了第三方复现。

## 四、公开研究中有没有相同方案

### 结论

根据我能找到的公开论文、实验室页面和开源仓库：

> 没有发现与 Cal 完全相同的公开实现，但每个主要组成部分都有强相关研究，而且其中一些在 2025–2026 已明显推进到真实视频、机器人或规划闭环。

“没有找到完全相同”不是对原创性的证明；未公开的实验室项目、不同术语下的工作以及最新未索引论文都可能遗漏。

### 最接近的公开方向（含 2025–2026 新进展）

| 项目/实验室 | 相似之处 | 主要差异 | 开源情况 |
|---|---|---|---|
| Embodied-SlotSSM、Object-Centric Structured World Models | per-slot 状态空间 + 关系注意力，在部分可观测环境中跨遮挡持续跟踪对象，做动作预测——**结构上目前最接近 Cal 的实体图** | 不强调从动作归因 self，也没有 Cal 的显式多假设/校准与语言读出链 | [OCSWM 综述](https://www.emergentmind.com/topics/object-centric-structured-world-models) |
| Object-Centric World Models × MCTS、Objects matter (RL) | 已把对象中心状态用于**规划/强化学习** | 正是 Cal 尚未做的闭环 | [OCWM×MCTS](https://arxiv.org/html/2601.06604)、[Objects matter](https://arxiv.org/html/2501.16443v1) |
| Meta FAIR V-JEPA 2 / 2-AC | 从视频学习世界表征；动作条件版本已在**真实 Franka 机械臂零样本 pick-and-place** | 大规模神经潜空间，不显式维护 self/身份/遮挡实体图 | [论文](https://arxiv.org/abs/2506.09985)、[官方代码](https://github.com/facebookresearch/vjepa2) |
| KUIS AI CarFormer | object-centric BEV、未来状态预测、CARLA 自动驾驶，与 FSD 路线非常接近 | 用 Slot Attention 学习车辆对象并直接驾驶，没有动作归因 self 和受控语言读出 | [论文/代码/模型公开](https://kuis-ai.github.io/CarFormer/) |
| Seoul National University LSlotFormer | 对象中心世界模型与语言结合，是高层结构上最接近 L0 的工作 | 它用语言指令**控制**未来状态预测；Cal 是从已有状态向外**读取**语言语义，因果方向相反 | [公开论文/ICLR Workshop](https://openreview.net/pdf?id=CMItmXqrue) |
| TUM/Columbia 机器人 self-model | 从动作—视觉或多模态感觉关系发现自身、学习自身动力学 | 主要研究机器人身体，不同时解决其他实体持续性、身份和语言 | [TUM](https://portal.fis.tum.de/de/publications/yielding-self-perception-in-robots-through-sensorimotor-contingen/)、[Columbia](https://doi.org/10.1038/s44182-025-00031-6) |
| Google PaLM-E | 把视觉、机器人状态和语言放入统一多模态模型 | 大规模预训练语言模型直接接收传感器嵌入，不要求先形成显式 self/entity belief graph（哲学与 Cal 相反） | [Google Research](https://research.google/blog/palm-e-an-embodied-multimodal-language-model/) |

如果聚焦 FSD，**CarFormer 仍是目前最值得认真对照的项目**：object-centric BEV 表征、车辆时序状态、未来预测、驾驶策略、CARLA 评价都已直接验证。

Cal 相对它的特点是：更强调从动作中发现 self；显式保持多假设、身份和遮挡状态；内部状态更容易解释；实验协议更重视因果负对照。但 CarFormer 在真实任务距离上明显更近，V-JEPA 2-AC 更已经把“真实视觉+闭环”变成公开既成事实，所以不能说 Cal 已经领先。

**竞争态势判断**：Cal 独特在“动作归因 self + 显式多假设/身份/遮挡 + 语言线性读出 + 极端证据纪律”这一组合；但每一个单独组件，公开工作都做得更大、更接近真实、更接近闭环。因此 Cal 的护城河是方法论与可解释性，不是能力领先。

## 五、最终判断与最能撬动价值的三条动作

客观地说：

> 这个方向值得继续，且不是闭门造车；它站在 object-centric learning、sensorimotor self-model、world model 和 grounded language 的交叉点上。

但也要避免两个过度结论：

1. 目前不能称为已经实现通用多模态学习器。
2. 目前不能称为已经验证的 FSD 或真实世界世界模型。

它现在最准确的定位是：

> **一个方向正确、实验纪律很强、已经形成可信合成证据，但尚未完成真实世界外部验证的实体认知研究原型。**

若要让 6.0 往 8+ 走，按杠杆排序，只有三件事真正重要：

1. **独立复现 I1**（ROADMAP 第一条）——证据治理再好也替代不了第三方跑通；这是把“单人四天冲刺”变成“可信成果”的最低成本动作。
2. **诊断 L0 V8 permanence 失败，并预注册一个全新外部留出**——不复用已消费留出。永久性对照没过是当前最实的科学缺口，直面它比开新阶段更值钱。
3. **迈出闭环第一步**：在实体图上做动作条件化的未来状态预测，并与神经/对象中心基线（如 Embodied-SlotSSM、OCWM×MCTS）对打。这是从“状态学习器”变成“world model”的唯一路径，也是唯一能与公开前沿正面对照的地方。
