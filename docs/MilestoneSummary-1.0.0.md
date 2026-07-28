先给结论：以“当前已经完成的系统”评分，而不是按未来设想评分。

- 方向：8.0 / 10
- 当前综合价值：6.0 / 10
- 潜在价值：如果跨过真实视觉、跨场景泛化和闭环行动三道门，可到 8.5 / 10；现在还不能按这个分数宣传。

这里的“价值”指已经形成、能被第三方验证和利用的价值，不是想象中的商业上限。

## 一、方向为什么是 8 分

方向是成立的，而且与公开研究前沿基本同向：

> 先通过视觉与行动建立自我、实体、身份和持续存在的内部状态，再让语言读取这些状态，最后才进入预测、规划与行动。

这条路线比“直接接一个大语言模型，让它看起来会说”更有科学意义，因为语言有机会对应一个真实、可追踪的内部世界状态。

当前做对了三件比较关键的事：

1. **把 self 建立在行动—感知因果关系上**

   系统不是依靠外观标签认出“我”，而是通过动作与视觉变化的对应关系识别 self。这与机器人领域长期研究的 sensorimotor contingency/self-perception 路线一致。慕尼黑工业大学的机器人自我感知研究同样利用动作造成的感觉变化区分 in-body 与 out-body；Columbia 的视觉自模型也研究如何仅从第一视角视觉学习自身动力学。[TUM sensorimotor self-perception](https://portal.fis.tum.de/de/publications/yielding-self-perception-in-robots-through-sensorimotor-contingen/)、[Columbia egocentric visual self-modeling](https://doi.org/10.1038/s44182-025-00031-6)

2. **使用统一实体状态，而不是互不相连的任务输出**

   self、其他实体、运动、遮挡、持续存在和身份历史都放入统一 Entity Belief Graph。这符合 object-centric learning 的核心思想：世界应被拆成持续存在、能够独立运动的实体，而不是只有一整块图像特征。

3. **语言建立在状态之上，而不是代替状态**

   L0 冻结 I1，再检查 self、空间、持续存在和身份语义是否能被简单线性读出。这是很干净的实验设计：如果简单读出器已经能读取，说明语义在底层状态中确实存在；如果必须靠大型语言模型“猜”，就很难判断能力来自哪里。

另外，当前项目对协议冻结、负对照、数据污染和 one-shot 证据的重视，高于大多数个人研究项目。这部分本身有研究方法价值。

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

现代公开研究中，“world model”通常还要求预测未来状态，有些还要用预测结果规划行动。Meta 的 V-JEPA 2、DreamerV3 等都把“根据当前状态和动作预测未来，并用于规划/控制”作为关键能力。[Meta V-JEPA 2](https://ai.meta.com/research/vjepa/)、[V-JEPA 2 官方开源仓库](https://github.com/facebookresearch/vjepa2)、[DreamerV3](https://github.com/danijar/dreamerv3)

当前系统已经有成为世界模型的“状态基础”，但还没有完成完整闭环。

另一个必须扣分的事实是：

- I1 已通过冻结 validation 和 holdout；
- L0 development 通过；
- V8 唯一 holdout 已运行，但最终为 `stop_and_report`：24 个 gate 中
  21 个通过，3 个 permanence/control-integrity gate 失败。

因此，现在可以说“部分受控语言语义在 development 和 holdout 中可读”，
但不能说“完整 L0 语言能力通过独立 holdout 验证”。尤其 permanence 没有
优于 raw-sensor 和 assume-all-visible 对照。这与项目自己的
[I1 报告](experiments/V2_I1_NEXT_ARCHITECTURE_RESULT.md) 和
[L0 报告](experiments/V2_L0_LANGUAGE_READOUT.md) 一致。

## 三、当前价值为什么是 6 分

我会这样拆分：

| 价值维度 | 分数 | 判断 |
|---|---:|---|
| 研究方向价值 | 8.0 | 问题分解正确，符合 object-centric、embodied、world-model 路线 |
| 实验方法价值 | 8.0 | 冻结协议、负对照、一次性证据和失败记录很扎实 |
| 当前算法证据 | 6.5 | 合成环境内强，但真实输入与跨环境证据不足 |
| 可复用工程价值 | 6.5 | 代码、测试、证据链和交互回放可复用 |
| 学术新颖性 | 5.0 | 组合有特色，但各核心组成部分已有相近研究 |
| 当前产品价值 | 2.5 | 尚不能直接用于 FSD、机器人或真实视觉产品 |

综合约为 **6.0/10**。

这不是低分。对一个探索阶段系统来说，6 分意味着：

> 已经不是概念或玩具代码，而是一个有可信内部结果的研究原型；但尚未形成经过外部 benchmark 验证的算法成果，更不是可部署产品。

当前最有价值的资产不是“它已经会语言”，而是：

- 建立了一条可解释的感知—自我—实体—语义链；
- 能用负对照解释能力为什么出现；
- 有严格证据治理；
- 后续真实场景实验可以明确知道是哪一层失败。

## 四、公开研究中有没有相同方案

### 结论

根据我能找到的公开论文、实验室页面和开源仓库：

> 没有发现与 Cal 完全相同的公开实现，但每个主要组成部分都有强相关研究，而且其中一些已经进入真实视频、机器人或自动驾驶。

“没有找到完全相同”不是对原创性的证明；未公开的实验室项目、不同术语下的工作以及最新未索引论文都可能遗漏。

### 最接近的公开方向

| 项目/实验室 | 相似之处 | 主要差异 | 开源情况 |
|---|---|---|---|
| Google DeepMind PLATO、Google SAVi/SAVi++ | 以对象为中心，跟踪视频中的实体，研究 object persistence | 不强调动作归因 self，也没有 Cal 的实体图→语言读出链 | [SAVi/SAVi++ 代码公开但已归档](https://github.com/google-research/slot-attention-video)；[PLATO 研究](https://deepmind.google/blog/intuitive-physics-learning-in-a-deep-learning-model-inspired-by-developmental-psychology/) |
| Meta FAIR V-JEPA 2/2.1 | 从视频学习世界表征；动作条件版本用于预测和机器人规划 | 大规模神经潜空间，不显式维护 self/身份/遮挡实体图 | [官方代码和模型开放](https://github.com/facebookresearch/vjepa2) |
| KUIS AI CarFormer | object-centric BEV、未来状态预测、自动驾驶，与你设想的 FSD 路线非常接近 | 用 Slot Attention 学习车辆对象并直接驾驶，没有动作归因 self 和受控语言读出 | [论文、代码和预训练模型公开](https://kuis-ai.github.io/CarFormer/) |
| Seoul National University LSlotFormer | 对象中心世界模型与语言结合，是高层结构上最接近 L0 的工作 | 它用语言指令控制未来状态预测；Cal 是从已有状态向外读取语言语义，因果方向相反 | [公开论文/ICLR Workshop](https://openreview.net/pdf?id=CMItmXqrue)，未找到对应官方开源代码 |
| TUM/Columbia 机器人 self-model | 从动作—视觉或多模态感觉关系发现自身、学习自身动力学 | 主要研究机器人身体，不同时解决其他实体持续性、身份和语言 | [TUM](https://portal.fis.tum.de/de/publications/yielding-self-perception-in-robots-through-sensorimotor-contingen/)、[Columbia](https://doi.org/10.1038/s44182-025-00031-6) |
| Google PaLM-E | 把视觉、机器人状态和语言放入统一多模态模型 | 大规模预训练语言模型直接接收传感器嵌入，不要求先形成显式 self/entity belief graph | [Google Research](https://research.google/blog/palm-e-an-embodied-multimodal-language-model/) |

如果聚焦你之前提到的 FSD，**CarFormer 是目前最值得认真对照的项目**。它已经直接验证：

- object-centric BEV 表征；
- 车辆时序状态；
- 未来预测；
- 驾驶策略；
- CARLA 自动驾驶评价。

而 Cal 当前相对它的特点是：

- 更强调从动作中发现 self；
- 显式保持多假设、身份和遮挡状态；
- 内部状态更容易解释；
- 实验协议更重视因果负对照。

但 CarFormer 在真实任务距离上明显更远，所以不能说 Cal 已经领先它。

## 最终判断

客观地说：

> 这个方向值得继续，且不是闭门造车；它站在 object-centric learning、sensorimotor self-model、world model 和 grounded language 的交叉点上。

但也要避免两个过度结论：

1. 目前不能称为已经实现通用多模态学习器。
2. 目前不能称为已经验证的 FSD 或真实世界世界模型。

它现在最准确的定位是：

> **一个方向正确、实验纪律很强、已经形成可信合成证据，但尚未完成真实世界外部验证的实体认知研究原型。**
