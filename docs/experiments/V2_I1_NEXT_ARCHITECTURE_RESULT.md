# V2-I1 下一代架构最终实验报告

日期：2026-07-27

## 1. 结论

V2-I1 下一代架构 **Unified Entity Belief Graph（统一实体信念图）**
通过了冻结协议 V4 的 calibration、一次性 validation 和一次性 holdout。
最终 runner 决策为：

```text
i1_next_generation_architecture_verified
```

这个结论的适用范围是冻结的合成环境：固定摄像头、三个视觉同构点实体、
静态遮挡物、每步只向 learner 提供可见占据和已执行动作副本。它证明该
架构在这个环境中能以同一个实体存储同时支持 self 归属、身份连续性和
遮挡下对象永久性；它不等价于现实视觉泛化或通用自我意识。

## 2. 冻结依据

- reviewed implementation commit：
  `395f22402e6706f262e3e34e1266b5e2b165462a`
- source SHA-256：
  `474071f6853288c7f49d6a0a027c03f393beac119b6d9bedcc5d80c2aee86550`
- protocol：
  `experiments/V2_I1_INTEGRATION_PROTOCOL_V4.json`
- protocol SHA-256：
  `f85dcd58787c697d424eae5efcd9a5cf4d98d523d61294707c31e11048c99b83`
- 所有正式 split 的 `run_start.git_dirty` 和最终 provenance
  `git_dirty` 均为 `false`。

## 3. 最终数值

| 指标 | Calibration 30000–30015 | Validation 32000–32007 | Holdout 31000–31015 | 冻结门 |
| --- | ---: | ---: | ---: | ---: |
| self F1 | 0.9404 | 0.9100 | **0.9435** | ≥ 0.90 |
| identity consistency | 0.9623 | 0.9417 | **0.9102** | ≥ 0.90 |
| visible identity coverage | 0.9943 | 0.9933 | **0.9938** | ≥ 0.90 |
| hidden distractor probability | 0.8200 | 0.7748 | **0.7571** | ≥ 0.55 |
| no-action self F1 | 0.0000 | 0.0000 | **0.0000** | 正式差值 ≥ 0.15 |
| shuffled-action self F1 | 0.0000 | 0.0000 | **0.0000** | 正式差值 ≥ 0.15 |
| assume-all-visible hidden probability | 0.0143 | 0.0129 | **0.0126** | ≤ 0.55 |
| formal 优于 visible control 的配对比例 | 1.0000 | 1.0000 | **1.0000** | ≥ 0.75 |

三个 split 的全部 13 个机制、结构、无标签和资源 gates 都为 `true`。
正式资源为 2,605 个可学习计数、56,527 bytes 活动状态和
3,997,392 MAC/步，分别低于 100k、64 KiB 和 5M 上限。

## 4. 一次性证据

Validation 在第一个 episode 前原子创建：

- `calmodel-i1-v4-validation-consumed`
  remote tag object：`b8a9b0f1df26e96ef976853b8ff6d7d3f6dbcdc1`
- `calmodel-i1-v4-validation-result`
  remote tag object：`a88cdfb56706f33571930d49bf375e39c73e9e25`
- result SHA-256：
  `6d41b4be7dddb4621c8c5bd96331aae5fc7abd0ebdca9485ef4d1fb3e3899fd8`
- immutable Git blob：
  `74174ae8b37fac936a4a85d7eb5edd0d9e2a9d75`

Holdout 在第一个 episode 前原子创建：

- `calmodel-i1-v4-holdout-consumed`
  remote tag object：`72b865c49177fc48f331e8ce589405f5467cb002`
- `calmodel-i1-v4-holdout-result`
  remote tag object：`ca67d50eb01406fb30cbcf7aab44bcedcc652cac`
- result SHA-256：
  `855fd1ec29b05efbe3ef2e35e263a3db8d6dc54611dd881db21e8e3416c313a8`
- immutable Git blob：
  `c7f4c57c23de8a74b1ee0bcffc745a7e14b854cb`

消费 tag 使用随机 nonce 和 expected-missing `--force-with-lease` CAS；
双独立 clone 的并发回归测试证明只有一个执行者能获锁。下一阶段从 remote
tag 指向的 blob 读取结果，验证 certificate/result SHA，并从 conditions
重算 aggregate、从当前 agent 重算资源、再精确重算全部 gates。

## 5. 子 Agent review 与修复

开发后分别从多目标跟踪、概率语义和协议完整性三个方向做了多轮独立
review。所有阻断发现均在运行新 validation 和 holdout 之前修复：

- 假设去重由 position-only 改为包含全部未来相关状态，等价分支用
  `logaddexp` 合并概率质量；
- self posterior 纳入全部存在实体和显式 null 类，移除 existence/age
  的阈值前过滤；
- 已建立运动历史的实体暂停时不会被静态背景学习吞掉；
- 同分支共址 occupancy 改为 Bernoulli union，保留 `other` 位移概率质量；
- 身份计分将漏报纳入分母，并强制两个真值实体使用不同的全局 ID；
- 活动状态和 MAC 改为保守上界，并用满 5×11 hypothesis bank 的
  deep-size 回归测试覆盖；
- prerequisite、V3 baseline、calibration/validation/holdout artifact
  均增加严格 hash、schema、seed、decision、commit/source 和 gate 校验；
- 本地 reservation 升级为共享 origin CAS consumption tag，正式结果升级为
  immutable Git blob evidence。

最终定点复核没有残余 P0/P1。冻结实现上的全量测试为 244 项，全部通过。

## 6. 可复核文件

- `results/V2-I1-v4-calibration-summary.json`
- `results/V2-I1-v4-validation-summary.json`
- `results/V2-I1-v4-validation-reservation.json`
- `results/V2-I1-v4-holdout-summary.json`
- `results/V2-I1-v4-holdout-reservation.json`

这些本地文件与远端证据的 SHA 一致。Validation 和 holdout 已消费，不得
删除 tag、调参后重跑或把同一 seed 集重新解释为新的未见验证。

## 7. 可重复交互式解释页

为了让后续维护者能直观看到 self 识别、身份连续性和遮挡下对象永久性如何
随时间变化，仓库增加了 calibration-only 交互式回放：

- 参考页面：
  `docs/experiments/assets/v2_i1_v4_replay_seed30000.html`
- 生成器：
  `cal/evaluation/v2_i1_replay.py`
- 使用与复现记录：
  `docs/experiments/V2_I1_REPLAY_GUIDE.md`
- 自动检查：
  `tests/test_v2_i1_replay.py`

回放只允许协议登记的 `30000–30015` calibration seeds。它会在仿真开始前
拒绝 validation、holdout 和未登记 seed，不会重新消费一次性证据。

seed 30000 参考页面包含 Formal、No action、Shuffled action 和
Assume all visible 四种条件。四者共享完全相同的世界轨迹和实际动作序列；
录制指标已逐字段对账冻结 runner。页面中的世界真值只供人类解释和离线
评分，始终在 agent 更新后读取，不进入 learner。

参考页在 `2026-07-27` 的逐字节 SHA-256 为：

```text
6445f0bda0263f9c25df621d0f64168a7f0f4b1b6fb87fbd38c6d83e304824f2
```

生成、校验、指标明细、文件大小及更新流程以
`V2_I1_REPLAY_GUIDE.md` 的“当前仓库参考页记录”为准。这个页面只解释已有
结论；正式结论仍以第 4、6 节列出的冻结 JSON、远端 evidence tags 和
immutable Git blobs 为准。
