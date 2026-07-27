# I1 V4 可重复交互式回放

## 1. 它解决什么问题

正式实验结果是 JSON，适合审计，但不容易直观看懂。这个回放把同一条
calibration 轨迹画成三个同步视角：

1. **世界真值**：评估器知道的 self、目标 A、目标 B 与遮挡物；
2. **系统观测与身份**：系统认为可见的区域、实际收到的占据格、统一实体图
   给出的轨迹 ID；
3. **系统信念**：系统对每个格子“这里有东西”的概率，以及当前被认作
   self 的轨迹。

页面是单个自包含 HTML，不加载网络资源。复制到另一台机器后，直接用浏览器
打开即可。

## 2. 生成

在仓库根目录执行：

```bash
uv run cal-v2-i1-replay \
  --seed 30000 \
  --output docs/experiments/assets/v2_i1_v4_replay_seed30000.html
```

可用 seed 只来自冻结 V4 协议的 calibration 集：

```bash
uv run cal-v2-i1-replay --list-seeds
```

生成器会明确拒绝：

- validation seeds `32000–32007`；
- holdout seeds `31000–31015`；
- 任何没有登记的 seed。

因此演示不会再次消费一次性 validation/holdout 证据。

## 3. 校验“同样输入得到同样页面”

```bash
uv run cal-v2-i1-replay \
  --seed 30000 \
  --check docs/experiments/assets/v2_i1_v4_replay_seed30000.html
```

`PASS` 表示当前源码、冻结协议和 seed 重新生成的 HTML 与仓库文件逐字节相同；
命令同时打印文件 SHA-256。任何数据、源码摘要、样式或脚本差异都会让校验
失败。

## 4. 页面怎么操作

- **播放 / 暂停**：连续观察 0–200 步；
- **上一步 / 滑块**：逐帧检查；
- **实验条件**：切换四种条件，世界轨迹和实际动作序列保持完全一致；
- **关键事件**：跳到 self 被识别、目标进入遮挡、重新出现、目标重合或
  轨迹身份改变的时刻；
- 键盘空格控制播放，左右方向键逐帧移动。

四种条件含义：

| 条件 | 改了什么 | 想回答的问题 |
|---|---|---|
| Formal | 真实动作 + 遮挡推断 | 完整系统能否工作 |
| No action | 系统不使用真实动作 | self 识别是否真的依赖动作 |
| Shuffled action | 动作错开 5 步 | 系统是否依赖正确的动作时序 |
| Assume all visible | 不推断遮挡 | 遮挡建模是否真的维持目标存在 |

切换条件时，页面上方四个数字显示该条件整段 200 步的正式指标。回放录制器
使用与冻结 runner 完全相同的计分算术，自动测试会逐字段对账。

## 5. 证据边界

这个页面是**解释工具，不是新的实验结论**。

学习器每一步实际只收到：

```text
局部 11×11 二值占据栅格 + 一个动作编号
```

左侧真值仅由回放记录器在 `agent.update(...)` 完成后读取，用于画图、标注
关键事件和离线评分；它不会传给学习器。页面内也明确保存
`evaluatorTruthUsedForLearning: false`。

最终结论仍由以下已归档证据支持：

- `results/V2-I1-v4-calibration-summary.json`
- `results/V2-I1-v4-validation-summary.json`
- `results/V2-I1-v4-holdout-summary.json`
- origin 上对应的 validation/holdout consumption 与 evidence tags

回放用于回答“系统是怎么做到的”；上述冻结结果用于回答“结论是否经过独立
validation 和 holdout 验证”。

## 6. 自动验证覆盖

`tests/test_v2_i1_replay.py` 检查：

- 只有 calibration seed 能启动仿真；
- 四个条件均有 201 帧，且共享同一条实际动作序列；
- 四个条件的指标逐字段等于冻结正式 runner；
- 代理更新接口只收到占据栅格与动作；
- 生成结果可逐字节复现；
- HTML 没有外部 URL、体积小于 2 MB；
- 内嵌 JavaScript 能通过 Node.js 语法检查；
- 仓库参考 HTML 与生成器输出完全一致。

运行：

```bash
uv run pytest tests/test_v2_i1_replay.py -q
```

## 7. 当前仓库参考页记录

记录日期：`2026-07-27`

参考文件：

```text
docs/experiments/assets/v2_i1_v4_replay_seed30000.html
```

固定输入和产物摘要：

| 项目 | 记录值 |
|---|---|
| seed | `30000`（calibration） |
| 步数 | `0–200`，每种条件 201 帧 |
| 条件数 | 4，共 804 帧 |
| 文件大小 | `1,025,429 bytes` |
| 文件 SHA-256 | `6445f0bda0263f9c25df621d0f64168a7f0f4b1b6fb87fbd38c6d83e304824f2` |
| 外部网络资源 | 无 |
| validation/holdout 消费 | 无 |

参考 seed 的四条件指标：

| 条件 | self F1 | 身份一致性 | 可见身份覆盖 | 遮挡目标概率 |
|---|---:|---:|---:|---:|
| Formal | 0.947712 | 0.953623 | 1.000000 | 0.913052 |
| No action | 0.000000 | 0.950725 | 1.000000 | 0.913470 |
| Shuffled action | 0.000000 | 0.950725 | 1.000000 | 0.913105 |
| Assume all visible | 0.866197 | 0.536232 | 1.000000 | 0.016926 |

完成时的验证记录：

```text
tests/test_v2_i1_replay.py: 12 passed
全项目测试: 256 passed
逐字节 replay check: PASS
JavaScript 语法检查: PASS
四条件指标与冻结正式 runner 逐字段相等: PASS
```

后续如果录制器、页面模板或模型源码发生变化，参考 HTML 的摘要可能随之改变。
维护者应重新生成页面、运行专用测试和全量测试，再更新本节的文件大小、
SHA-256 与验证记录；不能通过修改本节来“接受”未经测试的新页面。
