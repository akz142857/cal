# 项目复核报告：2026-07-26

范围：`git diff origin/main...HEAD`（本轮会话全部改动，约 2300 行新增/修改
Python）。方法：8 个独立探测角度（line-by-line、removed-behavior、
cross-file、reuse、simplification、efficiency、altitude、conventions），
每个候选发现单独派一个不知情的验证 agent 复核，只保留 CONFIRMED/PLAUSIBLE。

## 修复的 bug（6 个，全部 CONFIRMED）

**1. `MotionHypothesisFilter` 方向丢失** —— `__init__` 算出了正确的初始
方向 `d`，但从未存成 `self._direction`；`observe()` 首次重新出现时用
`getattr(self, "_direction", 1)` 回退到 +1，反向运动的目标第一次重新
可见就会被错误重锚定为正向。**修复**：构造函数里直接播种 `_direction`/
`_last_detection`，`observe()` 不再需要任何回退默认值。

**2. `MAX_FILTERS` 永久耗尽 + `retired` 死代码** —— `MotionHypothesisFilter.retired`
从未被置位，`occupancy.py` 里"如果 retired 就删除"的清理分支是死代码；
一旦累计出现 4 个不同 `(axis, row)` 键，后续所有新位置的遮挡永久失去
建模能力，且没有任何报错。**修复**：删除死掉的 `retired` 属性/分支，
改为按"最近一次被更新"做 LRU 淘汰——容量满时腾出最久未触碰的槽位，
而不是拒绝创建。M4 已消费的留出只有单一移动目标（永远只占 1 个键），
不受影响；这个 bug 只在 V2-I1 这类多实体世界里可达。

**3. `PauseLearner` 跨轴污染** —— 一个 `UnprivilegedOccupancyMemory` 的
所有 `MotionHypothesisFilter` 共享同一个 `PauseLearner`，但它只用裸整数
坐标当 key，不区分方向轴/行列。两个分属不同轴的实体如果坐标数值恰好
相同，会读到对方学到的暂停规律。**修复**：`record()`/`locations()` 都
改成 `(axis, position)` 复合键。同样只在多实体世界（V2-I1）可达，M4
留出（单实体）不受影响。

**4. 综合确认审计的哈希检查名不副实** —— `v1_development_matches_v1_protocol`
这个门名字听起来像是核对"v1 开发结果是否被篡改过"，实际只是核对该
文件自己声明的 `protocol_sha256` 字段，从未和 v2 协议
`amendment_record.prior_development_result_sha256` 这个冻结的历史哈希
做比对——而且验证时发现，直接做字节级比对**会当场炸掉**，因为这份
开发集产物本来就允许重复运行、每次重跑 provenance 时间戳都会变，
历史哈希从设计上就不该拿来做字节比对。**修复**：不是删掉这个不完整
的检查，而是加一个新门 `v1_development_internally_consistent`，核对
文件自身 `passed`/`decision`/`gates` 三者互相一致——这样"手动把某个
门从 False 改成 True 但没同步改 decision/passed"这类篡改会被抓到，
同时不需要对抗一个设计上就会变化的哈希。

**5. Dashboard 文案过期** —— `cal/dashboard/data.py` 里 V2-M4 那一行是
硬编码字符串"特权可见性占据 IoU"+"下一门：删除模拟器可见性掩码"，
但 `v2_stage_summary.py`（本轮改的）在无特权留出通过后，会把同一批
`key_results` 字段换成无特权数据——于是 dashboard 现在显示的数字明明
是去特权化之后的,文字却还在说"这是特权数据、下一步要去特权化"，
两者互相矛盾。**修复**：dashboard 也去检查无特权留出产物是否存在，
文案跟着数据来源切换，并加了对应测试。

**6. `_episode()` 的 `environment_version` 默认值是个陷阱** ——
函数自己的默认值是 2（对应 v2 环境），`run_v2_m4_unprivileged` 读协议
缺省值时用的是 1（对应 v1 环境），而当前冻结的 V3 协议实际用的是 3。
三个数字互不相同。目前没有任何调用方省略这个参数，所以不是活跃 bug，
但任何以后直接调用 `_episode()`（诊断脚本、notebook）的代码都会静默
拿到错误的世界版本、产出看似合理实则不代表冻结留出的数字。**修复**：
把这个参数改成必填，不留默认值——不确定该用哪个版本时报错，好过给
一个可能是错的答案。

## 严重性较低但也修了的发现（2 个，PLAUSIBLE）

**7. `_prune_runaway_tracks` 的越界淘汰与 `reacquisition_window` 脱节**——
越界检测本身不看时间，一条带噪声的 track 可能在远少于 40 步的时间内
就漂出边界被淘汰，实际上抵消了刚建好的更宽重识别窗口本该给它的机会。
**已修复**：把越界判据也并入同一个"必须先过 `stale_after`"的时间闸门，
不再独立触发——两个淘汰条件现在共享同一个时间前提，不会有 track 在
拿到承诺的整个窗口之前就被踢掉。

**8. 资源账目仍有小额低报** —— `IntegratedSelfWorldAgent.estimated_mac_per_step`
没算连通分量 flood fill 和 track 剪枝扫描的开销。已修复（补上两项），
但验证时算过：即使全额补上，相对 500 万 MAC/步的预算依然有 150+ 倍
余量，从未真正影响过任何门的判定——性质上和本 session 早前修过的
`occupancy.py` 资源低报（那次确实逼近预算）不是一回事。

## 修复带来的副作用，如实记录

`motion_hypotheses.py`/`occupancy.py` 不在任何协议的 `locked_source_sha256`
保护范围内（只有 M1-M3 相关文件被锁），所以这次修复不需要协议修订，
但**确实改变了 M4 无特权留出已经消费过的机制的当前行为**。用可重复的
开发集重新核对：

| 指标 | 修复前 | 修复后 | 门槛 |
| --- | ---: | ---: | ---: |
| 隐藏期运动目标概率（正式） | 0.7595 | 0.7600 | ≥0.55 |
| assume_all_visible 对照 | 0.4746 | **0.5384** | ≤0.55 |

正式门几乎不变，但**对照组的安全余量从 0.075 收窄到 0.012**——依然
通过，但边际薄了很多。已消费的一次性留出（`results/V2-M4-unprivileged-holdout-summary.json`）
保持不变、不会也不应该重跑；这里只是如实说明：如果现在重新（假设性地）
跑一次留出，数字会和已发布的不完全一样。M4 协议体系没有像 M1-M3 那样
的源码锁，这本身是一个值得记录的系统设计缺口——留给以后决定是否要
补上。

V2-I1（本来就未通过、留出未消费的探索性协议）的 `distractor_hidden_probability`
从 0.538 降到 0.220，`paired_formal_beats_visible_control` 从 100% 降到
81.25%——这是 LRU 淘汰替换永久锁死之后的真实副作用（自我实体频繁换轴
可能挤走本该继续预测干扰物的旧 filter）。门的通过/失败结果没有变化
（这两项本来就没过），只是数字变了，如实记录，不做进一步调参。

## 未修复项去向

V2-I1 的第五层摩擦（证据积累速率跟不上遮挡打断频率）仍是未解决状态，
与本次代码审查无关，是独立的机制设计问题，见
`docs/experiments/V2_I1_INTEGRATION_REPORT.md`。
