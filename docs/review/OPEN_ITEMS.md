# 未完成项清单

截至 2026-08-08（分支 `permanence-belief-free-gate-redesign`，PR #2）。

本文件汇总**已识别但尚未处理**的项，来源是三份评审报告与本轮执行中的发现。
每条都在编写时逐一复核过当前代码状态，而不是照抄评审报告——报告写于本轮
之前，其中一部分已在本轮解决。

来源标记：`P#` = [永久性评审](REVIEW_PERMANENCE_FREEZE_2026_08_08.md) 的发现编号；
`G#` = 该报告 §4.3 的系统性缺口编号。

---

## 一、阻断冻结（必须在第二阶段冻结前完成）

### O1（P1 / G7）永久性栈没有 source lock

**实测**：全库 7 份带 `locked_source_sha256` 的协议**全部是 M1–M3 确认协议，
永久性相关为 0 份**。

本轮加的 `audit_artifact_source_lock` 与漂移测试只能**发现**改动；M1–M3 的
`_verify_locked_sources` 是运行时**拒绝执行**。二者不等价——冻结若没有后者，
"锁"只是描述性的。

**要做的**：给永久性协议加 `locked_source_sha256`，并在 phase0 / phase-R 入口
加运行时校验。范围至少覆盖 14 个永久性模块 + `cal/env/` 的真值模拟器。

### O2（P1 / F6 后半条）对照可构造性只论证未实证

预注册 C.3 说明了 V8 对照因范式变更降级为非门控诊断，但评审要求的另一半——
**在 development split 上实跑全部对照的构造代码并附输出**——未执行。

**为什么不能省**：V5 留出正是因为身份打乱对照无法从留出事件中构造而中途停止，
且**不可重试**。论证不能替代实跑。

### O3 尚无候选实现

12 项确认门从未在真实候选上运行过。当前只有 `oracle` / `belief_free` /
`geometric` / `uniform` / `old_i1` 五个参照。

### O4 计划文档与预注册的关系未明确

`docs/experiments/V2_I1_STOCHASTIC_PERMANENCE_PLAN.md` 与预注册草案 C 节
对同一门系统各有描述，必须合并或声明其一取代另一。

---

## 二、非阻断但影响正确性（P2）

### O5（P2 / F10）`run_benchmark` 不校验 train/eval seed 重叠

**实测**：`permanence_forward_benchmark.py` 中 `disjoint`/`overlap` 出现 **0 次**。
守卫只存在于上层（`run_candidate_lifecycle`、`validate_disjoint_seed_sets`）。

CLI 可意外触发：`--train-seeds 101` 配默认 base 即让两者交叠。评审实测污染后
位置先验 top1 由 0.044 跳到 0.243。门控证据链受上层保护，但任何由核心函数或
CLI 直接产出的开发报告不受保护。

**要做的**：在 `run_benchmark` / `gru_capacity_sweep` / `run_diagnostic` 入口加断言。

### O6（P2 / F11）CLI 可覆盖注册表参数却仍盖 provenance 章

`--turn-probability` / `--steps` / `--warmup` 在读取注册表**之后**生效，产物
仍嵌入注册表 digest，且跳过 phase0/scan 强制的覆盖契约与 digest 复现检查。
另有软默认 `registry.get("selected_turn_probability", _DEFAULT)` 落在切分关键
参数上。

### O7（P2 / F13）`cal-index` 会截断已提交的 INDEX.json

**实测**：已提交的 `results/INDEX.json` 有 **702** 条 `path`；在干净检出上运行
README 推荐的 `uv run cal-index --results results` 只索引到 22 条——committed
INDEX 引用大量未入库的本地产物。**新环境执行该命令会静默摧毁历史索引。**

### O8（P2 / F15）`require_authorization` 只接受 schema 1

**实测**：`v2_artifacts.py:124` 硬要求 `result_schema_version == 1`，而 V8 结果
用的是 schema 2。当前无下游消费方，非活跃 bug，但校验器无法校验 schema-2 产物。

### O9（P2 / F12）27 份已提交结果的 provenance 为 dirty 或无 commit

含两份已消费 holdout 摘要与多份 `authorize_*` 授权产物。关键终局证据（V8、
I1 v4 holdout）干净。**历史产物不可重跑**，只能记录为已知限制。

### O10（P2 / F16）`autonomous_successors` 在非二值 static 概率下不是精确混合

数值探针实测 L1 偏差 **0.0231**。当前调用方全部喂二值网格故无害，但模块已预留
学习型 `static_probability` 槽位；接入时"精确推断"的假设会静默失效。

### O11（P2 / F17）`GridSpec` 的默认值陷阱

**实测**：`stochastic_motion_filter.py:34-35` 把 `grid_size=25`、`arena_low=7`
作为**默认值**，复制自评估世界且无运行时交叉校验。上游常量改动后，滤波器会
静默把界外单元当作确定墙体。

---

## 三、记录在案，不阻断（P3）

- **O12（F19）恒真门**：`fully_detached_safe`（`s_max >= H*E*K` 而 `s_max` 定义
  即 `H*E*K`，见 `stochastic_motion_filter.py:724`）、
  `shared_expansion_workspace_safe`（`12*k_max` 与自身比较）、
  `formal_research_budget_declared`、`branch_evidence_accounting`（残差恒为 0）。
- **O13（F20）容量验证器与 runner 容差不对称**：`np.isclose` rtol 1e-5 vs
  `math.isclose` abs_tol 1e-12，导致**诚实的 no-go 产物无法序列化**。
- **O14（F18）pairwise 负样本奇偶可分**（平衡准确率 0.99973）。已确认**不进入
  任何冻结门**。**附永久禁令：pairwise 正/负构造不得重新进入任何门。**
- **O15（F21–F23）** 零质量状态保留、`maximum_step_pruned_mass` 只写不读、
  `bayesian_no_detection_update` 无生产调用方、`replace_factor_atomic` 不校验
  code 唯一性、`maximum_tv_checkpoint` 记录最后而非最先、
  `run_scan` 硬编码 `steps=200/warmup=12` 而非读契约、RNG 流间距 40 000 无断言。

---

## 四、架构级缺口（评审 §4.3，本轮未动）

**实测**：以下四条在当前代码上仍然成立。

| # | 缺口 | 证据 |
| --- | --- | --- |
| G1 | 只有 `v2_m1_m3_confirmation.py` 调用 `_verify_locked_sources` | 全库唯一调用点 |
| G2 | `v2_m2.py` 省略 `--split` 时静默走无哈希验证的 legacy 路径 | `v2_m2.py:484` `split or "legacy_development"` |
| G5 | M4 体系整体无源码锁 | 已消费留出的机制被后续修复改变过行为（对照余量 0.075 → 0.012） |
| G6 | `capture_provenance` 的 `source_sha256` 只描述不校验 | 无任何校验方 |

G3（`v2_m1.py` 无锁）已部分变化：`v2_m1.py` 现被 V7 确认协议的
`locked_source_sha256` 覆盖，但仅在确认阶段运行时校验，单独运行不校验。
G4（锁未覆盖 `v2_m3.py` 与 `cal/env/` 模拟器）仍成立。

---

## 五、待定夺（需人类判断，非技术阻塞）

| # | 事项 | 现状 |
| --- | --- | --- |
| O16 | 闭合比例是否沿用 **0.40** | 当前实现即 0.40，理由是与既有 `6+/top1_closure_0.40` 同值、不新造阈值 |
| O17 | 留出规模 | `recommended_holdout_seed_count` 已由 2630 升至 **11078**，决定未来一次性留出的成本 |
| O18 | 冻结授权 | 本轮改动了预注册的锁定常量与门定义；按 [`REVIEW_PLAN.md`](REVIEW_PLAN.md) §8，冻结需评审 `pass` 且**人类 Gatekeeper 签署** |
| O19 | 留出密盐 | F9 的设计要求保管人持有密盐。**必须在生成任何留出 seed 之前就位**，事后改代码无法补救已生成的留出 |

---

## 本轮已解决（不在待办内，仅供对照）

P0：无信念候选过门（`belief_free` 成为参照下限）、草案门表与实现零重合、
草案无数值阈值与决策规则。
P1：phase0 验证盲区（验证器从原始数据重算）、恒真 floor 门（已删）、
绑定过期注册表（V4）、种子反演（HMAC 派生）。
P2：README/RESEARCH_STATUS 不提永久性专案。
方案自身评审的 F1–F8 全部落地（见
[补充说明](REVIEW_REVIEW_PLAN_SUPPLEMENT_2026_08_08.md) 的复核台账）。
