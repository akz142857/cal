# Cal V2 阶段报告

日期：2026-07-24

## 决策

**V2-A–C、V2-M1、冻结留出 M2 与 M3 通过；V2 总链停止在 M4 的无特权视觉门。**

这不是“唯一身体掩码已经可解”的结论。置换对称反例仍证明，即使 32 帧
视觉—动作历史相同，隐藏身份也可能不同。正式系统必须在这种状态下保留多个
假设，不能猜测模拟器的秘密标签。

## 逐级证据

| 阶段 | 核心结果 | 决策 |
| --- | --- | --- |
| V2-A–C | 镜像反例 32 帧逐像素多数 IoU 0.372 | 授权 M1 |
| V2-M1 | 轨迹 F1 1.000；主动步数减少 58.7% | 授权 M2 |
| V2-M2 | 节点 F1 1.000；交叉身份保持 1.000 | authorize_v2_m3 |
| V2-M3 | 真实完整图概率 0.999994；姿态投影 IoU 1.000 | authorize_v2_m4 |
| M1–M3 新数据综合确认 | M1 F1 0.9995；M2 交叉身份 0.9792；M3 真实图概率 0.999991 | confirm_m1_m3_and_authorize_unprivileged_v2_m4_design |
| V2-M4 | 特权可见性诊断：占据 IoU 0.952；遮挡召回 0.978 | stop_before_reconnection |

M1 的动作输入和在线失败更新消融都使 F1 下降至少 0.15。M2 的交叉压力
节点 F1 为 0.999。M3 在同构共享基座压力
中保持两张完整身体图各 0.5，并在破缺后最慢
2 步收敛；无因果似然开发消融
保持在机会水平。随后使用完全不同的新种子完成 M1–M3 综合确认，且没有读取
旧留出结果。M4 的移动目标隐藏期平均占据概率为
0.754，但仍读取模拟器可见性掩码。

## 失败闭环与资源

M1 为每步保存有界、无标签的预测—残差遥测；M2/M3 使用严格
预序列残差更新递推控制模型；M4 只保留小型概率格和短期运动假设。全部阶段
CPU 运行、经验零重放。资源通过不覆盖未通过的机制门。

## 不能外推的内容

- V2 proves staged mechanisms in small deterministic synthetic worlds, not general visual cognition.
- M2 and M3 consume deterministic sparse visual detections; a raw-pixel front end remains unvalidated.
- The mirrored shared-base stress is reported as multiple hypotheses because a unique hidden identity is unidentifiable.
- M4 receives a sensor visibility mask and uses a small two-dimensional allocentric grid.
- M3 pose projection uses a known analytic two-link renderer and validates graph selection, not learned raw-pixel segmentation.
- M4 still receives a simulator visibility mask and therefore remains a diagnostic rather than a formal visual-only stage.
- The fresh M1-M3 confirmation validates behavioral composition; it does not establish serialized M1 state handoff into M2.
- Only a future full V2 pass could authorize reconnection to the original object-permanence M2.

当前决定是停在 V2-M4：M3 已形成互斥、校准的完整身体图后验；下一步必须从视觉推断 free/occupied/unknown，删除模拟器可见性掩码。
不应把当前合成环境成绩解释为 FSD 等级能力。
