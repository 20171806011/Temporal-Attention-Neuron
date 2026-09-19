# TAN-II Sprint 4.2-C — 预注册(Routing → Binding minimal probe)

**项目**: TAN-II:组织机制与选择性计算的涌现
**阶段**: Sprint 4.2-C —— Routing → Binding 最小探针(不泛化、不学习、无 benchmark、无新拓扑)
**日期**: 2026-09-04
**前置**: 4.2-A/4.2-B 已冻结;TAN-I、4.1-C/D 冻结档案不可修改。

## 0. 目标口径(用户锁定)

> 被选中的对象不仅改变 attention winner,而且它所携带的 value 能否被正确传递到输出?
> 最小结构:(K_A,V_A),(K_B,V_B),历史中随机交换位置;给定 Q_A 或 Q_B,输出 V_A 或 V_B;
> 最重要的 counterfactual:**固定完全相同的 history,只改变 Q**。Q_A→V_A,Q_B→V_B。

## 1. 最小 (K,V) 解耦(唯一新增结构)

- 事件槽携带键 K(幅度,驱动注意力;键嵌入 φ̂₂ 与 4.2-B 完全相同)与值 V(传输通道):
  事件 A = (K_A, V_A),事件 B = (K_B, V_B);pad = (0, 0);
  探针槽键 = Q(**查询是控制信号,不是内容**),探针值 = 0(若探针携带自身值,输出永不可能
  是 V_A/V_B——这是 4.2-B amplitude-code leakage 教训的直接落实)。
- 规范历史:`[A=(1,5), 0, B=(2,9), 0]`(键沿用 4.2-B 规范历史;值预注册 V_A=5, V_B=9)。
- **值交换对照(value-swap control)**:`[A=(1,9), 0, B=(2,5), 0]`(键不变、值互换)。
- 位置随机化:12 个有序位置对;核无位置项(4.2-B),位置不影响读出——作为协议纪律执行并
  断言位置不变性(12 布局读出精确相等)。
- 模型:M2 主;M0 / M1(±符号)/ M2unnorm / M4 / M5-E 对照。动力学、惊奇、E-I、
  脉冲全部冻结(4.2-B 遗产)。

## 2. 绑定指标(预注册)

- **读出游出(主)**:`C_events(Q) = (α_A v_A + α_B v_B)/(α_A + α_B)`(仅事件槽归一化)。
- `C_full(Q)`(全窗口,pad/探针值 = 0);驱动 `A_E = tanh(S_E)·C_full`(单细胞);
  E-I 驱动 `A_E − w_EI·A_I`(M5-E/M4);条件固定点(脉冲标记,次级)。
- **反事实传递**:`T = C_events(Q_A) − C_events(Q_B)`。
- **winner 对齐**:`T` 的符号 = `sign(v_winner(Q_A) − v_winner(Q_B))`。
- **交换反对称**:`T_swap = −T`(读出游出跟随 (K,V) 关联,而不是键身份)。
- **硬度/软度**:`D(Q) = |C_events(Q) − v_winner(Q)|`(硬绑定 = D=0;软注意力 ⇒ D>0)。
- **基底对照**:M0(winner 恒 B,只锐化不路由);M2unnorm(注意力塌缩到探针,v=0 ⇒ 无传递);
  M1(±符号:winner 翻转 + 值传递,退化但近乎硬绑定)。
- 集合指标(键 x_A,x_B iid U(0.3,2.5),值按事件身份固定 A:5/B:9,位置随机):
  `E[|T|]`、`E[|T| | flip]`、`E[|T| | noflip]`、`E[T]`、flip 率(4.2-B 已冻结 0.444527)、
  **sign-matched alignment | flip**(定理:ᾱ_A = σ(c·S·û·Δφ̂) 单调 ⇒ = 1;作为定理核验,
  不作经验发现)。

## 3. 冻结预测(运行前闭式计算)

| 量 | 冻结值 |
|---|---|
| M2 C_events(Q=3) / C_events(Q=4) | 6.742691 / 7.155595 |
| M2 T = C_events(3) − C_events(4) | −0.412904(winner A→B,值 5→9,方向对齐) |
| M2 T_swap | +0.412904(精确反对称,T_swap = −T) |
| M2 D(Q=3) / D(Q=4) | 1.742691 / 1.844405(软绑定,非硬绑定) |
| M2 驱动传递 T_drive = A_E(3) − A_E(4) | −0.258552(单细胞门控**保持**方向) |
| M5-E 驱动传递 | +0.0985(**方向反转**;I 分支也读出值通道并相减——如实报告为边界发现) |
| M4 驱动传递 | −0.0865(winner 恒 B,锐化基底;Q=4 时注意力塌缩到探针) |
| M0 T | −0.084443(锐化基底,winner 恒 B) |
| M1(±符号反事实) | C_events(+1)=8.893612,C_events(−1)=5.106388;D=0.106388(近硬绑定,退化) |
| M2unnorm T | ≈ 0(探针占优,v=0 读出塌缩) |
| 集合基准 | E[T]≈0(对称);E[|T|]=0.400435;E[|T|\|flip]=0.465523;E[|T|\|noflip]=0.348347;alignment\|flip=1.0 |

判定:概率 |p̂−bench| ≤ 3SE+0.002;均值 |MC−bench| ≤ 3·SE_mean+0.01;确定性量 1e-9
(反对称 1e-12)。任一真实 FAIL → STOP AND DEBUG。

## 4. 声称纪律

- 只声称 **soft/partial binding(winner 对齐的值传递)** 在读出游出层面建立;
  **硬绑定(exact value delivery)未实现**(D≈1.7–1.8 > 0,软注意力的固有软度);
  E-I 驱动级方向反转是**如实报告的边界发现**,不掩盖。
- 不声称 generalized binding、不声称任何"内容理解/智能"。

## 5. 输出

`code/experiments/sprint4_2/sprint4_2_binding_probe.py`;
`results/sprint4_2/sprint4_2c_{summary.json,checks.csv,*.png}`;
`docs/TANII_SPRINT4_2C_{PREREGISTRATION,RESULTS,CLAIM_LEDGER}.md`。

*预注册完成;集合基准由 2001² 确定性求积冻结于 §3(先于正式 MC 运行)。*
