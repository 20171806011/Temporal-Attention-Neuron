# TAN-II Sprint 4.2-D — 主张账本(Claim Ledger)

**阶段**: Sprint 4.2-D(Hard Binding / WTA minimal probe)
**日期**: 2026-09-04
**等级**: ESTABLISHED / SUPPORTED / UNKNOWN / REJECTED。
**依据**: `docs/TANII_SPRINT4_2D_{PREREGISTRATION,RESULTS}.md`、
`docs/TANII_SPRINT4_2D_AMENDMENT_5.md`、`results/sprint4_2/sprint4_2d_*.json`。

---

## 1. 核心命题

| # | 主张 | 等级 | 依据与限定 |
|---|---|---|---|
| H1 | 有限竞争归一化(任意有限 γ)保持软绑定:D_events > 0(softmax 严格正性) | **ESTABLISHED** | 定理 + γ ∈ {1,2,5,10,100} 阶梯全部 D_events>0;γ=100 仍 >0(≈2e-11) |
| H2 | exact value delivery 恰好在一热 WTA 离散极限(γ=∞)实现:D=0、T=−4 精确 | **ESTABLISHED** | 确定性断言(1e-12);M2 event/full-WTA 均 (5,9) |
| H3 | WTA 不改变路由:winner、事件 flip rate(0.444527)、交换反对称在全部 γ 不变 | **ESTABLISHED** | winner (A,B) 全阶梯;flip rate 求积 γ 不变;T_swap=−T 每 γ 1e-12 |
| H4 | "exact value delivery 的最小组织自由度 = 一热离散选择" | **ESTABLISHED(限本模型族,由 H1–H3 合取)** | 连续→离散是唯一变化;无其他自由度介入 |
| H5 | E-I 电路重塑绑定后的值传递通路:γ<γ\*(≈5.098)方向反转,γ>γ\* 恢复 | **ESTABLISHED(如实报告的机制)** | T_drive 阶梯 +0.2367→−3.8674,γ\* 二分命中冻结值;I 分支被 WTA 钉在其自身 winner |
| H6 | routing selectivity 与 readout discreteness 是两个正交自由度 | **ESTABLISHED(概念框架的量化支撑)** | WTA 改变 discreteness 而不改变 routing(H3);无路由模型在 hard WTA 下交付 0(选择无对象) |
| H7 | D_full(Q=3) 在 γ≈1.198 的零点是相消伪精确,不是硬绑定 | **ESTABLISHED** | Amendment 5:该点 D_events≈1.5>0;结构硬绑定由 D_events 与极限定义 |
| H8 | M1 的"近硬绑定"(4.2-C D=0.106)在 event-WTA 下变精确、在 full-WTA 下变 0 | **ESTABLISHED** | event-WTA (9,5) 精确;full-WTA (0,0)(+1 探针赢、−1 pad 赢) |
| H9 | hard binding 在有限参数机制(可实现的有限增益)下可实现 | **REJECTED** | H1:有限 γ 恒软;精确性只在不可达极限 |
| H10 | Binding → Composition(多对象绑定后的组合计算) | **UNKNOWN** | 未进入实验范围(用户口径的下一刀候选) |
| H11 | E-I circuit implements binding | **REJECTED(表述不成立)** | E-I 只重塑传递通路(H5),选择由 WTA 完成 |

## 2. 协议合规记录

- Amendment 5(D_full 单调性预测错误):运行前发现、原文保留、追加修正、
  PREDICTION-CORRECTED-PRERUN;程序断言改为"具体形状断言"(D_events 单调、
  D_full(Q=4) 单调、D_full(Q=3) 过冲形状、γ_cross3 二分)并如实报告相消零点。
- 实现级 bug 修复(2 处,均记录):M1 硬极限检查的三元表达式写反;M2unnorm 基准脚本
  未传 norm=False(基准脚本内 bug,正式程序无此问题)。修复后 77/77。
- 停止规则:0 真实 FAIL;未发生任何运行后调参/改种子/改阈值/改采样。

## 3. 最终一页结论(口径锁定)

> **4.2-D 证明:在 (K,V) 解耦的 vector-QK TAN 上,exact value delivery 的最小
> 附加组织自由度是一热 winner-take-all(离散选择):有限竞争归一化只近似(软,
> D>0),精确交付恰好且仅在一热极限实现,且 WTA 不改变路由(翻转率/反对称/winner
> 全部不变)、并把 E-I 电路的传递方向在 γ\*≈5.098 处从反转恢复为正——同时确立了
> routing selectivity ⊥ readout discreteness 的正交双轴框架。**
> 完整阶梯:… Vector Q-K ⇒ Routing → K-V Decoupling ⇒ Soft Binding →
> **One-hot WTA ⇒ Exact Value Delivery(离散极限)**;Binding→Composition 留待后续 GO。
