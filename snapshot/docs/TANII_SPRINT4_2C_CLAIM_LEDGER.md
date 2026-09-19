# TAN-II Sprint 4.2-C — 主张账本(Claim Ledger)

**阶段**: Sprint 4.2-C(Routing → Binding minimal probe)
**日期**: 2026-09-04
**等级**: ESTABLISHED / SUPPORTED / UNKNOWN / REJECTED。
**依据**: `docs/TANII_SPRINT4_2C_{PREREGISTRATION,RESULTS}.md`、
`docs/TANII_SPRINT4_2C_AMENDMENT_4.md`、`results/sprint4_2/sprint4_2c_*.json`。

---

## 1. 核心命题

| # | 主张 | 等级 | 依据与限定 |
|---|---|---|---|
| B1 | (K,V) 解耦后,固定同一历史、只改变 Q 时,读出游出朝被选值移动(规范:6.7427→7.1556,更接近 5→9;交换后同样成立) | **ESTABLISHED** | 确定性闭式 42/42;距离判据两查询两历史全部通过 |
| B2 | 值传递的 winner 对齐(方向) | **ESTABLISHED(定理+核验)** | ᾱ_A = σ(c·S·û·Δφ̂) 单调 ⇒ 翻转历史上方向必然跟随;集合 sign-matched alignment = 1.000(三种子);定性声明:这是读出结构定理,非独立经验发现 |
| B3 | 读出游出跟随 (K,V) 关联而非键身份(值交换反对称) | **ESTABLISHED** | T_swap = −T 精确至 1e-12;交换历史下距离判据同样成立 |
| B4 | 传递幅度分级:路由模型的 |T| 显著大于锐化基底 | **ESTABLISHED** | 规范 |T_M2| = 0.413 > |T_M0| = 0.084;集合 flip 条件 0.466 vs noflip 0.348(MC 三种子 vs 冻结求积基准全部 ≤ 3SE+0.01) |
| B5 | 单细胞门控保持传递方向;E-I 电路反转传递方向 | **ESTABLISHED(如实报告的边界)** | M2 驱动传递 −0.2586(方向 ✓);M5-E +0.2367(反转,抑制分支二次读出值通道并相减;Amendment 4 修正数值,定性不变) |
| B6 | **软/部分绑定(soft/partial binding)在读出游出层面实现** | **ESTABLISHED(限本配置族)** | B1–B4 合取;D = 1.74/1.84 > 0 |
| B7 | **硬绑定(exact value delivery)实现** | **REJECTED(未实现)** | 软注意力读出游出为凸组合,D>0;对照 M1(±符号)D=0.106 更接近但仍非精确 |
| B8 | 探针值=0 的解耦消除 amplitude-code leakage(绑定侧) | **SUPPORTED** | M2 探针能量低于胜者(4.2-B);M2unnorm 塌缩到探针、无传递(AM2 绑定侧复现);限单一上下文族 |
| B9 | 硬绑定所需的下一个组织自由度 = winner-take-all 类选择 | **UNKNOWN** | 未测试;仅由 D>0 的归因提示 |
| B10 | generalized binding / 任意 (K,V) 族下的绑定 | **UNKNOWN** | 未做泛化 Probe(用户明令先做最小机制) |
| B11 | vector-QK ⇒ semantic binding / 内容理解 | **REJECTED(默认不接受)** | 只建立了 winner 对齐的软值传递 |

## 2. 协议合规记录

- Amendment 4(M5-E 驱动传递数值手算错误):运行前发现、原文保留、追加修正、
  标签 PREDICTION-CORRECTED-PRERUN;定性结论(方向反转)不变。
- 实现级 bug 修复(3 处,均记录):M4 误走向量核分支(标量核修复)、变量名覆盖
  (T/Ts 数组覆盖规范标量)、JSON 序列化 default——均不改变任何冻结基准或科学结论
  (修复后 42/42)。
- 停止规则:0 真实 FAIL;未发生任何运行后调参/改种子/改阈值/改采样。

## 3. 最终一页结论(口径锁定)

> **4.2-C 在铁门延续下证明:对同一历史只改变查询,(K,V) 解耦的 vector-QK TAN 的
> 读出游出朝被选值移动(距离判据全过、交换精确反对称、幅度分级 0.413 vs 0.084)——
> 即 soft/partial binding 成立;硬绑定(exact value delivery)因软注意力凸组合而未达
> (D≈1.7–1.8),单细胞门控保持传递方向而 E-I 电路反转方向(如实报告)。**
> 研究链更新为:… Vector Q-K ⇒ Routing → (K,V) decoupling ⇒ **Soft Binding**;
> **Hard Binding = 未达(候选组织自由度:winner-take-all,UNKNOWN,留待后续 GO)**。
