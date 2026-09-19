# TAN-II Sprint 4.2-B — 主张账本(Claim Ledger)

**阶段**: Sprint 4.2-B v1(vector-QK mechanism isolation)
**日期**: 2026-09-04
**等级**: ESTABLISHED / SUPPORTED / UNKNOWN / REJECTED。
**依据**: `docs/TANII_SPRINT4_2B_{PREREGISTRATION,RESULTS}.md`、
`docs/TANII_SPRINT4_2B_AMENDMENT_{2,3}.md`、`results/sprint4_2/sprint4_2b_*.json`。

---

## 1. 核心命题

| # | 主张 | 等级 | 依据与限定 |
|---|---|---|---|
| C1 | **向量-QK 组织在运行模型中实现反事实 winner 翻转(Level-2 realized routing,本配置族)**:固定历史 [1,0,2,0]、仅查询 3.0→4.0,M2/M5-E 的注意力 winner 从 A 翻到 B | **ESTABLISHED** | 确定性闭式 27/27(边距 +0.2587/−0.1559,交叉点 Q\*=3.7071 闭式与扫描一致);集合 MC 0.4425 vs 冻结基准 0.4445(3 种子,全部 ≤ 3SE+0.002);JSD>0 ∧ flip 合取成立 |
| C2 | 标量正查询族保持顺序锁定(M0/M4 FlipRate = 0,精确) | **ESTABLISHED** | 确定性 0;集合 MC 精确 0;重确认 TAN-I Probe-2 / 4.1 边界 |
| C3 | 带符号标量控制只给退化翻转(M1 = 1,精确;{0,1} 无中间值) | **ESTABLISHED** | 符号翻转即反转;全窗口 winner 退化为 pad |
| C4 | **归一化(等范数键)是翻转的机制使能条件**:未归一化对照 FlipRate = 0 精确 | **ESTABLISHED** | Amendment 2 的解析符号结构 `(x_A−x_B)[cosθ+(x_A+x_B)sinθ]` 在 ω=0.4 区间恒正 ⇒ 顺序锁定;数值精确 0。原预注册预测错误,已按 AM2 修正并保留原文 |
| C5 | 翻转不是 amplitude-code / 探针捷径 | **SUPPORTED** | 探针能量在两查询下均低于胜者(3.108<3.592,4.989<5.188);探针不入候选集;幅度/位置随机化;但限单一 ω、单一上下文族 |
| C6 | E-I 拓扑不改变排序边界(核级性质),只改变响应工作点 | **ESTABLISHED** | M4(标量)FlipRate=0;M5 E 支继承 M2 翻转、I 支滞后保持 A,A;E-I 耦合把 fp 压回亚阈值(M2 脉冲 3.40/4.52 vs M5 0.89/1.15) |
| C7 | 翻转是"查询方向扫过决策边界"的结构结果(ω 依赖,非偶然) | **ESTABLISHED** | ω∈{0.2,0.4,π/2,π} → (A,A)/(A,B)/(B,B)/(A,B),与闭式交叉点完全一致 |
| C8 | 在本机制族内,环境嵌入维超过最小 2-D 查询子空间并未增加路由容量,且在所测有限候选/查询配置下观测翻转率下降(M3:规范无翻转;集合率 0.0549→0.0284→0.0172) | **ESTABLISHED(限该表示族,措辞修正)** | 归因为 representation–candidate geometry mismatch(2-D 约束查询 × 矩曲线键;P13:d=8 时 Δφ̂ 移出查询子空间、质量被 pad 吸收);**不是**"高维抑制 routing"或"curse of dimensionality" |
| C9 | "d=2 是产生连续查询方向自由度的最小整数维度"(4.2-A 口径) | **ESTABLISHED(4.2-A 已建立,4.2-B 未改变)** | θ/π 曲线、排序容量阶梯;4.2-B 只在 4.2-A 框架内补充"实现" |
| C10 | d=2 ⇒ generalized content-addressable routing(Level-3) | **UNKNOWN** | 需查询统计/候选全随机化的独立泛化 Probe;本 Sprint 未做 |
| C11 | 向量-QK ⇒ semantic binding | **REJECTED(默认不接受,未测试)** | Level-1/2 不蕴含 binding;保持 4.2-A 的层级纪律 |
| C12 | "模型性能更高 ⇒ 机制更正确" | **REJECTED** | 无性能判据;M2 单细胞反而脉冲、M5 亚阈值——工作点差异不承载机制正确性 |

## 2. 协议合规记录

- 铁门逐项核对:窗口/惊奇门/E-I/膜/读出冻结,仅表示层变化;无学习/记忆/特征提取器 ✓。
- Amendment 2(未归一化对照预测错误)与 Amendment 3(fp 数值手算错误):均**运行前**发现、
  原文保留、追加修正、标签 PREDICTION-CORRECTED-PRERUN;未发生任何运行后调参/改种子/
  改阈值/改采样。
- 实现级 bug 修复(3 处,均记录):基准求积的范数平方笔误(x^k → x^{2k})、M1 网格对角平局
  处理、向量化 energy_of——全部只影响基准复算,不影响冻结基准与 MC 结论(修复后 48/48)。
- 停止规则:本 Sprint 无真实 FAIL;理论审计(4.2-A)1 项记录 + 本 Sprint 0 项 FAIL。

## 3. 最终一页结论(口径锁定)

> **4.2-B 在铁门约束下证明:把查询/键从标量因子提升为向量自由度(d=2,归一化矩曲线键 ×
> 旋转方向查询),即可在运行中的 TAN 里实现反事实 winner 翻转(Level-2 realized routing):
> 规范历史边距 +0.259/−0.156,集合翻转率 0.4425(vs 冻结基准 0.4445),标量族精确 0。**
> 同时得到两个边界性结论:(i) 归一化(等范数)是翻转的机制使能条件,不是装饰控制;
> (ii) 环境维不增加翻转容量。研究链由此推进到
> Memory → Saliency → Geometric Reorganization → Opponent Competition
> ⇏ Content Routing → **minimal vector-QK organization ⇒ realized reordering**;
> Routing→Binding 与 Level-3 泛化留待后续 Sprint(UNKNOWN,不宣称)。
