# TAN-II Sprint 4.2-B — 结果(Results,v1:vector-QK mechanism isolation)

**项目**: TAN-II:组织机制与选择性计算的涌现
**阶段**: Sprint 4.2-B v1 —— vector-QK 机制隔离(铁门:仅表示层变化)
**日期**: 2026-09-04
**预注册**: `docs/TANII_SPRINT4_2B_PREREGISTRATION.md`(保持原样)
**Amendment 2 / 3**: 运行前解析修正(未归一化对照 = 0;fp 数值修正),均已冻结
**输出**: `results/sprint4_2/sprint4_2b_*`(JSON/CSV/图)
**程序**: `code/experiments/sprint4_2/{sprint4_2_vector_qk, sprint4_2_routing_probe}.py`

## 0. 状态块

```
TAN-II SPRINT 4.2-B STATUS: COMPLETE
Canonical audit (deterministic):     27/27 PASS (0 FAIL)
Randomized routing probe:            48/48 PASS (0 FAIL)
Minimal success criterion:           MET -- fixed history [1,0,2,0,Q], only query changes:
                                     M2/M5-E: winner A(Q=3.0) -> B(Q=4.0), margins +0.2587/-0.1559
Scalar boundary retained:            M0/M4 FlipRate = 0 (exact); M1 = 1 (degenerate sign)
Ensemble FlipRate_CF:                M2/M5-E = 0.4425 (bench 0.4445) | M0/M4 = 0 | M1 = 1
Normalization as enabler (AM2):      unnormalized control FlipRate = 0 exactly
Iron gate:                           ONLY representation changed; window/surprise/E-I/membrane/
                                     readout frozen; no learning/memory/feature extractor
Post-hoc tuning: 0 | Frozen archives modified: 0 | Amendments: 2 (all pre-run, frozen)
```

---

## 1. 铁门合规声明

- 窗口 W=5、惊奇门 `S=[x_t−μ_t−ε]_+`(μ 含当前 tap)、E-I 拓扑(仅 M4/M5,w_EI=0.8 等)、
  膜动力学(λ=0.5,θ_sp=3.0,脉冲复位)、读出(注意力 argmax + 条件固定点)——**全部与
  TAN-I / 4.1 遗产逐字节一致**。
- 唯一变化:键 `x_i → φ̂_d(x_i)`,查询 `γ_t → c·S_t·û(ωS_t)`(d=2 旋转方向,ω=0.4 预注册)。
- 无学习、无额外记忆、无特征提取器;M1 为外部符号控制对照,M3 为环境维对照。
- 因此任何翻转都可**唯一归因于表示维度的变化**。

## 2. 规范反事实(确定性,27/27)

固定历史 `[1.0, 0.0, 2.0, 0.0]`,仅查询 Q∈{3.0, 4.0} 变化(win=历史槽 {A,B} 的能量 argmax):

| 模型 | winner(Q=3) | winner(Q=4) | Flip | 边距(E_A−E_B) | 备注 |
|---|---|---|---|---|---|
| M0 标量 | B | B | 0 | −3.60 / −5.20 | 探针能量 10.8/20.8 占优(amplitude-code leakage 复现) |
| M1 符号控制 | B(+1)/A(−1) | 同 | 1(退化) | — | 全窗口 winner = pad(能量基准 0) |
| **M2 向量 d=2** | **A** | **B** | **1** | **+0.2587 / −0.1559** | 探针能量 3.108/4.989 均**低于**胜者(无幅值捷径) |
| M2 未归一化(AM2) | B | B | 0 | 负/负 | 归一化是翻转的机制使能条件(见 §4) |
| M3 d=3,4,8 | A | A | 0 | + | 交叉点在查询区间外(θ*=1.945/2.197/2.347 > θ(4)=1.04) |
| M4 E-I 标量 | B | B | 0 | — | fp=1.359/1.640 亚阈值;重确认 4.1 边界 |
| **M5 E-I 向量** | **A** | **B** | **1**(E 支) | +0.2587/−0.1559 | **I 支保持 A,A(分支分歧)**;fp=0.893/1.152 亚阈值 |

- P1 交叉点 Q\* = 3.7071(闭式)与扫描一致:winner(Q≤3.70)=A,winner(Q≥3.75)=B。
- P10 JSD 合取:M0 `JSD>0 ∧ flip=0`(陷阱在 TAN 内复现);M2 `JSD>0 ∧ flip=1`;M1 `JSD>0 ∧ flip=1`。
- P11 M2 脉冲:fp(3)=3.404、fp(4)=4.516 ≥ θ_sp(响应幅度同时查询敏感);M4/M5 亚阈值。
- P12 ω 扫掠(确定性):ω=0.2 → (A,A);ω=0.4 → (A,B);ω=π/2 → (B,B);ω=π → (A,B)
  —— 翻转是"查询方向扫过决策边界"的结构结果,不是任何 ω 都有的偶然现象。
- P13:d=8 时 B 键能量 ≈ 基线(0.079 vs pad 0),注意力质量向 pad 迁移——高维矩曲线把
  Δφ̂ 移出查询子空间,如实报告。

## 3. 随机化集合探针(48/48,MC vs 冻结求积基准)

x_A,x_B iid U(0.3,2.5)、12 种有序位置、pad=0、探针 Q∈{3,4}:

| 模型 | 基准(2001² 求积,冻结) | MC(3 种子 × 10⁴) | 判定 |
|---|---|---|---|
| M0 | 0.000000 | 0.000000 ± 0.000000 | PASS |
| M1 | 1.000000 | 1.000000 ± 0.000000 | PASS |
| **M2** | **0.444527** | **0.442533 ± 0.004853** | PASS |
| M2unnorm | 0.000000 | 0.000000 | PASS |
| M3d3 | 0.054937 | 0.054267 ± 0.000634 | PASS |
| M3d4 | 0.028391 | 0.028367 ± 0.001109 | PASS |
| M3d8 | 0.017159 | 0.017800 ± 0.001233 | PASS |
| M4 | 0.000000 | 0.000000 | PASS |
| **M5-E** | **0.444527** | **0.442533 ± 0.004853** | PASS |
| M5-I | 0.405772 | 0.404833 ± 0.001914 | PASS |

> 解读:对随机化历史,把查询从 3.0 换成 4.0 会在 **~44% 的历史上改变被选候选**
> (M2/M5-E);标量族(M0/M4)精确为 0;未归一化精确为 0(AM2);符号控制精确为 1(退化)。

## 4. 机制发现(全部来自冻结输出,非事后解释)

1. **归一化 = 翻转的机制使能条件(Amendment 2 升级为正面结论)**:
   未归一化键 p(x)=(x,x²) 下 `û·Δp = (x_A−x_B)[cosθ + (x_A+x_B)sinθ]`,ω=0.4 区间内
   括号恒正 ⇒ winner 由幅度序锁定,FlipRate=0 **精确**。单位化键才让 ΔK 方向随候选对
   自由转动。与 4.2-A 5b(等范数≠布置对称)互相印证且更强:2 候选情形下
   **无归一化 ⇒ 容量 0;归一化 ⇒ 0.4445**。
2. **探针自抑制(无 amplitude-code leakage)**:M2 在 Q=3,4 处探针能量(3.108/4.989)
   均低于胜者能量——向量方向使大探针远离历史键方向,TAN-I Probe-2 的幅值泄漏在此
   配置下不出现;M0 对照组仍复现泄漏(探针 10.8/20.8 占优)。
3. **E-I 不改变排序边界、但改变响应工作点**:M4 FlipRate=0(重确认 4.1);M5 E 支继承
   M2 的翻转,同时 I 支因 ε_I 滞后保持 A,A(分支分歧),且 E-I 耦合把 fp 压回亚阈值
   (M2 单细胞 3.40/4.52 脉冲 vs M5 0.89/1.15 亚阈值)——注意力的排序能力是核级性质,
   增益/脉冲工作点是动力学级性质。
4. **高维≠更强(措辞修正版)**:In this mechanism family, increasing ambient embedding
   dimension beyond the minimal 2-D query subspace did not increase routing capacity
   and, under the tested finite candidate/query configuration, reduced the observed
   flip rate(M3:0.0549→0.0284→0.0172)。真正原因是 **representation–candidate geometry
   mismatch**(P13:d=8 时 Δφ̂ 移出 2-D 查询子空间,B 键能量退到 pad 基线、注意力质量
   被 pad 吸走)——不是"高维本身抑制 routing"、更不是"curse of dimensionality"类结论。

## 5. 核心问题的回答(用户口径)

> **"Does vector Q-K organization actually realize that capacity?"**
**是(Level-2 realized routing,在本配置族内)**:
- 固定同一历史、只改变 query 的反事实条件下,M2(最小向量 d=2)与 M5-E 的注意力
  winner 从 A 翻转到 B(边距 +0.2587/−0.1559,非数值噪声);翻转在 ω 扫掠中呈现
  结构依赖(0.2 无、0.4 有、π/2 无、π 有),与解析交叉点 Q\*=3.7071 一致;
  集合 FlipRate = 0.4425 vs 冻结基准 0.4445。
- 同一协议下标量族精确为 0——**4.2-A 的容量被 4.2-B 实现,且实现严格来自
  表示层变化(铁门)**。

## 6. 层级声称纪律(严格执行)

- 已建立:**Level-2 realized routing(本配置族)**——运行模型在反事实查询下实际改变
  注意力 winner,并通过 JSD>0 ∧ flip 合取、边距、探针非捷径审计。
- **不**声称:Level-3 generalized routing(需查询统计/候选全随机化的独立泛化 Probe,
  未做)、semantic binding(未做)、任何"智能/能力"表述。
- 不声称 d=2 在一切表示族中的最小性(本结果限"旋转方向查询 × 归一化矩曲线键 ×
  该 ω/上下文族";M3 的反例反而说明映射族选择的重要性)。

## 7. 研究链现状

Memory → Scalar Saliency → Geometric Reorganization → Opponent Competition
⇏ Content Routing(4.1)→ 4.2-A 给出几何容量标尺(θ/π、排序容量阶梯)→
**4.2-B 证明最小向量-QK 组织实现该容量(Level-2)**。
下一步(Routing→Binding 或 Level-3 泛化 Probe)为未来 Sprint,需另行 GO。

## 8. 复现

```
python code/experiments/sprint4_2/sprint4_2_vector_qk.py      # 27 checks, exit 0
python code/experiments/sprint4_2/sprint4_2_routing_probe.py  # 48 checks, exit 0
```
文件:`results/sprint4_2/sprint4_2b_{summary,probe_summary}.json`、
`results/sprint4_2/sprint4_2b_{checks,probe_checks}.csv`、
`sprint4_2b_{canonical,tuning,flip_q,ensemble,omega}.png`、`sprint4_2b_tuning.csv`。
环境:Python 3.12.7 / numpy 1.26.4 / scipy 1.13.1 / matplotlib 3.9.2;确定性。
