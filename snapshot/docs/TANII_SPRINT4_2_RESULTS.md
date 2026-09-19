# TAN-II Sprint 4.2-A — 结果(Results)

**项目**: TAN-II:组织机制与选择性计算的涌现
**阶段**: Sprint 4.2-A —— 纯查询–键几何(无 TAN、无 E-I)
**日期**: 2026-09-04
**预注册**: `docs/TANII_SPRINT4_2_PREREGISTRATION.md`(保持原样)
**理论审计**: `code/experiments/sprint4_2/sprint4_2_theory_audit.py`(43/44 PASS + 1 记录,保持原样)
**Amendment 1**: `docs/TANII_SPRINT4_2_AMENDMENT_1.md`(E4(iii):PROTOCOL-FAIL / NUMERICAL-PASS,已冻结)
**输出**: `results/sprint4_2/`(summary JSON、checks CSV、shortcut JSON/CSV、6 张图)
**程序**: `code/experiments/sprint4_2/{sprint4_2_theory_audit,sprint4_2_geometry,sprint4_2_shortcut_audit}.py`

## 0. 状态块

```
TAN-II SPRINT 4.2-A STATUS: COMPLETE
Theory audit:              44 checks, 43 PASS + 1 recorded FAIL (E4(iii), Amendment 1)
Geometry MC:               317 checks, 316 PASS + 1 PROTOCOL-FAIL/NUMERICAL-PASS, 0 FAIL
Shortcut audit:            ALL PASS
FlipRate_CF(theta) = theta/pi:  CONFIRMED for d in {2,3,4,8}, 11 angles x 3 seeds
Dimension independence:    CONFIRMED (theta/pi identical across d; F_d = 1/2 across d)
S0 -> S1 contrast:         CONFIRMED ({0,1} degenerate vs continuous [0,1])
Ordering capacity ladder:  1 / 2 / M(M-1) / M!   CONFIRMED (exact + LP + dense MC)
Seeds:                     20260904 / 20260905 / 20260906 (each passed independently)
Post-hoc tuning:           0 | Frozen archives modified: 0 | 4.2-B code: NOT written
```

---

## 1. 逐项"解析 vs Monte Carlo"对照(全部预注册判据)

判定准则(预注册):概率量 `|p̂ − p_an| ≤ 3·SE + 0.002`(SE = √(p̂(1−p̂)/N));
确定性量精确相等(1e-12)。三个种子各自满足才算通过。**任一真实 FAIL → STOP AND DEBUG**。

### 1.1 E1 成对 winner 测度(解析 1/2;Case A 为确定性 0/0/1)
- d ∈ {2,3,4,8} × 3 候选对(等范数正交 / 范数比 3 / 近共线)× 3 种子 = 36 项:PASS 36/36
  (代表值 0.494–0.508)。
- d=1:Case A(S1,S2→0;S3→1)**精确命中**;Case B ≈ 0.5(0.493–0.507,PASS);
  Case C ≈ 0.5(0.490–0.510,PASS)。
- **确认 Th3 的范数无关性**:不等范数、近共线均得 1/2(边界过原点)。

### 1.2 E2 查询诱发的 winner 变异 F_d(解析 1/2;Case A = 0)
- 按 d 汇总(每 d:3 对 × 3 种子 = 9 项均值):
  `d=2: 0.4996 | d=3: 0.4997 | d=4: 0.5033 | d=8: 0.4998` → **恒定 1/2(负对照成立)**。
- d=1:Case A 精确 0;Case B / Case C ≈ 0.5(PASS)。
- **确认 F_d 不随维度增长**:"维度越高 routing 越强"式的说法被此负对照排除。

### 1.3 E3 反事实翻转率(核心;解析 = θ/π)
pooled(每 θ 三个种子均值):

| d | θ/π = 0 | 1/12 | 1/6 | 1/4 | 1/3 | 1/2 | 2/3 | 3/4 | 5/6 | 11/12 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | .000 | .087 | .169 | .250 | .334 | .501 | .668 | .749 | .833 | .916 | 1.000 |
| 3 | .000 | .086 | .171 | .253 | .334 | .499 | .668 | .750 | .833 | .917 | 1.000 |
| 4 | .000 | .084 | .169 | .251 | .332 | .500 | .666 | .748 | .831 | .916 | 1.000 |
| 8 | .000 | .084 | .168 | .252 | .338 | .504 | .667 | .750 | .831 | .917 | 1.000 |
| 解析 | 0 | .083 | .167 | .250 | .333 | .500 | .667 | .750 | .833 | .917 | 1 |

- 132 项全部 PASS;**维度无关 + 随 θ 连续 [0,1]** 两项性质同时确认。
- **E3b(不等范数对照,范数比 10)**:θ ∈ {π/6, π/2, 5π/6} × 4 d × 3 种子 = 36 项 PASS
  (代表 0.168/0.503/0.831)⇒ 反事实翻转与候选范数无关(捷径审计 5a 的配套)。
- d=1:Case A 精确 0;Case B ≈ 0.5(PASS);**Case C:(+1,+1)→0、(+1,−1)→1 精确命中**。
- **S⁰ vs S¹ 对比确认**:带符号标量的翻转率只取退化端点 {0,1};d≥2 取连续区间 [0,1]。

### 1.4 E4 JSD 与 winner 重排的合取
- (i) 正标量陷阱:`JSD(σ(−1),σ(−4)) = 0.0751 > 0.05` 且 **flip = 0**(PASS)——查询改变
  注意力锐度/幅度,但**不改变 winner**。
- (ii) d=2 θ=2π/3:`JSD = 0.0647 > 0.05` 且 **flip = 1**(PASS)。
- (iii) d=2 θ=π/6:`JSD = 0.00045439`,`flip = 0` —— 按 Amendment 1 标记
  **PROTOCOL-FAIL / NUMERICAL-PASS**(原预测 0.00103 无效;对修正闭式基准偏差 < 1e-12;
  熵恒等式独立路径一致)。定性主张 **JSD>0 且无翻转** 成立并单独报告。
- (iv) 集总对照(θ=π/2,各向同性历史):`P[JSD>1e-9]` = 0.9998/0.9998/0.9999
  (解析 a.e. 1;断言 ≥ 0.999 通过)vs `P[flip]` = 0.502/0.506/0.494(解析 0.5,通过)。
  → **"分布改变"(≈100%)与"winner 重排"(=θ/π)之间存在一个可量化的缺口**:
  单独 JSD 会以 ~100% 的假阳性宣称"查询条件化再分配",合取判据才是对的。

### 1.5 E5 几何捷径审计
- 5a 2 候选范数无关性:r ∈ {1,2,5,10} 全部 ≈ 0.5(精确 0.5;MC 0.492–0.508)→ 2 候选
  点积 argmax 无范数捷径(Th3 的数值确认)。
- 5b distractor 范数泄漏(3 候选,精确布置积分 + MC 验证):
  | m | 0.5 | 1.0 | 2.0 | 5.0 | 10.0 |
  |---|---|---|---|---|---|
  | P[w=D] 精确 | 0.0907 | 0.1250 | 0.3407 | 0.4480 | 0.4758 |
  | MC 均值 | 0.0887 | 0.1262 | 0.3472 | 0.4513 | 0.4801 |
  - 注意 m=1(等范数)**非对称**布置下 P[D] = 0.125 ≠ 1/3 —— "等范数"本身不够,需要
    **布置对称或全随机化**;大范数 distractor 抢占查询空间份额(→ 渐近 0.5)。
    → 这就是 4.2-B Probe 必须用等范数 + 对称/随机化控制的原因。
- 5c 对称等范数 3 候选:{e₁, R₁₂₀e₁, R₂₄₀e₁}:精确 1/3 各,MC 0.326–0.340(PASS)。

### 1.6 E6 排序容量阶梯(精确 + LP + 稠密 MC)
- 精确 d=2 布置枚举:M=2 → 2;M=3 → 6;M=4(平面)→ 12(区域测度和 = 1)。
- LP 存在性:d=2 M=3 全 6 可行;M=4 恰好 12 可行且与布置胜者集**逐项相等**(另 12 不可行);
  d=3/8 M=3 → 6;d=3/8 M=4(满秩)→ 24;d=1:正标量仅自然序、带符号恰好自然+逆序。
- 稠密 MC(200,000 角度):观测相异排序数 == 精确计数(M=3:6/6;M=4:12/12)。
- **阶梯**:`1(正标量) → 2(带符号,任意 M) → M(M−1)(d=2) → M!(d≥3, M≤d+1)`,
  即 M=2,3,4 时 {1,2,2} / {1,2,6} / {1,2,12} / {1,2,24} / {1,2,24}。
  S⁰→S¹ 的拓扑跃迁被操作化为"查询可实现排序数"的跳变(2 → M(M−1) → M!)。

---

## 2. 十个问题(§19)的最终回答

1. **d=1 顺序不变性在什么条件下成立?** 查询域 ⊆ R\{0} 的单一符号类(Th1)。
   TAN 继承查询 γ_t = β W_q W_k [x_t − μ_t − ε]_+ ≥ 0 正是单一符号类 ⇒ 顺序锁定。
2. **放开 Q>0 后 d=1 是否仍不可能?** 否。Q→−Q 翻转 winner(Th2);
   实测 Case B ≈ 0.5、Case C 精确 {0,1}。d=1 的不可能定理是**域条件**命题。
3. **d=2 是否产生非平凡 winner 划分?** 是。边界为大圆,两个开半圆域,各测度 1/2;
   实测全部 d≥2、全部候选对 ≈ 0.5。
4. **是否存在解析 routing-capacity 证明?** 是。`FlipRate_CF(θ) = θ/π`(投影+均匀角证明,
   见 Theory Audit §4.3);另 `P[w=A]=1/2`(Th3)、`F_d=1/2`(Th4)、排序容量(布置+LP)。
5. **MC 是否与解析一致?** 一致:316/316 PASS + 1 PROTOCOL(Amendment 1);理论审计
   44 项中 43 项精确通过、1 项按修正闭式基准数值通过(合规性 FAIL 保留)。
6. **d=2 是否产生反事实 winner flip?** 是:`θ=π` → 1(端点精确),θ∈(0,π) 按 θ/π 连续;
   且与 d 无关(d=3,4,8 同曲线)。
7. **flip 是否只是 shortcut?** 两候选点积 argmax/软注意力只依赖 ΔK(范数无关,5a 实测);
   但 ≥3 候选时 distractor 范数泄漏存在(5b:0.091→0.476)且"等范数非对称布置"也不均匀
   (0.125≠1/3)→ 4.2-B 必须等范数+对称/全随机化。**配对 flip 本身不是捷径;泄漏来自
   多候选与布置不对称。**
8. **E-I 拓扑加入后是否改变边界?** 否(结构性:排序发生在 E-I 动力学之前的核能量上;
   E-I 只作用标量聚合)。4.1-C/D 已实测 E-I+标量注意力的 FlipRate=0;本 Sprint 的几何
   容量是核级性质,与 E-I 无关。
9. **d=2 证明的是哪一级?** 仅 **Level-1 geometric capacity**(解析 + MC)。Level-2
   (realized routing in vector-QK TAN)是 4.2-B 的对象;Level-3(泛化)需独立 Probe。
10. **能否合法写 d_min = 2?** 不能无条件写。合法表述(全部已验证):
    (a) "Within the positive-query scalar family (TAN-inherited), ordering is
    query-invariant (FlipRate=0)."
    (b) "Order invariance is broken exactly when the query can cross Q^TΔK = 0."
    (c) "d = 2 is the minimal dimension for a connected, rotationally-symmetric
    continuous query-direction manifold (S¹): the minimal d at which FlipRate_CF
    ranges continuously over (0,1) (= θ/π) rather than the degenerate {0,1} of S⁰."
    若用符号,写作 **d\* = 2** 并始终附带 (a)(b)(c)。

---

## 3. 本 Sprint 明确不宣称(No-claims)

- 不宣称 d_min = 2(无定语);不宣称"相变"(无数值与序参量定义);
- 不宣称 d=2 ⇒ content-addressable routing(Claim C,默认拒绝);
- 不宣称 realized/generalized routing(仅 Level-1 容量);
- 不宣称 E-I 拓扑与路由边界有关(边界是核级的);
- 不宣称三种候选物理实现(树突隔室/相位耦合/DA-5HT 平面)≡ R²(future hypotheses);
- 不把 E4(iii) 的合规性 FAIL 掩盖掉(Amendment 1 公开保留)。

---

## 4. 复现与文件

```
cd <repo>
python code/experiments/sprint4_2/sprint4_2_theory_audit.py      # exact audit (exit 1 by design: E4(iii) recorded FAIL)
python code/experiments/sprint4_2/sprint4_2_geometry.py          # MC suite (exit 0)
python code/experiments/sprint4_2/sprint4_2_shortcut_audit.py    # shortcut audit (exit 0)
```
- `results/sprint4_2/sprint4_2_summary.json`(317 项检查全表、参数、脚本哈希、版本)
- `results/sprint4_2/sprint4_2_checks.csv`(逐项:est / analytic / SE / CI / status / note)
- `results/sprint4_2/sprint4_2_shortcut_{summary.json,checks.csv}`
- 图:`sprint4_2_{winner_measure,fd,flip_theta,jsd,ordering,shortcut}.png`
- 环境:Python 3.12.7、numpy 1.26.4、scipy 1.13.1、matplotlib 3.9.2;确定性(种子固定)。

*理论审计 exit 1 是原始记录的保留形态(E4(iii) 原始断言未达);Amendment 1 对其
重新分类;不修改历史文件。*

## 5. 下一步(未开始,待 GO)

4.2-B(vector-QK TAN,`sprint4_2_vector_qk.py` / `sprint4_2_routing_probe.py`)在
Amendment 1 已冻结、本结果文档完成的前提下才可开始;其 Probe 设计必须落实本 Sprint
的 5b 结论(等范数、对称/全随机化、探针不入候选集)。本 Sprint 未写任何 4.2-B 代码。
