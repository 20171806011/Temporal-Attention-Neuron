# TAN-II Sprint 4.2-A — 预注册(Preregistration)

**项目**: TAN-II:组织机制与选择性计算的涌现
**阶段**: Sprint 4.2-A —— **纯查询–键几何**(无 TAN、无 E-I、无 surprise gate、无 LIF 动力学)
**前置**: `docs/TANII_SPRINT4_2_THEORY.md`(Theory Audit,已通过,含七条修正)
**状态**: 预注册(本文件先于任何运行完成;正式运行后不改任何科学设置)
**日期**: 2026-09-04
**目标措辞(已锁定)**:
> "验证在 TAN-compatible positive-query scalar family 之外,d=2 是产生连续 query-direction
> freedom 的最小整数维度,并验证其 query-conditioned ordering geometry。"
> (不声称"证明 d=2 是 routing 的 minimum dimension";不进入 4.2-B 的 TAN 改造。)

---

## 0. 采纳的七条修正(逐条落地)

1. 定理措辞锁定:一律写 "within the positive-query scalar family (TAN-inherited, Q≥0),
   candidate ordering is query-invariant";禁止写 "1-D attention cannot route"。
2. 三域分报:Case A 正标量 / Case B 无约束标量 / Case C ±1 归一化标量,互不混写。
3. 统计量以 `FlipRate_CF(θ) = θ/π` 为核心;`F_d = 1/2 ∀d≥2` 作为**维度无关负对照**,
   不作"维度跃迁"证据。
4. 捷径控制(几何层):2 候选范数无关性、distractor 范数泄漏、对称 3 候选 = 1/3 各、全部审计。
5. 分层声称:本 Sprint 只输出 **Level-1 geometric capacity**;不声称 realized/generalized routing。
6. Monte Carlo 逐项比对解析值,任何一项违反判定准则 → **STOP AND DEBUG**,不进入下一步。
7. 停止规则保留。

---

## 1. 数学对象(全部继承 Theory Audit 的记号)

- 候选键 `K_A, K_B ∈ R^d`(d ∈ {1,2,3,4,8});`ΔK = K_A − K_B ≠ 0`。
- 能量 `E_i(Q) = Q^T K_i`(d=1 为标量积)。
- 硬 winner `w(Q) = A ⟺ Q^TΔK > 0`;软注意力(2 候选)`α_A(Q) = σ(Q^TΔK)`,`σ` = logistic。
- 解析预言(Theory Audit 定理 1–5):
  - Th1:正标量族(D⊆R_{>0})内 `w(Q)` 常数;
  - Th2:带符号标量 `w(+1)=argmax_i K_i`,`w(−1)=argmin_i K_i`;
  - Th3:决策边界 `Q^TΔK=0` 是过原点的大超球面,`P_Q[w=A]=P_Q[w=B]=1/2`(**范数无关**);
  - Th4:`F_d = P[w(Q_1)≠w(Q_2)] = 1/2`(Q_1,Q_2 iid,d≥2 与 d=1 带符号;正标量族 = 0);
  - Th5:`FlipRate_CF(θ) = θ/π`(ΔK 各向同性,d≥2,θ = arccos(Q_A^TQ_B));
  - 排序容量:正标量 1;带符号标量 2(任意 M);d=2 为 M(M−1)(一般位置);
    d ≥ 3 且 M ≤ d+1(一般位置)为 M!。

---

## 2. 实验协议(全部预注册)

### 2.1 查询分布
- d ≥ 2:`Q ~ Unif(S^{d−1})`,用归一化高斯 `g/‖g‖`,`g ~ N(0, I_d)` 生成。
- d = 1:
  - Case A(正标量,TAN-compatible):`Q ~ U(0.01, 4.0)`;
  - Case B(无约束标量):`Q ~ N(0,1)`;
  - Case C(归一化标量):`Q ~ Uniform{−1, +1}`。
- 样本量:`N_Q = 10,000`/种子;种子 `{20260904, 20260905, 20260906}`(确定性,每个种子独立通过)。
- 独立查询对(F_d 与部分 E3)由 20,000 个查询两两配对得到(10,000 对)。

### 2.2 候选对
- d ≥ 2(每 d 嵌入环境维,前两坐标承载结构,其余为 0):
  - P1 等范数正交:`K_A = e_1`,`K_B = e_d`;
  - P2 不等范数(比率 3):`K_A = e_1`,`K_B = 3 e_d`;
  - P3 近共线(不等范数):`K_A = e_1`,`K_B = e_1 + 0.05 e_d`。
- d = 1:`S1: (K_A,K_B) = (1.0, 2.0)`;`S2: (−1.0, 1.0)`;`S3: (2.0, 1.0)`。
- E3 的历史:等范数主配置 `K_A = ΔK/2, K_B = −ΔK/2`(‖K_A‖=‖K_B‖=1/2,ΔK 各向同性);
  E3b 不等范数对照 `K_A = ΔK, K_B = −0.1 ΔK`(范数比 10,ΔK 方向不变)。
- 排序容量键集(d≥2):
  - M=3:`K = {e_1, e_2, (e_1+e_2)/√2}`;
  - M=4 平面(d=2):`K = {e_1, e_2, (e_1+e_2)/√2, (2e_1+e_2)/√5}`;
  - M=4 满秩(d=3,4,8):`K = {e_1, e_2, e_3, (e_1+e_2+e_3)/√3}`;
  - d=1 M=3:`K = {1.0, 2.0, 3.0}`。

### 2.3 实验清单(每项 = "解析 vs Monte Carlo" 一行)

- **E1 成对 winner 测度**:每 (d, 候选对, 种子) 估 `P̂[w=A]`;解析:d≥2 与 Case B/C 为 1/2;
  Case A 为确定性 `[K_A > K_B]`(S1,S2 → 0;S3 → 1)。
- **E2 查询诱发的 winner 变异 F_d**:每 (d, 对, 种子) 估 `F̂_d`;解析:d≥2 与 Case B/C 为 1/2;
  Case A 为 0(精确)。**负对照**:F̂_d 在 d ∈ {2,3,4,8} 恒定。
- **E3 反事实翻转率(核心)**:d≥2 × 角度网格 θ ∈ {0, π/12, π/6, π/4, π/3, π/2, 2π/3, 3π/4, 5π/6,
  11π/12, π}(Q_A = e_1,Q_B = (cosθ, sinθ, 0, …))× 3 种子;N_H = 10,000 个各向同性 ΔK;
  解析 = θ/π。对照:E3b(不等范数,θ ∈ {π/6, π/2, 5π/6},解析仍 θ/π);d=1:Case A → 0;
  Case B(Q_A,Q_B iid N(0,1))→ 1/2;Case C(+1,+1) → 0、( +1,−1) → 1(确定性)。
- **E4 JSD 与 winner 重排的合取**:
  (i) 正标量陷阱:固定 H:K_A=1.0,K_B=2.0(ΔK=−1);γ_A=1,γ_B=4:
  `α_A(γ)=σ(−γ)` ⇒ JSD(α(1),α(4)) 解析 ≈ 0.0751(断言 > 0.05)**且** flip = 0(断言精确 0)。
  (ii) d=2 翻转示例:固定 H:K_A=(0.5,0),K_B=(−0.5,0)(ΔK=e_1);Q_A=e_1,
  Q_B=(cos 2π/3, sin 2π/3):JSD ≈ 0.0647(断言 > 0.05)**且** flip = 1(断言)。
  (iii) d=2 不翻转但 JSD>0:同 H,Q_B=(cos π/6, sin π/6):JSD ≈ 0.00103(断言 > 5e-4)
  **且** flip = 0(断言)——证明"单独 JSD 不足"在任何族都成立,合取才是判据。
  (iv) 集总对照(θ=π/2,各向同性 H,N_H=10,000 × 3 种子):
  `P[JSD > 1e-9]`(解析 ≈ 1.0,断言 ≥ 0.99)vs `P[flip]`(解析 = 0.5,按判定准则)。
- **E5 几何捷径审计**(`sprint4_2_shortcut_audit.py`):
  (a) 2 候选范数无关性:d=2,`K_A=e_1, K_B=r·e_2`,r ∈ {1,2,5,10}:P̂[w=A] = 1/2(Th3);
  (b) distractor 范数泄漏:d=2,`{K_A=e_1, K_B=e_2, K_D=m·(e_1+e_2)/√2}`,m ∈ {0.5,1,2,5,10}:
  精确 d=2 布置积分给出 P̂[w=D](解析曲线)+ MC 验证;期望 m=1 时 ≠ 1/3(非对称布置)、
  随 m 单调升(渐近 ~1/2);
  (c) 对称等范数 3 候选:`{e_1, R_{2π/3}e_1, R_{4π/3}e_1}`:P̂[w=i] = 1/3 各(断言)。
- **E6 排序容量阶梯**:
  (a) 精确 d=2 布置枚举(边界角 `atan2` 解析,取中点读 winner,统计相异排序数):
  M=2 → 2;M=3 → 6;M=4(平面)→ 12(精确断言);
  (b) LP 存在性(scipy linprog,`|Q_l| ≤ 1`,相邻不等式 `Q^T K_{π_k} − Q^T K_{π_{k+1}} ≥ ε=1e-6`):
  d=2 M=3:6/6 可行;d=2 M=4:恰好 12 可行且与 (a) 的排序集合一致、另 12 不可行;
  d=3、8 M=3:6 可行;d=3、8 M=4(满秩):24 全可行;
  d=1:Case A(Q∈[ε,1])仅自然序可行;Case B(Q∈[−1,1])自然序 + 逆序可行、其余不可行;
  (c) 稠密采样交叉验证:角度 N_ang=200,000 均匀,观测相异排序数 == 精确计数(断言)。
- 输出图:`results/sprint4_2/sprint4_2_{winner_measure, fd, flip_theta, jsd, ordering, shortcut}.png`。

### 2.4 判定准则(预注册)
- 概率量:`|p̂ − p_an| ≤ 3·SE + 0.002`,`SE = sqrt(p̂(1−p̂)/N)`;同时报告 95% CI(正态近似)。
- 确定性量(d=1 符号结构、计数、LP 可行性集合、JSD 断言):精确相等/断言,容差 1e-12
  (计数:整数相等)。
- 三个种子**各自**满足准则才算通过;任一违反 → 该项 FAIL → **STOP AND DEBUG**(不静默通过)。
- 全部结果机读:`results/sprint4_2/sprint4_2_summary.json`(参数、解析值、MC 估计、SE、CI、
  PASS/FAIL 每项、脚本 SHA-256、numpy/scipy 版本)+ `results/sprint4_2/sprint4_2_checks.csv`
  (逐项检查表)+ 各实验 raw CSV。

### 2.5 预注册预测(可证伪)
1. E1:全部 d≥2 与 Case B/C ≈ 1/2;Case A = 0/0/1(按键序确定性)。
2. E2:F̂_d ≈ 1/2 对 d ∈ {2,3,4,8} 恒定;Case B/C = 1/2;Case A = 0。
3. E3:FlipRate_CF(θ) = θ/π,对 d ∈ {2,3,4,8} 维度无关、随 θ 连续 [0,1];
   不等范数对照相同;d=1:Case A 0 / Case B 1/2 / Case C {0,1} 退化端点。
4. E4:JSD>0 且 flip=0(正标量);JSD>0 且 flip=1(d=2, θ=2π/3);JSD>0 且 flip=0(d=2, θ=π/6);
   集总 P[JSD>1e-9] ≈ 1 而 P[flip] = θ/π。
5. E5:2 候选范数无关(1/2);distractor 泄漏曲线随 m 单调升;对称 3-set = 1/3 各。
6. E6:排序容量阶梯 1 / 2 / {2,6,12} / {2,6,24} / {2,6,24}(d = 1pos / 1signed / 2 / 3 / 8,
   M = {2,3,4});LP 可行集合精确匹配。

### 2.6 纪律与排除
- 只用 numpy / scipy / matplotlib;确定性种子;无 ML 框架。
- 不修改 TAN-I 冻结档案、不修改 TAN-II 4.1-C/D 结果。
- 不因结果改参数、不改 query 分布、不改候选编码;发现科学级 bug(公式/模型/条件)→ 停止并报告;
  仅实现级 bug 允许修复并记录。
- 本 Sprint 不写 `sprint4_2_vector_qk.py`、`sprint4_2_routing_probe.py`(4.2-B 内容)。

*预注册完成。所有后续运行必须与本文件一致。*
