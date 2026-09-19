# TAN-II Sprint 4.3-A — Theory / Preregistration(Minimal Composition Probe)

**项目**: TAN-II:组织机制与选择性计算的涌现
**阶段**: Sprint 4.3-A —— Binding → Composition 最小探针(**本文件只做理论+预注册,不写实验代码**)
**日期**: 2026-09-04
**前置冻结**: TAN-I;4.1-C/D;4.2-A/B/C/D(ACCEPTED / FROZEN);Hard Binding 已建立(离散 WTA 极限)。
**状态**: 理论/预注册待 GO;本文件冻结三项:(1) Composition 的定义;(2) 最小组合机制;
(3) Binding/Composition 可辨识性审计。

---

## 0. 目标口径(用户锁定)

> 在已经具备 exact binding 的系统中,最小增加什么组织自由度,才能让两个独立的 K-V
> 绑定结果发生组合,而不是只完成单次检索?
> 第一版任务:y = V_A + V_B(加法最干净;不做乘法、不做训练);
> 第二任务(非交换):y = V_A − V_B(检验组合顺序)。
> 预期负对照链:M0 FAIL;M1 PASS_binding ∧ FAIL_composition;M2 PASS_composition
> ⇒ Hard Binding ⇏ Composition;Parallel Binding ⇏ Composition;
> Binding + minimal combination node ⇒ Composition。
> 纪律:每次只加一个自由度;不扩展成小型 Transformer;不写代码(本轮)。

---

## 1. 冻结定义:Composition 是什么(Definition of Composition)

1. **组合任务**:给定两个独立事件 A=(K_A,V_A)、B=(K_B,V_B) 与算子
   `f ∈ {+, −}`,目标输出 `y = f(V_A, V_B)`。
2. **三误差判据(可辨识性核心)**:同时测量
   `E_bind,A = |Ĉ_1 − V_A|`、`E_bind,B = |Ĉ_2 − V_B|`、`E_comp = |y − f(V_A, V_B)|`。
   **PASS_composition ⇔ 三者同时 ≈ 0 且置换反事实全部成立**。
   只报 E_comp 会允许"输出碰巧对、binding 根本没发生"的 AM5 型伪成功——三误差把
   组合与绑定拆开测量。
3. **分离定理(预注册为定理核验)**:组合器节点在给定正确绑定下**不引入任何误差**:
   `E_comp | (both channels routed correctly) = 0 精确`。因此 M2 的无条件组合误差
   `E[E_comp] = 绑定(路由)误差`,与组合器无关——"组合能力"与"路由保真度"是两个
   正交量(与 4.2-D 的 routing ⊥ discreteness 同构的双轴框架)。
4. **两层 Composition 语义(诚实分层)**:
   - **Parallel Binding(M1)**:输出 (V_A, V_B) 对——两个独立检索结果,**不是组合**;
   - **Composition(M2)**:在平行绑定之上增加一个组合器节点 C = f(Ĉ_1, Ĉ_2)。
   不得把 M1 的平行检索称作 composition。

## 2. 冻结机制:最小组合机制(Minimal Composition Mechanism)

### 2.1 双查询窗口(唯一结构变化;W=5 冻结不变)

- 双查询窗口(预注册):`[A=(K_A,V_A), 0, B=(K_B,V_B), Q_1, Q_2]`(槽 0..4;
  两事件 + 一 pad + 两个查询槽)。W=5 严格保持。
- **查询不是候选**:通道只在事件槽 {0,2} 上做选择(继承 4.2-C 纪律);
  **查询槽值 = 0**(查询是控制信号,不携带内容);pad = (0,0)。
- 共享窗口均值 `μ = mean(5 槽,含两个查询)`;通道 c 的惊奇
  `S_c = [Q_c − μ − ε]_+`(ε=0),向量查询 `c·S_c·û(ωS_c)`,ω=0.4(4.2-B 冻结),
  键嵌入 `φ̂_2`(4.2-B 冻结),能量 `E_i^c = c·S_c·û(ωS_c)^T φ̂(K_i)`。

### 2.2 查询值(预注册,方向瞄准判据)

遗产查询(3.0, 4.0)是单查询反事实对,不适用于双查询窗口。**新查询对**
`(Q_1, Q_2) = (4.5, 5.3)`,选择判据(运行前推导):
> 要求 û(ωS_1) 的方向瞄准 φ̂(K_A)(K_A=1 ⇒ 方向 π/4),û(ωS_2) 瞄准 φ̂(K_B)
> (K_B=2 ⇒ 方向 arctan2≈1.1071)。双查询窗口 μ=(1+0+2+4.5+5.3)/5=2.56:
> S_1=4.5−2.56=1.94,θ_1=0.776(≈44.5°≈π/4 ✓);S_2=5.3−2.56=2.74,
> θ_2=1.096(≈62.8°≈63.4° ✓)。
手算校验(正式运行闭式复核):通道 1 能量差 ΔE = 3.8797−3.6679 = **+0.2118**
(winner A);通道 2 能量差 ΔE = 5.4801−5.2164 = **+0.2637**(winner B)。
单查询对照沿用遗产(3.0 → A,4.0 → B,即 4.2-C/D 冻结结果);注:4.5 单独出现时
μ 不同(S=3.0,θ=1.2)而检索 B——这是窗口上下文性质,如实记录。

### 2.3 三模型(每次一个自由度)

| 模型 | 结构 | 输出 | 预注册预测 |
|---|---|---|---|
| **M0** Single-Winner Binding(4.2-D 延续) | 单通道,查询 = **最后一个 tap** Q_2(冻结核键定当前输入);event-WTA 单 winner | 单值 = 胜者值 | E_bind,B=0;E_bind,A=|V_B−V_A|>0;E_comp=|V_A|>0(对 +;**不能组合**) |
| **M1** Multi-Winner / Parallel Binding | 两独立通道(槽 3→通道 1,槽 4→通道 2),各自 event-WTA | **对** (Ĉ_1, Ĉ_2)(无组合器) | E_bind,A=E_bind,B=0(规范);E_comp 按"无组合器、第二通道读出"约定 = |V_A|>0(**结构性 FAIL:组合未定义**) |
| **M2** Minimal Composition Node | M1 + 一个两输入组合器节点 | `y = f(Ĉ_1, Ĉ_2)`,f=+/− | E_bind=0 且 E_comp=0(规范精确);集合:E_comp \| both-routed = 0 精确(分离定理),E[E_comp] = 绑定误差(>0) |

- M0 的"最后一个 tap"键定:4.2-D 冻结核用当前输入 x_t;双查询窗口中 x_t = Q_2,
  故 M0 的注意力 = 通道 2 的注意力(闭合推导)。
- M2 的加法节点 = 单节点双输入加法器(最小组合自由度;不是网络)。

## 3. 冻结审计:Binding/Composition Identifiability Audit(反事实套件)

### 3.1 规范场景(双窗口 [A,0,B,Q_1,Q_2];Q_1=4.5,Q_2=5.3,μ=2.56)

| 场景 | (V_A,V_B) | y(+) | y(−) | M0 输出 | M0 E_comp(+) | M1 对输出 |
|---|---|---|---|---|---|---|
| S1(遗产值) | (5, 9) | 14 | −4 | 9 | 5 | (5,9) |
| S2(用户示例) | (7, −2) | 5 | 9 | −2 | 7 | (7,−2) |
| S3(改自用户 (2,3)) | (2, 6) | 8 | −4 | 6 | 2 | (2,6) |

> S3 用 6 替代 3:遗产单查询值 3.0 与候选值冲突会破坏 Q ∉ V 的泄漏约束
> (query leakage 审计);记此适配。

### 3.2 反事实套件(全部预注册;正式运行逐条断言)

1. **Query swap**:(Q_1,Q_2) → (Q_2,Q_1):`+`:y 不变(可交换);`−`:y → −y
   (通道 1 检索 V_B、通道 2 检索 V_A ⇒ C = V_B − V_A,精确反对称)。
2. **Value swap**:(K_A,V_B),(K_B,V_A):`+`:y 不变(**这是可交换 f 的正确行为**——
   对加法,值互换不改变和;pairing 敏感性改由通道对输出与 key swap 检验);
   `−`:y → −y(变化 ✓)。理论澄清:用户 §7"输出必须随 pairing 改变"对加法表现为
   通道级互换 + 结果不变,对减法表现为结果变号;两者都是 binding-aware 的正确预测。
3. **Key swap**:(K_B,V_A),(K_A,V_B)(键位置互换、值配对保持):`+`:y 不变 ✓
   (用户要求);`−`:y → −y。
4. **Single-query controls**:遗产 (3.0,4.0) 各自单独(4.2-D 窗口):3.0 → V_A、
   4.0 → V_B(冻结结果复现)——证明组合失败不是 binding 损坏。
5. **Position randomization**:事件在槽 {0,1,2} 的 6 个有序位置;断言全部读出
   逐位不变(1e-12,核无位置项)。
6. **Amplitude randomization**(集合):K_A,K_B iid U(0.3,2.5)(4.2-B 协议);
   V_A,V_B iid U(−3,3);位置随机;断言 E_comp | both-routed = 0 精确、交换反对称
   逐历史精确、E[E_comp] = 绑定(路由)误差基准(求积冻结,待 GO 后计算)。
7. **Query leakage**:Q ∈ {3,4,4.5,5.3} ∉ 候选值集合(S1/S2/S3 与 V∈[−3,3] 均满足);
   查询槽值 = 0、非候选(断言);查询不出现在任何输出表达式中。
8. **AM5 型陷阱防护(degenerate check)**:令 V_A = 0:M0 的 E_comp(+) = |V_B − V_B| = 0
   "伪成功",但 E_bind,A = |V_B| > 0 被三误差判据捕获——断言并报告为可辨识性审计的
   验证案例(不冒充组合成功)。

### 3.3 审计状态表(预注册)

| 判据 | M0 | M1 | M2 |
|---|---|---|---|
| 单绑定(遗产对照) | PASS(=4.2-D) | PASS | PASS |
| 平行绑定(E_bind,A=E_bind,B=0,规范) | FAIL(E_bind,A>0) | **PASS** | PASS |
| 组合(E_comp≈0 ∧ 反事实全过) | **FAIL** | **FAIL(结构性:无组合器)** | **PASS(规范精确;集合条件精确)** |
| 分离定理(E_comp \| both-routed = 0) | — | — | 断言(定理核验) |

⇒ 负对照链:H**Hard Binding ⇏ Composition**;**Parallel Binding ⇏ Composition**;
**Binding + minimal combination node ⇒ Composition(组合误差 = 绑定误差,组合器零误差)**。

## 4. 声称纪律与停止规则

- 可声称:上表负对照链、分离定理、交换/非交换组合的精确行为、"组合器不引入误差"。
- 不可声称:M1 是 composition;任何"general compositional intelligence / 语言语义组合";
  M2 的无条件集合组合误差(那是绑定误差);任何 benchmark。
- 停止规则:若规范断言(手算预测)不符 → STOP AND DEBUG;若 M2 的
  E_comp | both-routed ≠ 0 → 分离定理破缺,停止并报告;不因结果改查询值/键/值分布。

## 5. GO 后计划输出(本轮不创建)

- `code/experiments/sprint4_3/sprint4_3a_composition_probe.py`
  (M0/M1/M2、规范场景 S1–S3、±算子、反事实套件、集合 MC + 求积基准、
  可辨识性审计、图)
- `results/sprint4_3a/`(summary JSON、checks CSV、图)
- `docs/TANII_SPRINT4_3A_{RESULTS,CLAIM_LEDGER}.md`

*理论/预注册完成(3 项已冻结)。等待 GO;本阶段未写任何实验代码。*
