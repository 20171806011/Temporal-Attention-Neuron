# TAN-II Sprint 4.2-A — 主张账本(Claim Ledger)

**阶段**: Sprint 4.2-A(纯查询–键几何)
**日期**: 2026-09-04
**等级定义(本 Sprint)**:ESTABLISHED / SUPPORTED / UNKNOWN / REJECTED。
**依据文件**:`docs/TANII_SPRINT4_2_THEORY.md`(解析)、
`docs/TANII_SPRINT4_2_PREREGISTRATION.md`(协议)、
`docs/TANII_SPRINT4_2_AMENDMENT_1.md`(合规修正)、
`docs/TANII_SPRINT4_2_RESULTS.md`(数据)、
`results/sprint4_2/*.json / *.csv`(机读)。

---

## 1. 核心三命题(§20 的 Claim A / B / C)

| 命题 | 等级 | 依据与限定 |
|---|---|---|
| **A. 正标量查询 ⇒ 查询不变候选排序** | **ESTABLISHED** | Th1 + TAN 查询域推导(γ_t = βW_qW_k[x_t−μ_t−ε]_+ ≥ 0);d=1 Case A 确定性 0/0/1 精确命中;Case B/C 给出翻转恰证明命题的域条件性 |
| **B. d=2 在点积 QK 族中产生非零查询条件化排序容量** | **ESTABLISHED(限 Level-1 geometric capacity)** | 解析 `FlipRate_CF(θ)=θ/π`(证明见 Theory Audit §4.3)+ 根分解精确验证 + MC 316 项一致(d∈{2,3,4,8},θ 全网格,3 种子);`P[w=A]=1/2`、排序容量阶梯同证。限定:这是**容量**,不是实现,更不是泛化 |
| **C. d=2 ⇒ content-addressable routing** | **REJECTED(作为强命题,默认不接受)** | 本 Sprint 只建立 Level-1;Level-2(realized)与 Level-3(generalized)必须由 4.2-B 与独立 Probe 另行建立。不得由 B 外推 C |

## 2. 结构性与方法论命题

| 命题 | 等级 | 依据 |
|---|---|---|
| `FlipRate_CF(θ) = θ/π` 且**维度无关**(d≥2 同曲线) | **ESTABLISHED** | 解析(投影/均匀角)+ 根分解精确 + MC(d=2/3/4/8 同曲线,132+36 项 PASS) |
| `F_d = 1/2` 对 d≥2 **恒定**(负对照;不支持"维度越高 routing 越强") | **ESTABLISHED** | Th4 + MC:0.4996/0.4997/0.5033/0.4998(d=2/3/4/8) |
| `P_Q[w=A] = P_Q[w=B] = 1/2` **与候选范数无关**(2 候选点积) | **ESTABLISHED** | Th3(边界过原点)+ E1 全部候选对 + 5a(r∈{1,2,5,10}) |
| 带符号标量查询(d=1)可翻转 winner,但只有退化端点 {0,1}(S⁰) | **ESTABLISHED** | Th2 + Case C 精确 {0,1} + Case B ≈ 0.5 |
| 排序容量阶梯 1 / 2 / M(M−1) / M!(d = 1pos / 1sgn / 2 / ≥3, M≤d+1) | **ESTABLISHED** | 精确布置枚举 {2,6,12} + LP 可行集精确匹配 + 稠密 MC 200k 角度一致 + d=1 LP {1} / {2} |
| "S⁰→S¹ 拓扑/几何自由度改变"是 d=2 的恰当刻画 | **ESTABLISHED** | 翻转率取值集 {0,1} vs 连续 [0,1](=θ/π);排序容量 2 vs M(M−1)/M! |
| 单独 `JSD>0` 不足以宣称"查询条件化再分配" | **ESTABLISHED** | 正标量陷阱 JSD=0.0751 且 flip=0;集总对照 P[JSD>1e-9]≈1 vs P[flip]=θ/π(0.5);合取判据成立 |
| E-I 拓扑不改变该路由边界(边界是核级的,不是动力学级的) | **ESTABLISHED** | 结构性论证(排序发生在 E-I 之前)+ 4.1-C/D 实测 FlipRate=0 |
| distractor 范数泄漏(≥3 候选)存在,且"等范数"≠布置对称 | **ESTABLISHED** | 5b 精确曲线 0.091→0.476(m=0.5→10)与 MC 一致;m=1 非对称布置 P[D]=0.125≠1/3;对称布置 5c = 1/3 |
| "d_min = 2"(无定语,指最小翻转维度) | **REJECTED** | 带符号标量(d=1)已能翻转;翻转率由查询夹角而非维度控制。替换为 (a)(b)(c) 三条限定表述(见 Results §2 问题 10);符号写作 **d\* = 2** |
| "1-D attention cannot route"(一般命题) | **REJECTED** | 在 Case B/C 下为假;正确表述是域条件形式(Claim A) |
| "d=2 相变"(严格相变意义) | **REJECTED(不作正式 theorem)** | 无数值/极限行为与序参量定义;仅直观描述"最小整数维几何跃迁" |
| 三种候选物理实现(树突隔室 / 相位耦合 / DA-5HT 平面)≡ R² | **UNKNOWN(不作宣称)** | 仅为 future hypotheses;本 Sprint 未检验 |
| d=2 在完整 TAN(4.2-B)中是否实现 Level-2 routing | **UNKNOWN** | 4.2-B 未执行(待 GO);Amendment 1 已冻结为前置条件 |
| E4(iii) 预注册量级预测 0.00103 | **REJECTED(算术错误)** | 正确闭式值 0.00045439;合规处置见 Amendment 1(PROTOCOL-FAIL / NUMERICAL-PASS;定性主张不受影响) |

## 3. 协议合规记录

- 预注册七条修正:全部落实(结果文档 §0–§1 逐条对应)。
- E4(iii):原始预注册预测无效(算术错误);**不覆盖**原始文件,追加 `docs/TANII_SPRINT4_2_AMENDMENT_1.md`;
  后续输出标记 PROTOCOL-FAIL / NUMERICAL-PASS;理论审计脚本保持原样(exit 1 为原始记录形态)。
- 停止规则:本 Sprint 无真实 FAIL(316 PASS + 1 PROTOCOL + 0 FAIL;理论审计 43/44 + 1 记录);
  未发生任何"为通过而调参/改种子/改阈值/改采样"的操作。

## 4. 最终一页结论(口径锁定)

> **4.2-A 建立了(Level-1)查询条件化排序几何的精确标尺**:在 TAN 继承的正标量查询族内
> 排序查询不变(FlipRate=0);带符号标量只有退化端点 {0,1};d≥2 的翻转率 = 查询夹角/π、
> 维度无关、随夹角连续;排序容量阶梯 1/2/M(M−1)/M! 精确成立;两候选点积无范数捷径,
> 但多候选 distractor 范数泄漏存在且等范数≠布置对称。因此 **d=2 是 TAN 兼容正标量族之外,
> 产生连续查询方向自由度的最小整数维度(d\* = 2,带限定词)**;任何"路由"或"绑定"层级的
> 宣称均不属于本 Sprint 的证据范围。
