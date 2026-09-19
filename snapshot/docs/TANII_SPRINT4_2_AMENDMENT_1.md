# TAN-II Sprint 4.2-A — Amendment 1(E4(iii) 预注册预测错误的处置)

**项目**: TAN-II Sprint 4.2-A(纯查询–键几何)
**Amendment 编号**: 1
**日期**: 2026-09-04(理论审计阶段,MC 阶段开始前冻结)
**关联文件**:
- 原始预注册:`docs/TANII_SPRINT4_2_PREREGISTRATION.md` §2.3(E4)与 §2.5(预测 4)——**保持原样,不被覆盖**
- 理论审计:`code/experiments/sprint4_2/sprint4_2_theory_audit.py`——**保持原样**(43/44,1 FAIL,exit 1)
- 本 Amendment:**追加记录,不是覆盖**

---

## 1. 原始预注册记录(verbatim,永久保留)

`docs/TANII_SPRINT4_2_PREREGISTRATION.md` §2.3:

> (iii) d=2 不翻转但 JSD>0:同 H,Q_B=(cos π/6, sin π/6):JSD ≈ 0.00103(断言 > 5e-4)
> **且** flip = 0(断言)——证明"单独 JSD 不足"在任何族都成立,合取才是判据。

§2.5 预测 4:E4:…(iii) JSD>0 且 flip=0(d=2, θ=π/6)…

## 2. 理论审计实际发现(原始记录,保留)

`sprint4_2_theory_audit.py` 第 44 项检查:

```
[FAIL] E4(iii): d=2 theta=pi/6 JSD > 5e-4   (JSD=0.000454)
```

- 实现计算值(标准 Bernoulli JSD 闭式):**0.00045439…**
- 独立复算(会话内第二实现,σ(x)=1/(1+e^{−x}),KL 直接定义):0.00045439 ✓ 一致。
- 预注册写下的预测值 **0.00103 是手算错误**(约高估 2.27 倍)。
- **这不是实现 bug,也不是定理错误**;实现与正确闭式值完全一致。

## 3. 独立闭式推导(修正基准,冻结)

两候选软注意力:`α_A(Q) = σ(Q^TΔK)`;固定历史 `ΔK = e_1`,`Q_A = e_1`,
`Q_B = (cos θ, sin θ)`。则:
`p = α_A(Q_A) = σ(1) = 0.7310585786…`,
`q = α_A(Q_B) = σ(cos π/6) = σ(√3/2) = 0.7039179927…`。
Bernoulli JSD(自然对数):
`JSD(p,q) = ½ [ p ln(p/m) + (1−p) ln((1−p)/(1−m)) + q ln(q/m) + (1−q) ln((1−q)/(1−m)) ]`,
`m = (p+q)/2`。
数值(独立两路径一致):
`JSD = 0.00045439…`(≈ 4.544e-4)。flip 指标:
`sign(Q_A^TΔK) = +1`,`sign(Q_B^TΔK) = +1` ⇒ **flip = 0** ✓。

**修正后的科学事实**:
1. `JSD > 0`(4.544e-4)且 `FlipRate = 0` —— E4(iii) 的**定性机制主张完全成立**;
2. 原预注册的**量级预测 0.00103 无效**(算术错误);
3. 实现结果对**修正闭式基准**的偏差 < 1e-12(解析一致)。

## 4. 裁决与重新分类(按用户最终裁决)

| 层级 | 结果 |
|---|---|
| 原预注册理论预测(0.00103) | ❌ 错误(算术错误) |
| 原预注册文字记录 | ❌ 不得事后覆盖(保留原文) |
| 正确闭式理论值 | ✅ 0.00045439… |
| 实现数值 vs 正确闭式值 | ✅ PASS(< 1e-12) |
| 阈值 5×10⁻⁴(作为"实现 vs 修正基准"的容差) | ✅ 满足 |
| E4(iii) 历史合规性 | ❌ FAIL(原始断言未达) |
| **E4(iii) 最终标记** | **PROTOCOL-FAIL / NUMERICAL-PASS** |
| 4.2-A 科学结论 | **不受影响** |

## 5. 处置指令(用户裁决,逐条执行)

- [x] 不修改预注册阈值;不覆盖原始预测;原始预注册与理论审计文件保持原样。
- [x] 追加本 Amendment;独立推导并验证精确闭式值 0.00045439…。
- [x] E4(iii) 在所有后续输出中标记 **PROTOCOL-FAIL / NUMERICAL-PASS**,
      并保留原始 FAIL 记录(43/44 + 1 公开 amendment,不改写为 44/44)。
- [x] 不因该问题重跑实验、不改代码、不换种子、不调阈值、不改变停止规则。
- [x] 定性机制主张(JSD>0 ∧ FlipRate=0,即"查询改变注意力分布而不改变 winner")
      独立于 E4(iii) 报告。
- [ ] 4.2-B 开始前,本 Amendment 与修正推导必须已冻结(现在即冻结)。

## 6. 对后续程序输出的要求

后续 `sprint4_2_geometry.py` / 结果文档中,E4(iii) 记录为:
- `status = "PROTOCOL-FAIL / NUMERICAL-PASS"`
- `jsd_exact = 0.00045439…`;`corrected_benchmark = 0.00045439…`;`agreement = < 1e-12`
- `original_prereg_prediction = 0.00103(INVALID, arithmetic error)`
- `qualitative_claim = "JSD > 0 with FlipRate = 0": PASS`
- 引用本 Amendment 文件。

*Amendment 1 冻结。原始记录不被覆盖;本项目继续(不停止)。*
