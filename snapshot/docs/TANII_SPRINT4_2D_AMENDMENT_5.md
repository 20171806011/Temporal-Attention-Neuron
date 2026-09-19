# TAN-II Sprint 4.2-D — Amendment 5(运行前预测修正:D_full 单调性)

**项目**: TAN-II Sprint 4.2-D(Hard Binding / WTA minimal probe)
**Amendment 编号**: 5
**日期**: 2026-09-04(确定性闭式复核阶段,正式 MC 集合数据收集前)
**关联**: `docs/TANII_SPRINT4_2D_PREREGISTRATION.md` §2(保持原样,不被覆盖)

---

## 1. 原始预注册记录(verbatim,永久保留)

> D_ev、D_full 随 γ **单调不增**;**有限 γ 恒 D>0**(softmax 严格正性)——软绑定;
> **D=0 只在离散极限 γ=∞**——exact value delivery 需要一热 WTA。

## 2. 运行前复核发现:该表述对 D_full 不成立

- **D_events 单调非增且有限 γ 恒 >0:正确**(ᾱ_A = σ(γ·ld) 单调,软度 = 4·min(ᾱ,1−ᾱ) → 0⁺)。
- **D_full(Q=3) 非单调**:C_full(3,γ) = 5(α_A+α_B) + 4α_B:γ 增大时 A 份额上升、
  B 份额下降,全窗口读出游出从下方(γ=1:4.8906)穿过 V_A=5(γ_cross3 ≈ 1.197762904)
  过冲到上方(γ=2:5.2423;γ=5:5.4794),再回落(γ=10:5.2414)→ 5。
  即 D_full(3) 在有限 γ ≈ 1.198 处**恰好等于 0**。
- 该零点**不是结构硬绑定**:此时 D_events(3) ≈ 1.5 > 0(注意力仍软),C_full = 5 是
  "B 残值(9)与丢失质量(pad/探针)"的**相消伪精确**(cancellation artifact)。
  结构硬绑定只由 D_events 与极限行为定义(γ=∞ 时 winner 质量 → 1,C_full → 5 且
  D_events → 0)。
- D_full(Q=4) 单调递减(4.0568 → 3.7325 → 2.8218 → 1.5361 → 0),无误判。

## 3. 处置

- 原始文本保留;本 Amendment 取代该句的 D_full 部分:
  **D_events 单调非增、有限 γ 恒 >0;D_full(Q=3) 呈"过冲-回落"并在 γ≈1.1978 穿过 0
  (相消伪精确,如实报告);D_full(Q=4) 单调;两者在 γ=∞ 均 → 0(真正的硬绑定极限)。**
- 程序检查改为:断言 D_events 单调;断言 D_full 的具体非单调形状;报告 γ_cross3。
- 标签:`PREDICTION-CORRECTED-PRERUN`。

*Amendment 5 冻结(先于正式 MC 集合数据)。*
