# TAN-II Sprint 4.2-D — 预注册(Hard Binding minimal probe:winner-take-all)

**项目**: TAN-II:组织机制与选择性计算的涌现
**阶段**: Sprint 4.2-D —— Hard Binding 最小探针(候选机制:winner-take-all / competitive
normalization;不升维、不做 Level-3、无学习、无新递归拓扑)
**日期**: 2026-09-04
**前置**: 4.2-A/B/C 已冻结;用户裁决:下一刀 = Hard Binding minimal probe。

## 0. 目标口径(用户锁定)

> What is the minimal additional organizational freedom required for exact value
> delivery?(soft binding → hard binding? 连续加权 → 离散选择)
> 候选机制优先 winner-take-all / competitive normalization;d>2 已被排除。
> 最小机制链预期:Vector Q-K → Routing;K-V separation → Value transfer;
> WTA → Exact selection。

## 1. 唯一新增组织自由度:读出选择函数

在 4.2-C 的 (K,V) 解耦模型上,**只**改变读出游出的选择函数:
`α_i(γ) = softmax(γ·logit_i)`,γ ∈ **{1, 2, 5, 10, 100, ∞}**;
γ=∞ = 一热(argmax)WTA。γ=1 精确等于 4.2-C 冻结模型(连续性校验)。
其余全部冻结(窗口/惊奇/键/查询/E-I/膜/值/探针值=0)。
两个 WTA 候选集定义(预注册):
- **event-WTA(主,记忆选择)**:选择只在事件槽 {A,B} 上进行(pad/探针零权重);
  `C_events(γ) = (α_A v_A + α_B v_B)/(α_A+α_B)`;γ=∞ 时按事件 logit 的 argmax 取一热极限。
- **full-WTA(对照)**:全 5 槽选择;`C_full(γ) = Σ_i α_i(γ) v_i`(pad/探针值=0);
  γ=∞ 时全窗口 argmax。
- 软度:`D_events(γ,Q) = |C_events − v_winner_event|`;`D_full(γ,Q) = |C_full − v_winner_event|`;
  winner 事件在 γ 下不变(断言)。
- 传递:`T_events(γ) = C_events(γ,3) − C_events(γ,4)`;`T_full(γ)` 同式;交换反对称
  `T_swap(γ) = −T(γ)` 在每个 γ 断言(1e-12)。
- M5-E 驱动:`T_drive(γ) = A_E(γ,3) − A_E(γ,4)`,
  `A_E(γ,Q) = tanh(S_E)·C_fullE(γ,Q) − 0.8·tanh(S_I)·C_fullI(γ,Q)`。

## 2. 冻结预测(运行前闭式计算;γ 阶梯全部命中)

### M2 规范(值 5/9;winner 恒 A@Q=3、B@Q=4)

| γ | C_full(3) | C_full(4) | D_ev(3) | D_ev(4) | D_full(3) | D_full(4) | T_ev | T_full |
|---|---|---|---|---|---|---|---|---|
| 1 | 4.890550 | 4.943193 | 1.742691 | 1.844405 | 0.109450 | 4.056807 | −0.412904 | −0.052643 |
| 2 | 5.242303 | 5.267514 | 1.493762 | 1.690682 | 0.242303 | 3.732486 | −0.815556 | −0.025211 |
| 5 | 5.479412 | 6.178173 | 0.860900 | 1.257667 | 0.479412 | 2.821827 | −1.881433 | −0.698760 |
| 10 | 5.241441 | 7.463916 | 0.279808 | 0.695103 | 0.241441 | 1.536084 | −3.025090 | −2.222475 |
| 100 | 5.000000 | 8.999999 | ≈0(>0) | ≈0(>0) | ≈0(>0) | ≈0(>0) | −3.999999 | −3.999999 |
| ∞ | **5.000000** | **9.000000** | **0** | **0** | **0** | **0** | **−4** | **−4** |

- D_ev、D_full 随 γ **单调不增**;**有限 γ 恒 D>0**(softmax 严格正性)——软绑定;
  **D=0 只在离散极限 γ=∞**——exact value delivery 需要一热 WTA。
- 交换反对称在每个 γ 精确成立(T_swap(γ) = −T(γ))。

### M5-E 驱动阶梯(方向反转→恢复的交叉)

| γ | 1 | 2 | 5 | 10 | 100 | ∞ |
|---|---|---|---|---|---|---|
| T_drive | +0.236700 | +0.251584 | +0.015700 | −1.054093 | −3.803430 | **−3.867424** |

- **交叉点 γ\* = 5.098014585**(二分冻结);γ<γ\* 方向反转(4.2-C),γ>γ\* 方向恢复
  (抑制不再二次读出递增的值:I 分支 winner 恒 A、值恒 5)。
- 硬极限:A_E(∞,3) = 1.113437,A_E(∞,4) = 4.980861。

### 硬极限对照(event-WTA / full-WTA)

| 模型 | event-WTA C_events(∞) | full-WTA C_full(∞) |
|---|---|---|
| M2 | (5, 9)(exact) | (5, 9)(exact) |
| M0 | (9, 9)(winner 恒 B) | **(0, 0)(全窗口 winner = 探针,值 0)** |
| M1(±) | (9, 5)(exact,退化) | (0, 0)(s=+1 探针赢;s=−1 pad 赢) |
| M2unnorm | (9, 9)(winner 恒 B) | (0, 0)(探针赢) |

> 无路由模型在 hard WTA 下**交付不了任何值**——hard selection 不是修复路由缺失的捷径;
> 这是"routing selectivity ⊥ readout discreteness"正交框架的实验证据。

### 集合(γ 阶梯求积,冻结)

- flip rate(事件 winner)在**所有 γ 下不变 = 0.444527**(argmax 不变性,定理+求积断言)。
- E[|T_full|](γ):γ=1: 0.586346;2: 1.042547;5: 1.815325;10: 2.406862;
  100: 2.990795;**∞: 3.019684**(full-WTA 下 |v_winner3 − v_winner4| ∈ {0,4,5,9} 的混合)。
- MC 验证(3 种子 × 10⁴):γ=100 与 γ=∞:E[|T_full|] 对基准 ≤ 3·SE_mean+0.01;
  γ=100 flip rate 对 0.444527 ≤ 3·SE+0.002。

## 3. 声称纪律

- 可声称:"WTA(一热离散选择)⇒ 精确值传递(在离散极限,γ=∞)";"有限竞争归一化只近似,
  永不正合"(定理:softmax 严格正性)。
- 可声称:γ\* 交叉 = E-I 值传递通路被 WTA 重塑的机制证据(方向反转→恢复)。
- 不可声称:hard binding 在有限机制参数下实现;任何 benchmark/泛化。

## 4. 输出

`code/experiments/sprint4_2/sprint4_2_hard_binding.py`;
`results/sprint4_2/sprint4_2d_*`;`docs/TANII_SPRINT4_2D_{PREREGISTRATION,RESULTS,CLAIM_LEDGER}.md`。
判据:确定性量 1e-9(反对称 1e-12);求积复算 1e-6;MC 概率 3SE+0.002、均值 3SE+0.01。

*预注册完成;冻结值来自 §2(先于正式 MC 运行)。*
