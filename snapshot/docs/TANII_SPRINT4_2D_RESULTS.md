# TAN-II Sprint 4.2-D — 结果(Hard Binding minimal probe:winner-take-all)

**项目**: TAN-II:组织机制与选择性计算的涌现
**阶段**: Sprint 4.2-D —— Hard Binding 最小探针(WTA / competitive normalization)
**日期**: 2026-09-04
**预注册**: `docs/TANII_SPRINT4_2D_PREREGISTRATION.md`(保持原样)
**Amendment 5**: 运行前解析修正(D_full 单调性),已冻结
**输出**: `results/sprint4_2/sprint4_2d_*`
**程序**: `code/experiments/sprint4_2/sprint4_2_hard_binding.py`

## 0. 状态块

```
TAN-II SPRINT 4.2-D STATUS: COMPLETE
Checks:                   77/77 PASS (0 FAIL)
Minimal question answered: "What is the minimal additional organizational
                           freedom for exact value delivery?"
  -> one-hot winner-take-all (the DISCRETE limit gamma=inf): D = 0 exactly, T = -4
  -> every FINITE sharpening gain: D > 0 (softmax strict positivity) - soft
gamma-ladder:             gamma = 1(reproduces 4.2-C) / 2 / 5 / 10 / 100 / inf
D_events(3,4) at gamma=1:  1.7427 / 1.8444  ->  D(inf) = 0 / 0 (exact)
D_full(Q=3) overshoot:     crosses V_A at gamma = 1.19776 (cancellation artifact, AM5)
M5-E direction crossing:   gamma* = 5.09801 (T_drive: +0.2367 -> -3.8674)
Ensemble:                  flip rate gamma-invariant 0.444527;
                           E|T_full|: 0.586 -> 1.043 -> 1.815 -> 2.407 -> 2.991 (inf: 3.0197)
Amendments: 1 (AM5, pre-run) | Post-hoc tuning: 0 | Frozen archives modified: 0
```

## 1. 唯一变化与协议

- 在 4.2-C 冻结模型上**只**改变读出选择函数:`α_i(γ) = softmax(γ·logit_i)`,
  γ ∈ {1,2,5,10,100,∞(一热 argmax WTA)};γ=1 精确复现 4.2-C(连续性校验通过)。
- 两个候选集定义:event-WTA(记忆选择,主)/ full-WTA(全 5 槽对照;pad/探针值 = 0)。
- 铁门延续:其余全部冻结;无学习、无新递归拓扑、无升维。

## 2. 规范 γ 阶梯(确定性,全部命中冻结预测)

| γ | C_full(3)/(4) | D_events(3)/(4) | D_full(3)/(4) | T_events | T_full | winner |
|---|---|---|---|---|---|---|
| 1 (=4.2-C) | 4.8906/4.9432 | 1.7427/1.8444 | 0.1095/4.0568 | −0.4129 | −0.0526 | A,B |
| 2 | 5.2423/5.2675 | 1.4938/1.6907 | 0.2423/3.7325 | −0.8156 | −0.0252 | A,B |
| 5 | 5.4794/6.1782 | 0.8609/1.2577 | 0.4794/2.8218 | −1.8814 | −0.6988 | A,B |
| 10 | 5.2414/7.4639 | 0.2798/0.6951 | 0.2414/1.5361 | −3.0251 | −2.2225 | A,B |
| 100 | 5.0000/9.0000 | ≈0(>0) | ≈0(>0) | −4.0000 | −4.0000 | A,B |
| **∞** | **5/9** | **0/0** | **0/0** | **−4** | **−4** | A,B |

- **D_events 单调非增、有限 γ 恒 >0**(softmax 严格正性)⇒ 软绑定;
- **D = 0 只在离散极限 γ=∞** ⇒ exact value delivery 的最小组织自由度 = **一热 WTA**;
- **Amendment 5 的 D_full(Q=3) 过冲**:C_full(3,γ) 在 γ = 1.19776 穿过 V_A=5
  (D_full 瞬时 = 0)——这是"B 残值 + 丢失质量"的**相消伪精确**,此时 D_events ≈ 1.5
  仍 >0;结构硬绑定只由 D_events 与极限定义(如实报告,不冒充硬绑定)。
- 交换反对称 `T_swap(γ) = −T(γ)` 在**每个 γ 精确成立**(1e-12);winner (A,B) 全阶梯不变。

## 3. M5-E:WTA 重塑 E-I 值传递通路(方向反转 → 恢复)

| γ | 1 | 2 | 5 | 10 | 100 | ∞ |
|---|---|---|---|---|---|---|
| T_drive | +0.2367 | +0.2516 | +0.0157 | −1.0541 | −3.8034 | **−3.8674** |

- **交叉点 γ\* = 5.098014585**(二分,命中冻结值):γ<γ\* 时 E-I 电路的抑制分支
  二次读出递增的值、净传递方向反转(4.2-C 现象);γ>γ\* 时 I 分支被钉在其自身
  winner(A,值恒 5)上、不再递增,E 支的值传递(5→9)重新主导——**方向恢复**。
- 硬极限:A_E(∞,3) = 1.113437,A_E(∞,4) = 4.980861。
- 结论口径:E-I 不实现绑定,但**重塑绑定后的值传递通路**("E-I organization can
  reshape the effective value-transfer pathway after query-conditioned
  routing")。

## 4. 硬极限对照(无路由模型在 hard WTA 下交付不了值)

| 模型 | event-WTA C_events(∞) | full-WTA C_full(∞) |
|---|---|---|
| M2 | (5, 9) 精确 | (5, 9) 精确 |
| M0 | (9, 9)(winner 恒 B) | (0, 0)(探针赢,值 0) |
| M1(±) | (9, 5) 精确(退化) | (0, 0)(+1 探针赢;−1 pad 赢) |
| M2unnorm | (9, 9) | (0, 0)(探针赢) |

> hard selection 不是修复路由缺失的捷径:**routing selectivity ⊥ readout
> discreteness** 是两个正交自由度(用户提出的 conceptual framework 的量化证据)。

## 5. 集合(求积 + MC,3 种子 × 10⁴)

- flip rate(事件 winner)= 0.444527,**对全部 γ 不变**(argmax 不变性,定理+求积断言)。
- E[|T_full|](γ):γ=1: 0.5863;2: 1.0425;5: 1.8153;10: 2.4069;100: 2.9908;∞: 3.0197。
- MC 验证:γ=100 → 2.9607/2.9709/2.9705(vs 2.9908);γ=∞ → 2.9932/3.0019/3.0043
  (vs 3.0197);全部 ≤ 3·SE_mean+0.01;γ=100 flip rate 全部对 0.444527 通过。

## 6. 对总问题的回答

> What is the minimal additional organizational freedom required for exact
> value delivery?
**一热 winner-take-all(离散选择)**。理由链(全部冻结证据):
1. 任意有限竞争归一化(sharpening)只近似、永不正合(softmax 严格正性 ⇒ D>0);
2. D=0 恰好只在 γ=∞(argmax 一热)实现,且此时 winner/翻转率/交换反对称全部保持
   (WTA 只改变读出离散度,不改变路由);
3. WTA 同时把 E-I 电路的传递方向在 γ\*≈5.10 处反转回正(通路重塑的机制证据)。
4. 对照:无路由模型(M0/M1/unnorm)在 hard WTA 下交付 0(选择没有可交付的对象)。

## 7. 研究链更新(用户口径的完整阶梯)

```
Memory → Saliency → Geometry → Competition ⇏ Content Routing
→ Vector Q-K ⇒ Realized Routing (4.2-B)
→ K-V Decoupling ⇒ Soft/Partial Binding (4.2-C)
→ One-hot WTA ⇒ Exact Value Delivery (4.2-D, in the discrete limit)
正交轴: routing selectivity ⊥ readout discreteness
下一刀候选: Binding → Composition(未进入实验范围,待 GO)
```

## 8. 不宣称(No-claims)

- 不宣称有限机制参数下实现 hard binding(只在离散极限);
- 不把 γ=1.198 的相消零点冒充硬绑定(Amendment 5);
- 不宣称 E-I 实现绑定(只重塑传递通路);
- 不宣称 Composition / Level-3 泛化。

## 9. 复现

```
python code/experiments/sprint4_2/sprint4_2_hard_binding.py   # 77 checks, exit 0
```
文件:`results/sprint4_2/sprint4_2d_{summary.json, checks.csv, ladder.png}`。
环境:Python 3.12.7 / numpy 1.26.4 / matplotlib 3.9.2;确定性。
