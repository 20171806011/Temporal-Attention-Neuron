# TAN-II Sprint 4.3-A — 结果(Minimal Composition Probe)

**项目**: TAN-II:组织机制与选择性计算的涌现
**阶段**: Sprint 4.3-A —— Binding → Composition 最小探针(UNCONDITIONAL GO 执行)
**日期**: 2026-09-04
**预注册**: `docs/TANII_SPRINT4_3A_PREREGISTRATION.md`(最高实验规范,保持原样)
**Amendment 6**: 运行期边距基准 DEBUG 记录(手算→精确闭式),已冻结
**程序**: `code/experiments/sprint4_3/sprint4_3a_composition_probe.py`
**输出**: `results/sprint4_3/`

## 0. 状态块(§21 格式)

```
TAN-II SPRINT 4.3-A STATUS
Theory / Preregistration: FROZEN
Canonical:    S1: PASS | S2: PASS | S3: PASS
Binding:      M0: FAIL(bindA>0) | M1: PASS | M2: PASS
Composition:  M0: FAIL | M1: STRUCTURALLY_UNDEFINED (projection control FAIL)
              M2: PASS (canonical exact 0; E|both-routed = 0 exact)
Counterfactual: 1: PASS | 2: PASS | 3: PASS | 4: PASS | 5: PASS | 6: PASS
              7: PASS | 8: PASS
AM5 trap:     PASS (M0 E_comp=0 but E_bind,A>0 -> joint audit catches)
Monte Carlo:  N = 10,000 x 3 seeds
              P(both-routed) = 0.2207 (MC 0.22xx)
              E[E_comp+] = 1.1161 (MC 1.07-1.13)  E[E_comp-] = 2.0010 (MC 1.96-1.99)
              E[E_comp | both-routed] = 0 EXACT (max over 3x10^4 histories = 0.0e0)
Shortcut Audit: PASS
Post-hoc tuning: 0
Frozen archives modified: 0
```

## 1. Step 1:规范(确定性,79 项检查全过)

- **几何**:规范双查询窗口 μ=2.56(精确);边距闭式值 m_A=+0.210631046、
  m_B=+0.261878523(Amendment 6:预注册手算值 0.2118/0.2637 为 4 位手算估计,
  偏差 0.00117/0.00184 归因于手算三角取整;winner A/B 与全部下游行为不受影响,
  原文保留、偏差记录)。
- **三误差联合审计**(规范 S1/S2/S3 × {+,−}):

| 模型 | E_bind,A | E_bind,B | E_comp(+) | E_comp(−) | 判定 |
|---|---|---|---|---|---|
| M0 | \|V_B−V_A\|>0 | 0 | \|V_A\|>0 | \|2V_B−V_A\|>0 | **Composition FAIL**(单 winner 只交付一个操作数) |
| M1 | 0 | 0 | \|V_A\|>0 | \|2V_B−V_A\|>0 | **STRUCTURALLY_UNDEFINED**(无组合器;冻结的标量投影对照 = 第二通道读出,FAIL) |
| M2 | 0 | 0 | **0** | **0** | **Composition PASS(精确)** |

- M2 规范输出精确:y(+)=14/5/8,y(−)=−4/9/−4(S1/S2/S3);
- 位置随机化:6 个有序位置读出逐位不变(maxdiff < 1e-12);
- 泄漏:Q ∩ V = ∅;查询槽值=0、非候选(结构性)。

## 2. Step 2:8 项反事实(全部 PASS,逐项状态)

1. **Query Swap**:+ 不变、− 精确变号(通道 1→V_B、通道 2→V_A)。PASS
2. **Value Swap**:+ 不变(可交换 f 的正确行为)且通道对输出互换 (9,5);− 变号。PASS
3. **Key Swap**(值配对保持):+ 不变;− 变号。PASS
4. **Channel-output Swap**(组合器输入序):+ 不变;− 变号。PASS
5. **AM5 零值陷阱**:V_A=0:M0 E_comp=0 但 E_bind,A=5>0 → **联合审计捕获,判定
   Composition FAIL**;M2 E_comp=0 且 E_bind,A=0 → 真实 PASS。PASS
6. **Binding-preserving**(值平移 (5,9)→(7,11)+ 遗产单查询 3.0→A / 4.0→B):
   三误差保持 0、y=18 精确。PASS
7. **Binding-breaking**(双查询 (3.0,4.0)):通道 2 打偏 → E_bind,B=4 且 E_comp=4:
   **Routing → Wrong Operand → Composition Error** 因果链被实证。PASS
8. **Composition-sensitive relational**:y 跨场景精确等于 f 且减法区分关系
   (−4 / 9 / −4)。PASS

## 3. Step 3:集合 Monte Carlo(求积基准 vs MC,3 种子 × 10⁴)

| 量 | 求积基准 | MC(三种子) |
|---|---|---|
| P(A routed) | 0.49975 | 0.5044 / 0.5025 / 0.4997 |
| P(B routed) | 0.49975 | 0.4915 / 0.4988 / 0.5009 |
| P(both routed) | 0.22073 | ≈0.22(一致) |
| E[E_comp+] | 1.11610 | 1.1205 / 1.0746 / 1.1280 |
| E[E_comp−] | 2.00100 | 1.9893 / 1.9596 / 1.9788 |
| **E[E_comp \| both-routed]** | **0(解析)** | **0 精确(max=0.0e0,3×10⁴ 历史)** |

- **分离测试(核心)**:both-routed 条件下组合误差**逐历史精确为 0** → 组合器
  **内在误差 = 0**;无条件 E[E_comp] > 0 完全来自上游路由误差
  (Routing Error → Wrong Operand → Composition Error)。STOP-2 不触发。
- 查询顺序随机化第二遍:逐历史 + 不变、− 精确反对称(结构性不变量)。PASS
- 捷径审计:全部泄漏控制通过(§15 清单)。

## 4. 最终科学问题的回答(§24)

> 在已经具备正确 Binding 的情况下,增加一个最小二输入组合节点,是否足以产生
> 可识别的 Composition?
**是(在本受控机制族内,sufficiency result)**:M2 的规范组合误差精确为 0
(三误差联合:绑定 0 ∧ 组合 0),且 8 项反事实全部通过;分离测试证明**组合器零误差**,
无条件误差全部由绑定/路由保真度决定。同时 M0(单 winner)与 M1(平行绑定,无组合器)
精确地不能组合——负对照链闭合:
`Hard Binding ⇏ Composition;Parallel Binding ⇏ Composition;
Binding + minimal two-input combiner ⇒ Composition(组合误差 = 绑定误差)`。

## 5. 组织自由度阶梯(§17)

```
M0: Single Winner → (+Parallel Representation) → M1: Parallel Binding
  (⇏ Composition, STRUCTURALLY_UNDEFINED)
→ (+Minimal Two-input Combiner) → M2: Binding + Composition
  (sufficiency within the tested mechanistic family only)
因果顺序: Routing → Binding → Operand Delivery → Composition(§16,实证于 CF7)
```

## 6. 范围纪律(§18,严格执行)

本 Sprint 只证明:在冻结的 TAN-II 最小机制族中,最小二输入组合节点足以把已实现的
双通道 Binding 转化为显式 Composition。**不**声称智能/认知/语义理解/大脑原理/
composition 的唯一实现/所有网络遵循该阶梯。

## 7. 复现

```
python code/experiments/sprint4_3/sprint4_3a_composition_probe.py   # 79 checks, exit 0
```
- `results/sprint4_3/{canonical_results,counterfactual_results,monte_carlo_results,
  audit_summary}.json`、`sprint4_3a_checks.csv`、`sprint4_3a_composition.png`
- 确定性记录:seeds {20260904,20260905,20260906}、N_H=10⁴、参数/容差/版本全在
  `audit_summary.json`;脚本 SHA-256 记录于内。
- 实现级 bug 修复记录(3 处,均不涉科学设置):CSV 编码(GBK→utf-8)、死代码清理、
  边距基准常量更新(AM6,非 post-hoc tuning——不改变任何模型/查询/判据)。
