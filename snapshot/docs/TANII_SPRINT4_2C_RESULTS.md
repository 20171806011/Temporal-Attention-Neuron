# TAN-II Sprint 4.2-C — 结果(Routing → Binding minimal probe)

**项目**: TAN-II:组织机制与选择性计算的涌现
**阶段**: Sprint 4.2-C —— Routing → Binding 最小探针
**日期**: 2026-09-04
**预注册**: `docs/TANII_SPRINT4_2C_PREREGISTRATION.md`(保持原样)
**Amendment 4**: 运行前解析修正(M5-E 驱动传递数值),已冻结
**输出**: `results/sprint4_2/sprint4_2c_*`
**程序**: `code/experiments/sprint4_2/sprint4_2_binding_probe.py`

## 0. 状态块

```
TAN-II SPRINT 4.2-C STATUS: COMPLETE
Checks:                    42/42 PASS (0 FAIL)
Canonical counterfactual:  [A=(1,5), 0, B=(2,9), 0], only Q changes 3.0 -> 4.0:
  C_events: 6.7427 -> 7.1556   (moves toward the SELECTED value: 5 -> 9)
  winner:   A -> B              (routing, from 4.2-B)
  T = -0.412904 (winner-aligned) | T_swap = +0.412904 (exact antisymmetry, 1e-12)
Softness: D = 1.7427 / 1.8444  (soft/partial binding; hard binding D=0 NOT reached)
Drive transmission: M2 keeps direction (-0.2586); M5-E INVERTS it (+0.2367, AM4)
Ensemble: E|T| = 0.4004 (MC 0.3977/0.3967/0.4021); flip-cond 0.4655 (0.4685/0.4624/0.4712)
Amendments: 1 (AM4, pre-run) | Post-hoc tuning: 0 | Frozen archives modified: 0
```

## 1. 最小 (K,V) 解耦与协议执行

- 事件槽键 K(驱动注意力,与 4.2-B 完全同构)与值 V(传输通道)解耦;pad=(0,0);
  探针键 = Q、**探针值 = 0**(查询是控制信号,不是内容)。
- 规范历史 `[A=(1,5), 0, B=(2,9), 0]`;值交换对照 `[A=(1,9), 0, B=(2,5), 0]`;
  12 种位置排列——**位置不变性断言通过**(12 布局 C_events 逐位相等,1e-12)。
- 铁门延续:窗口/惊奇/E-I/膜/读出冻结;无学习、无新拓扑、无 benchmark。

## 2. 规范反事实结果(确定性,全部命中冻结预测)

| 量 | 冻结预测 | 实测 | 判定 |
|---|---|---|---|
| C_events(Q=3) | 6.742691 | 6.742691 | PASS |
| C_events(Q=4) | 7.155595 | 7.155595 | PASS |
| T = C(3) − C(4) | −0.412904 | −0.412904 | PASS(方向 = sign(v_winner(3)−v_winner(4)) = −) |
| T_swap | +0.412904 | +0.412904;**T_swap = −T 至 1e-12** | PASS(读出游出跟随 (K,V) 关联,不是键身份) |
| 距离判据 | \|C(3)−5\| < \|C(3)−9\|;\|C(4)−9\| < \|C(4)−5\| | 1.74<2.26;1.84<2.16 | PASS(Q_A 输出更接近 V_A,Q_B 更接近 V_B) |
| D(软度) | 1.7427 / 1.8444 | 同 | PASS(**D>0:软绑定,硬绑定未达**) |
| M2 驱动传递 | −0.258552 | −0.258552 | PASS(单细胞门控**保持**传递方向) |
| M5-E 驱动传递 | +0.236700(AM4) | +0.236700 | PASS(**方向反转**;抑制分支也读出值通道并相减) |
| M4 驱动(标量 E-I) | −0.0869 / −0.0004 | 同 | PASS(winner 恒 B;锐化基底,无路由即无传递) |
| M0 T | −0.084443 | 同 | PASS(锐化基底;|T_M0| < |T_M2| ✓) |
| M1(±符号) | C(+1)=8.8936,C(−1)=5.1064;D=0.1064 | 同 | PASS(退化但近硬绑定:D_M1=0.106 < D_M2=1.74) |
| M2unnorm | T ≈ 0 | −0.000215 | PASS(注意力塌缩到探针 v=0,无值传递——AM2 的绑定侧复现) |

## 3. 集合结果(MC vs 冻结求积基准,3 种子 × 10⁴)

| 量 | 基准 | MC(三种子) | 判定 |
|---|---|---|---|
| E[T] | ≈ 0(对称) | +0.0075 / −0.0016 / +0.0062 | PASS |
| E[|T|] | 0.400435 | 0.3977 / 0.3967 / 0.4021 | PASS |
| E[|T| \| flip] | 0.465523 | 0.4685 / 0.4624 / 0.4712 | PASS |
| E[|T| \| noflip] | 0.348347 | 0.3408 / 0.3437 / 0.3488 | PASS |
| flip rate | 0.444527 | 0.4454 / 0.4465 / 0.4357 | PASS |
| sign-matched alignment \| flip | 1.0 | 1.000(三种子) | PASS(**定理核验**) |

## 4. 诚实分层:什么是 4.2-C 真正的证据

1. **方向对齐是读出结构的定理**(ᾱ_A = σ(c·S·û·Δφ̂) 对 logit 差单调 ⇒ 翻转历史上的值传递
   方向必然跟随 winner)。集合 alignment = 1.0 是**定理核验**,不作经验发现。
2. **4.2-C 的实质证据是四条**:
   (i) 规范反事实下输出更接近被选值(两查询、两历史,距离判据全部通过);
   (ii) **值交换反对称精确**(T_swap = −T,1e-12)——读出跟随 (K,V) 关联而非键身份;
   (iii) **传递幅度分级**:路由模型的 |T| = 0.413 显著大于锐化基底 M0 的 0.084;
   集合上 flip 条件传递 0.466 vs noflip 0.348;
   (iv) **传递路径**:单细胞门控保持方向(−0.259),E-I 电路因抑制分支的第二次值读出
   **反转方向**(+0.237,Amendment 4)——这是如实报告的边界发现。
3. **软绑定 vs 硬绑定**:软注意力读出游出是凸组合,D = 1.74/1.84 > 0 ⇒
   **硬绑定(exact value delivery)未实现**;对照显示 M1(±符号,能量铺展大)反而
   D = 0.106 更接近硬绑定。硬绑定需要 winner-take-all 类组织(未来)。

## 5. 用户口径的回答

> 固定完全相同的 history,只改变 Q:Q_A→V_A,Q_B→V_B?
- **是(soft/partial 意义)**:Q_A=3 时读出游出 6.7427 更接近 V_A=5(1.74 vs 2.26),
  Q_B=4 时 7.1556 更接近 V_B=9(1.84 vs 2.16);值交换后判据同样成立且 T 精确反对称。
- **但不是 exact**:读出是软混合,D>0。routing 与 binding 因此被严格拆开测量:
  routing(winner 翻转)= 4.2-B 已建立;binding(值传递)= 4.2-C 建立为**软绑定**,
  其硬绑定缺口(D)被量化并归因于软注意力结构。

## 6. 研究链更新

Memory → Saliency → Geometry → Competition ⇏ Routing →
Vector Q-K ⇒ **Routing(Level-2, 4.2-B)** → **(K,V) 解耦 ⇒ Soft/Partial Binding(4.2-C)**;
Hard Binding(exact value delivery)= 未达,归因于软注意力凸组合,候选解 = winner-take-all
(未来组织自由度,UNKNOWN,不宣称)。

## 7. 不宣称(No-claims)

- 不宣称硬绑定 / 精确值传递;不宣称 generalized binding;
- 不宣称 E-I 驱动级传递成功(方向反转如实报告);
- 不宣称任何"内容理解/智能";不把定理核验(alignment=1)包装成经验发现。

## 8. 复现

```
python code/experiments/sprint4_2/sprint4_2_binding_probe.py   # 42 checks, exit 0
```
文件:`results/sprint4_2/sprint4_2c_{summary.json, checks.csv, canonical.png, ensemble.png}`。
环境:Python 3.12.7 / numpy 1.26.4 / matplotlib 3.9.2;确定性。
