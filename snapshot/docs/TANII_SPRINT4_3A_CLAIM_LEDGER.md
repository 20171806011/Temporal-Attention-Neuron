# TAN-II Sprint 4.3-A — 主张账本(Claim Ledger)

**阶段**: Sprint 4.3-A(Minimal Composition Probe)
**日期**: 2026-09-04
**等级**: ESTABLISHED / SUPPORTED / REJECTED / UNKNOWN。
**依据**: `docs/TANII_SPRINT4_3A_{PREREGISTRATION,AMENDMENT_6,RESULTS}.md`、
`results/sprint4_3/*.json`(79 项检查、8 反事实、集合 MC)。

---

## 1. 核心命题

| # | 主张 | 等级 | 依据与限定 |
|---|---|---|---|
| C1 | 单 winner 系统(M0)在双查询组合场景下不能组合:E_bind,A=\|V_B−V_A\|>0、E_comp=\|V_A\|>0 | **ESTABLISHED** | 规范三场景精确;一次 WTA 只有一个 winner,只交付一个操作数 |
| C2 | 平行双通道(M1)实现精确平行绑定(E_bind,A=E_bind,B=0)但不构成组合 | **ESTABLISHED** | 输出是对 (V_A,V_B);组合在结构上未定义(STRUCTURALLY_UNDEFINED;冻结的标量投影对照 E_comp=\|V_B−f\|>0) |
| C3 | 增加最小二输入组合器节点(M2)⇒ 显式 Composition(规范三误差联合 = 0 精确,8 反事实全过) | **ESTABLISHED(限本机制族,sufficiency)** | 确定性 79/79;y=f(V_A,V_B) 逐场景精确;不是"泛化组合智能" |
| C4 | 组合器内在误差 = 0(分离定理):E[E_comp \| both-routed] = 0 精确 | **ESTABLISHED** | 3 种子 × 10⁴ 历史 max = 0.0e0;理论:组合器对正确绑定值做精确算术 |
| C5 | 无条件组合误差 = 上游绑定/路由误差的传播 | **ESTABLISHED** | E[E_comp+]=1.116(基准)≈MC;结构分解:E_comp+=\|V_A−V_B\|·[恰一通道错];CF7 实证因果链 Routing→Wrong Operand→Composition Error |
| C6 | 交换/非交换组合的精确行为:加法可交换(query/value/key swap 不变),减法反对称(三项 swap 精确变号) | **ESTABLISHED** | 8 反事实 + 集合逐历史断言(1e-12);含 channel-output swap |
| C7 | AM5 型伪成功被三误差联合审计捕获 | **ESTABLISHED** | V_A=0:M0 E_comp=0 但 E_bind,A>0 → FAIL;M2 真 PASS |
| C8 | 负对照链:Hard Binding ⇏ Composition;Parallel Binding ⇏ Composition;Binding + combiner ⇒ Composition | **ESTABLISHED(限本机制族)** | C1+C2+C3 合取 |
| C9 | "Composition 只需要一个 combiner(对所有系统/所有任务)" | **REJECTED(未主张)** | 仅为受控机制族内的 sufficiency result |
| C10 | M2 实现 generalized compositional intelligence / 语义组合 / 认知 | **REJECTED(未主张,范围纪律 §18)** | 无泛化 Probe、无训练、无 benchmark |
| C11 | 规范路由边距的预注册手算值(0.2118/0.2637) | **REJECTED(数值基准修正)** | 精确闭式 0.210631046/0.261878523(Amendment 6,DEBUG 归因手算取整;定性预测不变) |
| C12 | 双查询窗口下遗产查询 (3.0,4.0) 的绑定破坏行为 | **ESTABLISHED** | CF7:通道 2 打偏,E_bind,B=4、E_comp=4(窗口上下文性质,预注册记录) |
| C13 | 组合任务在更复杂算子/更多操作数/泛化分布下的行为 | **UNKNOWN** | 本 Sprint 仅 {+,−} × 2 操作数 × 冻结查询对;未做 |

## 2. 协议合规记录

- Amendment 6(运行期边距基准):按 GO 指令 STOP → DEBUG → 记录原因 → 再继续 处理;
  原文保留、独立复核、PREDICTION-CORRECTED-AT-RUN 标记;非 post-hoc tuning。
- 实现级 bug 修复(3 处,记录在案):CSV 编码、死代码、边距常量(AM6)。
- 停止规则:最终 0 FAIL;STOP-2(分离测试)不触发;无泄漏、无捷径、无 post-hoc。
- 理论预测与实测分离:预测仅见于预注册;实测全部来自程序输出(独立报告)。

## 3. 最终一页结论(口径锁定)

> **4.3-A 在冻结的最小机制族内证明:单 winner 不能组合;平行绑定本身不是组合;
> 在其上增加唯一的最小二输入组合节点后,规范三误差联合审计精确为 0、8 项反事实
> 全过、组合器内在误差逐历史为 0——即 Binding + minimal combiner ⇒ Composition
> (sufficiency),且无条件组合误差完全由绑定/路由保真度决定(分离测试)。**
> 完整阶梯至此:Memory → Saliency → Geometry → Competition ⇏ Routing →
> Vector Q-K ⇒ Routing → K-V Decoupling ⇒ Soft Binding → One-hot WTA ⇒
> Exact Delivery → **Parallel Binding + Minimal Combiner ⇒ Composition**。
> 下一步(若有)将是 Composition 的泛化/多操作数/新算子问题(UNKNOWN,待 GO)。
