# TAN conceptual latest-value range audit: v2 concept package

这是 AI 辅助的 Chat 沙箱有边界概念架构比较，不是原 TAN/P1-B/v11 复现，不改变它们的结论。未训练或调参。同一 agent 的复跑和核验不是外部独立验证。历史来源哈希见 previous_artifacts_sha256.json，复核见 REPLAY_CHECK.json。公开版复现和哈希说明见 PUBLIC_REPRODUCTION.md。

## 快速运行

已实际运行环境：Python 3.13.5，NumPy 2.3.5，float64。主网格没有随机数，没有训练和参数搜索，所以不需要随机种子。

```bash
python range_consistency_audit_v2.py --out rerun_results_v2.json
python verify_results_v2.py --results rerun_results_v2.json --out rerun_verification.json
```

脚本默认读取同目录 frozen_spec.json，SHA-256 不匹配会停止。相同环境重复运行已得到逐字节一致 JSON；不承诺不同 BLAS/平台逐位一致。检验脚本是同一 Chat 环境内的另一路核验，不是用户或第三方的独立重跑。下载哈希也不是实验复现。

## 冻结网格

先保存并输出 frozen_spec.json 的哈希，然后才运行算法。状态维数4；初始两键 f1=e1，f2=e1+a*e2，a∈{1,.01}；初值(.25,-.5)。随后以同一个固定受扰 f1 反复覆盖到1，共16次，再加入 e3→0，最后用干净 f2 覆盖到-.75。

- 两档绝对特征值截断：1e-10、1e-4。
- 七个矩阵条件：无故障；weak-null 对称耦合、弱模侵蚀、零模正向抬升，各用 .1α、.6α 的算子范数预算。故障在干净初始化后只注入一次；后续累积写入保留该故障，不逐次重新注入。
- 五个键噪声条件：零；.001、.02 倍 e3；.001、.02 倍(.8e2+.6e3)。固定扰动重复，不是iid噪声。
- 共140场景×4组=560条轨迹。不是独立抽样成功率或总体频率估计。它是小容量机制诊断，不是高维大容量性能测试。

初始化使用干净特征、截断1e-12、span阈值1e-9；故障阶段统一span阈值.25。这个阶段变化是事先固定条件，不是向学习器告知新旧标签。主程序只按span残差路由。

## 四种旧写

U：不投影；P：投影整个b及增量；PN：P加leverage归一化；TK：与PN完全相同，但只在当前输入处使用真键。TK没有正确矩阵或正确逆矩阵，也不保存真key表，所以它不是无条件性能上界。

学习器只保存C、b、计数和固定数值参数；真key、目标表、真实几何只在评分器中使用。全b重投影会删掉已存在的弱方向内容；不要把这个算法混同于“仅投影增量”。

所有组的新写均为C+=xx^T，b+=v*x，不在新写前自动投影旧b。已满容量、leverage<=1e-12或状态范数超过1e12时记录拒绝/停止。范数上限在写后检查：首个越界状态保留作故障诊断，后续写被阻断，不宣称其已满足状态范围约束。

## 主要结果

1. 无矩阵故障、a=1、键噪声.02e3时，U在加入新e3→0后读出.015；P/PN/TK为0，旧键保持(1,-.5)。
2. 无矩阵故障、a=.01、混合噪声.02时，U/P在固定受扰地址重复写会放大残差。PN在该地址输出1，但真键输出约(-.15068493,.56849315)，而目标是(1,-.5)。TK在正确几何下保护真键。
3. a=.01且截断1e-4时，有效弱方向被删除。即使TK，真键输出也约(1,1.00005)，不能保持第二值-.5。
4. 专门的截断恢复fixture显示：一旦全b投影删掉弱方向，之后调回正确截断也不能恢复；U保留的隐藏内容则可以恢复。该恢复操作仅是预先列出的诊断fixture，不是主网格的参数调优。
5. 旋转投影fixture中，Qhat的幂等残差约5.55e-17，却与真实Q相差.6。它将.001e3污染映成.00048e2+.00036e3，留下真实零空间污染。
6. 原(-.2,.4)反例保留并重新执行：normalized更新在受扰地址稳定正确，但真实键错误。

主网格：U/P各5条轨迹触发范数停止；PN/TK没有触发。所有140条PN轨迹在固定受扰观测上的最终残差<1e-9，却不能据此认定真键恢复。later-new路由计数和数值误差分别保留；新目标恰为0可能掩盖误路由的当前数值后果。

## 关于置信与证书

没有bootstrap、p值或总体成功率。原冻结的充分截断证书delta<cutoff<alpha-delta仅覆盖本网格的零故障组合；这项设计局限已保留，不改阈值“制造通过”。在结果之后另加了同rank、已知误差界下的代数核验，检查180个条件，其中120个有非零矩阵故障。此项是已有记录的补充核验，不是新增实验配置。详见 verification_v2.json 与 THEORY.md。

结果初版曾把“已停止”归入any_misroute标签；仅修正汇总标签，所有网格和fixture数值逐项未变。SCIENTIFIC_CORRECTIONS.json记录初版哈希，REPLAY_CHECK.json记录一致性。没有隐藏失败或重新搜索参数。

## 文件

- frozen_spec.json / SPEC_SHA256.txt：运行前冻结规则。
- range_consistency_audit_v2.py：完整主脚本，仅依赖NumPy。
- results_v2.json：全部560条轨迹、16次旧写详细记录与额外fixture；未压缩约16MB。
- grid_summary.csv：逐场景逐组汇总，不是统计推断。
- summary_v2.json：缩略结果、固定切片及全部fixture。
- verify_results_v2.py / verification_v2.json：另一路代数、合同及有理数反例核验。
- THEORY.md：不变量证明、近似投影风险、局部充分误差合同及资源界限。
- REPLAY_CHECK.json / SCIENTIFIC_CORRECTIONS.json / previous_artifacts_sha256.json：复跑与构建来源。
- HISTORICAL_SHA256SUMS.txt：原包历史哈希；PUBLIC_SHA256SUMS.txt：当前公开文件校验。

## 可防守结论

正确range投影消除“旧写落入真实零空间、以后扩容再激活”的特定漏洞。估计range投影可能删掉有效内容或保留错误污染。Leverage归一化保证的是固定观测上的步长控制，不是地址身份。稳健latest覆盖仍需地址/输入承诺、误差与谱预算、容量准入和适当拒绝；没有由这些小型结果推出生物可行性或神经元普遍不可能性。
