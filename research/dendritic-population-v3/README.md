# v3：先确认原生脉冲网络能学会简单任务

**两组网络都通过了可学习性检查。** 保留原生硬阈值发放，在三个种子、每个 512 个新测试样本上，树突群体和普通脉冲群体均达到 100% 准确率与覆盖率，没有弃答、全静默或双输出饱和。

这是持续输入通道识别任务，不是 v2 的延迟 XOR，也不证明树突、循环交互或任何单一神经元具有普遍优势。两组配对准确率差为 0；有限样本全部成功并不等于未来必然成功或模型等价。

训练前冻结协议，24 次开发候选均完整保留。开发集上，原分类损失和目标发放率损失都能学习；按预定规则，为两组共选目标发放率损失、学习率 0.01。冻结选择后才生成新测试数据。v1/v2 未改动。

- [完整报告](REPORT.md)：结果、原因边界、计算成本和后续瓶颈。
- [冻结协议](PROTOCOL.md)与[源码哈希](FREEZE.json)：提交 `44ca749`，先于开发数据生成。
- [模型选择记录](results/SELECTION.json)：提交 `93a239b`，先于测试生成。
- [测试结果](results/summary.json)、[图表](results/comparison.png)、[全部训练记录](results/training_log.json)。
- [初始梯度检查](GRADIENT_CHECK.json)、[训练后梯度审计](results/gradient_audit.json)、[独立数值复核](results/verification.json)。

两组同为 20 个连续状态、176 个可训练参数，匹配数据、矩阵运算与训练搜索预算；细胞数、标量操作和生物能耗不相等。前向是原生二值脉冲；替代导数只用于反向训练，不能称为硬脉冲函数的真实导数。可微辅助图仅用于梯度核验。

复核通过：9,921 项数值检查、6,144 个完整模型试次；另有 704 项初始、1,056 项训练后参数导数检查。这里的“独立”指另一套方程实现，不是外部团队复现。

## 复现

本目录需要 Python 与 NumPy，确切版本见模型选择记录；作图另需 Matplotlib。可运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 audit_gradients.py
```

后一个脚本还读取相邻、保持不变的 v2/model.py，仅用于前向方程一致性检查。重跑核验会更新耗时记录并改变对应文件哈希；原交付哈希见 `MANIFEST.json`。

重新训练时，将本目录源代码、协议、引用说明和 FREEZE.json 复制至新目录，不复制 results/，运行 check_gradients.py、train.py。只有 SELECTION.json 中 gate_pass 为 true 才运行 evaluate.py；测试拒绝重复执行。最后运行 verify.py。保持父目录布局才可运行含 v2 一致性检查的 audit_gradients.py。

下一步应在开发数据上逐步加入延迟与条件绑定，固定任务后比较训练方法；当前结果没有解决 v2 难题。本轮已保存在本地独立分支 `research/dendritic-population-v3-2026-11-03`，未推送 GitHub，主分支未改。
