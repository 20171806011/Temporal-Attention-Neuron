# v2：可学习连接与群体交互

已在独立分支完成一轮冻结方案的实验。保留神经元原生发放，加入可学习的输入、循环和输出连接；v1 与旧 TAN 代码未改动。

**这轮没有证明有效的树突优势，也没有稳定学会延迟 XOR。** 三个种子各有 512 个新测试样本：

| 种子 | 树突群体 | 普通脉冲群体 |
|---|---:|---:|
| 0 | 0.00% | 0.00% |
| 1 | 0.00% | 47.27% |
| 2 | 0.00% | 23.83% |

树突模型的输出全部平局：两个种子静默，一个种子双输出饱和。平局按预先规则计为弃答错误。普通网络也未稳定超过均衡二分类的 50% 强制猜测参考。切断跨神经元连接会改变隐藏发放，说明交互确实存在；目前没有形成稳定的任务收益。

两组均为 20 个连续状态、176 个可训练参数、相同矩阵运算规模和训练搜索预算。细胞数、标量操作和能耗不等同。原生阈值/复位保留；训练采用数值 SPSA，不声称生物学习。

- [完整报告](REPORT.md)：方法、负结果、置信区间、学习失败与资源边界。
- [冻结协议](PROTOCOL.md)、[代码哈希](FREEZE.json)：提交 `8f6a06e`，先于训练数据生成。
- [模型选择记录](results/SELECTION.json)：提交 `634285f`，先于测试数据生成。
- [全部结果](results/summary.json)、[图表](results/comparison.png)、[完整训练日志](results/training_log.json)。
- [复核记录](results/verification.json)：1,473 项检查通过，9,216 个模型试次完整重放；另有[检查器修正记录](VERIFICATION_ATTEMPTS.json)。
- [描述性诊断](results/diagnostics.json)：没有用于调参或改变主要结论。

可从本目录运行 `python3 verify.py` 复核保存的数据。需要 Python 与 NumPy；版本见模型选择记录。图表另外使用 Matplotlib。保存的 NPZ 是输入与完整数值轨迹，不只是摘要。

要重跑实验，请把本目录的 `.py` 文件、`PROTOCOL.md`、`FREEZE.json` 复制到新的空目录，不复制 `results/`，依次运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 test_model.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 train.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 evaluate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 verify.py
```

训练拒绝覆盖已有 `results/`；测试拒绝重复执行。复核会更新其耗时记录，因此重跑后文件哈希可能变化，原始交付哈希以 `MANIFEST.json` 为准。该清单不包含自身和 Python 缓存。

下一瓶颈是稳定训练、有效读出与循环增益控制。任何新方案都应先在开发数据上建立可学习性，再使用新的冻结测试集；本轮不继续调参。研究保存在本地分支 `research/dendritic-population-v2-2026-11-03`，没有推送或合并 GitHub 主分支。
