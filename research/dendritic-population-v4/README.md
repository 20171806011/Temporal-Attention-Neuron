# V4：原生脉冲网络可学习性边界与结构对照

已按批准的草案完成 A → B → C，并完成一次新初始化、新输入的测试。**本轮观察到的 L3 差异仅适用于 delayed XOR、共用 RATE/.01 配置、200 次更新和各 5 个开发/测试初始化种子。** 没有改变 V2 的前向方程或原生硬脉冲，也没有据此作一般性结构结论。

| 问题 | 实际结果及边界 |
|---|---|
| Q1：换训练方法是否解释 V2 失败？ | 配对 Adam 运行中有明显改善，但主/次预算的共用选择都未让两个结构的所有开发种子通过；不能把历史失败唯一归因于 SPSA。完整 CE 同损失/同学习率对照与 RATE 搜索分列于报告。 |
| Q2：在哪一级失效？ | 开发阶段 POINT 首次在 L2 出现未通过种子，DEND 到 L3 仍通过；新初始化测试中 L2 两组都通过，故该开发失效点不稳定。L3 的全部预定差异判据得到满足，限于上述配方。 |
| Q3：跨细胞循环是否必要？ | L2 无跨细胞连接重训对照两组都通过；直接切断既有模型会下降，因此不能把既有解的依赖写成连接不可替代。 |

## 一次性新测试

每组每级 5 个新初始化种子，每个种子 512 条新测试输入。平票/静默计错，未丢弃任何种子。

| 等级 | DEND 平均准确率 | POINT 平均准确率 | 配对差（百分点）与探索性 95% 区间 |
|---|---:|---:|---|
| L2：4 步空隔 | 100.00% | 99.53% | +0.47 [0.00, 1.48]；不满足差异判据 |
| L3：delayed XOR | 99.18% | 61.45% | +37.73 [17.50, 57.74]；满足冻结的任务内判据 |

L0/L1 两组每个开发种子均为 100%：**预期达到天花板，未进行测试**。这些区间未做多重比较校正，5 个初始化种子仍有限；不构成总体等价、能耗、生物学或任务外推论。

## 文件

- [完整报告](REPORT.md)与[所有候选/逐种子结果表](TABLES.md)。
- [冻结协议](PROTOCOL.md)、[源码冻结哈希](FREEZE.json)、[运行提交顺序](RUN_PROVENANCE.json)。
- [图](results/comparison.png)、[机器可读新测试结果](results/TEST_SUMMARY.json)。
- [开发锁定记录](results/DEVELOPMENT_LOCK.json)、[新初始化复制检查点](results/REPLICATION_LOCK.json)。
- [资源/计算量汇总](results/aggregation.json)：A100/A200 共用轨迹去重，不把更新次数等同算力。
- [独立数值验证](results/verification.json)、[选择后梯度审计](results/gradient_audit.json)、[训练前检查](GRADIENT_CHECK.json)。
- [旧文件完整性](HISTORICAL_INTEGRITY.json)、[V3 作者更正附注版](../dendritic-population-v3/REPORT_WITH_APPENDIX.md)。原模块误认已关闭，原报告/代码/清单未被改写。

数值验证 57,138 项通过，完整重放 20,480 个测试模型试次；最大状态残差 1.33e-15。训练前 704 项和选择后 16,400 项参数导数检查通过；420 个训练更新抽样重算一致。所有 Adam 更新与 SPSA 更新序列也重建通过。这些是同一 AI 辅助工作流内检查，不是外部独立复现或同行评议。

前向始终使用硬二值脉冲；代理导数是训练规则，不是硬脉冲函数精确导数。完整模型 176 个可学习参数，重训消融模型 116 个；DEND/POINT 在同一条件内匹配。训练总计 140 条独立轨迹、26,000 次更新（24,000 Adam、2,000 SPSA），包含全部失败候选及新初始化复制。CPU 成本、矩阵工作量和不匹配资源详见报告。

## 复现与审计

Python/NumPy 版本在 DEVELOPMENT_LOCK.json；图表需要 Matplotlib。在本目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 verify_v4.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 audit_gradients.py
```

审计会更新耗时 JSON；原交付文件哈希以 MANIFEST.json 为准。重跑实验应在新的隔离检出中使用空结果目录，保留相邻 v2/model.py 作为冻结依赖。顺序是 `check_gradients.py` → `runner.py develop` → 审查并提交开发锁定 → `runner.py replicate` → 提交复制检查点 → `evaluate.py`，不得覆盖原结果或用已见测试结果重选配置。训练/测试命令均包含防重复写入检查。

草案材料保留作历史记录，DRAFT_MANIFEST.json 中原始哈希对应草案提交 `74a1ec1`；最终完整交付见 MANIFEST.json。V4 仍在本地分支 `research/dendritic-population-v4-2026-11-03`，未推送、未创建公开 PR、未合并主分支。V4 是独立实验线，未合并 TAN 主线论文。
