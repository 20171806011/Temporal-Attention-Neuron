# 来源台账与证据层级

固定仓库：20171806011/Temporal-Attention-Neuron
固定提交：85213f0d417d54e4f48b301ed1c0e4200ef3c421

本轮亲自读取：
1. snapshot/code/experiments/temporal_distractor.py，60–160行：支持、生成器、读出特征。
   Git blob 440daff8aced676aa66e97e99196cb9bd8e185ca。
   https://github.com/20171806011/Temporal-Attention-Neuron/blob/85213f0d417d54e4f48b301ed1c0e4200ef3c421/snapshot/code/experiments/temporal_distractor.py
2. snapshot/code/experiments/natural_collision.py，225–285行：未平移指数+epsilon，u/h分离及reset。
   Git blob 5bb1358e5ea35ef7077e2d0c65bda4661409b5ab。
   https://github.com/20171806011/Temporal-Attention-Neuron/blob/85213f0d417d54e4f48b301ed1c0e4200ef3c421/snapshot/code/experiments/natural_collision.py
3. snapshot/code/model/tan.py，65–135行：稳定化softmax、先入窗、S、原类forward。
   Git blob d4e0ba873955b6e6b1f10bd36403f79c2666de51。
   https://github.com/20171806011/Temporal-Attention-Neuron/blob/85213f0d417d54e4f48b301ed1c0e4200ef3c421/snapshot/code/model/tan.py
4. 当前附件task-grounded Probe1包的SOURCE_REVIEW.md、frozen_spec.json、run_probe1.py。

运行来源：
- source/tan.py 从此前已归档包提取到新目录，重新核对Git blob；实际调用原类forward。为了记录pre-reset u，仅覆写softmax以捕获原函数返回的权重，原forward方程未改。
- 历史任务递推是本轮对所读natural_collision函数的明确转写；不冒称导入完整原模块。
- source/task_excerpt.py 是此前已披露的转写，保留供查阅，本轮不训练、不调用其中生成器或优化器。
- 前轮约33%clean结果来自已归档执行，不在本轮重新估计。

证据层级：
A. 源码事实：本轮实际访问的文件/片段与已核验原类字节。
B. 数学结论：显式映射、分支、rank/单调性/碰撞存在性及局部逆证书，在文中假设下推出。
C. 本轮执行：固定fixture、根括区、Fraction有理区间、原类与转写数值核对；不是原任务全量重跑。
D. 未执行解释：mixed训练可能造成clean负迁移；非线性Bayes最优准确率；全域已知cue逆问题。

既有v11、P1B结果本轮不验证/更改；不能从其不同observable的结论外推本clean映射。
