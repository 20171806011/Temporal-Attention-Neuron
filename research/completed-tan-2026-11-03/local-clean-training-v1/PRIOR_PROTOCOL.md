# 下一最小因果对照（本轮未执行）

问题：旧读出在clean上接近机会水平，是否部分来自clean/distractor混合训练，而不仅是clean响应结构？

仅比较同一原类source response、同一33参数线性softmax、同一原clean测试集。
不添加cue、target或任何新observable，不比较多种nonlinear模型，不改变生成器/目标/动力学。

固定三个原seed；使用原已存训练数据。每seed两个训练臂：
- Clean-only：每类500个clean训练trial，共1500。
- Mixed：每类250个clean +250个distractor，共1500。

各类从一个事先固定的共同训练索引排列取样（默认固定RandomState 2026110351+seed）；mixed的250个clean为clean-only的500个子集。
若某类不足所需样本则停止，不扩大/调配。组标识保持原源记录；选择不看测试值或模型输出。
均保持原训练统计标准化、零初始化、L2=1e-4、L-BFGS-B maxiter800/ftol1e-12。
每臂一次拟合，不筛选初始化/参数。不复活历史已取消的shuffle审计门。

唯一主要比较：同一clean测试输入上的balanced3差（clean-only minus mixed），并保留每类recall/混淆及全部seed。
配对统计单位保持原history/复用组规则；区间仅条件于这些固定训练数据与拟合，不宣称新独立盲测。
训练样本量/各类数量相同，避免用“只用一半样本”混淆训练组成的解释。

解释：
- Clean-only明显提升：支持混合训练目标存在clean干扰；不消除本轮证明的精确响应碰撞。
- 不提升：只能说这次同类线性读出干预没有解决，不能证明Bayes上限为1/3或没有剩余信息。

这是一项未来分析方案，不是已经执行的结果或已验证结论。
