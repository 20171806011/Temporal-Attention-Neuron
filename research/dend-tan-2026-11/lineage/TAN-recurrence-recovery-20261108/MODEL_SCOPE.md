> Historical archival document. Original language preserved. Personal paths/host metadata and unavailable media links were sanitized; consult README.md and ARCHIVE_INDEX.json.

# 本批模型身份与原始 TAN 的代码核对

**本批没有使用原始 Temporal Attention Neuron 的时间注意力机制。** 当前实验研究的是 separately specified DEND/POINT 群体电路。目录的 TAN 前缀标识研究项目及历史来源，不能据此把当前网络称为“全部由原始 TAN 神经元组成”。此前目录名与报告标题保留原样，以免破坏已封存数据；这份说明纠正其可支持的命名和结论范围。

## 只读证据及版本

最新研究仓库 HEAD 为 `7dcddee41490d8d917936891989eeb7ddb67e0da`。其 README 第5行明确说明有原始 TAN 与另行定义的树突/点神经元群体两条研究线，第27行明确写出群体电路不同于原始单元模型。当前稀疏检出的目录不包含原始 snapshot 工作文件，因此只读了另一现有工作副本；原始 tan.py 与 habituation.py 的 Git blob 与最新 HEAD 中对应路径逐一相同，并非仅凭旧目录名称推断。

选定原文已复制到本目录的 model_scope_sources/，没有编辑原仓库：

- `original_tan.py`：原始 snapshot/code/model/tan.py，blob `d4e0ba873955b6e6b1f10bd36403f79c2666de51`。
- `habituation_excerpt.txt`：snapshot/code/experiments/habituation.py 的67–136行；对应完整文件 blob `692be3cfed27b1a76943a620454fa0afc0ef24d4`，与上一实现相互印证。
- `population_v10_model.py`：当前电路的 V10 来源定义，blob `19f22cde11f30af913b214c834ccaab242e6b84c`。
- `repository_README.md`：仓库对两条研究线的明确声明，blob `9891dd7431a09f17236586f30978f20b75e2a12e`。

路径、SHA-256、HEAD 和 blob 对照见 MODEL_SCOPE.json。此核对没有运行原始 TAN，也没有补充任何注意力模块。

## 原始 TAN-full 实际包含什么

original_tan.py 的 make_variant('TAN-full') 在147–148行保留全部标志，默认参数在134–143行。forward 的顺序是：

1. 74–75行滚动历史窗口并写入当前输入；默认窗口长度5。
2. 80–82行计算 `S=max(0, current_input−mean(history)−noise_tolerance)`。
3. 91–99行构造 `Q=S*W_q`、`K=history*W_k`、`V=history*W_v`，由历史位置上的 softmax(beta*Q*K) 得到加权上下文 C。
4. 104–108行用 `tanh(S)*C` 门控注入。
5. 116–123行进行膜电位泄漏、阈值发放和复位。

因此原始 TAN 的“时间注意力”有具体的历史缓存、惊讶量、Q/K/V、时间位置权重与门控注入，不是只要包含跨步状态和脉冲就具备。

## 当前 DEND/POINT 的真实定义

当前 model.py 的12–16行只解包输入权重 Wi、递归权重 Wr、分支/隐藏/输出偏置和输出权重 Wo；26–56行是完整原生前向路径。实际实验的 common.py 明确设置 rho_d=.95、rho_s=.85，覆盖 model.py 的函数默认值；本轮 engine.py 使用这两个实际常数。

共同候选状态为 `dc=.95*d+.25*(x*Wi+p*Wr+bd)`。

- **DEND**：model.py 第42–43行使用 `local=ReLU(dc)`，保留连续分支状态 d=dc；每两个分级分支以固定 .5 权重汇入一个隐藏胞体。胞体和两个输出单元按严格 >.5 阈值产生二值脉冲并硬复位。
- **POINT**：44–46行把对应的2h个首层单元改为阈值发放的点单元，`branch_spike=1(dc>.5)`、`d=dc*(1−branch_spike)`；下游 h 个隐藏点单元及2个输出点单元仍以原生脉冲计算。这里变量 d 代表对应首层点单元膜电位，不表示 POINT 也拥有 DEND 那种分级树突。
- 两者都有上一隐藏脉冲 p 经 Wr 返回首层的跨单元递归，及 p 经 Wo 驱动输出。查询八步的两个输出脉冲计数决定类别，平票未判定并计错。当前恢复实验只是对该递归驱动施加冻结的时间窗屏蔽。

POINT 是**等参数的多细胞脉冲群体对照**，不是单个 LIF，也不是把原始 TAN 的某个 attention 开关关掉。h 是隐藏胞体/中层单元数：DEND 有 h+2 个脉冲细胞和2h个分级分支；POINT 有3h+2个脉冲细胞。h96 时分别为98个脉冲细胞+192分级分支，对照290个脉冲细胞；每个模型20,066个训练参数。

## 缺失的原始 TAN 组件与容易混淆之处

当前前向图中没有滚动输入历史窗口 H，没有 current_input−mean(H) 的惊讶计算，没有 Wq/Wk/Wv 的历史寻址，没有跨历史位置的 softmax 权重，也没有 tanh(S)*C 的门控上下文注入。

model.py 第61行确实有 exp 归一化，但它在 **CE 训练损失**中作用于两个输出计数；它不是神经元内的时间注意力。查询输入通道也只是固定时间标记，不是注意力的 Q。前向缓存 cache 用于训练反向传播或保存轨迹，原生更新没有读取过去 cache 来做注意力。

当前模型沿用了 V10 群体电路的拓扑与硬脉冲规则，后来明确扩展了宽度、按 sqrt(6/h) 缩放的递归/输出初始化、可配置保持系数、写入/保持/查询时序及干预接口。这些是已冻结协议中的群体电路研究扩展，不能描述成原始 TAN 时间注意力模型的直接复现或简单等价改写。

## 结论范围与暂停点

现有数据可支持本批明确电路、任务和训练预算中的优化交互、原生延迟行为、状态移植/清零效应以及递归中断后的任务表现。它们不能证明原始 TAN 的注意力、惊讶门控、历史寻址是否有效，也不能说明原始 TAN 网络扩大后的表现，更不能自动转移到原始 TAN 的记忆容量或优越性结论。

用户随后明确授权：“下一步我们先把原始的时间注意力重新放进神经元群体里面……你就这么做”，取代了等待澄清的暂停要求。先完成并封存当前恢复实验，再另起有界群体 pilot，明确原始 TAN-full 的单元等价部分、群体连接与训练适配，并设置 no-attention 对照。当前已完成的 DEND/POINT 模型没有加入注意力，全部历史结果不会回溯改名为原始 TAN 结果。
