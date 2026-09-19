# TAN 研究现状审计与下一阶段执行计划

审计日期：2026-09-17。对象：`C:\Users\李则徐\Downloads\TAN_Final_Submission\TAN_Final_Submission` 及可访问的关联本地材料。

本报告区分三件事：文件是否存在、历史运行是否有证据、科学主张是否被该证据支持。`COMPLETE`、`PASS`、文件名中的 `Final` 均不单独构成科学验收。本次没有重跑冻结科学实验、训练模型、调整参数、修改原始结果或提交论文；新增内容仅为本报告。

## 1. 总裁决

**真实进度：TAN-I 的机制实验与负结果已经归档；TAN-II 已执行到 Sprint 4.3-A 的双通道绑定加显式算术组合器；两篇独立机制论文均已编译成稿，但尚未通过科学一致性和投稿前验收。**

不能把现状描述成“尚未开始向量查询／绑定／组合”，也不能描述成“已经建立泛化关系计算或神经元组合智能”。目前的前沿是：在特定手工设计的查询、键映射、候选掩码及读出规则下，完成了一条受控机制族内的构造性充分条件链。

**下一步优先级：P0 证据与公式勘误 → P1 查询任务可辨识性与动力学承载审计 → P2 有限精度和干扰边界 → 条件满足后才考虑更多操作数或嵌套组合。** 现在不应直接追加乘法、更多头、更大网络或常规分类 benchmark。

以下是研究计划，不是新实验结果。计划中的新文件、验收门槛和预算均为本次建议，尚未执行，也不得冒充历史预注册。

## 2. 扫描范围与证据边界

### 2.1 实际覆盖

- 对当前用户目录下可访问的研究候选位置建立了 46,584 个文件的路径索引，排除操作系统、依赖包、缓存、凭据及无关通信内容；对 1,986 个候选文本、代码、笔记本和日志文件做了 TAN 关键词检索。这是分层发现扫描，不是宣称逐字阅读了 46,584 个文件。
- 主目录共有 367 个非缓存文件：39 个 Python、43 个 Markdown、43 个 TeX、54 个 CSV、12 个 JSON、4 个 NPZ，另含论文、图片、构建日志和脚本。核对了全部 JSON/CSV/NPZ 的结构与关键数值；审阅了核心模型、分析、实验入口、理论、预注册、amendment、claim ledger 和论文相关段落。
- 39 个 Python 文件通过只读语法编译检查。`paper/scripts/qa_pdf.py` 有一个无效转义的 SyntaxWarning，无语法错误。语法通过不等于运行复现。
- 8 个 TAN-II 结果文件记录的脚本 SHA-256 均与当前对应脚本一致；TAN-I Probe-2 的唯一已记录脚本哈希也一致，共 9/9。20 对 `data/experiment_results` 与 `results/tables` 同名 CSV 字节一致。未把这些检查扩大解释为“所有结果自生成以来均未改动”。
- 检查了 `Downloads/Paper_Project (1)/Paper_Project`、`Downloads/新建项目/Paper_Project`、主目录下 TAN/Untitled21/Untitled22 笔记本、散落的 stage2 脚本、旧版论文和总结，以及 `C:/home/z/my-project/download/figures_en` 的早期结果。PycharmProjects、Documents/Codex 等候选工作空间的发现扫描未给出更晚的 TAN 主线。
- 检查了 TAN、Paper_Project、旧项目及研究会话 ZIP 的目录记录；旧 `TAN_Final_Submission.zip` 是 2026-08-16 的 163-entry 包，不包含之后的完整 TAN-I/TAN-II 机制档案，不能作为当前全量备份。
- 检查了早期 Stage 2 Markdown、18 页 PPTX、两份 DOCX 总结的文本；它们属于较早研究口径，不能替代当前状态。
- 读取了两份导出 JSONL 和最新压缩研究会话，重建了 25 轮研究时间线。最新压缩记录中 2026-09-11 只有模型选择事件，没有更晚的科学执行；最后一轮论文完成事件在 2026-09-09。
- 提取了主目录三份 PDF 全文：旧行为论文 39 页、TAN-I 28 页、TAN-II 25 页；另检查了 9 份散落旧 PDF 的首页与版本。生成了关键页预览，但当前会话的图像查看工具拒绝图像输入，因此**没有完成视觉 QA，不能声称排版已全部通过**。

### 2.2 不可覆盖或尚未验证

- OneDrive 中两份无关课程笔记本因云文件提供程序未运行而不可读。未登录或下载未同步云端材料，也没有访问未连接的远程计算集群。
- 其他本地工作空间的发现扫描未找到比 4.3-A 更晚的 TAN 实验。结论限于列明的本地可访问范围，不声称所有外部账户、远端服务器均不存在新结果。
- 未进行干净环境全量实验重放、完整 PDF 视觉检查、外部同行评审或截至今日的系统性新颖性检索。
- 原始网页搜索接口未返回结果；文献定位改为直接读取作者／出版方原始 PDF 和页面。第 7 节仅列已核对的相关先例，不能据此宣称“最新”或“首创”。

## 3. 真实时间线

日期以会话事件和文件证据交叉核对。多份 TAN-II 文档自署 `2026-09-04`，但会话明确记录其创建和执行发生在 9 月 8—9 日。应追加时间线澄清，不回写旧日期，也不据此直接指控数据造假。

| 实际阶段 | 可核对日期 | 已完成的工作 | 当前边界 |
|---|---|---|---|
| 早期行为验证与 Stage 2 | 4—8 月的本地版本 | 习惯化、噪声、趋光、基线、消融、参数扫掠、动力学与 MI；主包含 400/1000/800/2480 行相应指标 | LIF 校准、配对统计、独立训练基线等旧问题仍有记录，不重新当作新主线 |
| TAN-I Phase 1 | 2026-09-04 | 自然状态碰撞、相同未来输入、复位再分叉 | h 的状态非充分性，不是所有 Markov 系统都收缩 |
| TAN-I Phase 2 | 2026-09-04 | 3 seeds、374 events 的局部响应几何及幅度控制 | 存在判定分支缺口；有效秩不等于内禀维度 |
| Probe-1 v3 与 amended audit | 2026-09-04 | 3-seed 重建与有限支持结果；保留退化单元格 | 不是干扰鲁棒性或普遍记忆优势 |
| Probe-2 | 2026-09-04 | 可辨识性审计结束，`TERMINATED / AUDIT_FAILED` | 正标量核的排序限制；不是待调参的未完成实验 |
| TAN-I 独立论文 | 2026-09-04 | `paper/`，28 页，8 主图、6 表、58 条参考文献记录 | 成稿，不等于投稿验收完成 |
| TAN-II 4.1-C/D | 2026-09-08 | 静态 E-I 调谐与离散动力学、零翻转对照 | 有调谐不等于有路由 |
| TAN-II 4.2-A/B/C/D | 2026-09-09 | 几何容量、向量核重排、软值传递、WTA 极限 | 均有明确机制族和读出层级限制 |
| TAN-II 4.3-A | 2026-09-09 | 双通道绑定、显式加减组合器、反事实与 Monte Carlo | 条件正确，不是泛化组合计算 |
| TAN-II 独立论文 | 2026-09-09 | `paper2/`，25 页、10 节、6 主图、3 表、16 个 bibliography entries | 存在公式、统计转录和主张范围问题 |

根目录 README、CHANGELOG、`NEXT_AGENT_BRIEF.md` 和总账主要停在 TAN-I。它们是历史快照；TAN-II 的现状必须合并各 sprint 的账本，而不是按根目录交接文档从头再做。

## 4. 已完成的科学内容

| 证据块 | 直接证据 | 本次可接受的表述 |
|---|---|---|
| Phase 1 | 160,000 条候选历史，每模型 1,500 碰撞对；另有 843 对共同复位子集 | 膜电位 h 单独不是完整状态；窗口本身已提供历史依赖，非 attention 独有 |
| Phase 2 | zF：B4 dD=2.088918，B3=1.483525；zC：B4=1.589085，B1=1.673962 | 所测局部几何发生变化；不支持“比 LIF 更高的内禀维数”或由几何直接推出能力优势 |
| Probe-1 | B2 distractor 约 0.843；B4 约 0.389；归档 pooled 对比 +0.057041，CI [0.038749,0.075775] | 对不完全健康的 B3 的有限条件性优势；保留 seed 衰减和 class-coverage 限制 |
| Probe-2 | B2 0.7789、B3 0.7094 捷径；同历史 argmax flip=0；JSD 检验 p=0.9201 | 冻结正标量核没有通过 query-conditioned routing 可辨识性审计 |
| 4.1-C/D | C1 峰 x=1.00、h=0.200；C2-C4 按冻结阈值无合格峰；6 布局零翻转 | 静态对手结构可产生非单调调谐；该核级排序边界没有被 E-I 打破 |
| 4.2-A | 317 checks：316 PASS，1 PROTOCOL-FAIL / NUMERICAL-PASS；另 36 shortcut checks | 对已说明分布和点积族的几何性质进行解析与数值核验；不是全绿的原始预注册 |
| 4.2-B | 27+48 checks；规范边距 +0.258743/-0.155910；集合翻转 0.442533 对网格基准 0.444527 | 受测核实际发生反事实重排；不能把“翻转率”当作“正确寻址率” |
| 4.2-C | 42 checks；C_events=6.742691/7.155595，软误差约 1.74/1.84 | 在事件归一化读出层实现配对敏感的软值传递；E-I 净驱动可反转其方向 |
| 4.2-D | 77 checks；一热分支精确交付，gamma*=5.098015 | 本 softmax 锐化族中软／硬读出可分；一热选择不修复错误路由 |
| 4.3-A | 79 checks；3×10,000 历史；双正确路由 6,608 条；该子集组合误差为零 | 两个正确操作数送入显式加减器后正确计算；无条件组合仍受上游绑定限制 |

其中“检查数”包括结构断言、公式核验、数值比较和部分直接设为 True 的构造性声明，不等于同等数量的独立经验检验。

## 5. 必须优先处理的审计发现

### A01 · P0 · 查询旋转公式与实际代码不一致

`paper2/sections/s2_framework.tex:199` 同时写 `Q=cS·u(omega S)` 与 `u(s)=(cos(omega s),sin(omega s))`；相同重复出现在 `docs/TANII_SPRINT4_2B_PREREGISTRATION.md:38`。字面上角度是 omega²S。实际代码 `sprint4_2_vector_qk.py:97` 和 `:123` 使用角度 omega S，预注册同文件第 63 行的预测也采用后者。

本次直接计算闭式表达式得到：

| 查询 | 代码与数值预测的边距 | 论文公式字面边距 | 后果 |
|---|---:|---:|---|
| Q=3 | +0.2587427243 | +0.7055409799 | 两者均选 A |
| Q=4 | -0.1559101334 | +0.8425586282 | 代码选 B，字面公式仍选 A |

这会使读者按论文复现时失去核心翻转。最小勘误是明确 `u(theta)=(cos theta,sin theta)`，保留 `Q=cS·u(omega S)`；也可采用等价的另一套记号，但必须全篇一致。应追加勘误说明原预注册内部不一致，而不是改代码使其匹配错误文字。

### A02 · P0 · Phase 2 的 D 判定是分支覆盖缺口

`effective_dimension.py:967` 的 stable 由现存 CSV 算得 True：dD_delta=1.088918，CI 下界=1.873931。A 因 B4-zC<B1-zC 失败；B 因 B4-zF>B3-zF 失败；C 因 B4-zC 与 C0 的差 0.534135 大于 0.5 失败。第 984 行把所有剩余情形直接分给 D，第 1000 行却将 D 解释为“没有稳定事件锁定扩张”。

因此原日志确实记录 `OUTCOME: D`，但其说明不等于数据证明没有扩张。这不是重新发现正结果：原始数值保留，问题是判定类别不完备。应新增 `UNCLASSIFIED_MIXED_CASE` 或等价解释性勘误，保留原始 D、冻结脚本和日期。未来判定器必须显式区分 `not stable` 与 `stable but not A/B/C`。

另：事件 log-trace 差=1.094463；各模型 event-minus-quiet 变化量之差=1.354674。两个数字并不互相冲突，应明确统计对象，不把后者含糊写成一项独立“配对检验”。

### A03 · P0 · 论文发生均值／中位数、实验层级和样本量混写

1. `paper2/sections/s3_state_geometry.tex:29` 把 K 的均值及其 CI 写成 median。CSV 的真实值如下，不能仅修改某个数字而保留错误统计名称。

| 模型 | K_mean | K_median |
|---|---:|---:|
| LIF | 0.500000 | 0.500000 |
| Buffer | 4469.536489 | 895.529570 |
| TAN-noAttn | 12347.886007 | 1369.079508 |
| Full TAN | 20465.855578 | 2947.724430 |

2. `paper2/sections/s5_routing_geometry.tex:185` 把纯几何两候选示例的 JSD=0.0751/0.0647 放进运行模型的描述。4.2-B JSON 中全窗口规范 JSD 实际为 M0=0.009462259、M2=0.008726529。定性结论不变，数字必须归还各自实验和候选集。
3. 132 个 E3 条件仍满足冻结判据，但最大单 seed 绝对偏差是 0.009266667；按三个 seed 先平均后的最大偏差是 0.004666667。论文的“maximum deviation ≈0.004”未区分统计口径。机读字段 `theta_deg` 在这些行存的是 theta/pi（例如 0.6667），不是度数，也需在派生表中标注。
4. Probe-1 的 6,000 是全部测试试次数。生成器每 seed 的 distractor-present=666、distractor-catch=333，共 999；三 seed distractor 条件共 2,997。bootstrap 从 6,000 条抽样再按条件过滤并非自动无效，但结果应同时报告总数、条件数及独立 seed 数。
5. 4.3-A 的条件零误差覆盖 6,608 条 both-routed 历史（2188/2258/2162），不是 30,000 条全部历史都正确。全体历史的加法平均误差约 1.116、减法约 2.001，不能省略。

### A04 · P0 · 理论定义和最小性表述需要收窄

- `paper/sections/methods.tex:37` 写 Markovian claim 预测 K<=1 收缩。Markov 性不蕴含收缩，例如 h_next=2h 本身是 Markov 的。状态非充分性应由完整状态／投影的构造性论证及精确复位对照支撑；大 K 或近碰撞本身不是对所有 Markov 模型的排除证明。
- 需要始终写“膜电位 h 单独不足”，而非“完整状态不 Markov”。若记忆被定义为完整即时状态无法包含的历史，就错误排除了普通有记忆的 Markov 状态模型。
- softmax 在所有有限 logits 上都有正支持，所以“支持集不随 query 改变”不能区分 saliency 与 routing。应使用排序、目标匹配和配对传递等各自指标。“排序不变”等价于“核必然 rank-1 可分”也过强，至多有特定充分条件。
- 标量排序定理须单独处理 S=0 平局；软混合障碍须限定两个不同值、相应归一化候选集等条件。多候选可能发生数值相消，不能把某次输出相等当成结构性精确绑定。
- 单 winner 的冻结读出不能同时交付两个独立操作数，不等于所有单头结构都不能计算两值函数。例如两值均匀平均再乘 2 就能求和；该反例不实现本文三误差定义的绑定，但足以说明必须区分功能性求和与“经两个已绑定操作数计算”的机制定义。
- `paper2/sections/s9_ladder.tex:141` 声称全阶梯没有增加 state space/nonlinearity，不符合引入 E/I 两单元、向量表示、K/V 解耦与双通道的事实。固定 W 不等于资源或状态维度不变。

### A05 · P0 · 声称继承的 softmax 约定不符合代码

`paper2/sections/s2_framework.tex:25` 称每个 sprint 均继承分母 delta=1e-9；实际 4.2-B 的 `softmax` 使用 exp(logits-max)/sum(exp)，没有该偏移。TAN-I 的偏移位于稳定化之后，与论文未减 max 的写法在严格数值上也不同。小偏差通常不改变主结论，但“逐字节相同”“严格凸组合”“精确归一化”不可混写。应把数学定义、数值稳定化、偏移和容差分别列出，不回写冻结代码。

### A06 · P0 · 时间与 amendment 的证据等级不能混用

原始会话支持理论／设计先于后续正式 MC 的工作顺序，但不是所有 amendment 都先于任何运行。例如 AM3 明确在规范审计首次运行后、正式 MC 前形成；AM4/AM5 也有确定性检查和调试记录。应分别标记“确定性开发阶段修正”“正式 Monte Carlo 前冻结”“运行期修正”，不能统一宣称六项都在任何运行前完成。本地预注册也不是外部时间戳注册。

哈希一致只能证明当前脚本与记录对应，不能证明所有历史数字独立于开发调试、或全档案从未修改。新建 manifest 的时间必须是现在，不倒填成实验开始日期。

### A07 · P0 · 成稿不等于可直接投稿

- TAN-I PDF 首页保留 email placeholder；TAN-II PDF 首页同时保留 correspondence placeholder 和 `Draft v0.1 — Sections 1–2 only`，尽管正文已有十节。
- README 指向 `main.pdf`，当前实际 PDF 已重命名。根目录构建脚本仍主要编译旧 39 页行为论文；“run all”也只覆盖早期六组行为实验，不覆盖当前机制链。
- TAN-II 有 16 个 bibliography entries，其中含两个本地档案条目；没有据此核实投稿、公开归档 DOI 或同行评审状态。不能把内部 ACCEPTED/FROZEN 当期刊接收。
- 原稿的 QA 主要是文本／编译检查。TAN-I 日志仍有 11 个 overfull hbox，TAN-II 有 2 个，并有 underfull 提示。文本能提取或编译零错误不等于完整视觉验收。
- 旧总结、旧 PPT 的性能和生物类比表述不能直接复制到当前机制论文。若继续以旧行为论文投稿，其独立基线、校准和配对统计问题需要另开工作包，不占当前主线的默认预算。

## 6. 真正尚未解决的研究问题

### 6.1 当前组合证据停在算子／读出层

`code/experiments/sprint4_3/sprint4_3a_composition_probe.py:96` 的 `run_history` 直接计算窗口均值、两个键得分、winner、被选值，随后第 110 行执行 `C1+C2` 或 `C1-C2`。这一路径没有膜电位的逐时间递推、spike/reset 或通过神经元输出完成的算术运算。4.2-B 的关键证据也来自窗口核和条件固定点计算；4.1 确实有离散／情节动力学，不能据此替代后续节点的动态验证。

因此 `E_comp | both-routed = 0` 是正确但受限的构造性充分性：准确操作数送入准确加减器，结果准确。它不能独自建立学习到的组合能力、泛化组合智能，或膜／脉冲动力学对组合的因果必要性。所谓两个“正交轴”目前是模块化依赖关系，不是已经估计出的普遍统计独立性。

部分 shortcut flags 被直接设为 True。若构造确实排除了某通道，它们可以作为结构性说明；不能替代独立泄漏检测器。未来需把 `STRUCTURAL_BY_CONSTRUCTION`、`NUMERICALLY_TESTED`、`EXTERNALLY_CHECKED` 分开。

### 6.2 固定查询与随机键之间缺少泛化寻址的身份约定

4.3-A 的 Monte Carlo 固定 Q1=4.5、Q2=5.3，却让 KA/KB 独立同分布随机化。A/B 是生成器身份，现有固定数值 cue 没有携带任意新键的地址。交换 A/B 标签与配对后，输入的可观察结构不告诉系统哪一项“应叫 A”。按交换对称性，单通道的身份命中率约 1/2 并不意外。

实测 both-routed 为 0.2188/0.2258/0.2162；网格基准 0.220725。**这不能被解读为已通过语义寻址任务后的约 22% 泛化准确率，也不能只靠调 omega 来修复。** 该随机化实验本来验证的是条件误差分离。若下一步要声称 generalized routing，必须先说明 query 怎样指向新 key，同时不泄漏 value。

优先研究问题应是：

> 当目标地址、惊奇强度和历史上下文分别受控时，哪种最小兼容性函数能够稳定选择所指对象？选出的值在哪一层被实际保留和使用，神经元动力学是否提供了静态检索器没有的作用？

### 6.3 表示变换不等于“单独提高维度”

现有成功同时依赖旋转查询、非线性矩曲线键和归一化；同为 d=2 的未归一化控制就不翻转。因此可归因于这套表示机制，而不能唯一归因于 ambient dimension。`phi(x)` 和旋转 query 各由一个标量参数生成，还应区分嵌入维度、内禀参数自由度与物理状态资源。

候选最小对照应包含标量距离分数 `-(q-k)^2`，而不只比较正标量乘积与二维点积。前者在 q 等于目标键、键互异时本就可以内容匹配；这不是对当前冻结数据的修改，而是未来比较族必须说明的边界。

### 6.4 原始数据与复现仍有缺口

TAN-II 当前主要保留汇总 JSON、逐检查 CSV、曲线和图，未见完整保存的 30,000 条 composition 输入／逐试次输出。固定种子和代码允许再生成，但“可以再生成”不等于“原始逐试次档案已齐全”。新实验必须保存输入、目标、候选 mask、原始 logits、选择、各层输出、条件标记和 run manifest；不能只存 PASS 表。

## 7. 文献与贡献定位

已直接核对的原始来源：

- **R1** Moses S. Charikar, *Similarity Estimation Techniques from Rounding Algorithms*, STOC 2002, §3 Random Hyperplane Based Hash Functions。原文给出随机超平面哈希同号概率 `1-theta/pi`；异号概率即 `theta/pi`。所以 TAN-II 的角度翻转律应明确作为经典几何恒等式在当前机制中的应用／复核，而非未经定位的独立数学新定律。
- **R2** André Martins and Ramón Astudillo, *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification*, ICML 2016, PMLR 48:1614–1623。该工作给出可产生稀疏概率的替代映射。因此“有限 softmax 锐化不能给出一热”不能扩大成“所有有限机制都不能精确选择”。
- **R3** Imanol Schlag, Kazuki Irie and Jürgen Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021, PMLR 139:9355–9366。用于对照 attention、动态记忆和有限容量的已知联系；其线性 attention 结论不自动适用于本项目的 softmax TAN。
- **R4** Hubert Ramsauer et al., *Hopfield Networks is All You Need*, arXiv:2008.02217v3，2021-04-28 版本。用于对照 attention 与现代关联记忆的联系；不能由类比替代 TAN 的机制证据。

当前 Paper 2 bibliography 未包含以上这些直接相关的比较项。至少应补充 R1/R2；R3/R4 用来确定“attention + neuron/memory”叙事的范围。尚需专项检索核查最近研究，不能把这四项当作完整 related work。

最可能保留的贡献定位是：**一套受约束机制族的可辨识性审计方法、阴性对照和清楚的结构边界，以及 E-I 对绑定后值传递通路的条件性影响。** 是否足以支撑独立论文的新颖性，仍需完成文献矩阵与外部复核；本报告不承诺目标期刊或接收概率。

## 8. 执行路线与验收门

T0 定义为用户批准实施该计划的日期。以下人日和 CPU 预算是建议上限，不是已测工时。每个工作包先交付可检查证据再进入下一包。执行者与复核者是责任角色，不表示本次已使用多代理。

| 优先级 / 包 | 时间预算 | 输入 | 必须交付 | 验收及停止条件 |
|---|---|---|---|---|
| P0-A 统一现状与保全 | 0.5–1 人日 | 当前 367-file 档案、会话、各 ledger | 当前态索引、内容哈希 manifest、chronology、统一 claim ledger | 100% 主张连到文件和适用模型；来源不明或相同 claim 引用冲突则停止投稿与新实验 |
| P0-B 勘误与独立核对 | 1–2 人日 | 第 5 节 A01–A07 | 独立数值核验表、原文→勘误对照、定义和定理的假设清单 | omega/候选集/归一化/统计单位/分支逻辑全部一致；保留旧 FAIL；未解决科学矛盾不得进入 P1 |
| P0-C 贡献与稿件检查 | 1–2 人日，可与保全后工作并行 | 两篇论文、R1–R4、补充检索 | 文献重叠矩阵、带来源的正文修订副本、构建与 QA 记录 | 无占位符、无越界必要性／智能主张；公式与数字对应；视觉 QA 未过不得标 submission-ready |
| P1-A 地址可辨识性 | 2–3 人日；首轮 ≤2 CPU 小时 | 已冻结的新任务定义与最小比较族 | 新预注册、oracle/shortcut 审计、逐试次数据、seed 级结果 | 先过 G1 再跑正式矩阵；若 target 身份不可由合法输入确定，停止并判任务无效，不判模型失败 |
| P1-B 动力学承载 | 2–3 人日；≤2 CPU 小时 | P1-A 的合法任务、明确读出端口 | static vs dynamic 同输入对照、各层因果消融、值传递追踪 | 若 C 层外接算术已解释所有结果，停止“神经动力学实现组合”主张；可按静态算子研究收口 |
| P2 有限精度与干扰 | 2–3 人日；≤2 CPU 小时 | 通过 P1 的任务与固定机制 | margin/gain/noise/候选数边界，负结果，误差传播图表 | 无结果驱动调参；若不存在预先定义的可用区间，报告边界并停止扩展 |
| P3 条件性扩展 | 另行批准，不预分配算力 | P0–P2 全部裁决 | 三操作数或一个新算子的独立研究协议 | 不因 4.3-A 的 79 PASS 自动 GO；修改 W、端口、读出或记忆均算新自由度 |

### P0 的具体落地要求

1. 新增主索引，不覆盖历史 NEXT_AGENT_BRIEF。每个工件记录相对路径、字节数、当前 SHA-256、生成时间证据、来源脚本、父 run、冻结状态。既有缺失哈希标 `NOT_RECORDED_HISTORICALLY`，禁止补造历史哈希。
2. 新总账至少含：claim ID、精确命题、适用模型／域、理论假设、证据文件与行／字段、原状态、本次审查状态、可允许措辞、反例／shortcut、未完成检查、停止条件。将 NOT_TESTED、NOT_CLAIMED 与被反例推翻的 REJECTED 分开。
3. 为 A01/A02/A03/A05 建立独立读取现存数值的审计程序；输出新报告，不覆盖原 summary。`theta/pi` 标签、候选集合、logTr 定义、mean/median、条件样本数均必须机器检查。
4. 修订论文必须在副本或明确授权的派生稿中进行，保留旧版与勘误记录。Phase 2 改“解释与分类”不是重新筛选实验；Probe-1 的试次 bootstrap 不可升级为跨未知种子的一般性结论。
5. 确定 Paper 2 是“算子级构造与审计框架”还是要主张“动态神经元实现”。如选择后者，P1-B 为投稿前置；如选择前者，应立即收窄相应标题、摘要和结论，不必为了保留旧叙事强造新实验。
6. 文献矩阵按“已知定理／本项目复核／新增机制事实／未证猜想”逐项归类。若核心只剩经典恒等式和显式算术节点，则停止扩大理论创新主张，转为方法性或技术报告定位。

## 9. 下一项科学实验的具体协议草案

建议名称：**TAN-II 4.3-B：查询地址可辨识性与计算承载层审计**。必须是新分支／新目录，不覆盖 4.3-A；本节参数为待批准的前瞻设计。

### 9.1 先冻结问题，再写代码

分开三个判据：

1. **Reordering**：换 query 是否改变 winner。
2. **Address correctness**：winner 是否是 query 真正指定的对象。
3. **Delivered computation**：该对象的值是否经声明的 C/A/h/spike 端口进入组合，且满足三误差判据。

三者不互换。4.2-B 已测第 1 项，4.3-A 主要测正确绑定条件下的第 3 项；新任务补第 2 项及端口因果性。

### 9.2 数据生成与身份契约

- 候选数先取 N=2、3。保留 W=5：两 query 时最多放 3 个事件。N>=4 必须增加 W 或改变事件编码，属于独立的新资源变化，本轮不做。
- 键 KA... 独立采自 U(0.3,2.5)，初始无噪校准限定最小键间隔 0.05；值独立采自 U(-3,3)，不由键、位置或 query 决定。间隔约束是采样前的任务定义，记录 rejection count，不得测试后筛选。
- 在每条历史中均匀选择两个不同目标索引 j1/j2，query 的内容 cue 分别是所请求的 key，不携带 value。角色 A/B 必须来自这项显式请求，不能继续使用不可观察的生成器内部名称。
- 将内容 cue 和产生 S 的 novelty/amplitude 通道分别记录。独立地址端口是新增自由度，必须明确记账；不把它描述为冻结 TAN 已有功能。纯核检查可固定 S=1 作为独立控制，完整模型则报告自然 gate-open coverage。
- 事件位置、目标角色顺序、键值配对、无关 distractor 独立随机化。query 槽值为零，候选资格由输入类型给出；模型不得读取隐藏 target ID、未送入系统的 V 或审计器中间量。
- 样本外分布预先限定：低幅键 U(0.05,0.3)、高幅键 U(2.5,3.5)，同样记录间隔约束；不将这些测试分布用于调整映射或选择 omega。
- 观测 query 噪声取标准差 sigma×2.2，sigma∈{0,0.01,0.05}，在同一原始 scalar key cue 上加噪后再交给各表示映射。q<=0 等域外情况的处理必须在协议中固定，并计入覆盖率与无条件错误，不能静默丢弃。

### 9.3 比较族与归因纪律

至少保留以下角色，参数数量、状态维度、候选 mask、可见端口逐项列明：

| 模型角色 | 实现／审计目的 | 不允许的解释 |
|---|---|---|
| Query-blind buffer/uniform control | 对同一历史输出不依赖地址，用作泄漏和依赖性阴性对照 | 不能故意隐藏其合法可用信息后宣称普遍优势 |
| 正标量乘积核 | score=cS·k；复核固定排序边界 | 不能把其失败推广到全部一维兼容性函数 |
| 带符号标量核 | 符号控制须在运行前固定；用 N=3 中间键检验仅极值反转的限制 | 不把二候选 flip=1 当任意内容地址能力 |
| 冻结 S-旋转＋归一化键参考 | 保留旧代码为参考，明确其没有独立地址 port | 不因新任务失败修改旧 omega 或删除该对照 |
| 标量 metric attention | score=-(q-k)^2，先证明无噪 q=k_target 时可检索 | 若它成功，不再声称二维向量在所有族中必要 |
| 显式地址 vector-QK | query=phi(q)，key=phi(k)，与旧 S 驱动方向分开 | 不把端口、非线性映射、归一化的联合变化只归因于维度 |

先做核层同输入比较，再考察 gate 和动态读出。不要一次改变输入端口、兼容性函数、normalization 和膜方程后声称识别了某一个因素。为使草案可直接落地，符号阴性对照默认固定 s(q)=+1 当 q>=1.4，否则为 -1；1.4 是校准键区间中点，不从测试表现选择。该对照用于边界诊断，不声称是最优标量方法。相同分数／零 gate 默认 abstain，计入无条件错误及覆盖率，不静默选一个方便的候选；如需研究其他 tie 规则，应单独注册。

### 9.4 样本量、保存项与统计

- 新种子建议 `2026091700` 到 `2026091709`，10 个独立 seed；不是历史三 seed 的再包装。每个主要条件每 seed 1,000 条历史，先进行无噪 N=2/3 校准，再跑声明的噪声和样本外矩阵。
- 主端点为 joint correctness：两目标均正确寻址、两 binding errors 与 composition error 均在阈值内。硬选择／结构不变量用绝对容差 1e-12；已舍入的闭式基准用 1e-9。有限 softmax 的性能误差另用相对值范围的容差，不混成“数学精确”。
- 同时报 per-target accuracy、both-routed 分子／分母、gate coverage、abstention/tie、选错时与选对时的输出误差、无条件误差、margin 分布。条件零误差必须同时给出条件覆盖率。
- 同一 seed 和历史用于各模型配对比较。保存 seed 级结果与 paired differences；概率的区间和跨 seed 的不确定性分开，不能用数万相关试次冒充数万个独立模型复现。主检验与探索性切片分开，多重比较方案在运行前固定。
- 每条记录保存：run_id、seed、原始序列、key/value、query content、novelty、mask、target ID（仅审计器）、logits、winner、C/A/h_pre/h_post/spike、组合输出、真值、反事实组 ID、错误分解、排除／abstain 原因。新原始数据使用 CSV/NPZ 等现有可读格式，无需引入大型框架。

### 9.5 G1 可辨识性关卡

正式矩阵开始前必须全部通过：

- 合法 oracle 只读取同样的 key cue、候选 keys 和 values；无噪、互异键校准下得到 100% 正确身份和输出。失败则判协议／实现问题并停止。
- 固定 keys/query，仅交换 values，真值和模型应有正确的 pairing 响应；固定 values，随机 target/request，应排除常数、位置、query-only 和幅值排序捷径。
- 隐藏 target ID 只供评分，绝不传入模型。若合法输入对应多个不同目标且不能区分，判 `TASK_UNIDENTIFIABLE`，停止，不给模型记能力失败。
- 阴性泄漏检测器若明显高于其已推导 chance，应首先审计生成器和端口。建议预注册触发线：95% 区间下界超过 chance+0.02；该线只用于泄漏调查，不能结果出来后改。
- Query swap、value swap、key-pair permutation、position permutation、零值／相等值陷阱、gate-closed／tie 均有单独断言。结构性 True flag 不计为已运行测试。

通过只表示任务可解释，不表示 TAN 获胜。模型失败可以成为有效的边界结果。

## 10. 动力学、有限精度与后续扩展

### 10.1 P1-B：确认计算发生在哪里

当前 `run_history` 的依赖关系已经表明其组合输出不依赖 h/spike，因此无需再跑大量消融来“发现”这一点。先决定论文是限定为算子级研究，还是另建真正的在线动态实现。只有选择后者才执行以下新实验：

1. 明确 C、门控电流 A、复位前 h、复位后 h、spike train 五个端口，不再把组合实数输出和二值 spike 都模糊记成同一个 y。
2. 对同一事件流比较静态 key-value 检索加算术、完整逐步状态更新、移除／重置膜记忆、移除窗口历史、gate 控制。每次只改变一个载体，不改变目标编码。
3. 加入预先声明的读出延迟 {0,1,2}，记录事件是否仍在 W=5 窗口。不得把已滑出窗口与尚在窗口的样本混报，也不得让响应分类器直接读取原始 value。
4. 若使用 spike train 表示连续值，先给出编码、读取窗口和解码误差定义；单个二值 spike 不可能被直接宣称为任意实数的精确交付。若增加外部解码器，单独列为新模块。
5. 判断动力学贡献采用等价性界限或确切依赖关系，不以“不显著”替代“完全相同”。若静态模块已解释全部所测端点，停止神经动力学必要性主张，按该结果收口。

### 10.2 P2：把有限误差边界算清楚

先用解析式节省无意义扫参。对两不同值、winner margin m>0、事件归一化 softmax：

`D = |VA-VB| / (1 + exp(gamma*m))`。

因此可预先推导给定 epsilon 所需的 gamma*m，再验证有限增益的范围；不需要反复调 gamma 直到出现浮点意义上的“零”。对于更多候选，使用 losing mass 与最小 margin 的误差上界，并区分真正的一热算子、指数下溢和数值舍入。

- 扫描 gamma∈{1,2,5,10,100}，一热 argmax 作为独立理想参照，不当作已实现的生物有限增益极限。
- 同时记录 logits margin、键间距、噪声、S 和 normalization；小 margin、tie、S=0 必须报告。有限网格求积结果与解析真值分开，使用加密网格／独立积分验证误差，不能称固定 2001² 网格本身为严格闭式。
- 比较 sparsemax 等替代选择规则仅用于检验“softmax 族边界”，不将替换结果写回旧 WTA 实验；若不研究必要性，可保留为理论反例而不扩展实验。
- 算子只保留加／减，报告 binding 错误传到 composition 的机制；不因为 addition 对交换不变就认为缺少 pairing，也不因 error cancellation 判断绑定成功。

### 10.3 进入 P3 的条件

只有在 G1 通过、目标地址契约稳定、无噪校准全部正确、关键端点及计算载体已明确、噪声／精度边界已解释后，才申请 P3。首项仅选择“三操作数”或“一种新算子”之一。增加输入数需要预算 W 和状态存储；改变运算需要说明算子由外部指定、学习还是系统内生成。

如果所选新算子仅是继续调用正确的内置算术函数，而没有新的可辨识机制问题，则不启动该 sprint。无新机制问题是合理停止理由。

## 11. 可执行的复现入口

以下命令是 **P0 获批后的隔离复核步骤**，本次未执行。禁止在原目录直接调用旧 `run_all_experiments.bat` 来“更新”结果；它会写入旧结果路径，而且不是当前研究链的完整入口。

### 11.1 建立一次性副本

```powershell
$ErrorActionPreference = 'Stop'
$tanSource = (Resolve-Path -LiteralPath 'C:\Users\李则徐\Downloads\TAN_Final_Submission\TAN_Final_Submission').Path
$tanReview = Join-Path $env:TEMP ('tan_review_' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tanReview | Out-Null
$tanSnapshot = Join-Path $tanReview 'snapshot'
Copy-Item -LiteralPath $tanSource -Destination $tanSnapshot -Recurse
Set-Location -LiteralPath $tanSnapshot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = Join-Path $tanReview 'matplotlib_config'
python -X utf8 -B -c 'import sys,numpy,scipy,matplotlib; print(sys.version); print(numpy.__version__,scipy.__version__,matplotlib.__version__)'
```

历史科学运行环境为 Python 3.12.7、NumPy 1.26.4、SciPy 1.13.1、Matplotlib 3.9.2。环境不同则先记录并决定兼容性复核范围，不自动改系统环境或安装 torch。`requirements.txt` 是宽松下限，不是环境锁。副本只用于新复核，不替换原归档。

### 11.2 重放 TAN-II 的现有审计

```powershell
$tanJobs = @(
    @{Script='sprint4_1cd_ei_dynamics.py'; Name='s41cd'},
    @{Script='code/experiments/sprint4_2/sprint4_2_geometry.py'; Name='s42a'},
    @{Script='code/experiments/sprint4_2/sprint4_2_shortcut_audit.py'; Name='s42_shortcuts'},
    @{Script='code/experiments/sprint4_2/sprint4_2_vector_qk.py'; Name='s42b'},
    @{Script='code/experiments/sprint4_2/sprint4_2_routing_probe.py'; Name='s42b_probe'},
    @{Script='code/experiments/sprint4_2/sprint4_2_binding_probe.py'; Name='s42c'},
    @{Script='code/experiments/sprint4_2/sprint4_2_hard_binding.py'; Name='s42d'},
    @{Script='code/experiments/sprint4_3/sprint4_3a_composition_probe.py'; Name='s43a'}
)
foreach ($tanJob in $tanJobs) {
    $tanOut = Join-Path $tanReview $tanJob.Name
    $tanLog = Join-Path $tanReview ($tanJob.Name + '.log')
    $tanOutput = & python -X utf8 -B $tanJob.Script --outdir $tanOut 2>&1
    $tanExit = $LASTEXITCODE
    $tanOutput | Tee-Object -FilePath $tanLog
    if ($tanExit -ne 0) { throw ('STOP: ' + $tanJob.Name + ' exit=' + $tanExit) }
}
```

退出码零只是第一层验收，还必须逐项比较新旧 JSON 数值、参数和检查集合，不比较 PNG/PDF 的二进制哈希来判科学数值复现：

| 入口 | 期望审计形态 |
|---|---|
| s41cd | C1 PASS，C2–C4 无合格峰；P2 零翻转 |
| s42a | 316 PASS + 1 PROTOCOL-FAIL / NUMERICAL-PASS，总数 317 |
| s42_shortcuts | 36 PASS |
| s42b | 27 PASS |
| s42b_probe | 48 PASS |
| s42c | 42 PASS |
| s42d | 77 PASS |
| s43a | 79 PASS；同时保存约 22% both-routed 和非零无条件误差 |

原始理论审计单独运行，因为预期退出码是 1，不能被普通的“全部退出码必须零”覆盖：

```powershell
$tanTheory = & python -X utf8 -B code/experiments/sprint4_2/sprint4_2_theory_audit.py 2>&1
$tanTheoryExit = $LASTEXITCODE
$tanTheory | Tee-Object -FilePath (Join-Path $tanReview 's42_theory_original.log')
$tanFailures = @($tanTheory | Where-Object { $_ -match '^\[FAIL\]' })
if ($tanTheoryExit -ne 1 -or $tanFailures.Count -ne 1 -or $tanFailures[0] -notmatch 'E4\(iii\)') {
    throw 'STOP: original theory-audit failure set changed'
}
```

保留这一失败及 AM1；如输出格式改变，人工核对原始 failure set，不修改科学阈值使脚本变绿。TAN-I 的冻结实验本轮不要求重跑；Phase 2 分支问题通过读取现存 summary 与代码即可核对。若后续需要再生成其数据，必须另行明确授权并在副本里保存新 run。

### 11.3 论文构建在副本中完成

先完成勘误，再在副本的 `paper` 或 `paper2` 中分别执行：

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

每条命令都检查退出码，随后检查 undefined references/citations、占位符、页数及全部页面。包缺失时报告依赖，不静默联网安装或使用 `--enable-installer`。新 `main.pdf` 是派生输出，不覆盖重命名的历史 PDF。

### 11.4 待建工件清单

以下是 P0/P1 批准后才创建的文件，不应声称现在已存在：

- `docs/CURRENT_RESEARCH_STATE.md`、`docs/MASTER_CLAIM_LEDGER.csv`、`docs/RESEARCH_CHRONOLOGY.md`。
- 独立 audit 目录中的 `manifest.json`、`claim_alignment.json`、`numeric_diff.json`、`environment.txt`、`execution.log`。
- `docs/TANII_SPRINT4_3B_PREREGISTRATION.md` 与版本化的新生成器／独立 oracle／只读评分器。
- 新 run 目录中的 `trials.npz`、`per_seed.csv`、`counterfactuals.csv`、`summary.json`、`RUN_MANIFEST.json`，包含 protocol/code/input/result hashes 和真实生成时间。

## 12. 全局停止条件

### 硬停止

- 原文件哈希与已记录版本不符，或同一 claim 对应多个不一致来源：停止发布和扩展，先确定证据版本。
- 科学公式与实现不一致、oracle 无噪失效、目标不可辨识、隐藏标签或 value 泄漏：停止正式运行；保留失败，修复须新 amendment／新 run。
- 为通过而更换种子、丢弃失败单元格、改查询编码、改阈值或只报条件成功：停止该轮的确认性解释；如继续则标探索性研究并使用新的冻结协议。
- 以数值下溢、均值相消、gate-closed、空条件子集或 Q=0 平局伪造 exact binding：停止并单独报告退化情形。
- 任一模型的计算依赖了本不该可见的未来输入／未传入端口：停止因果主张。
- 每个 P1/P2 工作包超过 2 CPU 小时、峰值内存超过 2 GB，或同一科学级问题连续两次修改仍不能解释：停止扩大规模，先提交最小复现与资源报告。无需 GPU；不自动购买算力或提交外部服务。

### 科学 NO-GO，仍然可以形成结果

- Metric 或静态检索器解释了全部功能：停止 TAN 独有或二维全局必要性的主张，按比较与边界结果写作。
- h/spike 消融不改变声明端点：停止该任务中“神经动力学必需”的主张，而非隐藏阴性结果。
- 明确的地址任务上当前 S-旋转映射不能稳定寻址：归档为泛化边界，不通过不断调 omega 改写历史成功条件。
- 所有新结果只是验证写进程序的算术节点：停止添加更多算子，优先整理方法、定义和限制。
- 文献复核显示主要定理均为已知结果，且没有足够新增机制事实：停止独立理论创新包装，选择方法性／技术报告定位或等待新的可辨识问题。

### 允许完成并收口

P0 全部阻断项有明确勘误或撤回、证据总账可追踪、结论与载体／比较族一致、复核日志完整、稿件无未声明边界，就可以结束这一轮。**不需要所有假说都通过，也不需要继续升级模型才能算研究完成。** 投稿动作、公开上传、外部协作及增加算力另需用户指示。

## 13. 建议排程与首轮交付

| 时间 | 唯一重点 | 当日／阶段交付 | 下一步决定 |
|---|---|---|---|
| T0–T0+1 | 保全与证据总账 | 当前态、manifest、时间线、问题清单 | 有版本冲突先停 |
| T0+1–T0+3 | A01–A07 勘误、文献定位 | 逐项数值核对、定义清单、可审阅稿件差异 | P0 未过不启动新科学运行 |
| T0+4–T0+6 | P1-A | 地址契约、oracle、反事实、逐试次保存 | G1 决定任务是否有效 |
| T0+7–T0+9 | P1-B，或按算子级研究收口 | 动态端口依赖／消融与静态基线对照 | 不存在动态新增作用则收窄主张 |
| T0+10–T0+12 | 有必要时 P2 | 有限精度与干扰边界 | 形成边界结论或申请 P3 |
| T0+13–T0+14 | 综合与复核 | 一份可追踪论文／技术报告包、未完成项与停止记录 | 决定投稿、暂停或独立新阶段 |

最有价值的下一次工作请求是：**只执行 P0，优先处理双 omega、Phase 2 判定缺口、论文数字与样本量口径，建立主账；不改冻结实验，不启动 4.3-B。** 这样能在 1–3 个工作日内先把现有成果变成可信的研究基线，再决定后续投入。

## 14. 可追踪证据索引

路径以主研究目录为基准，行号用于快速定位；JSON 以字段名为权威。以下只列最承重证据，不把旧交接意见当作高于原代码／数值的事实。

| 编号 | 文件／字段 | 支撑事项 |
|---|---|---|
| E01 | `docs/NEXT_AGENT_BRIEF.md`、`docs/CLAIM_LEDGER.md`、`docs/PHASE4_COMPLETION_REPORT.md` | TAN-I 历史冻结与根交接入口过时 |
| E02 | `results/tables/natural_collision_summary.csv:1`、`results/logs/natural_collision.log` | K 的均值／中位数、碰撞与复位统计 |
| E03 | `results/tables/effective_dimension_summary.csv:1`、`results/logs/effective_dimension.log` | 374 events、effective rank、log-trace、原始 OUTCOME D |
| E04 | `code/experiments/effective_dimension.py:967`、`:984`、`:1000` | stable=True 与 D 说明矛盾的分支覆盖原因 |
| E05 | `audit_v3_amended/PER_SEED_AUDIT.md`、`POOLED_PROBE1_V3.md`、`MACHINE_READABLE_AUDIT.json` | Probe-1 限定结果和无效单元格 |
| E06 | `code/experiments/temporal_distractor.py:103`、`code/analysis/probe1_v3_amended_audit.py:187` | 总样本 6000 与 distractor 2997 的区别、条件 bootstrap |
| E07 | `code/experiments/probe2/identifiability/AUDIT_OUTPUT.txt`、`THEORY_PRECHECK.md` | Probe-2 的失败、捷径与零翻转 |
| E08 | `docs/TANII_SPRINT4_1CD_THEORY.md`、`RESULTS` 对应文件、`results/sprint4_1cd/sprint4_1cd_summary.json` | E-I 阶段的静态与动态结果 |
| E09 | `results/sprint4_2/sprint4_2_summary.json` 的 checks；`docs/TANII_SPRINT4_2_AMENDMENT_1.md` | 316+1 的真实状态、E3 偏差范围 |
| E10 | `paper2/sections/s2_framework.tex:199`、`docs/TANII_SPRINT4_2B_PREREGISTRATION.md:38` 与 `:63` | 双 omega 与预注册内部记号矛盾 |
| E11 | `code/experiments/sprint4_2/sprint4_2_vector_qk.py:84`、`:97`、`:123` | 实际 softmax 和 query 旋转公式 |
| E12 | `results/sprint4_2/sprint4_2b_summary.json:186`、`:277` | M0/M2 规范全窗口 JSD |
| E13 | `results/sprint4_2/sprint4_2c_summary.json`、`sprint4_2d_summary.json` 与各 claim ledger | 软绑定、硬选择和 E-I 值传递 |
| E14 | `results/sprint4_3/audit_summary.json`、`monte_carlo_results.json:5` | 79 checks、双正确覆盖率、条件／无条件误差 |
| E15 | `code/experiments/sprint4_3/sprint4_3a_composition_probe.py:96`、`:110`、`:198`、`:443` | 算术组合的真实执行路径与构造性 flags |
| E16 | `paper2/sections/s3_state_geometry.tex:29`、`s5_routing_geometry.tex:118`、`:185` | 统计名称、最大偏差和 JSD 的转录问题 |
| E17 | `paper/sections/methods.tex:37`、`paper2/sections/s6_binding.tex`、`s7_wta.tex`、`s9_ladder.tex:141` | Markov／收缩、softmax／单头／状态资源的命题范围 |
| E18 | `paper/main.tex:36`、`paper2/main.tex:34`、`:35`；对应 PDF 全文与 main.log | 占位符、旧版本日期、实际 28/25 页与构建状态 |
| E19 | 用户目录下 `.dsh/sessions/...TAN_Final_Submission.../session-9e1ecfa5-4cb3-4947-ac66-e70fe72bc208/session.v3.jsonl.zstd`；Downloads 同会话 JSONL 导出 | 25 轮实际研究时间线、开发期修正、最后科学活动 |
| E20 | `Downloads/REVISION_DELIVERABLES.md`、旧 Stage 2 计划／PPT／DOCX、旧 Paper_Project | 旧主线未处理的问题与已被新机制研究取代的下一步 |

### 14.1 已匹配的历史脚本哈希

完整哈希见原 JSON／档案；此处列前 16 位供检索。9/9 指当前脚本匹配其历史记录，不代表新增了全数据的历史哈希。

| 脚本 | SHA-256 前 16 位 |
|---|---|
| Probe-2 identifiability | aabf8ebbb81e911e |
| sprint4_1cd_ei_dynamics | 324548198e50f604 |
| sprint4_2_geometry | cfb201a9ba26ecab |
| sprint4_2_shortcut_audit | 5cdac1369597bde5 |
| sprint4_2_vector_qk | 3328362c8e956ff6 |
| sprint4_2_routing_probe | 8f0982c971018f1e |
| sprint4_2_binding_probe | 5d55652500be0949 |
| sprint4_2_hard_binding | e5178f7713386518 |
| sprint4_3a_composition_probe | 8155552b8a22e695 |

## 15. 最终口径

现有成果不是空白，也不是只停在最初的标量 TAN。已经积累了完整的边界探索、失败审计、有限机制实现和两份论文成稿。问题在于归档入口落后于实际进度，论文的某些定义／公式／统计转录超过或偏离了可追踪证据。

最合理的下一步是先把已有成果纠正成一个可靠、可复核、可交接的基线，再问“明确地址的检索是否成立、哪一层真的在计算”。不要用继续堆叠显式算术节点代替这两项审计。

本轮输出：此报告。后续 P0 修订、隔离实验重放、4.3-B 和投稿动作均未启动。
