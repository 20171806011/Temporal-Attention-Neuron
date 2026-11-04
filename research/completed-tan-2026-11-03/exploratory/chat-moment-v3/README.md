# Scalar-key moment associative memory v3

AI 辅助的 Chat 沙箱概念研究。不是 TAN/P1-B/v11 重跑，不覆盖 v1 或 v2。同一 agent 的复跑和核验不是外部独立验证。公开版说明见 PUBLIC_REPRODUCTION.md。

## 本轮成果

1. 用f(k)=(1,k,...,k^Nmax)和N<=Nmax，精确算术下任何真正新scalar key都在已有span外，满容量仍可识别为新后拒绝。
2. Hankel键矩和latest值矩支持distinct-key新写与旧键残差覆盖。
3. 实现带根区间、地址、谱和内容误差证书的候选事务：通过才commit；拒绝时计数、全部矩、值矩和误差预算字节不变。绝不先投影/修补已存内容。
4. 非零输入噪声与非零定点舍入下证书非空；密集键的coverage较差且部分拒绝纯属保守证书未通过，不是已观察到数值错误。

## 运行

参考环境 Python 3.13.5 / NumPy 2.3.5 / SymPy 1.14.0。

```bash
python scalar_moment_v3.py --spec frozen_spec.json --out rerun_results_v3.json
python verify_v3.py --spec frozen_spec.json --results rerun_results_v3.json --out rerun_verification_v3.json
sha256sum -c PUBLIC_SHA256SUMS.txt
```

无训练、无随机抽样、无种子：所有噪声偏移都是冻结有理数。3条件×3方法×12请求=108个事务；这不是总体成功率实验。

## 输入及成功合同

canonical真实键最小间隔Delta由输入协议保证，包括未来新键。每次带来源保证的zeta给出传感误差上界。各方法都只获observed scalar/value/zeta，不获评分器真key。
新登记保存首次观测锚点（定点法再量化），用global sigma连接canonical身份和锚点。重复写不漂移锚点。理想对照只消除内部算术误差，仍使用同样的带噪观测，并拒绝歧义和满容量新键。
严格latest指已commit的canonical身份。拒绝写入不算完成；requested targets的未满足差额另报。读取时先认证地址，再输出相应登记节点的恢复值；不是对任意带噪q直接调用无条件C伪逆。

## 冻结集合

- separated_q24：canonical A=-0.5, B=0.5, C=0；Delta=0.2；p=24。
- clustered_q24：A=-0.02, B=0.02, C=0；Delta=0.015；p=24。
- clustered_q40：同密集集合，p=40。

普通传感误差上界1e-4，实际偏移为其预定非零分数；另有一次传感区间跨两个旧键的请求。后续D/E检验未来新插入及满容量识别。允许误差epsilon=0.01。

## 结果摘要

| 条件 | 矩记忆commit/请求 | 表commit/请求 | 理想同噪声commit/请求 | 矩被接受latest最大误差 | 其他已commit记忆最大保留误差 |
|---|---:|---:|---:|---:|---:|
| separated_q24 | 9/12 | 9/12 | 9/12 | 7.21215e-8 | 9.24783e-8 |
| clustered_q24 | 4/12 | 9/12 | 9/12 | 8.86491e-6 | 1.12829e-5 |
| clustered_q40 | 6/12 | 9/12 | 9/12 | 1.83076e-9 | 1.83076e-9 |

表和理想对照本样本数值误差为0；目标values是二进制有理数，因此表恰能精确表示，不是任意实值下零量化误差的结论。

所有已commit键普通查询都获认证。若把曾请求但未成功插入的键也纳入分母，矩方法分别返回29/34、25/34、29/34次value；表和理想均29/34。这个返回率不是requested latest全部满足率：被拒绝的覆盖可能读出旧值。

例如clustered_q40最后A保持0.75，而最后请求为0.125；C保持-0.25，而请求0.875。两次未完成目标不能记作成功。
clustered_q24拒绝两个中间键C的插入后，后来E仍获得新登记；拒绝未删除先前两个记忆或消耗名额。

## 保守性与不利结果

10个算出后被拒绝的候选，评分器事后发现其具体点预测误差都小于0.01；但它们没有通过统一区间/谱/内容预算。代码没有用这个评分反馈挽救候选，也没有调参。拒绝的含义是“未获得充分保证”，不等于该具体候选数值必然错误。

本实现beta更新放弃误差相关与抵消，使密集键预算可能快速变松，甚至在p=40下阻止后续覆盖。没有无限写入coverage保证。相同每word精度下，表更小、接受更多；矩编码在这个集合没有量化效率优势。

## 证据层级

- 主脚本实际执行了固定集合和有理数精确fixture。
- verify_v3.py不导入学习器代码，按保存结果重建评分器参考，完成3412个计数断言；这是同一agent的另一路核验，不是第三方独立重跑。
- 同一参考环境重复主执行的JSON字节一致，见REPRODUCIBILITY.md。
- 不是硬件故障校准、总体分布评估、原TAN复现或生物可行性证明。

## 文件

frozen_spec.json与SPEC_INITIAL_SHA256.txt记录运行前参数冻结；results_v3.json保留全部状态、预算、请求、候选和query；summary_v3.json是摘要；refusals_v3.json保留全部拒绝及未完成目标；THEORY.md给证明；RESOURCES.md列资源；SOURCES.md列访问到的一手资料与失败检索；verification_v3.json是另一路核验记录。
