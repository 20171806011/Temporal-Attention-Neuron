> Historical archival document. Original language preserved. Personal paths/host metadata and unavailable media links were sanitized; consult README.md and ARCHIVE_INDEX.json.

# 从较大网络探索锁定的跨规模猜想

该文件在探索独立核验通过后、任何 confirm_tune/confirm 科学数据生成之前写入。进入规则、优先顺序与阈值来自原冻结 PLAN.md，未按观察修改。

## 进入验证门槛

| 候选 | 是否达到冻结门槛 | 是否进入本轮 |
|---|---|---|
| H1 | False | False |
| H2 | False | False |
| H3 | False | False |
| H4 | False | False |

观测数值与全部基准种子记录在 HYPOTHESIS_LOCK.json；未达门槛的候选不改名、不放宽标准。

## 锁定预测

没有候选达到事先门槛。本轮停止针对性科学验证；仅报告完整探索及这一阴性进入结果。

## 统计单位与独立数据

验证为全新五个训练/测试块，n=5。四种情形共用对应块的数据以作配对；测试例、条件数和 bootstrap 重采样不增加独立训练样本数。confirm_tune 与 confirm 使用冻结 common.py 中的新种子命名空间；调参两个块和最终训练五个块也分开。每个模型固定 200 更新、不重启、不替换失败。

每个预测输出全部五个差值、均值和 20,000 次完整种子块重采样的逐项百分位 95% 区间，分析 RNG=3009000001。没有多重性校正；n 小、配置选择不确定性未包含，结论是受限范围的独立跟进证据。达到门槛称本轮支持；未达到说明该预测未获支持或基准不足，绝不推为一般不可能。

## 跨规模与限制的解释

这是从 h=24 的规律到 h=6 的跨规模检验。小规模不复现不等于否定大规模效应，小规模成功也不等于解释大网络。rd=0 从训练开始去掉分支跨步记忆，并改变写入充电；D=0 校准必须与 D=24 保持一起报告。瞬时分支清零与 rd=0 后重新训练逻辑不同。完整模型和受限模型分别独立调参；两类模型都有同等预算。
