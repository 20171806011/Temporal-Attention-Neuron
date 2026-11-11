"""Apply only the candidate gates frozen in PLAN.md; never generate validation data."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'HYPOTHESIS_LOCK.json').exists()
assert not (R/'confirm_tune').exists() and not (R/'confirm').exists()
assert read(R/'AUDIT_explore.json')['status']=='PASS'
rows=read(R/'explore/ROWS.json')
def get(s,delay=24,cond='clean',iv='baseline',arm='DEND',h=24):
    return next(r for r in rows if (r['seed'],r['h'],r['arm'],r['delay'],r['condition'],r['intervention']['name'])==(s,h,arm,delay,cond,iv))
def passes(s,delays,arm='DEND'):
    return all(get(s,d,arm=arm)['balanced_accuracy']>=.9 and get(s,d,arm=arm)['coverage']>=.9 for d in delays)
metrics={
 'H1':[get(s,iv='zero_v')['balanced_accuracy']-get(s,iv='zero_d')['balanced_accuracy'] for s in range(3)],
 'H2_DEND_hold_events':[get(s)['hold_hidden_events'] for s in range(3)],
 'H2_POINT_hold_events':[get(s,arm='POINT')['hold_hidden_events'] for s in range(3)],
 'H3':[get(s,8)['balanced_accuracy']-get(s,48)['balanced_accuracy'] for s in range(3)],
 'H4':[get(s)['balanced_accuracy']-get(s,cond='interference')['balanced_accuracy'] for s in range(3)]}
gates={
 'H1':all(passes(s,[0,24]) and metrics['H1'][s]>=.15 for s in range(3)),
 'H2':all(passes(s,[0,24],arm) and get(s,arm=arm)['hold_hidden_events']>=1 for s in range(3) for arm in ('DEND','POINT')),
 'H3':all(passes(s,[0,8]) and metrics['H3'][s]>=.15 for s in range(3)),
 'H4':all(passes(s,[0,24]) and metrics['H4'][s]>=.10 for s in range(3))}
selected=[h for h in ('H1','H2','H3','H4') if gates[h]][:2]
descriptions={
 'H1':'h=6 新 full DEND 解在 D=24 clean 的胞体清零 BA 减分支清零 BA 为正，平均至少 .15。需相应基准至少 4/5 块通过 D=0 与 D=24 的 BA/覆盖率 .90 双门槛，且配对差的逐项 95% 区间下界>0。',
 'H2':'h=6 独立调参 rd=0 DEND 至少 4/5 新块在 D=0 和 D=24 clean 同时通过 .90 双门槛；full 基准也至少 4/5 通过；D=24 full−rd0 BA 差的逐项 95% 区间上界<=.05。成功个例与稳定替代预测分开报告。',
 'H3':'h=6 full DEND 的 D=8 clean BA 减 D=48 clean BA 平均至少 .15，逐项 95% 区间下界>0；至少 4/5 基准通过 D=0 和 D=8 双门槛。D=48 明确是延迟分布外。',
 'H4':'h=6 full DEND 的 D=24 clean BA 减同基础例 interference BA 平均至少 .10，逐项 95% 区间下界>0；至少 4/5 基准通过 D=0 和 D=24 clean 双门槛。'}
lines=['# 从较大网络探索锁定的跨规模猜想','',
'该文件在探索独立核验通过后、任何 confirm_tune/confirm 科学数据生成之前写入。进入规则、优先顺序与阈值来自原冻结 PLAN.md，未按观察修改。','',
'## 进入验证门槛','',
'| 候选 | 是否达到冻结门槛 | 是否进入本轮 |','|---|---|---|']
for k in gates:lines.append(f'| {k} | {gates[k]} | {k in selected} |')
lines+=['','观测数值与全部基准种子记录在 HYPOTHESIS_LOCK.json；未达门槛的候选不改名、不放宽标准。','',
'## 锁定预测','']
if selected:
    for k in selected:lines +=[f'### {k}', '',descriptions[k],'']
else:lines+=['没有候选达到事先门槛。本轮停止针对性科学验证；仅报告完整探索及这一阴性进入结果。','']
lines+=['## 统计单位与独立数据','',
'验证为全新五个训练/测试块，n=5。四种情形共用对应块的数据以作配对；测试例、条件数和 bootstrap 重采样不增加独立训练样本数。confirm_tune 与 confirm 使用冻结 common.py 中的新种子命名空间；调参两个块和最终训练五个块也分开。每个模型固定 200 更新、不重启、不替换失败。','',
'每个预测输出全部五个差值、均值和 20,000 次完整种子块重采样的逐项百分位 95% 区间，分析 RNG=3009000001。没有多重性校正；n 小、配置选择不确定性未包含，结论是受限范围的独立跟进证据。达到门槛称本轮支持；未达到说明该预测未获支持或基准不足，绝不推为一般不可能。','',
'## 跨规模与限制的解释','',
'这是从 h=24 的规律到 h=6 的跨规模检验。小规模不复现不等于否定大规模效应，小规模成功也不等于解释大网络。rd=0 从训练开始去掉分支跨步记忆，并改变写入充电；D=0 校准必须与 D=24 保持一起报告。瞬时分支清零与 rd=0 后重新训练逻辑不同。完整模型和受限模型分别独立调参；两类模型都有同等预算。','']
(R/'HYPOTHESES.md').write_text('\n'.join(lines),encoding='utf-8')
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gates=gates,selected=selected,run_validation=bool(selected),include_rd0='H2' in selected,exploration_metrics=metrics,
    baseline_rows=[get(s,d,arm=a) for a in ('DEND','POINT') for s in range(3) for d in (0,8,24,48)],
    hypotheses_sha256=sha(R/'HYPOTHESES.md'),selection_code_sha256=sha(Path(__file__)),exploration_freeze=sha(R/'EXPLORE_FREEZE.json'),exploration_rows_sha256=sha(R/'explore/ROWS.json'),audit_sha256=sha(R/'AUDIT_explore.json'),confirmation_data_generated=False)
(R/'HYPOTHESIS_LOCK.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(gates=gates,selected=selected,metrics=metrics),indent=2))
