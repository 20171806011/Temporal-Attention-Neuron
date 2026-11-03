"""Build the manuscript's source ledger; does not run scientific experiments."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREFIX='research/completed-tan-2026-11-03/'
groups={
'E1':('Static addressing, not dynamic storage',[
'audit-v11/Temporal-Attention-Neuron/p1_experiments/results/run_p1a_rev2/condition_summary.csv',
'audit-v11/Temporal-Attention-Neuron/p1_experiments/THEORETICAL_PROPOSITION_1D_METRIC.md',
'audit-v11/tan_audit/project_reassessment_v8/P1A_EVIDENCE_REVIEW.md',
'audit-v11/tan_audit/project_reassessment_v8/P1A_LEDGER_CHECKS.json']),
'E2':('Exact supported distractor-swap response collision; not a clean-only theorem',[
'audit-v11/tan_audit/project_reassessment_v8/REPORT.md',
'audit-v11/tan_audit/project_reassessment_v8/probe1_results.json',
'audit-v11/tan_audit/project_reassessment_v8/theory_certificates.json',
'audit-v11/tan_audit/project_reassessment_v8/THEORY_SCOPE_REVIEW.md']),
'E3':('Locally executed six-fit training-composition intervention; conditional inference',[
'local-clean-training-v1/REPORT.md','local-clean-training-v1/FROZEN_PROTOCOL.md',
'local-clean-training-v1/results.json','local-clean-training-v1/README.md',
'local-clean-training-v1/upstream_probe1/source/tan.py',
'local-clean-training-v1/upstream_probe1/run_probe1.py']),
'E4':('Fixed calibration and paired noise comparison, not equal total resources',[
'audit-v11/tan_audit/gate_mechanism_v11/REPORT.md',
'audit-v11/tan_audit/gate_mechanism_v11/metric_summary.json',
'audit-v11/tan_audit/gate_mechanism_v11/PREREGISTRATION.md']),
'E5':('Modified carrier under restricted input and observation resources',[
'native-eight-stage/EXPERIMENT_SERIES_SUMMARY.md',
'native-eight-stage/timescale-heldout/REPORT.md']),
'E6':('Classical interpolation with extra writer/controller/readout resources',[
'audit-v11/tan_audit/interpolation_control/REPORT.md',
'audit-v11/tan_audit/precision_binding/REPORT.md']),
'E7':('Chat-sandbox conceptual experiments; not independently replicated here',[
'exploratory/chat-latest-value/README.md','exploratory/chat-range-v2/README.md',
'exploratory/chat-moment-v3/README.md','exploratory/chat-minimal-v4/README.md',
'exploratory/chat-whq-v1/README.md'])}
ledger={'baseline_commit':'222caa0201feff1e37a06049aa44c0e59e1cbaba','groups':{}}
lines=['# Evidence and manuscript relationship','','The manuscript synthesizes prior evidence. No prior manuscript or scientific result is overwritten.','',
'Existing manuscript families: `snapshot/manuscript`, `snapshot/paper`, and `snapshot/paper2`. The 174-entry inventory includes draft/report sources and is not a claim of reading every file. Earlier stage-localization manuscript patches are retained as corrections rather than counted as separate experiments.','',
'## Unresolved evidence boundary','','A separate clean-only map collision certificate and a complete higher-moment task-benefit experiment report were not located in this published snapshot. The paper presents the native analytical sensitivity, the verified distractor-swap collision, and the saved training control separately. No missing numerical result is inferred from code alone.','']
for group,(scope,files) in groups.items():
 entries=[];lines += ['## '+group,'',scope+'.','']
 for f in files:
  p=ROOT/PREFIX/f
  entries.append({'path':PREFIX+f,'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  url='https://github.com/20171806011/Temporal-Attention-Neuron/blob/'+ledger['baseline_commit']+'/'+PREFIX+f
  lines.append('- ['+f+']('+url+')')
 ledger['groups'][group]={'scope':scope,'files':entries};lines.append('')
(HERE/'evidence_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
(HERE/'EVIDENCE.md').write_text('\n'.join(lines).rstrip()+'\n')
print('Evidence ledger:',sum(len(x['files']) for x in ledger['groups'].values()),'sources')
