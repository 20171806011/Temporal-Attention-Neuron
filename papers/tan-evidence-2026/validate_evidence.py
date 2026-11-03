"""Read-only publication checks; no model execution, fitting, or new trials."""
from pathlib import Path
import csv,hashlib,json,math
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SNAP=ROOT/'research/completed-tan-2026-11-03'
ledger=json.loads((HERE/'evidence_ledger.json').read_text())
checks=[]
def require(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
for group in ledger['groups'].values():
 for f in group['files']:
  p=ROOT/f['path'];b=p.read_bytes()
  require(len(b)==f['size_bytes'] and hashlib.sha256(b).hexdigest()==f['sha256'],'Evidence hash: '+f['path'])
snapshot=json.loads((SNAP/'SNAPSHOT_MANIFEST.json').read_text())
require(len(snapshot['files'])==680,'Snapshot has 680 scientific files')
for f in snapshot['files']:
 b=(SNAP/f['path']).read_bytes()
 require(len(b)==f['size_bytes'] and hashlib.sha256(b).hexdigest()==f['sha256'],'Snapshot hash: '+f['path'])
paper=(HERE/'manuscript.md').read_text()
d=json.loads((SNAP/'local-clean-training-v1/results.json').read_text())
for arm in ['clean_only','mixed']:
 v=sum(x['arms'][arm]['balanced3'] for x in d['runs'])/3
 require(math.isclose(v,d['means'][arm],abs_tol=1e-14),'Equal-seed mean '+arm)
 require(f'{v*100:.4f}' in paper,'Paper mean '+arm)
 for run in d['runs']:
  a=run['arms'][arm];cm=a['confusion']
  ba=sum(cm[i][i]/sum(cm[i]) for i in range(3))/3
  require(math.isclose(ba,a['balanced3'],abs_tol=1e-14),'Confusion to BA '+str(run['seed'])+arm)
  require(a['optimizer']['success'],'Optimizer convergence '+str(run['seed'])+arm)
  require(f'{ba*100:.4f}' in paper,'Paper per-seed BA '+str(run['seed'])+arm)
require(f"{100*(d['means']['clean_only']-d['means']['mixed']):.4f}" in paper,'Paper training difference')
for v in d['conditional_percentile95']:require(f'{100*v:.4f}' in paper,'Paper conditional interval endpoint')
v=json.loads((SNAP/'audit-v11/tan_audit/gate_mechanism_v11/metric_summary.json').read_text())
for noise in ['iid','ar09']:
 for model in ['TAN','ADAPT']:
  require(f"{100*v[noise+'__'+model+'__BA']['estimate']:.4f}" in paper,'Paper BA '+noise+model)
  miss=round(1024*(1-v[noise+'__'+model+'__TPR']['estimate']))
  alarm=round(1024*v[noise+'__'+model+'__FPR']['estimate'])
  require(f'| {miss} | {alarm} |' in paper,'Paper miss/alarm counts '+noise+model)
for key in ['contrast__Delta_AR','contrast__Interaction']:
 require(f"{100*v[key]['estimate']:.4f}" in paper,'Paper contrast '+key)
 for x in v[key]['percentile_interval']:require(f'{100*x:.4f}' in paper,'Paper interval '+key)
c=json.loads((SNAP/'audit-v11/tan_audit/project_reassessment_v8/probe1_results.json').read_text())
require(c['cases']['total_pairs']==545,'Collision denominator')
for arm in c['arms'].values():
 require(f"| {arm['literal_equal_response_pairs']} | {arm['differing_spike_feature_pairs']} |" in paper,'Collision table row')
rows=list(csv.DictReader((SNAP/'audit-v11/Temporal-Attention-Neuron/p1_experiments/results/run_p1a_rev2/condition_summary.csv').open()))
for model in ['ScalarMetricAttention','VectorQKAddressAttention']:
 row=next(r for r in rows if r['Condition']=='C2_N3_zero_noise' and r['Model']==model)
 require(float(row['MiddleKeyChannelAcc_%'])==100,'Static middle query '+model)
row=next(r for r in rows if r['Condition']=='C2_N3_zero_noise' and r['Model']=='SignedScalarKernel')
require(float(row['MiddleKeyChannelAcc_%'])==0,'Signed scalar middle query')
report={'status':'PASS','check_count':len(checks),'scope':'Source and snapshot identity; saved confusion arithmetic; selected headline values and tables. No simulation, training or external replication.','checks':checks}
(HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS:',len(checks),'publication consistency checks')
