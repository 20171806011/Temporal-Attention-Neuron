"""Posthoc descriptions of already saved counts/gradients; zero neural reruns."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
rows=[]
for row in read(R/'evaluate/ROWS.json'):
    if row['condition']!='nominal':continue
    c=np.load(R/'evaluate'/f'{row["id"]}_nominal.npz')['counts'].astype(float);y=np.load(R/'evaluate'/f'test_s{row["seed"]}.npz')['y'];s=c.sum(1);sign=np.eye(2)[y]*2-1;d=(c*sign).sum(1);pat,ns=np.unique(c,axis=0,return_counts=True)
    if np.all(c==0):category='all_silent'
    elif np.all(c==8):category='all_saturated'
    elif np.all(c[:,0]==c[:,1]):category='all_intermediate_tie'
    else:category='some_decisions'
    train=np.load(R/'train'/f'{row["id"]}.npz');cg=train['component_gradients'];ng=np.linalg.norm(cg,axis=2);total=np.linalg.norm(cg.sum(1),axis=1)
    item={k:row[k] for k in ('id','seed','arm','loss','balanced_accuracy','coverage','silence')}
    item.update(category=category,common_loss=float(np.mean((s-8)**2/32)),mean_sum=float(s.mean()),mean_signed_margin=float(d.mean()),fraction_4_4=float(np.mean(np.all(c==4,axis=1))),fraction_sum8=float(np.mean(s==8)),patterns=[dict(count=cc.astype(int).tolist(),n=int(n)) for cc,n in zip(pat,ns)],diagnostic_clipping_fraction=float(np.mean(total>1)),last_discriminative_norm=float(ng[-1,0]),last_common_norm=float(ng[-1,1]),last_l2_norm=float(ng[-1,2]));rows.append(item)
aggregates={}
for kind in ('CE','COUNT','CECOM'):
    rr=[r for r in rows if r['loss']==kind];aggregates[kind]={k:float(np.mean([r[k] for r in rr])) for k in ('balanced_accuracy','coverage','silence','common_loss','mean_sum','mean_signed_margin','fraction_4_4','fraction_sum8','diagnostic_clipping_fraction')};aggregates[kind]['categories']={k:sum(r['category']==k for r in rr) for k in ('all_silent','all_saturated','all_intermediate_tie','some_decisions')}
    aggregates[kind]['note']='Equal-weight descriptive average over12 models (4 architectures x3 paired blocks), not12 independent training blocks and not a new inferential endpoint.'
pairs={kind:[] for kind in ('COUNT','CECOM')}
for alt in pairs:
    for r in [r for r in rows if r['loss']==alt]:
        ce=next(z for z in rows if z['seed']==r['seed'] and z['arm']==r['arm'] and z['loss']=='CE');pairs[alt].append(dict(seed=r['seed'],arm=r['arm'],ba_delta=r['balanced_accuracy']-ce['balanced_accuracy'],coverage_delta=r['coverage']-ce['coverage']))
out=dict(status='POSTHOC_SAVED_RECORDS_ONLY',new_training_runs=0,new_neural_runs=0,aggregates=aggregates,rows=rows,paired_differences=pairs)
(R/'SAVED_ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(aggregates,indent=2))
