"""Protocol-defined analyses of saved audited records; no model execution."""
from pathlib import Path
import csv,json
import numpy as np
R=Path(__file__).resolve().parent
ARMS=('DEND','FULL','NOATT','GAIN2');RATES=(.01,.0025)
CONDITIONS=('nominal','clean','strong','cue_gap3','tag_removed','query_only')
RESOURCE_KEYS=('soma_events','output_events','recurrent_deliveries','output_deliveries','hold_soma_events','hold_output_events')
BOOT=np.random.default_rng(3981900001).integers(0,3,(20000,3))
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(name,x):(R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ci(values):
    x=np.asarray(values,dtype=float);assert x.shape==(3,)
    return dict(values=x.tolist(),mean=float(x.mean()),min=float(x.min()),max=float(x.max()),n=3,ci95=np.quantile(x[BOOT].mean(1),[.025,.975]).tolist())
def csvwrite(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def main():
    assert read(R/'AUDIT.json')['status']=='PASS'
    raw=read(R/'evaluate/ROWS.json');training=read(R/'train/ROWS.json');assert len(raw)==144 and len(training)==24
    rows=[];negative=[]
    for source in raw:
        row=dict(source);assert len(row['mean_resources'])==6;row.update(zip(RESOURCE_KEYS,row['mean_resources']))
        row['internal_pulse_deliveries']=row['recurrent_deliveries']+row['output_deliveries']
        assert row['hold_soma_events']==row['hold_output_events']==0
        if row['condition'] in ('tag_removed','query_only'):
            residual=row['balanced_accuracy']-.5*row['coverage'];assert abs(residual)<1e-12
            row['negative_control_ba_minus_half_coverage']=residual
            negative.append({k:row[k] for k in ('id','arm','seed','lr','condition','balanced_accuracy','coverage','silence','negative_control_ba_minus_half_coverage')})
        rows.append(row)
    groups={};metrics=('balanced_accuracy','accuracy','coverage','silence')+RESOURCE_KEYS+('internal_pulse_deliveries',)
    for arm in ARMS:
        for lr in RATES:
            group={};groups[f'{arm}_lr{lr}']=group
            for condition in CONDITIONS:
                rr=sorted((r for r in rows if (r['arm'],r['lr'],r['condition'])==(arm,lr,condition)),key=lambda r:r['seed'])
                assert [r['seed'] for r in rr]==[0,1,2]
                group[condition]={metric:ci([r[metric] for r in rr]) for metric in metrics}
                group[condition]['ba90_count']=sum(r['balanced_accuracy']>=.9 for r in rr)
    def values(arm,lr,condition):return np.asarray(groups[f'{arm}_lr{lr}'][condition]['balanced_accuracy']['values'])
    effects=[]
    for lr in RATES:
        for comparator in ('NOATT','DEND','GAIN2'):
            for condition in CONDITIONS:
                effects.append(dict(lr=lr,condition=condition,comparison='FULL_minus_'+comparator,primary=comparator=='NOATT' and condition=='nominal',negative_control=condition in ('tag_removed','query_only'),**ci(values('FULL',lr,condition)-values(comparator,lr,condition))))
    condition_effects=[]
    for arm in ARMS:
        for lr in RATES:
            for condition in ('clean','strong','cue_gap3'):
                condition_effects.append(dict(arm=arm,lr=lr,condition=condition,comparison=condition+'_minus_nominal',primary=False,**ci(values(arm,lr,condition)-values(arm,lr,'nominal'))))
    diagnostics=[]
    for tr in training:
        z=np.load(R/'train'/f'{tr["id"]}.npz');base=np.load(R/'train'/f'base_s{tr["seed"]}.npz');y=base['y']
        assert z['counts'].shape==(400,64,2) and z['history'].shape==(400,4)
        for k in (0,1,2,5,10,25,50,100,200,399):
            c=z['counts'][k];pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1));h=z['history'][k]
            diagnostics.append(dict(id=tr['id'],arm=tr['arm'],lr=tr['lr'],seed=tr['seed'],completed_updates=k,next_update=k+1,condition='nominal',balanced_accuracy=float(sum(np.mean(pred[y==j]==j) for j in (0,1))/2),accuracy=float(np.mean(pred==y)),coverage=float(np.mean(pred>=0)),silence=float(np.mean(c.sum(1)==0)),query_output_spikes_mean=float(c.sum(1).mean()),loss=float(h[0]),gradient_norm=float(h[1]),task_gradient_norm=float(h[2]),regularizer_gradient_norm=float(h[3])))
    assert len(diagnostics)==240 and len(effects)==36 and len(condition_effects)==24 and len(negative)==48
    result=dict(study_status='EXPLORATORY',conditions=list(CONDITIONS),groups=groups,effects=effects,primary_effects=[e for e in effects if e['primary']],secondary_effects=[e for e in effects if not e['primary']],within_arm_condition_effects=condition_effects,negative_controls=dict(status='PASS',rows=negative,identity='Balanced accuracy = 0.5 * coverage; identical opposite-label inputs force identical deterministic counts. Ties are errors, so BA need not equal 50%.',max_abs_residual=max(abs(r['negative_control_ba_minus_half_coverage']) for r in negative)),inference='Three paired independently fitted seeds; 20,000 whole-block bootstrap samples, seed3981900001. Pointwise unadjusted95% intervals. Two nominal FULL-minus-NOATT primary comparisons;34 other between-arm and24 within-arm comparisons are secondary. All settings retained; no confirmation claim or winner selection.',diagnostic_timing='Saved counts precede update k+1, after k completed updates. Final held-out test weights have completed400 updates.',resource_scope='All conditions have q16,T24 and no separate hold interval; hold event columns are structurally zero. Native events/fanout differ from dense operations and graded branch transfers.')
    dump('RESULTS.json',result);dump('DIAGNOSTICS.json',diagnostics);csvwrite('metrics.csv',rows);csvwrite('diagnostics.csv',diagnostics)
    print(json.dumps(dict(primary=result['primary_effects'],means={key:{c:g['balanced_accuracy']['mean'] for c,g in group.items()} for key,group in groups.items()},negative_control_max_residual=result['negative_controls']['max_abs_residual']),indent=2))
if __name__=='__main__':main()
