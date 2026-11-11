"""Frozen descriptive analyses of saved records; no model imports or execution."""
from pathlib import Path
import csv,json
import numpy as np
R=Path(__file__).resolve().parent
ARMS=('DEND','FULL','NOATT','GAIN2');RATES=(.01,.0025);DELAYS=(0,24,48,96)
RESOURCE_KEYS=('soma_events','output_events','recurrent_deliveries','output_deliveries','hold_soma_events','hold_output_events')
BOOT=np.random.default_rng(3971900001).integers(0,3,(20000,3))
def read(p):return json.loads(p.read_text(encoding='utf-8'))
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
    raw=read(R/'evaluate/ROWS.json');training=read(R/'train/ROWS.json');assert len(raw)==96 and len(training)==24
    rows=[]
    for source in raw:
        r=dict(source);assert len(r['mean_resources'])==6;r.update(zip(RESOURCE_KEYS,r['mean_resources']))
        r['internal_pulse_deliveries']=r['recurrent_deliveries']+r['output_deliveries'];rows.append(r)
    groups={};metrics=('balanced_accuracy','accuracy','coverage','silence')+RESOURCE_KEYS+('internal_pulse_deliveries',)
    for arm in ARMS:
        for lr in RATES:
            group={};groups[f'{arm}_lr{lr}']=group
            for D in DELAYS:
                rr=sorted((r for r in rows if (r['arm'],r['lr'],r['delay'])==(arm,lr,D)),key=lambda r:r['seed'])
                assert [r['seed'] for r in rr]==[0,1,2]
                group[str(D)]={metric:ci([r[metric] for r in rr]) for metric in metrics}
                group[str(D)]['ba90_count']=sum(r['balanced_accuracy']>=.9 for r in rr)
    def values(arm,lr,D):return np.asarray(groups[f'{arm}_lr{lr}'][str(D)]['balanced_accuracy']['values'])
    effects=[]
    for lr in RATES:
        for comparator in ('NOATT','DEND','GAIN2'):
            for D in DELAYS:
                effects.append(dict(lr=lr,delay=D,comparison='FULL_minus_'+comparator,primary=comparator=='NOATT' and D==24,**ci(values('FULL',lr,D)-values(comparator,lr,D))))
    diagnostics=[]
    for tr in training:
        z=np.load(R/'train'/f'{tr["id"]}.npz');assert z['counts'].shape==(200,128,2) and z['history'].shape==(200,4)
        bases=[np.load(R/'train'/f'base_s{tr["seed"]}_g{g}.npz') for g in (0,1)]
        for k in (0,1,2,5,10,25,50,100,199):
            for g,D in enumerate((0,24)):
                c=z['counts'][k,64*g:64*(g+1)];y=bases[g]['y'];pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1));h=z['history'][k]
                diagnostics.append(dict(id=tr['id'],arm=tr['arm'],lr=tr['lr'],seed=tr['seed'],completed_updates=k,next_update=k+1,delay=D,balanced_accuracy=float(sum(np.mean(pred[y==j]==j) for j in (0,1))/2),accuracy=float(np.mean(pred==y)),coverage=float(np.mean(pred>=0)),silence=float(np.mean(c.sum(1)==0)),query_output_spikes_mean=float(c.sum(1).mean()),combined_training_loss=float(h[0]),gradient_norm=float(h[1]),task_gradient_norm=float(h[2]),regularizer_gradient_norm=float(h[3])))
    assert len(diagnostics)==432
    result=dict(groups=groups,effects=effects,primary_effects=[e for e in effects if e['primary']],secondary_effects=[e for e in effects if not e['primary']],attention_effects=[e for e in effects if e['comparison']=='FULL_minus_NOATT'],secondary_cross_architecture=[e for e in effects if e['comparison']!='FULL_minus_NOATT'],inference='Three paired independently trained seeds. 20,000 whole-block bootstrap resamples, seed 3971900001; pointwise unadjusted 95% intervals. Two fixed-rate D24 FULL-minus-NOATT primary effects. All conditions retained, no winner selection.',diagnostic_timing='Saved counts precede update k+1, after k completed updates; final test weights have completed all 200 updates. Loss and gradient columns describe the joint D0/D24 training objective, not each group separately.',resource_scope='Native soma/output events and pulse fanout are distinct from dense MACs and graded branch transfers; all retained in metrics.csv.')
    dump('RESULTS.json',result);dump('DIAGNOSTICS.json',diagnostics);csvwrite('metrics.csv',rows);csvwrite('diagnostics.csv',diagnostics)
    print(json.dumps(dict(primary=result['primary_effects'],means={key:{D:g['balanced_accuracy']['mean'] for D,g in group.items()} for key,group in groups.items()}),indent=2))
if __name__=='__main__':main()
