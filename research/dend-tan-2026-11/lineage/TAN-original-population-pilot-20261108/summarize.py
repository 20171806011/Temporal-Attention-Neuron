from pathlib import Path
import json,csv
import numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def ci(v):
    x=np.array(v,float);ix=np.random.default_rng(3951900001).integers(0,len(x),(20000,len(x)))
    return dict(values=x.tolist(),mean=float(x.mean()),min=float(x.min()),max=float(x.max()),n=len(x),ci95=np.quantile(x[ix].mean(1),[.025,.975]).tolist())
def csvwrite(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def main():
    assert read(R/'AUDIT.json')['status']=='PASS';rows=read(R/'evaluate/ROWS.json');groups={};effects=[];cross=[];diagnostics=[]
    for arm in ('TAN_FULL','TAN_NOATT','DEND','POINT'):
        for lr in (.01,.0025):
            g={};groups[f'{arm}_lr{lr}']=g
            for D in (0,24,48,96):
                rr=[r for r in rows if (r['arm'],r['lr'],r['delay'])==(arm,lr,D)];assert [r['seed'] for r in rr]==[0,1,2]
                g[str(D)]={k:ci([r[k] for r in rr]) for k in ('balanced_accuracy','accuracy','coverage','silence','first_layer_events','hidden_events','output_events','hold_first_layer_events','hold_hidden_events','hold_output_events','internal_deliveries')}
                g[str(D)]['ba90_count']=sum(r['balanced_accuracy']>=.9 for r in rr)
    def vals(arm,lr,D):return np.array(groups[f'{arm}_lr{lr}'][str(D)]['balanced_accuracy']['values'])
    for lr in (.01,.0025):
        for D in (0,24,48,96):effects.append(dict(lr=lr,delay=D,primary=D==24,comparison='TAN_FULL_minus_TAN_NOATT',**ci(vals('TAN_FULL',lr,D)-vals('TAN_NOATT',lr,D))))
        for comparator in ('DEND','POINT'):
            for D in (24,96):cross.append(dict(lr=lr,delay=D,comparison='TAN_FULL_minus_'+comparator,**ci(vals('TAN_FULL',lr,D)-vals(comparator,lr,D))))
    for tr in read(R/'train/ROWS.json'):
        z=np.load(R/'train'/f'{tr["id"]}.npz');bases=[np.load(R/'train'/f'base_s{tr["seed"]}_g{g}.npz') for g in (0,1)]
        for k in (0,1,2,5,10,25,50,100,199):
            for g,D in enumerate((0,24)):
                c=z['counts'][k,g*64:(g+1)*64];y=bases[g]['y'];p=np.where(c[:,0]==c[:,1],-1,c.argmax(1));diagnostics.append(dict(id=tr['id'],arm=tr['arm'],lr=tr['lr'],seed=tr['seed'],completed_updates=k,delay=D,balanced_accuracy=float(sum(np.mean(p[y==j]==j) for j in (0,1))/2),silence=float(np.mean(c.sum(1)==0)),coverage=float(np.mean(p>=0)),query_output_spikes_mean=float(c.sum(1).mean()),gradient_norm=float(z['history'][k,1]),loss=float(z['history'][k,0])))
    result=dict(groups=groups,attention_effects=effects,secondary_cross_architecture=cross,inference='Three independently fitted paired blocks; 20000 pointwise unadjusted bootstrap samples. Two fixed-rate D24 primary effects. No tuning, recipe/seed selection or held-out-pattern generalization claim.')
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');(R/'DIAGNOSTICS.json').write_text(json.dumps(diagnostics,indent=2)+'\n');csvwrite('metrics.csv',rows);csvwrite('diagnostics.csv',diagnostics)
    print(json.dumps(dict(primary=[e for e in effects if e['primary']],means={k:{D:s['balanced_accuracy']['mean'] for D,s in g.items()} for k,g in groups.items()}),indent=2))
if __name__=='__main__':main()
