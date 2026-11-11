from pathlib import Path
import json,csv,numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def ci(v):
    x=np.asarray(v,float);ix=np.random.default_rng(3931900001).integers(0,len(x),(20000,len(x)))
    return dict(values=x.tolist(),mean=float(x.mean()),min=float(x.min()),max=float(x.max()),n=len(x),ci95=np.quantile(x[ix].mean(1),[.025,.975]).tolist())
def writecsv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader()
        for row in rows:wr.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def key(arm,h,lr):return f'{arm}_h{h}_lr{lr}'
def main():
    assert read(R/'AUDIT.json')['status']=='PASS'
    rows=read(R/'evaluate/ROWS.json');trs=read(R/'evaluate/TRANSFERS.json');groups={};diag=[]
    for arm in ('DEND','POINT'):
        for h in (24,96):
            for lr in (.01,.0025):
                g=dict(conditions={},transfers={});groups[key(arm,h,lr)]=g
                for D in (0,24,48,96):
                    rr=[r for r in rows if (r['arm'],r['h'],r['lr'],r['delay'],r['condition'])==(arm,h,lr,D,'baseline')];assert [r['seed'] for r in rr]==[0,1,2]
                    g['conditions'][str(D)]={k:ci([r[k] for r in rr]) for k in ('balanced_accuracy','coverage','accuracy','silence','branch_events','hidden_events','output_events','hold_branch_events','hold_hidden_events','hold_output_events','internal_deliveries')}
                    g['conditions'][str(D)]['ba90_count']=sum(r['balanced_accuracy']>=.9 for r in rr)
                for D in (24,96):
                    g['transfers'][str(D)]={}
                    for part in ('d','vp','dvp','o','all'):
                        tt=[r for r in trs if (r['arm'],r['h'],r['lr'],r['delay'],r['part'])==(arm,h,lr,D,part)];assert [r['seed'] for r in tt]==[0,1,2]
                        g['transfers'][str(D)][part]=dict(macro=ci([r['macro_contrast'] for r in tt]),directions={})
                        for j in (0,1):
                            dd=[r['directions'][j] for r in tt];g['transfers'][str(D)][part]['directions'][str(j)]=dict(contrast=ci([r['contrast'] for r in dd]),opposite=ci([r['opposite_target_fraction'] for r in dd]),same=ci([r['same_target_fraction'] for r in dd]),conditional_flip=[r['conditional_flip'] for r in dd],conditional_n=[r['both_correct_n'] for r in dd])
    vals=lambda a,h,l,D:np.array(groups[key(a,h,l)]['conditions'][str(D)]['balanced_accuracy']['values'])
    effects=[]
    for a in ('DEND','POINT'):
        for D in (24,96):
            d96=vals(a,96,.0025,D)-vals(a,96,.01,D);d24=vals(a,24,.0025,D)-vals(a,24,.01,D)
            effects.extend([dict(arm=a,delay=D,comparison='h96_low_minus_high',primary=D==24,**ci(d96)),dict(arm=a,delay=D,comparison='width_x_rate_interaction',primary=D==24,**ci(d96-d24))])
    for r in read(R/'train/ROWS.json'):
        z=np.load(R/'train'/f'{r["id"]}.npz');bases=[np.load(R/'train'/f'base_s{r["seed"]}_g{g}.npz') for g in (0,1)]
        for ix,k in enumerate(z['completed_updates']):
            for g,D in enumerate((0,24)):
                c=z['counts'][k,g*64:(g+1)*64];y=bases[g]['y'];pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1));oc=z[f'oc_D{D}'][ix,:,16+D:,:];sl=1/(1+5*np.abs(oc-.5))**2
                diag.append(dict(id=r['id'],seed=r['seed'],h=r['h'],arm=r['arm'],lr=r['lr'],completed_updates=int(k),delay=D,balanced_accuracy=float(sum(np.mean(pred[y==j]==j) for j in (0,1))/2),silence=float(np.mean(c.sum(1)==0)),coverage=float(np.mean(pred>=0)),prediction_counts=[int(sum(pred==j)) for j in (-1,0,1)],output_candidate_mean=float(oc.mean()),output_candidate_min=float(oc.min()),output_candidate_max=float(oc.max()),output_slope_mean=float(sl.mean()),output_slope_median=float(np.median(sl)),output_slope_below_001=float(np.mean(sl<.01)),gradient_norm=float(z['history'][k,1])))
    result=dict(groups=groups,effects=effects,inference='Three paired seed blocks. 20000 pointwise unadjusted percentile bootstrap resamples; all factorial cells reported; no selection.')
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');(R/'DIAGNOSTICS.json').write_text(json.dumps(diag,indent=2)+'\n')
    writecsv('metrics.csv',rows);writecsv('transfers.csv',[dict(id=r['id'],seed=r['seed'],h=r['h'],arm=r['arm'],lr=r['lr'],delay=r['delay'],part=r['part'],macro=r['macro_contrast'],**d) for r in trs for d in r['directions']]);writecsv('diagnostics.csv',diag)
    print(json.dumps(dict(effects=effects,means={k:{d:v['balanced_accuracy']['mean'] for d,v in g['conditions'].items()} for k,g in groups.items()}),indent=2))
if __name__=='__main__':main()
