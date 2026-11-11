"""Frozen comparisons over the five independent seed blocks."""
from pathlib import Path
import json,csv
import numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def ci(values):
    x=np.asarray(values,float);ix=np.random.default_rng(3921900001).integers(0,len(x),(20000,len(x)))
    return dict(values=x.tolist(),mean=float(x.mean()),min=float(x.min()),max=float(x.max()),n=len(x),ci95=np.quantile(x[ix].mean(1),[.025,.975]).tolist())
def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader()
        for r in rows:wr.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def main():
    rows=read(R/'evaluate'/'ROWS.json');trans=read(R/'evaluate'/'TRANSFERS.json');groups={};width=[];architecture=[]
    for h in (24,48,96):
        for arm in ('DEND','POINT'):
            key=f'{arm}_h{h}';rr=[r for r in rows if r['h']==h and r['arm']==arm];g={'conditions':{},'transfers':{}}
            for D,kind in [(0,'clean'),(24,'clean'),(48,'clean'),(96,'clean'),(24,'interference'),(96,'interference')]:
                r=[x for x in rr if (x['delay'],x['kind'],x['condition'])==(D,kind,'baseline')];check=[x['seed'] for x in r];assert check==list(range(5))
                g['conditions'][f'd{D}_{kind}']={k:ci([x[k] for x in r]) for k in ('balanced_accuracy','coverage','accuracy','silence','branch_events','hidden_events','output_events','hold_branch_events','hold_hidden_events','hold_output_events','internal_deliveries')}
                g['conditions'][f'd{D}_{kind}']['ba90_count']=sum(x['balanced_accuracy']>=.9 for x in r)
            for D in (24,96):
                g['transfers'][str(D)]={}
                for part in ('d','vp','dvp','o','all'):
                    r=[x for x in trans if (x['h'],x['arm'],x['delay'],x['part'])==(h,arm,D,part)];assert [x['seed'] for x in r]==list(range(5))
                    g['transfers'][str(D)][part]={'macro':ci([x['macro_contrast'] for x in r]),'directions':{}}
                    for j in (0,1):
                        g['transfers'][str(D)][part]['directions'][f'{j}->{1-j}']={k:ci([x['directions'][j][k] for x in r]) for k in ('opposite_target_fraction','same_target_fraction','contrast')}
                        g['transfers'][str(D)][part]['directions'][f'{j}->{1-j}'].update(conditional_flip_by_seed=[x['directions'][j]['conditional_flip'] for x in r],conditional_n_by_seed=[x['directions'][j]['both_correct_n'] for x in r])
            vals=lambda name:np.array(g['conditions'][name]['balanced_accuracy']['values'])
            g['delay_loss_d24_to_d96']=ci(vals('d24_clean')-vals('d96_clean'))
            g['interference_loss']={str(D):ci(vals(f'd{D}_clean')-vals(f'd{D}_interference')) for D in (24,96)};groups[key]=g
    for arm in ('DEND','POINT'):
        for h in (48,96):
            v=np.array(groups[f'{arm}_h{h}']['conditions']['d96_clean']['balanced_accuracy']['values'])-groups[f'{arm}_h24']['conditions']['d96_clean']['balanced_accuracy']['values']
            width.append(dict(arm=arm,h=h,reference_h=24,condition='d96_clean',**ci(v)))
    for h in (24,48,96):
        for D in (24,96):
            v=np.array(groups[f'DEND_h{h}']['conditions'][f'd{D}_clean']['balanced_accuracy']['values'])-groups[f'POINT_h{h}']['conditions'][f'd{D}_clean']['balanced_accuracy']['values'];architecture.append(dict(h=h,delay=D,**ci(v)))
    result=dict(groups=groups,primary_width_effects=width,secondary_DEND_minus_POINT=architecture,selection=read(R/'selection'/'SELECTION_LOCK.json')['selected'],inference='Five seed blocks; 20,000 pointwise unadjusted percentile bootstrap resamples; all widths reported, no selected-width confirmation.')
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');csvwrite(R/'metrics.csv',rows);csvwrite(R/'transfers.csv',[dict(id=r['id'],h=r['h'],arm=r['arm'],seed=r['seed'],delay=r['delay'],part=r['part'],macro=r['macro_contrast'],**d) for r in trans for d in r['directions']]);csvwrite(R/'tuning.csv',read(R/'selection'/'ROWS.json'))
    print(json.dumps({'selection':result['selection'],'primary_width_effects':width,'means':{k:{c:d['balanced_accuracy']['mean'] for c,d in g['conditions'].items()} for k,g in groups.items()}},indent=2))
if __name__=='__main__':main()
