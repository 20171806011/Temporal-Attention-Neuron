"""Prespecified seed-block bootstrap; no new network runs or selection."""
from pathlib import Path
import json,csv
import numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def interval(v):
    a=np.asarray(v,float);ix=np.random.default_rng(3111900001).integers(0,len(a),(20000,len(a)));boot=a[ix].mean(1)
    return dict(values=a.tolist(),mean=float(a.mean()),min=float(a.min()),max=float(a.max()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(a))
def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def main():
    result={}
    for stage in ('old','new'):
        rows=read(R/stage/'ROWS.json');trans=read(R/stage/'TRANSFERS.json');probes=read(R/stage/'PROBES.json');names=list(dict.fromkeys(r['id'].split('_',1)[1] for r in rows));groups={}
        for name in names:
            rr=[r for r in rows if r['id'].split('_',1)[1]==name];tt=[r for r in trans if r['id'].split('_',1)[1]==name];pp=[r for r in probes if r['id'].split('_',1)[1]==name and r['time']==40]
            baseline=[r for r in rr if r['condition']=='baseline'];g={'ba24':interval([r['balanced_accuracy'] for r in baseline]),'coverage24':interval([r['coverage'] for r in baseline]),'transfers':{},'probe40':{}}
            for metric in ('hold_hidden_events','hold_branch_events','hidden_events','branch_events','output_events','internal_deliveries'):
                g[metric]=interval([r[metric] for r in baseline])
            g['ba24_ge90_count']=sum(r['balanced_accuracy']>=.9 for r in baseline)
            g['mean90_min85_descriptor']=g['ba24']['mean']>=.9 and g['ba24']['min']>=.85
            d0=[r for r in rr if r['condition']=='d0']
            if d0:
                g['ba0']=interval([r['balanced_accuracy'] for r in d0]);g['delay_drop']=interval([a['balanced_accuracy']-b['balanced_accuracy'] for a,b in zip(d0,baseline)])
                g['both_delays_ba_cov_ge90']=[a['seed'] for a,b in zip(d0,baseline) if min(a['balanced_accuracy'],a['coverage'],b['balanced_accuracy'],b['coverage'])>=.9]
            for part in ('d','vp','dvp','o','all'):
                t=[r for r in tt if r['part']==part];p=[r for r in pp if r['part']==part]
                g['transfers'][part]={'macro_contrast':interval([r['macro_contrast'] for r in t]),'directions':{}}
                for j in (0,1):
                    g['transfers'][part]['directions'][f'{j}->{1-j}']={k:interval([r['directions'][j][k] for r in t]) for k in ('opposite_target_fraction','same_target_fraction','contrast','donor_prediction_agreement')}
                    g['transfers'][part]['directions'][f'{j}->{1-j}']['conditional_flip_by_seed']=[r['directions'][j]['conditional_flip'] for r in t]
                    g['transfers'][part]['directions'][f'{j}->{1-j}']['conditional_n_by_seed']=[r['directions'][j]['both_baseline_correct_n'] for r in t]
                g['probe40'][part]={k:interval([r[k] for r in p]) for k in ('real_accuracy','shuffled_accuracy')}
            g['geometry40']={k:interval([r[k] for r in pp if r['part']=='d']) for k in ('d_centroid_cosine','d_centroid_distance','d_within_rms')}
            groups[name]=g
        result[stage]=groups
        csvwrite(R/f'{stage}_metrics.csv',rows)
        csvwrite(R/f'{stage}_transfers.csv',[dict(id=r['id'],seed=r['seed'],part=r['part'],macro_contrast=r['macro_contrast'],**d) for r in trans for d in r['directions']])
        csvwrite(R/f'{stage}_probes.csv',probes)
    new=result['new'];full=new['DEND_full'];rd0=new['DEND_rd0'];point=new['POINT_full']
    result['primary_paired']={
        'ba24_full_minus_rd0':interval(np.array(full['ba24']['values'])-rd0['ba24']['values']),
        'vp_contrast_rd0_minus_full':interval(np.array(rd0['transfers']['vp']['macro_contrast']['values'])-full['transfers']['vp']['macro_contrast']['values'])}
    result['secondary_paired']={'ba24_DEND_minus_POINT':interval(np.array(full['ba24']['values'])-point['ba24']['values'])}
    result['inference_limits']='n=3 historical and n=5 new paired seed blocks; 20,000 percentile bootstrap resamples; pointwise unadjusted intervals; task and inherited fixed recipes, not architecture impossibility or energy.'
    (R/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:{n:{'ba24':g['ba24'],'d_transfer':g['transfers']['d']['macro_contrast'],'vp_transfer':g['transfers']['vp']['macro_contrast'],'d_probe40':g['probe40']['d'],'geometry40':g['geometry40']} for n,g in result[k].items()} for k in ('old','new')},indent=2))
if __name__=='__main__':main()
