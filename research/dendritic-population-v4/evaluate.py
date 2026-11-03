"""Only one test invocation, after development and fresh-replication locks."""
import time,json
import numpy as np
import model
from data import make
from common import R,OUT,dump,sha,mask,check_freeze

def interval(ds,level):
    rng=np.random.default_rng(2026115000+level);draws=[]
    for _ in range(2000):
        vals=[]
        for s in rng.integers(0,5,5):
            d,p=ds[s];ids=np.concatenate([rng.choice(np.flatnonzero(p==c),int(np.sum(p==c))) for c in np.unique(p)])
            vals.append(d[ids].mean())
        draws.append(np.mean(vals))
    return {'difference':float(np.mean([d.mean() for d,p in ds])),'percentile95':np.quantile(draws,[.025,.975]).tolist()}

def main():
    check_freeze();dev=json.loads((OUT/'DEVELOPMENT_LOCK.json').read_text());rep=json.loads((OUT/'REPLICATION_LOCK.json').read_text())
    assert rep['development_lock_sha256']==sha(OUT/'DEVELOPMENT_LOCK.json')
    for f,h in rep['files'].items():assert sha(OUT/f)==h,f
    assert not (OUT/'TEST_STARTED.json').exists(),'One test invocation only'
    dump(OUT/'TEST_STARTED.json',{'development_lock_sha256':sha(OUT/'DEVELOPMENT_LOCK.json'),'replication_lock_sha256':sha(OUT/'REPLICATION_LOCK.json'),'eligible_levels':rep['eligible_levels']})
    start=time.perf_counter();rows=[];correct={};pairs={};hashes={}
    for level in rep['eligible_levels']:
        for s in range(5):
            seed=2026114002+100*level+10*s;x,y,p=make(level,512,seed);pairs[level,s]=p
            file=f'test_L{level}_s{s}.npz';np.savez_compressed(OUT/file,x=x,y=y,pair=p,seed=np.array(seed));hashes[file]=sha(OUT/file)
            for arm in ('DEND','POINT'):
                conditions=['intact','retrained','cut'] if level==dev['c_level'] else ['intact']
                for cond in conditions:
                    q=next(r for r in rep['rows'] if (r['level'],r['seed'],r['arm'],r['condition'])==(level,s,arm,'intact' if cond=='cut' else cond))
                    t=np.load(OUT/q['file'],allow_pickle=False)
                    if cond=='cut':t=t*mask()
                    tick=time.perf_counter();v=model.metrics(x,y,t,arm);seconds=time.perf_counter()-tick
                    file=f'test_L{level}_s{s}_{arm}_{cond}.npz';np.savez_compressed(OUT/file,**{k:v[k] for k in ('counts','prediction','spikes','states','candidates')});hashes[file]=sha(OUT/file);correct[level,s,arm,cond]=v['prediction']==y
                    rows.append(dict(level=level,seed=s,arm=arm,condition=cond,source_file=q['file'],file=file,accuracy=v['accuracy'],coverage=v['coverage'],abstentions=v['abstentions'],abstention_rate=v['abstentions']/512,silent_fraction=float(np.all(v['counts']==0,1).mean()),both_saturated_fraction=float(np.all(v['counts']==8,1).mean()),events_mean=v['events_mean'],rate_error=model.loss_counts(v['counts'],y,'RATE')[0],inference_seconds=seconds))
    differences={};interventions={};claims={}
    logs=json.loads((OUT/'development_log.json').read_text())
    for level in rep['eligible_levels']:
        ds=[(correct[level,s,'DEND','intact'].astype(float)-correct[level,s,'POINT','intact'],pairs[level,s]) for s in range(5)];d=interval(ds,level);differences[str(level)]=d
        bg=next(z['gates'] for z in dev['levels'] if z['level']==level)
        tg={a:all(r['accuracy']>=.9 and r['coverage']>=.9 for r in rows if r['level']==level and r['condition']=='intact' and r['arm']==a) for a in ('DEND','POINT')}
        valdiff=[next(r['validation']['accuracy'] for r in logs if (r['stage'],r['level'],r['seed'],r['arm'])==('B',level,s,'DEND'))-next(r['validation']['accuracy'] for r in logs if (r['stage'],r['level'],r['seed'],r['arm'])==('B',level,s,'POINT')) for s in range(5)]
        positive=all(z>0 for z in valdiff) and all(a.mean()>0 for a,p in ds) and bg['DEND'] and not bg['POINT'] and tg['DEND'] and not tg['POINT'] and d['percentile95'][0]>0
        negative=all(z<0 for z in valdiff) and all(a.mean()<0 for a,p in ds) and bg['POINT'] and not bg['DEND'] and tg['POINT'] and not tg['DEND'] and d['percentile95'][1]<0
        claims[str(level)]={'development_gates':bg,'test_gates':tg,'difference_criteria_met':bool(positive or negative)}
        if level==dev['c_level']:
            for arm in ('DEND','POINT'):
                for control in ('retrained','cut'):
                    interventions[f'L{level}_{arm}_intact_minus_{control}']=interval([(correct[level,s,arm,'intact'].astype(float)-correct[level,s,arm,control],pairs[level,s]) for s in range(5)],level)
    summary=dict(rows=rows,differences=differences,interventions=interventions,claim_checks=claims,hashes=hashes,seconds=time.perf_counter()-start,replication_lock_sha256=sha(OUT/'REPLICATION_LOCK.json'),test_levels=rep['eligible_levels'])
    dump(OUT/'TEST_SUMMARY.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','hashes')},indent=2))
if __name__=='__main__':main()
