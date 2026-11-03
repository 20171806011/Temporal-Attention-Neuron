import json,time
import numpy as np
import model
from data import make
from common import R,OUT,dump,sha,brief,freeze_check

def seed_interval(values,rng_seed):
    v=np.asarray(values,dtype=float);assert v.shape==(10,)
    rng=np.random.default_rng(rng_seed);draw=rng.integers(0,10,size=(20000,10));samples=v[draw].mean(1)
    return {'seed_pair_differences':v.tolist(),'mean':float(v.mean()),'percentile95':np.quantile(samples,[.025,.975]).tolist(),'unit':'independently trained seed pair','n_seed_pairs':10}

def main():
    freeze_check();pre=json.loads((OUT/'PREPARE_LOCK.json').read_text());fin=json.loads((OUT/'FINAL_LOCK.json').read_text())
    assert fin['prepare_lock_sha256']==sha(OUT/'PREPARE_LOCK.json')
    for lock in (pre,fin):
        for f,h in lock['files'].items():assert sha(OUT/f)==h,f
    assert not (OUT/'TEST_STARTED.json').exists();dump(OUT/'TEST_STARTED.json',{'prepare_lock_sha256':sha(OUT/'PREPARE_LOCK.json'),'final_lock_sha256':sha(OUT/'FINAL_LOCK.json')})
    start=time.perf_counter();rows=[];hashes={};records=pre['rows']+fin['rows'];scored={}
    for phase in ('A','B'):
        for s in range(10):
            seed=(2026116100 if phase=='A' else 2026118100)+10*s;x,y,p=make(512,seed);f=f'test_{phase}_s{s}.npz';np.savez_compressed(OUT/f,x=x,y=y,pair=p,seed=np.array(seed));hashes[f]=sha(OUT/f)
            for arm in ('DEND','POINT'):
                for role in (['fixed'] if phase=='A' else ['fixed','selected']):
                    q=next(r for r in records if (r['stage'],r['seed'],r['arm'],r['role'])==(phase,s,arm,role));theta=np.load(OUT/q['file'],allow_pickle=False)
                    for mode in ('intact','no_history'):
                        if time.perf_counter()-start>300:raise RuntimeError('Test/artifact wall cap exceeded; no automatic retry')
                        key=(phase,s,arm,q['file'],mode)
                        if key in scored:
                            source=scored[key];row=dict(source,role=role,reused=True);rows.append(row);continue
                        xx=x if mode=='intact' else x.copy()
                        if mode=='no_history':xx[:,:8,:]=0
                        tick=time.perf_counter();v=model.metrics(xx,y,theta,arm);seconds=time.perf_counter()-tick
                        if mode=='no_history':
                            assert np.array_equal(v['states'][:,7],np.repeat(v['states'][:1,7],512,axis=0))
                            assert np.array_equal(v['spikes'][:,7,12:18],np.repeat(v['spikes'][:1,7,12:18],512,axis=0))
                        f=f'test_{phase}_s{s}_{arm}_{role}_{mode}.npz';np.savez_compressed(OUT/f,**{k:v[k] for k in ('counts','prediction','spikes','states','candidates')});hashes[f]=sha(OUT/f)
                        row=dict(phase=phase,seed=s,arm=arm,role=role,mode=mode,source_file=q['file'],file=f,reused=False,inference_seconds=seconds,**brief(v,y));rows.append(row);scored[key]=row
    def values(phase,role,arm,mode='intact'):
        return np.array([next(r['accuracy'] for r in rows if (r['phase'],r['role'],r['arm'],r['mode'],r['seed'])==(phase,role,arm,mode,s)) for s in range(10)])
    intervals={}
    for phase,role,seed in [('A','fixed',2026119001),('B','selected',2026119002),('B','fixed',2026119003)]:intervals[f'{phase}_{role}_DEND_minus_POINT']=seed_interval(values(phase,role,'DEND')-values(phase,role,'POINT'),seed)
    for arm in ('DEND','POINT'):intervals[f'B_{arm}_selected_minus_fixed']=seed_interval(values('B','selected',arm)-values('B','fixed',arm),2026119003)
    delta=(values('B','fixed','DEND')-values('B','fixed','POINT'))-(values('B','selected','DEND')-values('B','selected','POINT'))
    intervals['B_paired_gap_attenuation']=seed_interval(delta,2026119003)
    for phase,role in [('A','fixed'),('B','fixed'),('B','selected')]:
        for arm in ('DEND','POINT'):intervals[f'{phase}_{role}_{arm}_intact_minus_no_history']=seed_interval(values(phase,role,arm)-values(phase,role,arm,'no_history'),2026119004)
    ia=intervals['A_fixed_DEND_minus_POINT'];ib=intervals['B_selected_DEND_minus_POINT']
    result=dict(rows=rows,intervals=intervals,primary='B_selected_DEND_minus_POINT',stable_direction=bool(ia['percentile95'][0]>0 and ib['percentile95'][0]>0),stable_practically_meaningful=bool(ia['percentile95'][0]>.05 and ib['percentile95'][0]>.05),practical_threshold=.05,hashes=hashes,seconds=time.perf_counter()-start,final_lock_sha256=sha(OUT/'FINAL_LOCK.json'),unique_model_evaluations=len(scored),unique_test_histories=20*512)
    dump(OUT/'TEST_SUMMARY.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','hashes')},indent=2))
if __name__=='__main__':main()
