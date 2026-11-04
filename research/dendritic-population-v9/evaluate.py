"""One guarded test generation/evaluation after checkpoint commit."""
import json,time
import numpy as np
import model
from data import make
from common import R,OUT,REGISTRY,dump,sha,freeze_check,describe,save_native,analyze

def main():
    freeze_check();lock=json.loads((OUT/'TRAIN_LOCK.json').read_text());assert not (OUT/'TEST_STARTED.json').exists()
    for f,h in lock['files'].items():assert sha(OUT/f)==h,f
    dump(OUT/'TEST_STARTED.json',{'train_lock_sha256':sha(OUT/'TRAIN_LOCK.json')});start=time.perf_counter();rows=[];hashes={}
    for seed in REGISTRY['pairs']:
        s=seed['seed'];x,y,pair=make(512,seed['test_seed']);df=f'test_s{s}.npz';np.savez_compressed(OUT/df,x=x,y=y,pair=pair,seed=np.array(seed['test_seed']));hashes[df]=sha(OUT/df)
        for source in [r for r in lock['rows']+lock['cut_rows'] if r['seed']==s]:
            if time.perf_counter()-start>300:dump(OUT/'TEST_FAILURE.json',{'reason':'300s test cap','completed_rows':len(rows)});raise RuntimeError('No automatic test retry')
            theta=np.load(OUT/source['file'],allow_pickle=False);tick=time.perf_counter();v=model.metrics(x,y,theta,source['arm']);secs=time.perf_counter()-tick
            f=source['id']+'_test_native.npz';save_native(OUT/f,v);hashes[f]=sha(OUT/f)
            row=dict(id=source['id'],seed=s,arm=source['arm'],condition=source['condition'],data_file=df,source_file=source['file'],file=f,inference_seconds=secs,**describe(v,y));rows.append(row);dump(OUT/'TEST_LOG.json',rows)
    summary=dict(rows=rows,statistics=analyze(rows,lock['rows']),hashes=hashes,train_lock_sha256=sha(OUT/'TRAIN_LOCK.json'),seconds=time.perf_counter()-start,unique_test_histories=5*512,model_test_episodes=30*512)
    dump(OUT/'TEST_SUMMARY.json',summary);print(json.dumps(summary['statistics'],indent=2))
if __name__=='__main__':main()
