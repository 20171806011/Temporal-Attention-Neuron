from pathlib import Path
import json,hashlib,time
import numpy as np
from data import make
from model import metrics
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bootstrap(ds):
    r=np.random.default_rng(2026110599);draw=[]
    for _ in range(2000):
        m=[]
        for s in r.integers(0,3,3):
            d,y=ds[s];idx=np.concatenate([r.choice(np.flatnonzero(y==c),256) for c in (0,1)]);m.append(d[idx].mean())
        draw.append(np.mean(m))
    return dict(difference=float(np.mean([d.mean() for d,y in ds])),percentile95=np.quantile(draw,[.025,.975]).tolist())
def main():
    out=R/'results';lock=json.loads((out/'SELECTION.json').read_text());assert lock['gate_pass'],'Development gate failed; test forbidden'
    assert not (out/'TEST_STARTED.json').exists(),'Test already attempted'
    for z in lock['selected']:assert sha(out/z['file'])==z['sha256']
    (out/'TEST_STARTED.json').write_text(json.dumps({'selection_sha256':sha(out/'SELECTION.json'),'gate_checked_before_generation':True},indent=2)+'\n')
    begin=time.perf_counter();rows=[];correct={};labels={};hashes={}
    for s in range(3):
        x,y=make(512,2026110502+10*s);labels[s]=y;f=out/f'test_{s}.npz';np.savez_compressed(f,x=x,y=y);hashes[f.name]=sha(f)
        for arm in ('DEND','POINT'):
            choice=next(z for z in lock['selected'] if (z['seed'],z['arm'])==(s,arm))
            for mode in ('trained','initial'):
                t=np.load(out/(choice['file'] if mode=='trained' else f'initial_{s}.npy'),allow_pickle=False)
                tick=time.perf_counter();v=metrics(x,y,t,arm);elapsed=time.perf_counter()-tick
                f=out/f'test_{s}_{arm}_{mode}.npz';np.savez_compressed(f,**{k:v[k] for k in ('counts','prediction','spikes','states','candidates')});hashes[f.name]=sha(f)
                correct[s,arm,mode]=v['prediction']==y
                rows.append(dict(seed=s,arm=arm,mode=mode,accuracy=v['accuracy'],coverage=v['coverage'],abstentions=v['abstentions'],events_mean=v['events_mean'],inference_seconds=elapsed,silent_fraction=float(np.mean(np.all(v['counts']==0,axis=1))),both_saturated_fraction=float(np.mean(np.all(v['counts']==8,axis=1)))))
    ds=[(correct[s,'DEND','trained'].astype(float)-correct[s,'POINT','trained'],labels[s]) for s in range(3)]
    learning={a:bootstrap([(correct[s,a,'trained'].astype(float)-correct[s,a,'initial'],labels[s]) for s in range(3)]) for a in ('DEND','POINT')}
    result=dict(rows=rows,paired_difference=bootstrap(ds),learning=learning,sanity_pass=all(z['accuracy']>=.9 and z['coverage']>=.9 for z in rows if z['mode']=='trained'),seconds=time.perf_counter()-begin,hashes=hashes,selection_sha256=sha(out/'SELECTION.json'))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','hashes')},indent=2))
if __name__=='__main__':main()
