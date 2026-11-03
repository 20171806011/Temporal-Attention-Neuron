import json,hashlib,time
from pathlib import Path
import numpy as np
from model import forward
from data import make
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def interval(delta):
    rng=np.random.default_rng(2026110499);samples=[]
    for _ in range(2000):
        values=[]
        for s in rng.integers(0,3,3):
            d,p=delta[s];idx=np.concatenate([rng.choice(np.flatnonzero(p==k),128) for k in range(4)])
            values.append(d[idx].mean())
        samples.append(np.mean(values))
    return {'difference':float(np.mean([d.mean() for d,p in delta])),'percentile95':np.quantile(samples,[.025,.975]).tolist()}
def main():
    out=ROOT/'results'; lock=json.loads((out/'SELECTION.json').read_text());target=out/'TEST_STARTED.json'
    assert not target.exists(),'Refuse repeated test run'
    for r in lock['selected']:assert sha(out/r['file'])==r['sha256']
    target.write_text(json.dumps({'selection_sha256':sha(out/'SELECTION.json'),'selected_checkpoints_verified_before_test_generation':True},indent=2)+'\n')
    begin=time.perf_counter();rows=[];correct={};pairs={};hashes={}
    for s in range(3):
        x,y,p=make(512,2026110402+10*s);pairs[s]=p
        f=out/f'test_{s}.npz';np.savez_compressed(f,x=x,y=y,pair=p);hashes[f.name]=sha(f)
        for arm in ('DEND','POINT'):
            r=next(z for z in lock['selected'] if z['seed']==s and z['arm']==arm);theta=np.load(out/r['file'],allow_pickle=False)
            for mode in ('intact','cut_cross','initial'):
                if time.perf_counter()-begin>120:raise RuntimeError('evaluation budget exceeded')
                par=np.load(out/f'initial_{s}.npy',allow_pickle=False) if mode=='initial' else theta
                tick=time.perf_counter();v=forward(x,par,arm,cut_cross=mode=='cut_cross',trace=mode!='initial');seconds=time.perf_counter()-tick
                f=out/f'test_{s}_{arm}_{mode}.npz';np.savez_compressed(f,**v);hashes[f.name]=sha(f)
                c=v['prediction']==y;correct[s,arm,mode]=c
                rows.append(dict(seed=s,arm=arm,mode=mode,accuracy=float(c.mean()),coverage=float(np.mean(v['prediction']>=0)),correct=int(c.sum()),abstentions=int(np.sum(v['prediction']<0)),events_mean=v['events'].mean(0).tolist(),inference_seconds=seconds))
    primary=interval([(correct[s,'DEND','intact'].astype(float)-correct[s,'POINT','intact'],pairs[s]) for s in range(3)])
    interactions={a:interval([(correct[s,a,'intact'].astype(float)-correct[s,a,'cut_cross'],pairs[s]) for s in range(3)]) for a in ('DEND','POINT')}
    learning={a:interval([(correct[s,a,'intact'].astype(float)-correct[s,a,'initial'],pairs[s]) for s in range(3)]) for a in ('DEND','POINT')}
    result=dict(rows=rows,primary=primary,useful_advantage=bool(primary['difference']>.05 and primary['percentile95'][0]>0),interaction=interactions,learning=learning,seconds=time.perf_counter()-begin,hashes=hashes,selection_sha256=sha(out/'SELECTION.json'))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','hashes')},indent=2))
if __name__=='__main__':main()
