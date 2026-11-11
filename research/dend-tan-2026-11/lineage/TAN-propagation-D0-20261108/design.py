from pathlib import Path
import itertools,json,hashlib
import numpy as np
R=Path(__file__).resolve().parent
ARMS=('TAN_FULL','TAN_NOATT');GS=(1.,4.);LS=(.5,2.);REGS=(.0001,0.);LR=.0025
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def assemble(b):
    x=np.zeros((len(b['y']),24,6));x[:,8:16,:4]=b['write'];x[:,16:,4]=1;return x
def conditions():
    return [dict(id=f's{s}_a{a}_g{g}_l{l}_r{r}',seed=s,arm=arm,G=G,L=L,reg=reg) for s in range(3) for g,G in enumerate(GS) for l,L in enumerate(LS) for r,reg in enumerate(REGS) for a,arm in enumerate(ARMS)]
def probes():
    rng=np.random.default_rng(3964010000);n=16;out={};amps=np.linspace(.8,1.2,n)
    z=np.zeros((n,24,6));out['external_zero']=z.copy()
    x=z.copy();x[:,:,:4]=amps[:,None,None];out['external_constant']=x
    x=z.copy();x[:,8:16,:4]=amps[:,None,None];out['external_step']=x
    x=z.copy();x[:,8:,:4]=(np.arange(16)%2)[None,:,None]*amps[:,None,None];out['external_alternating']=x
    x=z.copy();x[:,8:,:4]=rng.uniform(0,1.2,(n,16,4));out['external_irregular']=x
    bits=rng.integers(0,2,(n,4));amp=np.where(bits,rng.uniform(.8,1.2,(n,4)),rng.uniform(0,.1,(n,4)));x=z.copy();x[:,8:16,:4]=np.maximum(0,amp[:,None,:]+rng.normal(0,.03,(n,8,4)));x[:,16:,4]=1;out['task_shaped_unlabelled']=x
    return out
