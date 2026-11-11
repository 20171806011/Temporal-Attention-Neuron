from pathlib import Path
import time,numpy as np
import model,engine,base_data,independent
from common import *
tic=time.perf_counter();checks=0;maxerr=0.
def check(v):
    global checks
    assert v;checks+=1
def close(a,b):
    global maxerr
    e=float(np.max(np.abs(a-b)));maxerr=max(maxerr,e);check(e<1e-8)
def candidate(v,w,h):
    wo=w[-2-2*h:-2].reshape(h,2);ob=w[-2:];n,T,_=v['states'].shape
    prev_o=np.concatenate((np.zeros((n,1,2)),v['states'][:,:-1,3*h:]),1)
    prev_p=np.concatenate((np.zeros((n,1,h)),v['spikes'][:,:-1,2*h:3*h]),1)
    return .85*prev_o+np.einsum('nti,ij->ntj',prev_p,wo,optimize=False)+ob
source=read(R/'SOURCE_LOCK.json');old=(R/source['directory']).resolve()
check(sha(old/'MANIFEST.json')==source['manifest_sha256'])
for name,hs in source['files'].items():check(sha(R/name)==hs and sha(old/name)==hs)
check(SIZES==(24,96) and RATES==(.01,.0025) and len(CONDITIONS)==4)
for s in range(3):check(seeds('train',s)['initial']==3931030000+s)
base=base_data.make_base(16,79941001);x,q=assemble(base,24,'clean')
for h in SIZES:
    w=model.initialize(79942001,h)
    for arm in ARMS:
        c,k=model.forward(x[:8],w,arm,h,ret=RET,query=q);v=independent.replay(x[:8],w,arm,h,q)
        check(np.array_equal(c,v['counts']));close(np.stack([a['oc'] for a in k],1),candidate(v,w,h))
        loss,g,cc=model.loss_grad(x,base['y'],w,arm,'CE',h,ret=RET,query=q);check(np.isfinite(g).all());g/=max(1,np.linalg.norm(g))
        m=.1*g;vv=.001*g*g;direction=(m/(1-.9))/(np.sqrt(vv/(1-.999))+1e-8)
        high=w-.01*direction;low=w-.0025*direction;close(high-w,4*(low-w));check(max(abs(high).max(),abs(low).max())<2)
        for lr,weights in zip(RATES,(high,low)):
            c,k=model.forward(x[:8],weights,arm,h,ret=RET,query=q);v=independent.replay(x[:8],weights,arm,h,q);check(np.array_equal(c,v['counts']));close(np.stack([a['oc'] for a in k],1),candidate(v,weights,h))
check(not (R/'train').exists() and not (R/'evaluate').exists())
dump(R/'PRECHECK.json',dict(status='PASS',checks=checks,max_numeric_error=maxerr,wall_seconds=time.perf_counter()-tic,scientific_seed_used=False,fixture_seed=[79941001,79942001],inherited_fixture_source=source['directory']))
print(read(R/'PRECHECK.json'))
