"""Independent count-space objectives; no neural or training-module imports."""
from pathlib import Path
import json,time
import numpy as np
OBJECTIVES=('CE','COUNT','CECOM')

def components(counts,y):
    c=np.asarray(counts,dtype=float);y=np.asarray(y,dtype=int);n=len(y)
    assert c.shape==(n,2) and n>0 and np.all((y==0)|(y==1))
    z=.5*c;z-=np.max(z,axis=1,keepdims=True);e=np.exp(z);p=e/e.sum(1,keepdims=True)
    ce=-np.log(p[np.arange(n),y]).mean();gce=p.copy();gce[np.arange(n),y]-=1;gce*=.5/n
    summed=c.sum(1)-8.;common=np.mean(summed*summed/32)
    gcommon=np.repeat((summed/(16*n))[:,None],2,axis=1)
    sign=np.full_like(c,-1.);sign[np.arange(n),y]=1.
    margin=np.sum(c*sign,axis=1)-4.;margin_loss=np.mean(margin*margin/32)
    gmargin=sign*(margin/(16*n))[:,None]
    target=np.full_like(c,2.);target[np.arange(n),y]=6.
    count=np.sum((c-target)**2)/(16*n);gcount=(c-target)/(8*n)
    return {'CE':dict(loss=float(ce),gradient=gce),
            'COMMON':dict(loss=float(common),gradient=gcommon),
            'MARGIN':dict(loss=float(margin_loss),gradient=gmargin),
            'COUNT':dict(loss=float(count),gradient=gcount),
            'CECOM':dict(loss=float(ce+common),gradient=gce+gcommon)}

def loss_grad(counts,y,objective):
    assert objective in OBJECTIVES
    part=components(counts,y)[objective];return part['loss'],part['gradient']

def decomposition(counts,y,objective):
    assert objective in OBJECTIVES
    parts=components(counts,y);disc=parts['MARGIN' if objective=='COUNT' else 'CE']
    common=parts['COMMON'] if objective!='CE' else dict(loss=0.,gradient=np.zeros_like(parts['COMMON']['gradient']))
    return dict(discriminative_loss=disc['loss'],common_loss=common['loss'],task_loss=disc['loss']+common['loss'],discriminative_gradient=disc['gradient'],common_gradient=common['gradient'],total_gradient=disc['gradient']+common['gradient'])

def fixture():
    """Fast continuous count-space finite differences, not spike derivatives."""
    tic=time.perf_counter();rng=np.random.default_rng(79992007);c=rng.uniform(-1,9,(7,2));y=rng.integers(0,2,7);records=[];checks=0
    for objective in (*OBJECTIVES,'COMMON','MARGIN'):
        p=components(c,y)[objective];g=p['gradient']
        for _ in range(8):
            direction=rng.normal(size=c.shape);direction/=np.linalg.norm(direction);eps=1e-6
            numeric=(components(c+eps*direction,y)[objective]['loss']-components(c-eps*direction,y)[objective]['loss'])/(2*eps)
            analytic=float(np.sum(g*direction));error=abs(numeric-analytic);assert error<2e-8
            records.append(dict(objective=objective,error=error));checks+=1
    for cc in (c,np.zeros((7,2)),np.full((7,2),8.)):
        p=components(cc,y)
        assert np.allclose(p['COUNT']['loss'],p['COMMON']['loss']+p['MARGIN']['loss'],rtol=0,atol=1e-12)
        assert np.allclose(p['COUNT']['gradient'],p['COMMON']['gradient']+p['MARGIN']['gradient'],rtol=0,atol=1e-12)
        assert np.allclose(p['CECOM']['gradient'],p['CE']['gradient']+p['COMMON']['gradient'],rtol=0,atol=1e-12)
        assert np.allclose(p['CE']['gradient'].sum(1),0,rtol=0,atol=1e-14)
        checks+=4
    # On equal count pairs, COUNT and CE+COMMON have the same count gradient.
    for cc in (np.zeros((7,2)),np.full((7,2),8.)):
        p=components(cc,y);assert np.allclose(p['COUNT']['gradient'],p['CECOM']['gradient'],rtol=0,atol=1e-14);checks+=1
    return dict(status='PASS',checks=checks,fixture_seed=79992007,scientific_seeds_used=False,max_finite_difference_error=max(r['error'] for r in records),wall_seconds=time.perf_counter()-tic,scope='Independent count-space loss decomposition and directional derivatives only; no claim of a true hard-spike derivative.',directions=records)

if __name__=='__main__':
    result=fixture();(Path(__file__).resolve().parent/'LOSS_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='directions'}))
