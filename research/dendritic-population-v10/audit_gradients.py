"""Audit saved updates and all final surrogate derivatives; no new data/training."""
import json,time
import numpy as np
import model
from common import OUT,dump,REGISTRY

def main():
    start=time.perf_counter();lock=json.loads((OUT/'TRAIN_LOCK.json').read_text());sampled=0;maxsample=0;checks=[]
    for r in lock['rows']:
        h=r['h'];P=model.parameter_count(h);tr=np.load(OUT/r['trace'],allow_pickle=False);d=np.load(OUT/r['data_file'],allow_pickle=False)
        t=np.load(OUT/r['initial_file'],allow_pickle=False).copy();m=np.zeros(P);v=np.zeros(P)
        for k,g in enumerate(tr['clipped_gradients']):
            assert time.perf_counter()-start<300,'300s audit cap'
            if k in REGISTRY['sampled_updates']:
                loss,raw,c=model.loss_grad(d['x'],d['y'],t,r['arm'],r['loss'],h);norm=np.linalg.norm(raw);expected=raw/max(1,norm);err=float(np.max(abs(expected-g)));maxsample=max(maxsample,err)
                assert err<1e-12 and abs(loss-tr['history'][k,0])<1e-12 and abs(norm-tr['history'][k,1])<1e-12
                assert np.array_equal(c,tr['train_counts'][k]);sampled+=1
            m=.9*m+.1*g;v=.999*v+.001*g*g;t=np.clip(t-r['lr']*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
        assert np.array_equal(t,np.load(OUT/r['file'],allow_pickle=False))
        x=d['x'][:2];y=d['y'][:2];_,g,_=model.loss_grad(x,y,t,r['arm'],r['loss'],h);_,anchor=model.forward(x,t,r['arm'],h);err=0;eps=1e-6
        for j in range(P):
            assert time.perf_counter()-start<300,'300s audit cap'
            plus=t.copy();minus=t.copy();plus[j]+=eps;minus[j]-=eps
            fd=(model.objective(x,y,plus,r['arm'],r['loss'],h,anchor)-model.objective(x,y,minus,r['arm'],r['loss'],h,anchor))/(2*eps);err=max(err,abs(fd-g[j]))
        assert err<1e-6,(r['id'],err);checks.append(dict(id=r['id'],parameters=P,max_chain_rule_residual=float(err)))
    result=dict(status='PASS',sampled_updates=sampled,max_sampled_gradient_residual=maxsample,final_checkpoints=len(checks),parameter_derivative_checks=sum(r['parameters'] for r in checks),
        max_chain_rule_residual=max(r['max_chain_rule_residual'] for r in checks),rows=checks,seconds=time.perf_counter()-start,
        limits='Finite differences of locally anchored surrogate graph, not hard-spike derivative. Sampled gradients reuse frozen analytic backward. Native/data/Adam/statistics checker separately written.')
    dump(OUT/'gradient_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
