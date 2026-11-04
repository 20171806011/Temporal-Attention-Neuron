"""Audit saved updates and final surrogate chain rule; no training/new data."""
import json,time
import numpy as np
import model
from common import OUT,dump,REGISTRY

def main():
    start=time.perf_counter();lock=json.loads((OUT/'TRAIN_LOCK.json').read_text());sampled=0;maxsample=0;checks=[]
    for r in lock['rows']:
        tr=np.load(OUT/r['trace'],allow_pickle=False);d=np.load(OUT/r['data_file'],allow_pickle=False);t=np.load(OUT/r['initial_file'],allow_pickle=False).copy();m=np.zeros(176);v=np.zeros(176)
        for k,g in enumerate(tr['clipped_gradients']):
            if k in REGISTRY['sampled_updates']:
                loss,raw,c=model.loss_grad(d['x'],d['y'],t,r['arm'],r['loss']);norm=np.linalg.norm(raw);expected=raw/max(1,norm);err=float(np.max(abs(expected-g)));maxsample=max(maxsample,err)
                assert err<1e-12 and abs(loss-tr['history'][k,0])<1e-12 and abs(norm-tr['history'][k,1])<1e-12
                assert np.array_equal(c,tr['train_counts'][k]);sampled+=1
            m=.9*m+.1*g;v=.999*v+.001*g*g;t=np.clip(t-r['lr']*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
        assert np.array_equal(t,np.load(OUT/r['file'],allow_pickle=False))
        x=d['x'][:2];y=d['y'][:2];_,g,_=model.loss_grad(x,y,t,r['arm'],r['loss']);_,anchor=model.forward(x,t,r['arm']);err=0;eps=1e-6
        for j in range(176):
            plus=t.copy();minus=t.copy();plus[j]+=eps;minus[j]-=eps
            fd=(model.objective(x,y,plus,r['arm'],r['loss'],anchor)-model.objective(x,y,minus,r['arm'],r['loss'],anchor))/(2*eps);err=max(err,abs(fd-g[j]))
        assert err<1e-6,(r['id'],err);checks.append(dict(id=r['id'],parameters=176,max_chain_rule_residual=float(err)))
    result=dict(status='PASS',sampled_updates=sampled,max_sampled_gradient_residual=maxsample,final_checkpoints=len(checks),parameter_derivative_checks=len(checks)*176,max_chain_rule_residual=max(r['max_chain_rule_residual'] for r in checks),rows=checks,seconds=time.perf_counter()-start,limits='Finite differences on locally anchored surrogate graph, not hard-spike derivative. Sampled gradients reuse frozen analytic backward. Native/data/Adam/statistics checker separately written.')
    dump(OUT/'gradient_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
