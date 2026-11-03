"""Postselection audit, without training or data generation."""
import json,time
import numpy as np
import model
from common import OUT,dump

def main():
    start=time.perf_counter();pre=json.loads((OUT/'PREPARE_LOCK.json').read_text());fin=json.loads((OUT/'FINAL_LOCK.json').read_text());rows=pre['rows']+fin['rows'];unique={r['trace']:r for r in rows};samples=0;maxsample=0
    for trace,r in unique.items():
        tr=np.load(OUT/trace,allow_pickle=False);d=np.load(OUT/r['data_file'],allow_pickle=False);t=np.load(OUT/r['initial_file'],allow_pickle=False).copy();m=np.zeros(176);v=np.zeros(176)
        for k,g in enumerate(tr['clipped_gradients']):
            if k in (0,99,199):
                loss,raw,_=model.loss_grad(d['x'],d['y'],t,r['arm'],r['loss']);norm=np.linalg.norm(raw);expected=raw/max(1,norm);err=float(np.max(abs(expected-g)));maxsample=max(maxsample,err);assert err<1e-12
                assert abs(loss-tr['history'][k,0])<1e-12 and abs(norm-tr['history'][k,1])<1e-12;samples+=1
            m=.9*m+.1*g;v=.999*v+.001*g*g;t=np.clip(t-r['lr']*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
    chosen={}
    for r in rows:
        if r['stage']=='search':
            q=pre['selection']['selected'][r['arm']]
            if (r['loss'],r['lr'])!=(q['loss'],q['lr']):continue
        chosen[r['file']]=r
    checks=[]
    for file,r in chosen.items():
        d=np.load(OUT/r['data_file'],allow_pickle=False);x=d['x'][:2];y=d['y'][:2];t=np.load(OUT/file,allow_pickle=False);_,g,_=model.loss_grad(x,y,t,r['arm'],r['loss']);_,anchor=model.forward(x,t,r['arm']);err=0;eps=1e-6
        for j in range(176):
            plus=t.copy();minus=t.copy();plus[j]+=eps;minus[j]-=eps
            fd=(model.objective(x,y,plus,r['arm'],r['loss'],anchor)-model.objective(x,y,minus,r['arm'],r['loss'],anchor))/(2*eps);err=max(err,abs(fd-g[j]))
        assert err<1e-6,(file,err);checks.append({'file':file,'max_chain_rule_residual':float(err),'parameters':176})
    result=dict(status='PASS',sampled_updates=samples,max_sampled_gradient_residual=maxsample,unique_selected_weights=len(checks),parameter_derivative_checks=len(checks)*176,max_chain_rule_residual=max(r['max_chain_rule_residual'] for r in checks),rows=checks,seconds=time.perf_counter()-start,limits='Locally anchored surrogate chain rule, not true hard-spike derivative; sampled gradients recomputed by frozen analytic code, independent native/optimizer replay in verify.py. Aliases deduplicated.')
    dump(OUT/'gradient_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
