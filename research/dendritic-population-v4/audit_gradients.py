"""Post-selection audit only. No training, data generation, or parameter selection."""
import json,time
import numpy as np
import model
from common import R,OUT,mask,dump

def main():
    start=time.perf_counter();dev=json.loads((OUT/'DEVELOPMENT_LOCK.json').read_text());rows=json.loads((OUT/'development_log.json').read_text())
    if (OUT/'replication_log.json').exists():rows+=json.loads((OUT/'replication_log.json').read_text())
    unique={r['trace']:r for r in rows if r['trace']};sampled=0;max_sample=0
    for trace,r in unique.items():
        d=np.load(OUT/r['data_file'],allow_pickle=False);tr=np.load(OUT/trace,allow_pickle=False);theta=np.load(OUT/r['initial_file'],allow_pickle=False).copy();active=mask() if r['condition']=='retrained' else np.ones(176);m=np.zeros(176);v=np.zeros(176)
        probes=(0,99,199) if r['method']=='ADAM' else (0,49,99)
        for k,g in enumerate(tr['clipped_gradients']):
            if k in probes:
                if r['method']=='ADAM':
                    loss,raw,_=model.loss_grad(d['x'],d['y'],theta,r['arm'],r['loss']);raw*=active;norm=np.linalg.norm(raw);expected=raw/max(1,norm)
                    assert abs(loss-tr['history'][k,0])<1e-12 and abs(norm-tr['history'][k,1])<1e-12
                else:
                    direction=tr['directions'][k];c=.15/(k+1)**.101
                    plus=model.objective(d['x'],d['y'],theta+c*direction,r['arm'],r['loss']);minus=model.objective(d['x'],d['y'],theta-c*direction,r['arm'],r['loss'])
                    assert abs(plus-tr['history'][k,0])<1e-12 and abs(minus-tr['history'][k,1])<1e-12
                    expected=np.clip((plus-minus)/(2*c)*direction,-5,5)
                err=float(np.max(abs(expected-g)));max_sample=max(max_sample,err);assert err<1e-12;sampled+=1
            if r['method']=='ADAM':
                m=(.9*m+.1*g)*active;v=(.999*v+.001*g*g)*active;theta=np.clip(theta-r['lr']*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)*active
            else:theta=np.clip(theta-r['lr']/(k+1)**.602*g,-2,2)
    selected=[]
    for r in rows:
        if r['method']!='ADAM':continue
        if r['stage']=='A':
            cfg=next(z['selected'] for z in dev['selections'] if z['budget']==r['steps'])
            if (r['loss'],r['lr'])!=(cfg['loss'],cfg['lr']):continue
        selected.append(r)
    derivatives=0;checks=[]
    for r in selected:
        d=np.load(OUT/r['data_file'],allow_pickle=False);x=d['x'][:2];y=d['y'][:2];theta=np.load(OUT/r['file'],allow_pickle=False)
        _,raw,_=model.loss_grad(x,y,theta,r['arm'],r['loss']);_,anchor=model.forward(x,theta,r['arm']);active=mask() if r['condition']=='retrained' else np.ones(176);indices=np.flatnonzero(active);errors=[];eps=1e-6
        for j in indices:
            plus=theta.copy();minus=theta.copy();plus[j]+=eps;minus[j]-=eps
            fd=(model.objective(x,y,plus,r['arm'],r['loss'],anchor)-model.objective(x,y,minus,r['arm'],r['loss'],anchor))/(2*eps);errors.append(abs(fd-raw[j]))
        err=float(max(errors));assert err<1e-6,(r['id'],err);derivatives+=len(indices)
        checks.append({'id':r['id'],'parameters_checked':len(indices),'max_chain_rule_residual':err})
    result=dict(status='PASS',sampled_training_updates=sampled,max_sampled_gradient_residual=max_sample,selected_checkpoints=len(checks),parameter_derivative_checks=derivatives,max_chain_rule_residual=max(z['max_chain_rule_residual'] for z in checks),rows=checks,seconds=time.perf_counter()-start,limits='Local anchored surrogate chain rule, not exact hard-spike loss derivative. Samples use frozen analytic code; independent dynamics/optimizer reconstruction are in verify_v4.py. No post-test tuning.')
    dump(OUT/'gradient_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
