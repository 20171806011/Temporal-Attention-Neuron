"""Selected-checkpoint local chain rule, plus sampled logged training gradient audit."""
from pathlib import Path
import json,time,importlib.util
import numpy as np
from model import loss_grad,objective,forward,loss_counts,initialize,P
R=Path(__file__).resolve().parent

def main():
    begin=time.perf_counter();out=R/'results';lock=json.loads((out/'SELECTION.json').read_text());log=json.loads((out/'training_log.json').read_text());rows=[];sampled=0;max_sample=0
    # Confirm unchanged native dynamics against the preserved V2 implementation.
    spec=importlib.util.spec_from_file_location('preserved_v2',R.parent/'dendritic-population-v2/model.py');v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
    fixture=np.random.default_rng(9).uniform(0,1,(3,24,6));theta=initialize(19)
    for arm in ('DEND','POINT'):
        counts,cache=forward(fixture,theta,arm);v=v2.forward(fixture,theta,arm,trace=True)
        assert np.array_equal(counts,v['counts']) and np.array_equal(np.stack([q['states'] for q in cache],1),v['states'])
    for z in log:
        d=np.load(out/f"development_{z['seed']}.npz",allow_pickle=False);tr=np.load(out/z['trace'],allow_pickle=False);t=np.load(out/f"initial_{z['seed']}.npy",allow_pickle=False).copy();m=np.zeros(P);v=np.zeros(P)
        for k,g in enumerate(tr['clipped_gradients']):
            if k in (0,99,199):
                loss,raw,c=loss_grad(d['x'],d['y'],t,z['arm'],z['loss']);norm=np.linalg.norm(raw);clipped=raw/max(1,norm);err=float(np.max(abs(clipped-g)));max_sample=max(max_sample,err)
                assert err<1e-12 and abs(loss-tr['history'][k,0])<1e-12 and abs(norm-tr['history'][k,1])<1e-12
                pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1));assert float(np.mean(pred==d['y']))==tr['history'][k,3];sampled+=1
            m=.9*m+.1*g;v=.999*v+.001*g*g;t=np.clip(t-z['lr']*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
    for z in lock['selected']:
        d=np.load(out/f"development_{z['seed']}.npz",allow_pickle=False);x=d['x'][:2];y=d['y'][:2];t=np.load(out/z['file'],allow_pickle=False);loss,g,c=loss_grad(x,y,t,z['arm'],z['loss']);_,anchor=forward(x,t,z['arm']);eps=1e-6;fd=np.zeros(P)
        for j in range(P):
            plus=t.copy();minus=t.copy();plus[j]+=eps;minus[j]-=eps
            fd[j]=(objective(x,y,plus,z['arm'],z['loss'],anchor)-objective(x,y,minus,z['arm'],z['loss'],anchor))/(2*eps)
        error=float(np.max(abs(fd-g)));assert error<1e-6,(z['seed'],z['arm'],error)
        rows.append(dict(seed=z['seed'],arm=z['arm'],loss=z['loss'],max_chain_rule_residual=error,parameters=P))
    controls=[]
    for name,counts in [('silence',[0.,0.]),('both_saturated',[8.,8.]),('correct_target',[6.,1.])]:
        c=np.array([counts]);controls.append(dict(pattern=name,counts=counts,CE=loss_counts(c,np.array([0]),'CE')[0],RATE=loss_counts(c,np.array([0]),'RATE')[0]))
    result=dict(status='PASS',native_v2_parity=True,sampled_training_updates=sampled,max_sampled_gradient_residual=max_sample,selected_parameter_derivative_checks=6*P,rows=rows,analytic_loss_controls=controls,seconds=time.perf_counter()-begin,limits='Local surrogate chain rule, not true hard-spike derivative. Sampled gradient recomputation uses frozen analytic code; independent native recurrence and Adam replay are in verify.py. No post-test training or tuning.')
    (out/'gradient_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
