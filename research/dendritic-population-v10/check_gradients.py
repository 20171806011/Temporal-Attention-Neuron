"""Finite differences of the anchored surrogate graph, not hard-spike derivatives."""
from pathlib import Path
import json,time,numpy as np
import model
R=Path(__file__).resolve().parent

def main():
    start=time.perf_counter();x=np.random.default_rng(991).uniform(0,1,(2,24,6));y=np.array([0,1]);rows=[]
    for h in (6,4,2,1):
        theta=model.initialize(992,h);P=model.parameter_count(h)
        for arm in ('DEND','POINT'):
            _,g,_=model.loss_grad(x,y,theta,arm,'RATE',h);_,anchor=model.forward(x,theta,arm,h);eps=1e-6;finite=np.zeros(P);hard=np.zeros(P)
            for j in range(P):
                assert time.perf_counter()-start<300,'300s audit cap'
                plus=theta.copy();minus=theta.copy();plus[j]+=eps;minus[j]-=eps
                finite[j]=(model.objective(x,y,plus,arm,'RATE',h,anchor)-model.objective(x,y,minus,arm,'RATE',h,anchor))/(2*eps)
                hard[j]=(model.objective(x,y,plus,arm,'RATE',h)-model.objective(x,y,minus,arm,'RATE',h))/(2*eps)
            error=float(np.max(abs(finite-g)));assert error<1e-6,(h,arm,error)
            harderror=float(np.max(abs(hard-.0002*theta/P)));assert harderror<1e-7,(h,arm,harderror)
            _,gz,_=model.loss_grad(x,y,theta,arm,'RATE',h,True);assert np.allclose(gz,.0002*theta/P,rtol=0,atol=1e-15)
            rows.append(dict(h=h,arm=arm,loss='RATE',max_surrogate_chain_rule_error=error,max_hard_loss_minus_regularizer_error=harderror,parameter_derivatives=P))
    result=dict(status='PASS',finite_difference_parameter_checks=sum(r['parameter_derivatives'] for r in rows),seconds=time.perf_counter()-start,
        meaning='Central differences of a locally anchored differentiable graph validate the surrogate chain rule, NOT a derivative of the hard-spike model.',rows=rows)
    (R/'GRADIENT_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
