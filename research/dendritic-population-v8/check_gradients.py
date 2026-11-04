"""Check local surrogate chain rule, never equate it to a hard-spike derivative."""
from pathlib import Path
import json,numpy as np
from model import initialize,forward,objective,loss_grad,P,unpack
R=Path(__file__).resolve().parent

def main():
    rng=np.random.default_rng(991);x=rng.uniform(0,1,(2,24,6));y=np.array([0,1]);theta=initialize(992);rows=[]
    # Native quiet state, strict boundary, reset and binary spikes.
    zero=np.zeros(P);z=np.zeros_like(x)
    for arm in ('DEND','POINT'):
        c,a=forward(z,zero,arm);assert not c.any() and not any(e['hs'].any() for e in a)
        unpack(zero)[5][:]=.5;c,a=forward(z,zero,arm)
        assert not a[0]['os'].any() and a[1]['os'].all() and not a[1]['states'][:,18:].any()
        zero[:]=0
        for kind in ('CE','RATE'):
            l,g,c=loss_grad(x,y,theta,arm,kind);_,anchor=forward(x,theta,arm)
            eps=1e-6;finite=np.zeros(P);hard=np.zeros(P)
            for j in range(P):
                plus=theta.copy();minus=theta.copy();plus[j]+=eps;minus[j]-=eps
                finite[j]=(objective(x,y,plus,arm,kind,anchor)-objective(x,y,minus,arm,kind,anchor))/(2*eps)
                hard[j]=(objective(x,y,plus,arm,kind)-objective(x,y,minus,arm,kind))/(2*eps)
            error=float(np.max(abs(finite-g)));assert error<1e-6,(arm,kind,error)
            # At this non-boundary fixture true hard-output loss is locally constant apart from L2.
            harderror=float(np.max(abs(hard-.0002*theta/P)));assert harderror<1e-7
            _,gz,_=loss_grad(x,y,theta,arm,kind,True);assert np.allclose(gz,.0002*theta/P,rtol=0,atol=1e-15)
            assert all(np.isin(a[name],[0.,1.]).all() for a in anchor for name in ('bs','hs','os'))
            rows.append(dict(arm=arm,loss=kind,max_surrogate_chain_rule_error=error,max_hard_loss_minus_regularizer_error=harderror,gradient_norm=float(np.linalg.norm(g)),parameter_derivatives=P))
    result={'status':'PASS','finite_difference_parameter_checks':4*P,'native_fixture_checks':'quiet, strict threshold, reset, binary forward','meaning':'Central differences on a locally anchored differentiable graph validate the explicit surrogate chain rule. They are NOT derivatives of the hard spike network.','rows':rows}
    (R/'GRADIENT_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
