import time,numpy as np
import training,independent_loss,task_data,model,population
from common import *
tic=time.perf_counter();checks=0;errors=[];component_errors=[]
def check(v,message):
    global checks
    checks+=1;assert bool(v),message
def close(a,b,message):check(np.max(np.abs(np.asarray(a)-np.asarray(b)))<1e-9,message)
def main():
    b=task_data.make_base(64,79997001);xx,q=task_data.assemble(b);x=xx[:8];y=b['y'][:8];w=population.initialize(79997002,6);rng=np.random.default_rng(79997003)
    for arm in ARMS:
        c,anchor=model.forward(x,w,arm,q)
        for kind in training.LOSSES:
            loss,g,c,terms,grads=training.loss_grad(x,y,w,arm,q,kind,True)
            il,ig=independent_loss.loss_grad(c,y,kind);disc,common,gd,gc=training.count_terms(c,y,kind)
            close(disc+common,il,'independent objective');close(gd+gc,ig,'independent count gradients');close(grads.sum(0),g,'parameter gradient decomposition');close(loss,training.objective(x,y,w,arm,q,kind,anchor),'anchored objective')
            close(grads[0],training.backward(x,w,arm,q,anchor,gd),'BPTT linearity for discriminative term')
            if kind=='CE':
                old=model.loss_grad(x,y,w,arm,q);close(old[0],loss,'unchanged CE loss');close(old[1],g,'unchanged CE BPTT');close(old[2],c,'unchanged CE counts')
            for j in range(6):
                d=rng.normal(size=176);d/=np.linalg.norm(d);eps=1e-6
                numeric=(training.objective(x,y,w+eps*d,arm,q,kind,anchor)-training.objective(x,y,w-eps*d,arm,q,kind,anchor))/(2*eps);error=abs(numeric-g@d);errors.append(float(error));check(error<2e-5+2e-4*max(abs(numeric),abs(g@d)),'objective proxy finite difference')
                for term in (0,1):
                    vals=[]
                    for ww in (w+eps*d,w-eps*d):
                        cc=model.forward(x,ww,arm,q,anchor)[0];vals.append(training.count_terms(cc,y,kind)[term])
                    numeric=(vals[0]-vals[1])/(2*eps);error=abs(numeric-grads[term]@d);component_errors.append(float(error));check(error<2e-5+2e-4*max(abs(numeric),abs(grads[term]@d)),'component proxy finite difference')
    dump(R/'GRADIENT_CHECK.json',dict(status='PASS',checks=checks,directions=len(errors),component_directions=len(component_errors),max_fd_error=max(errors),max_component_fd_error=max(component_errors),wall_seconds=time.perf_counter()-tic,scientific_seeds_used=False,scope='Independent count-space formulas; unchanged CE loss and BPTT; separate component BPTT linearity; finite differences of anchored surrogate extension, not hard-spike true derivatives.'));print('GRADIENT CHECK PASS',checks,max(errors))
if __name__=='__main__':
    try:main()
    except Exception as e:dump(R/f'GRADIENT_CHECK_FAILURE_{len(list(R.glob("GRADIENT_CHECK_FAILURE_*.json")))+1}.json',dict(reason=str(e),checks=checks));raise
