import time,numpy as np
import task_data,independent_data,model,population
from common import *
def main():
    tic=time.perf_counter();rows=[];errs=[]
    for n,seed in ((64,79993001),(256,79993002)):
        a=task_data.make_base(n,seed);b=independent_data.make_base(n,seed)
        assert set(a)==set(b)
        for k in a:assert np.array_equal(a[k],b[k]),k
        for condition in task_data.CONDITIONS:
            x,q=task_data.assemble(a,condition);rec=independent_data.check(b,x,condition);rows.append(dict(seed=seed,condition=condition,**rec))
    b=task_data.make_base(64,79993001);x,q=task_data.assemble(b);w=population.initialize(79994001,6);rng=np.random.default_rng(79995001)
    for arm in ARMS:
        loss,g,c=model.loss_grad(x,b['y'],w,arm,q);_,anchor=model.forward(x,w,arm,q);assert np.linalg.norm(g-.0002*w/176)>1e-6,'nontrivial fixture task gradient'
        for j in range(4):
            d=rng.normal(size=176);d/=np.linalg.norm(d);eps=1e-6;numeric=(model.objective(x,b['y'],w+eps*d,arm,q,anchor)-model.objective(x,b['y'],w-eps*d,arm,q,anchor))/(2*eps);e=abs(numeric-g@d);assert e<2e-5+2e-4*max(abs(numeric),abs(g@d));errs.append(float(e))
        for condition in ('tag_removed','query_only'):
            xx,qq=task_data.assemble(b,condition);z=model.evaluate(xx,w,arm,qq);m=basic(z['counts'],b['y']);assert abs(m['balanced_accuracy']-.5*m['coverage'])<1e-12
            for i,j in independent_data.pairs(b):assert np.array_equal(z['counts'][i],z['counts'][j])
    for name,hs in read(R/'SOURCE_LOCK.json')['sources'].items():assert sha(R/name)==hs,'unchanged source '+name
    dump(R/'DATA_CHECK.json',dict(status='PASS',wall_seconds=time.perf_counter()-tic,scientific_seeds_used=False,fixture_data_sets=rows,gradient_directions=len(errs),max_task_gradient_fd_error=max(errs),negative_control_native_checks=8,scope='Independent generation, exact counterbalanced inputs, query/label/amplitude controls, identical native predictions and BA=.5coverage on missing information; unchanged source hashes; nontrivial task-shaped proxy derivatives.'));print('DATA CHECK PASS',max(errs))
if __name__=='__main__':
    try:main()
    except Exception as e:dump(R/f'DATA_CHECK_FAILURE_{len(list(R.glob("DATA_CHECK_FAILURE_*.json")))+1}.json',dict(reason=str(e)));raise
