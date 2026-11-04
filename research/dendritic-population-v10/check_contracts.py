"""Deterministic non-scientific fixtures; never calls the task data generator."""
from pathlib import Path
import importlib.util,json,inspect
import numpy as np
import model
R=Path(__file__).resolve().parent

def main():
    spec=importlib.util.spec_from_file_location('frozen_v8',R.parent/'dendritic-population-v8/model.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    x=np.random.default_rng(141).uniform(0,1,(4,24,6));perm=np.array([2,0,3,1]);checks=[]
    assert np.array_equal(model.initialize(142,6),old.initialize(142))
    for h in (6,4,2,1):
        theta=model.initialize(142,h)
        for arm in ('DEND','POINT'):
            c,history=model.forward(x,theta,arm,h)
            if h==6:
                co,ho=old.forward(x,theta,arm);assert np.array_equal(c,co)
                assert all(np.array_equal(p[key],q[key]) for p,q in zip(history,ho) for key in p)
                y=np.array([0,1,0,1]);lo,go,co=old.loss_grad(x,y,theta,arm,'RATE');ln,gn,cn=model.loss_grad(x,y,theta,arm,'RATE',h)
                assert lo==ln and np.array_equal(go,gn) and np.array_equal(co,cn)
            cp,hp=model.forward(x[perm],theta,arm,h);assert np.array_equal(c[perm],cp)
            for i in range(4):
                ci,hi=model.forward(x[i:i+1],theta,arm,h);assert np.array_equal(ci[0],c[i])
                assert all(np.allclose(p['states'][i],q['states'][0],rtol=0,atol=1e-12) for p,q in zip(history,hi))
            assert set(inspect.signature(model.forward).parameters)=={'x','t','arm','h','anchor'}
            z=np.zeros((2,24,6));zero=np.zeros(model.parameter_count(h));params=model.unpack(zero,h);params[3][:]=.5;params[5][:]=.5
            _,edge=model.forward(z,zero,arm,h)
            assert np.all(edge[0]['vc']==.5) and np.all(edge[0]['oc']==.5)
            assert not edge[0]['hs'].any() and not edge[0]['os'].any()
            assert edge[1]['hs'].all() and edge[1]['os'].all() and not edge[1]['states'][:,2*h:].any()
            assert all(np.isin(step[key],[0.,1.]).all() for step in edge for key in ('bs','hs','os'))
            if arm=='POINT':
                zero[:]=0;model.unpack(zero,h)[2][:]=2;_,edge=model.forward(z,zero,arm,h)
                assert np.all(edge[0]['dc']==.5) and not edge[0]['bs'].any()
                assert edge[1]['bs'].all() and not edge[1]['states'][:,:2*h].any()
            checks.append(dict(h=h,arm=arm,native_parity='exact frozen V8 forward/backward/init' if h==6 else 'same equations, dimension generalized',
                permutation_split_batch='PASS',strict_threshold_reset_binary='PASS',label_argument='absent'))
    (R/'CONTRACT_CHECK.json').write_text(json.dumps(dict(status='PASS',scope='Deterministic fixtures only; no scientific split generated',rows=checks),indent=2)+'\n')
    print('V10 contract fixtures PASS')
if __name__=='__main__':main()
