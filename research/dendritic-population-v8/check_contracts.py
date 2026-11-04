from pathlib import Path
import importlib.util,json
import numpy as np
import model
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v4',R.parent/'dendritic-population-v7/model.py');v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4)
x=np.random.default_rng(141).uniform(0,1,(4,24,6));t=model.initialize(142);perm=np.array([2,0,3,1]);checks=[]
for a in ('DEND','POINT'):
 c,h=model.forward(x,t,a);c4,h4=v4.forward(x,t,a);assert np.array_equal(c,c4)
 assert all(np.array_equal(p['states'],q['states']) for p,q in zip(h,h4))
 cp,hp=model.forward(x[perm],t,a);assert np.array_equal(c[perm],cp)
 for i in range(4):
  ci,hi=model.forward(x[i:i+1],t,a);assert np.array_equal(ci[0],c[i]);assert all(np.allclose(p['states'][i],q['states'][0],rtol=0,atol=1e-12) for p,q in zip(h,hi))
 no=x.copy();no[:,:8]=0;_,hn=model.forward(no,t,a);assert np.array_equal(hn[7]['states'],np.repeat(hn[7]['states'][:1],4,0))
 assert set(__import__('inspect').signature(model.forward).parameters)=={'x','t','arm','anchor'}
 # Explicit native boundary fixtures; these are not scientific datasets.
 z=np.zeros((2,24,6));zero=np.zeros(model.P);params=model.unpack(zero);params[3][:]=.5;params[5][:]=.5
 _,edge=model.forward(z,zero,a)
 assert np.all(edge[0]['vc']==.5) and np.all(edge[0]['oc']==.5)
 assert not edge[0]['hs'].any() and not edge[0]['os'].any()
 assert edge[1]['hs'].all() and edge[1]['os'].all() and not edge[1]['states'][:,12:].any()
 assert all(np.isin(step[key],[0.,1.]).all() for step in edge for key in ('bs','hs','os'))
 if a=='POINT':
  zero[:]=0;model.unpack(zero)[2][:]=2;_,edge=model.forward(z,zero,a)
  assert np.all(edge[0]['dc']==.5) and not edge[0]['bs'].any()
  assert edge[1]['bs'].all() and not edge[1]['states'][:,:12].any()
 checks.append({'arm':a,'native_parity':'PASS','permutation_split_batch':'PASS','no_history_step8_state_identical':'PASS','label_argument':'absent'})
(R/'CONTRACT_CHECK.json').write_text(json.dumps({'status':'PASS','scope':'Deterministic fixtures only; no scientific split generated','rows':checks},indent=2)+'\n');print('Contract checks PASS')
