"""Deterministic pre-data parity and masked proxy checks, not task datasets."""
from pathlib import Path
import importlib.util,json
import numpy as np
import model
from common import mask
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('preserved_v2',R.parent/'dendritic-population-v2/model.py')
v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
x=np.random.default_rng(993).uniform(0,1,(2,24,6));y=np.array([0,1]);initial=model.initialize(994);active=mask();rows=[]
assert active.sum()==116 and (active==0).sum()==60
assert np.array_equal(active[72:144].reshape(6,12),np.arange(6)[:,None]==np.arange(12)[None,:]//2)
for arm in ('DEND','POINT'):
 for cut in (False,True):
  theta=initial*active if cut else initial;c,cache=model.forward(x,theta,arm);old=v2.forward(x,initial,arm,cut_cross=cut,trace=True)
  assert np.array_equal(c,old['counts'])
  assert np.array_equal(np.stack([a['states'] for a in cache],1),old['states'])
 theta=initial*active;_,grad,_=model.loss_grad(x,y,theta,arm,'RATE');grad*=active;_,anchor=model.forward(x,theta,arm);errors=[]
 for j in np.flatnonzero(active):
  plus=theta.copy();minus=theta.copy();plus[j]+=1e-6;minus[j]-=1e-6
  fd=(model.objective(x,y,plus,arm,'RATE',anchor)-model.objective(x,y,minus,arm,'RATE',anchor))/2e-6;errors.append(abs(fd-grad[j]))
 assert max(errors)<1e-6 and not grad[active==0].any()
 rows.append(dict(arm=arm,native_intact_and_masked_parity='PASS',active_derivative_checks=116,max_proxy_derivative_error=float(max(errors)),fixed_coordinates=60))
(R/'MASKED_CHECK.json').write_text(json.dumps(dict(status='PASS',scope='Deterministic fixtures only, no scientific data',rows=rows),indent=2)+'\n')
print(json.dumps(rows))
