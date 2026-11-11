import time,numpy as np
import model,engine,independent,base_data
from common import *
tic=time.perf_counter();checks=0;maxerr=0.
def check(v):
    global checks
    assert v;checks+=1
def close(a,b):
    global maxerr
    e=float(np.max(abs(a-b)));maxerr=max(maxerr,e);check(e<1e-8)
source=read(R/'SOURCE_LOCK.json');old=(R/source['directory']).resolve();check(sha(old/'MANIFEST.json')==source['manifest_sha256']);check(read(old/'AUDIT.json')['status']=='PASS')
for n in ('model.py','base_data.py'):check(sha(R/n)==sha(old/n))
planned=cases();check(len(planned)==36);check(len({(c['delay'],c['condition']) for c in planned})==36)
for c in planned:
    if 'cut' in c:check(c['cut'][0]==28 and c['cut'][1]<=16+c['delay'])
base=base_data.make_base(16,79951001)
for h in (24,96):
    w=model.initialize(79952001,h)
    for arm in ('DEND','POINT'):
        chosen=[c for c in planned if (c['delay'],c['condition']) in ((24,'baseline'),(24,'zero_d'),(24,'zero_vp'),(24,'zero_all'),(24,'identity'),(24,'cut_L12_R0'),(28,'cut_L4_R12'),(28,'keepoff_L4_R12'))]
        chosen+=[dict(delay=24,condition='whole_hold_fixture',cut=[16,40]),dict(delay=24,condition='empty_cut_fixture',cut=[28,28])]
        for case in chosen:
            x,q=assemble(base,case['delay'],'clean');got,cost=engine.evaluate(x,w,arm,h,q,case);ref=independent.replay(x,w,arm,h,q,case=case)
            for k in ('counts','spikes'):check(np.array_equal(got[k],ref[k]))
            for k in ('states','resources'):close(got[k],ref[k])
            expected_cut=sum(case.get('cut',(-1,-1))[0]<=t<case.get('cut',(-1,-1))[1] for t in range(len(x[0])));check(cost['cut_steps']==expected_cut)
            name=case['condition'];inter=None
            if name in ('zero_d','zero_all'):inter=dict(kind='zero',tick=28,part=name.split('_')[1])
            if name=='identity':inter=dict(kind='sham',tick=28)
            if name=='whole_hold_fixture':inter=dict(kind='cut_recurrence')
            if name in ('baseline','empty_cut_fixture','zero_d','zero_all','identity','whole_hold_fixture'):
                c,k=model.forward(x,w,arm,h,ret=RET,query=q,intervention=inter);check(np.array_equal(c,got['counts']));close(np.stack([a['states'][:8] for a in k],1),got['states'])
check(not (R/'evaluate').exists());dump(R/'PRECHECK.json',dict(status='PASS',checks=checks,max_numeric_error=maxerr,wall_seconds=time.perf_counter()-tic,scientific_seeds_used=False,fixture_seeds=[79951001,79952001],scope='Native and independent window timing/reset/cost fixtures plus inherited baseline/full-hold/zero/sham equations'))
print(read(R/'PRECHECK.json'))
