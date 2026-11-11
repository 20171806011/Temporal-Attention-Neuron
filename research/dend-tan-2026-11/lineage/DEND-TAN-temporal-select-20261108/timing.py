import time,numpy as np
import model,population,task_data
from common import *
tic=time.perf_counter();b=task_data.make_base(64,79991001);x,q=task_data.assemble(b);rows=[]
for arm in ARMS:
    w=population.initialize(79992001,6);m=w*0;v=w*0;times=[]
    for k in range(12):
        assert time.perf_counter()-tic<90;t=time.perf_counter();_,g,_=model.loss_grad(x,b['y'],w,arm,q);g/=max(1,np.linalg.norm(g));m=.9*m+.1*g;v=.999*v+.001*g*g;w=np.clip(w-.0025*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);times.append(time.perf_counter()-t)
    rows.append(dict(arm=arm,seconds=times))
forecast=2*400*6*sum(max(r['seconds'][6:]) for r in rows)+100;dump(R/'TIMING.json',dict(status='PASS' if forecast<840 else 'INFEASIBLE',forecast_with_margin=forecast,wall_seconds=time.perf_counter()-tic,fixture_updates=48,scientific_seeds_used=False,rows=rows));assert forecast<840;print('Forecast with 2x margin +100s:',forecast)
