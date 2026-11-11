import time,numpy as np
import model,population,base_data
from common import *
tic=time.perf_counter();groups=[]
for j,D in enumerate((0,24)):
    b=base_data.make_base(64,79981001+j);x,q=assemble(b,D);groups.append((x,b['y'],q))
rows=[]
for arm in ARMS:
    w=population.initialize(79982001,6);m=w*0;v=w*0;times=[]
    for k in range(8):
        assert time.perf_counter()-tic<90;t=time.perf_counter();g=w*0
        for x,y,q in groups:g+=model.loss_grad(x,y,w,arm,q)[1]/2
        g/=max(1,np.linalg.norm(g));m=.9*m+.1*g;v=.999*v+.001*g*g;w=np.clip(w-.0025*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);times.append(time.perf_counter()-t)
    rows.append(dict(arm=arm,seconds=times))
forecast=2*200*6*sum(max(row['seconds'][4:]) for row in rows)+100
dump(R/'TIMING.json',dict(status='PASS' if forecast<840 else 'INFEASIBLE',forecast_with_margin=forecast,wall_seconds=time.perf_counter()-tic,fixture_updates=32,scientific_seeds_used=False,rows=rows));assert forecast<840;print('Forecast with 2x margin and 100s eval allowance:',forecast)
