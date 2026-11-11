import time,numpy as np
import model,population,base_data
from design import *
tic=time.perf_counter();b=base_data.make_base(64,79971001);x=assemble(b);rows=[]
for arm,G,L in itertools.product(ARMS,GS,LS):
    w=population.initialize(79972001,6);m=w*0;v=w*0;times=[]
    for k in range(8):
        assert time.perf_counter()-tic<90
        t=time.perf_counter();_,g,_=model.loss_grad(x,b['y'],w,arm,G,L,.0001);g/=max(1,np.linalg.norm(g));m=.9*m+.1*g;v=.999*v+.001*g*g;w=np.clip(w-LR*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);times.append(time.perf_counter()-t)
    rows.append(dict(arm=arm,G=G,L=L,seconds=times))
maxima={arm:max(max(row['seconds'][4:]) for row in rows if row['arm']==arm) for arm in ARMS};forecasts={str(U):1.75*U*24*sum(maxima.values())+90 for U in (150,100,60)};choice=next((U for U in (150,100,60) if forecasts[str(U)]<=840),None)
dump(R/'TIMING.json',dict(status='PASS' if choice else 'INFEASIBLE',scientific_seeds_used=False,wall_seconds=time.perf_counter()-tic,fixture_updates=64,rows=rows,forecasts=forecasts,chosen_updates=choice));assert choice
dump(R/'DESIGN.json',dict(fits=48,updates=choice,learning_rate=LR,arms=list(ARMS),gains=list(GS),couplings=list(LS),regularizers=list(REGS),seeds=3,train_n=64,test_n=256,delay=0,science_cap_seconds=840,audit_cap_seconds=300));print(json.dumps(dict(chosen_updates=choice,forecasts=forecasts)))
