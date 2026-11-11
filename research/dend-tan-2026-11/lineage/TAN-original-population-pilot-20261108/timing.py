"""Fixture-only wall timing; no scientific seeds, selection or retained fitted model."""
import time,json
from pathlib import Path
import numpy as np
import native_tan,population,base_data
R=Path(__file__).resolve().parent
def main():
    gs=[]
    for g,D in enumerate((0,24)):
        b=base_data.make_base(64,79965001+g);q=16+D;x=np.zeros((64,q+8,6));x[:,8:16,:4]=b['write'];x[:,q:,4]=1;gs.append((x,b['y'],q))
    rows=[];tic=time.perf_counter()
    for arm in native_tan.ARMS:
        w=population.initialize(79966001,6);m=np.zeros_like(w);v=np.zeros_like(w);times=[]
        for k in range(8):
            if time.perf_counter()-tic>90:raise TimeoutError('90-second fixture timing cap')
            start=time.perf_counter();grad=np.zeros_like(w)
            for x,y,q in gs:grad+=native_tan.loss_grad(x,y,w,arm,6,q)[1]/2
            grad/=max(1,np.linalg.norm(grad));m=.9*m+.1*grad;v=.999*v+.001*grad*grad;w=np.clip(w-.0025*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);times.append(time.perf_counter()-start)
        rows.append(dict(arm=arm,fixture_updates=8,seconds=times,sustained_max_seconds=max(times[4:]),predicted_200_update_fit_seconds=200*max(times[4:])))
    forecast=sum(r['predicted_200_update_fit_seconds']*6 for r in rows)
    out=dict(status='COMPLETE',scientific_seeds_used=False,scientific_fits=0,fixture_updates=32,wall_seconds=time.perf_counter()-tic,rows=rows,forecast_24fits_seconds=forecast,forecast_with_50percent_margin_and_150s_evaluation=1.5*forecast+150,forecast_is_not_guaranteed=True)
    (R/'TIMING.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
