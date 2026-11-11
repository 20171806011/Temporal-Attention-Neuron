"""Performance-blind timing fixtures, disjoint from scientific seeds."""
from pathlib import Path
import time,json,ctypes,sys,os,platform
import numpy as np
import model,base_data
R=Path(__file__).resolve().parent
def peak():
    class M(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    m=M();m.cb=ctypes.sizeof(M);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(m),m.cb);return m.PeakWorkingSetSize
def main():
    assert not (R/'PILOT.json').exists();start=time.perf_counter();records=[]
    groups=[]
    for g,D in enumerate((0,24)):
        b=base_data.make_base(64,7993100+g);x,q=base_data.assemble(b,D,'clean');groups.append((x,b['y'],q))
    for h in (24,48,96):
        for arm in ('DEND','POINT'):
            w=model.initialize(7993200+h,h);m=np.zeros_like(w);v=m.copy();samples=[]
            for k in range(13):
                tic=time.perf_counter();g=np.zeros_like(w)
                for x,y,q in groups:
                    _,gg,_=model.loss_grad(x,y,w,arm,'CE',h,ret=(.95,.85),query=q);g+=gg/2
                assert np.isfinite(g).all();g/=max(1,np.linalg.norm(g));m=.9*m+.1*g;v=.999*v+.001*g*g
                w=np.clip(w-.01*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
                if k:samples.append(time.perf_counter()-tic)
                if time.perf_counter()-start>90:raise TimeoutError('90s pilot cap')
            row=dict(h=h,arm=arm,timed_updates=12,warmup_updates=1,seconds_per_update=float(np.mean(samples)),p95_update_seconds=float(np.quantile(samples,.95)),projected_200_update_fit_seconds=200*float(np.mean(samples))+.5,process_peak_working_set_bytes=peak(),parameters=model.parameter_count(h))
            records.append(row);print(json.dumps(row),flush=True)
    result=dict(machine=platform.node(),python=sys.version,executable=sys.executable,numpy=np.__version__,threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        data_seeds=[7993100,7993101],init_seed_rule='7993200+h',scientific_data_generated=False,performance_or_accuracy_used_for_selection=False,fixture_updates=78,wall_seconds=time.perf_counter()-start,records=records)
    (R/'PILOT.json').write_text(json.dumps(result,indent=2)+'\n');print('PILOT COMPLETE',result['wall_seconds'])
if __name__=='__main__':main()
