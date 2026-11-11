from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,ctypes,os
import numpy as np
R=Path(__file__).resolve().parent
CONDITIONS=('nominal','tag_removed','query_only')
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def memory():
    class M(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    m=M();m.cb=ctypes.sizeof(M);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(m),m.cb);return dict(current_working_set_bytes=m.WorkingSetSize,peak_working_set_bytes=m.PeakWorkingSetSize)
def native(c,y):
    pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1));forced=(c[:,1]>c[:,0]).astype(int)
    def ba(p):return float((np.mean(p[y==0]==0)+np.mean(p[y==1]==1))/2)
    silent=np.all(c==0,1);sat=np.all(c==8,1);eq=c[:,0]==c[:,1]
    return dict(strict_ba=ba(pred),forced_ba=ba(forced),coverage=float(np.mean(~eq)),silence=float(silent.mean()),saturated_tie=float(sat.mean()),intermediate_tie=float(np.mean(eq&~silent&~sat)))
class Budget:
    def __init__(self):
        for name,h in read(R/'FREEZE.json')['files'].items():assert sha(R/name)==h,'frozen '+name
        assert not (R/'RUN_STATE.json').exists(),'one shot';self.tic=time.perf_counter();self.cpu=time.process_time();self.stage='init';self.capture_files=0;self.steps=0;self.fits=0;self.mark('running')
    def mark(self,status,**extra):dump(R/'RUN_STATE.json',dict(status=status,stage=self.stage,utc=utc(),wall_seconds=time.perf_counter()-self.tic,**extra))
    def guard(self):
        if time.perf_counter()-self.tic>840:raise TimeoutError('840 second scientific cap')
        if memory()['current_working_set_bytes']>1024**3:raise MemoryError('1GiB working set')
    def captured(self,n):self.capture_files+=1;self.steps+=n*24;self.guard()
    def finish(self):
        self.guard();assert self.capture_files==144 and self.fits==1332 and self.steps==1548288
        size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file());assert size<1.5*1024**3
        dump(R/'SCIENCE_SUMMARY.json',dict(status='COMPLETE',utc=utc(),models=36,neural_training_updates=0,probe_configs=1332,probe_targets=1332*21,probe_evaluations=3996,native_evaluations=108,neural_capture_files=self.capture_files,forward_sample_steps=self.steps,dense_forward_macs=self.steps*156,wall_seconds=time.perf_counter()-self.tic,cpu_seconds=time.process_time()-self.cpu,cap_seconds=840,artifact_bytes_at_science_end=size,**memory()));self.mark('complete')
