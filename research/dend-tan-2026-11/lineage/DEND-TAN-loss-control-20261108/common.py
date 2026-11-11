from pathlib import Path
from datetime import datetime,timezone
import json,time,ctypes,hashlib,os
import numpy as np
R=Path(__file__).resolve().parent
ARMS=('DEND','FULL','NOATT','GAIN2');RATES=(.0025,);H=6;DELAYS=(0,24,48,96);UPDATES=400
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def memory():
    class M(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    m=M();m.cb=ctypes.sizeof(M);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(m),m.cb);return dict(current_working_set_bytes=m.WorkingSetSize,peak_working_set_bytes=m.PeakWorkingSetSize)
def assemble(base,D):
    assert D in DELAYS;q=16+D;x=np.zeros((len(base['y']),q+8,6));x[:,8:16,:4]=base['write'];x[:,q:,4]=1;return x,q
def basic(c,y):
    p=np.where(c[:,0]==c[:,1],-1,c.argmax(1));rec=[float(np.mean(p[y==j]==j)) for j in (0,1)]
    return dict(balanced_accuracy=sum(rec)/2,accuracy=float(np.mean(p==y)),coverage=float(np.mean(p>=0)),silence=float(np.mean(c.sum(1)==0)),recalls=rec,prediction_counts=[int(sum(p==j)) for j in (-1,0,1)])
def structural_cost(arm,h,T):
    att=arm in ('FULL','NOATT');full=arm=='FULL';N=2*h
    return dict(parameters=2*h*h+17*h+2,spiking_cells=h+2,graded_branches=N,membrane_or_branch_slots=3*h+2,history_slots=5*N if att else 0,persistent_continuous_slots=3*h+2+(5*N if att else 0),persistent_array_bytes_per_episode=(4*h+4+(5*N if att else 0))*8,dense_forward_macs=T*(2*h*h+14*h),state_candidate_updates=T*(3*h+2),graded_local_transfers=T*N,fixed_pair_multiplications=T*h,injection_additions=T*N,history_roll_copy_and_overwrite_writes=6*N*T if att else 0,attention_unit_steps=N*T if att else 0,qkv_multiplications=11*N*T if att else 0,surprise_mean_additions=4*N*T if att else 0,surprise_mean_divisions=N*T if att else 0,gate_tanh_evaluations=N*T if att else 0,gate_context_multiplications=N*T if att else 0,score_multiplications=5*N*T if full else 0,beta_multiplications=5*N*T if full else 0,max_and_cached_argmax_comparisons=8*N*T if full else 0,score_centering_subtractions=5*N*T if full else 0,exponentials=5*N*T if full else 0,normalization_divisions=5*N*T if full else 0,denominator_additions=5*N*T if full else 0,context_multiplications=5*N*T if full else 0,context_additions=4*N*T if att else 0,noatt_context_divisions=N*T if arm=='NOATT' else 0)
class Budget:
    def __init__(self):
        for n,hs in read(R/'FREEZE.json')['files'].items():assert sha(R/n)==hs,'frozen source '+n
        assert not (R/'RUN_STATE.json').exists(),'one-shot science only';self.tic=time.perf_counter();self.cpu=time.process_time();self.stage='initial';self.rows=[];self.macs=0;self.steps=0;self.mark('running')
    def mark(self,status,**other):dump(R/'RUN_STATE.json',dict(status=status,stage=self.stage,pid=os.getpid(),utc=utc(),wall_seconds=time.perf_counter()-self.tic,**other))
    def guard(self,disk=False):
        if time.perf_counter()-self.tic>840:raise TimeoutError('840 second aggregate scientific cap')
        if memory()['current_working_set_bytes']>1024**3:raise MemoryError('1 GiB working-set cap')
        if disk and sum(p.stat().st_size for p in R.rglob('*') if p.is_file())>1.5*1024**3:raise OSError('1.5 GiB new artifact cap')
    def begin(self,stage):
        self.guard();self.stage=stage;self.out=R/stage;self.out.mkdir();self.rows=[];self.stic=time.perf_counter();self.scpu=time.process_time();self.smac=self.macs;self.ssteps=self.steps;self.mark('running')
    def charge(self,n,T):self.macs+=n*T*(2*H*H+14*H);self.steps+=n*T
    def flush(self):
        if self.stage in ('train','evaluate'):dump(self.out/'ROWS.json',self.rows)
    def finish_stage(self,**extra):
        self.guard(disk=True);self.flush();dump(self.out/'SUMMARY.json',dict(utc=utc(),wall_seconds=time.perf_counter()-self.stic,cpu_seconds=time.process_time()-self.scpu,**memory(),dense_forward_macs=self.macs-self.smac,sample_steps=self.steps-self.ssteps,**extra))
    def finish(self):
        self.guard(disk=True);dump(R/'SCIENCE_SUMMARY.json',dict(status='COMPLETE',utc=utc(),fits=36,updates=14400,evaluations=216,wall_seconds=time.perf_counter()-self.tic,cpu_seconds=time.process_time()-self.cpu,**memory(),dense_forward_macs=self.macs,sample_steps=self.steps,cap_seconds=840));self.mark('complete')
    def fail(self,e):
        self.flush();self.mark('stopped',reason=str(e));dump(R/'SCIENCE_FAILURE.json',dict(utc=utc(),stage=self.stage,reason=str(e),exception=type(e).__name__,wall_seconds=time.perf_counter()-self.tic,**memory()))
