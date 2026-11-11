from pathlib import Path
import json,time,ctypes,hashlib,os
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parent
SIZES=(24,96);ARMS=('DEND','POINT');RATES=(.01,.0025);RET=(.95,.85)
CHECKPOINTS=(0,1,2,5,10,25,50,100,199)
CONDITIONS=((0,'clean'),(24,'clean'),(48,'clean'),(96,'clean'))
PARTS={'d':(0,),'vp':(1,3),'dvp':(0,1,3),'o':(2,),'all':(0,1,2,3)}
GLOBAL_CAP=840
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def memory():
    class M(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    m=M();m.cb=ctypes.sizeof(M);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(m),m.cb)
    return dict(current_working_set_bytes=m.WorkingSetSize,peak_working_set_bytes=m.PeakWorkingSetSize)
def seeds(stage,s):
    b=3931010000
    return dict(train=[b+100*s+10*g for g in (0,1)],test=b+10000+100*s,initial=b+20000+s,donors=b+30000+100*s)
def assemble(base,D,condition):
    assert D in (0,24,48,96) and condition in ('clean','interference') and (D or condition=='clean')
    q=16+D;x=np.zeros((len(base['y']),q+8,6));x[:,8:16,:4]=base['write'];x[:,q:,4]=1
    if condition=='interference':x[:,18:22,5]=base['distractor']
    return x,q
def basic(c,y):
    p=np.where(c[:,0]==c[:,1],-1,c.argmax(1));recalls=[float(np.mean(p[y==j]==j)) for j in (0,1)]
    return dict(balanced_accuracy=sum(recalls)/2,accuracy=float(np.mean(p==y)),coverage=float(np.mean(p>=0)),silence=float(np.mean(c.sum(1)==0)),
                recalls=recalls,prediction_counts=[int(np.sum(p==j)) for j in (-1,0,1)]),p
def donors(y,seed):
    rng=np.random.default_rng(seed);opp=np.zeros(len(y),int);same=opp.copy()
    for j in (0,1):
        ids=np.flatnonzero(y==j);pool=rng.permutation(np.flatnonzero(y!=j));opp[ids]=np.resize(pool,len(ids));order=rng.permutation(ids);same[order]=np.roll(order,1)
    return dict(opposite=opp,same=same)
class Budget:
    def __init__(self):
        for name,h in read(R/'FREEZE.json')['files'].items():assert sha(R/name)==h,'frozen source '+name
        assert not (R/'RUN_STATE.json').exists(),'one-shot scientific run only'
        self.tic=time.perf_counter();self.cpu=time.process_time();self.macs=0;self.sample_steps=0;self.rows=[];self.trs=[];self.overhead=[]
        self.stage='initial';self.out=R
        self.mark('running')
    def mark(self,status,**extra):
        dump(R/'RUN_STATE.json',dict(stage=self.stage,status=status,pid=os.getpid(),utc=utc(),global_wall_seconds=time.perf_counter()-self.tic,**extra))
    def begin(self,stage):
        self.guard();self.stage=stage;self.out=R/stage;self.out.mkdir();self.stage_tic=time.perf_counter();self.stage_cpu=time.process_time();self.stage_macs=self.macs;self.stage_steps=self.sample_steps
        self.rows=[];self.trs=[];self.overhead=[];self.mark('running')
    def guard(self,disk=False):
        if time.perf_counter()-self.tic>GLOBAL_CAP:raise TimeoutError('840 second aggregate scientific cap')
        if memory()['current_working_set_bytes']>1024**3:raise MemoryError('1 GiB current working-set cap')
        if disk and sum(p.stat().st_size for p in R.rglob('*') if p.is_file())>1.5*1024**3:raise OSError('1.5 GiB new artifact cap')
    def forward_cost(self,n,T,h):
        self.macs+=n*T*(2*h*h+14*h);self.sample_steps+=n*T
    def flush(self):
        if self.stage in ('train','evaluate'):
            dump(self.out/'ROWS.json',self.rows)
            if self.stage=='evaluate':
                dump(self.out/'TRANSFERS.json',self.trs);dump(self.out/'OVERHEAD.json',self.overhead)
    def finish_stage(self,**extra):
        self.guard(disk=True);self.flush()
        dump(self.out/'SUMMARY.json',dict(utc=utc(),wall_seconds=time.perf_counter()-self.stage_tic,cpu_seconds=time.process_time()-self.stage_cpu,**memory(),forward_macs=self.macs-self.stage_macs,sample_steps=self.sample_steps-self.stage_steps,**extra))
    def finish(self):
        self.guard(disk=True)
        dump(R/'SCIENCE_SUMMARY.json',dict(status='COMPLETE',utc=utc(),wall_seconds=time.perf_counter()-self.tic,cpu_seconds=time.process_time()-self.cpu,**memory(),forward_macs=self.macs,sample_steps=self.sample_steps,scientific_fits=24,optimizer_updates=4800,evaluations=624,cap_seconds=GLOBAL_CAP))
        self.mark('complete')
    def fail(self,e):
        self.flush();self.mark('stopped',reason=str(e),exception=type(e).__name__)
        dump(R/'SCIENCE_FAILURE.json',dict(stage=self.stage,utc=utc(),reason=str(e),exception=type(e).__name__,wall_seconds=time.perf_counter()-self.tic,cpu_seconds=time.process_time()-self.cpu,**memory(),forward_macs=self.macs,sample_steps=self.sample_steps))
