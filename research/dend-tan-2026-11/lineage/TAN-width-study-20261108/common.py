from pathlib import Path
import json,time,ctypes,hashlib,os
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parent
SIZES=(24,48,96);ARMS=('DEND','POINT');LOSSES=('RATE','CE');RET=(.95,.85)
CONDITIONS=((0,'clean'),(24,'clean'),(48,'clean'),(96,'clean'),(24,'interference'),(96,'interference'))
PARTS={'d':(0,),'vp':(1,3),'dvp':(0,1,3),'o':(2,),'all':(0,1,2,3)}
CAPS={'tune':300,'train':400,'evaluate':140}
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
    b=3910010000 if stage=='tune' else 3911010000
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
    def __init__(self,stage):
        self.stage=stage;self.out=R/stage
        for name,h in read(R/'FREEZE.json')['files'].items():assert sha(R/name)==h,'frozen source '+name
        if (R/'RUN_STATE.json').exists():assert read(R/'RUN_STATE.json')['status']=='complete','another stage active or interrupted'
        self.out.mkdir();self.tic=time.perf_counter();self.cpu=time.process_time();self.macs=0;self.sample_steps=0
        dump(R/'RUN_STATE.json',dict(stage=stage,status='running',pid=os.getpid(),utc=utc()))
    def guard(self,disk=False):
        if time.perf_counter()-self.tic>CAPS[self.stage]:raise TimeoutError(self.stage+' wall budget')
        if memory()['current_working_set_bytes']>1024**3:raise MemoryError('1 GiB current working-set cap')
        if disk and sum(p.stat().st_size for p in R.rglob('*') if p.is_file())>1.5*1024**3:raise OSError('1.5 GiB artifact cap')
    def forward_cost(self,n,T,h):
        self.macs+=n*T*(2*h*h+14*h);self.sample_steps+=n*T
    def finish(self,**other):
        self.guard(disk=True);dump(self.out/'SUMMARY.json',dict(utc=utc(),wall_seconds=time.perf_counter()-self.tic,cpu_seconds=time.process_time()-self.cpu,**memory(),forward_macs=self.macs,sample_steps=self.sample_steps,**other))
        dump(R/'RUN_STATE.json',dict(stage=self.stage,status='complete',pid=os.getpid(),utc=utc()))
