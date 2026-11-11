from pathlib import Path
import json,time,ctypes,hashlib,os
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parent
RET=(.95,.85)
CONDITIONS=((0,'clean'),(24,'clean'),(48,'clean'),(96,'clean'))
PARTS={'d':(0,),'vp':(1,3),'dvp':(0,1,3),'o':(2,),'all':(0,1,2,3)}
GLOBAL_CAP=600
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
    b=3941010000
    return dict(train=[b+100*s+10*g for g in (0,1)],test=b+10000+100*s,initial=b+20000+s,donors=b+30000+100*s)
def assemble(base,D,condition):
    assert D in (16,24,28,36,48,64,72,84,96) and condition=='clean'
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
def cases():
    rows=[]
    for D in (16,24,28,36,48,64,72,84,96):rows.append(dict(condition='baseline',delay=D))
    for L in (4,12,24):
        for recovery in (0,12,48):
            D=12+L+recovery
            rows.append(dict(condition=f'cut_L{L}_R{recovery}',delay=D,cut=[28,28+L],L=L,recovery=recovery))
            if recovery:rows.append(dict(condition=f'keepoff_L{L}_R{recovery}',delay=D,cut=[28,28+L+recovery],L=L,recovery=recovery))
    for D in (24,48,96):
        for part in ('d','vp','all'):rows.append(dict(condition='zero_'+part,delay=D,zero_tick=28,zero_part=part))
        rows.append(dict(condition='identity',delay=D,sham_tick=28))
    assert len(rows)==36
    return rows

class Budget:
    def __init__(self):
        for n,h in read(R/'FREEZE.json')['files'].items():assert sha(R/n)==h,'frozen source '+n
        assert not (R/'RUN_STATE.json').exists(),'no restart of the frozen experiment'
        self.tic=time.perf_counter();self.cpu=time.process_time();self.rows=[];self.macs=0;self.nominal_macs=0;self.sample_steps=0
        self.out=R/'evaluate';self.out.mkdir();self.mark('running')
    def mark(self,status,**extra):dump(R/'RUN_STATE.json',dict(status=status,pid=os.getpid(),utc=utc(),wall_seconds=time.perf_counter()-self.tic,**extra))
    def guard(self,disk=False):
        if time.perf_counter()-self.tic>600:raise TimeoutError('600 second scientific cap')
        if memory()['current_working_set_bytes']>1024**3:raise MemoryError('1 GiB current working-set cap')
        if disk and sum(p.stat().st_size for p in R.rglob('*') if p.is_file())>1.5*1024**3:raise OSError('1.5 GiB new artifact cap')
    def flush(self):dump(self.out/'ROWS.json',self.rows)
    def finish(self):
        self.guard(disk=True);self.flush();dump(R/'SCIENCE_SUMMARY.json',dict(status='COMPLETE',utc=utc(),models=24,evaluations=864,new_training=0,wall_seconds=time.perf_counter()-self.tic,cpu_seconds=time.process_time()-self.cpu,**memory(),executed_forward_macs=self.macs,nominal_forward_macs=self.nominal_macs,sample_steps=self.sample_steps,cap_seconds=600));self.mark('complete')
    def fail(self,e):
        self.flush();self.mark('stopped',reason=str(e));dump(R/'SCIENCE_FAILURE.json',dict(utc=utc(),reason=str(e),exception=type(e).__name__,wall_seconds=time.perf_counter()-self.tic,completed_cases=len(self.rows)))
