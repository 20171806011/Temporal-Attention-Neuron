from pathlib import Path
import json,hashlib,time,ctypes,os
import numpy as np
import model
R=Path(__file__).resolve().parent
ARMS=('DEND','POINT');SIZES=(6,24)
STAGES=('explore_tune','explore','confirm_tune','confirm')
CAPS={'explore_tune':360,'explore':240,'confirm_tune':300,'confirm':240}
def utc():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze_check():
    f=read(R/'EXPLORE_FREEZE.json')
    for n,h in f['files'].items():assert sha(R/n)==h,n
    return sha(R/'EXPLORE_FREEZE.json')
def candidates(restricted=False):
    return [dict(id=i,loss=loss,lr=.01,ret=[0 if restricted else rd,rs]) for i,(loss,rd,rs) in enumerate([('RATE',.95,.85),('CE',.95,.85),('RATE',.99,.95),('CE',.99,.95)])]
def seeds(stage,s,h=6):
    offsets={'explore_tune':0,'explore':100000,'confirm_tune':1000000,'confirm':1100000}
    b=3000010000+offsets[stage]
    return dict(train=[b+100*s+10*g for g in range(4)],evaluation=b+10000+100*s,initial=b+20000+1000*h+s)
def interventions(delay=24):
    return [dict(name='baseline',kind='none'),dict(name='sham',kind='sham',tick=16+delay//2),
            *[dict(name='zero_'+p,kind='zero',tick=16+delay//2,part=p) for p in ('d','v','all')],
            dict(name='cut_recurrence',kind='cut_recurrence')]
def peak():
    class P(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('peak','working','a','b','c','d','e','f')]
    p=P();p.cb=ctypes.sizeof(p);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb)
    return int(p.peak) if ok else None
def start(stage):
    f=freeze_check()
    if stage.startswith('confirm'):
        lock=read(R/'HYPOTHESIS_LOCK.json');assert sha(R/'HYPOTHESES.md')==lock['hypotheses_sha256'];assert lock['run_validation']
    out=R/stage;out.mkdir(exist_ok=False)
    previous=sum(read(R/s/'SUMMARY.json')['wall_seconds'] for s in STAGES if (R/s/'SUMMARY.json').exists())
    dump(out/'STARTED.json',dict(utc=utc(),freeze=f,prior_scientific_seconds=previous,stage_cap=CAPS[stage],total_cap=900))
    return out,time.perf_counter(),time.process_time(),previous
def guard(stage,tic,prior):
    elapsed=time.perf_counter()-tic
    if elapsed>CAPS[stage] or prior+elapsed>900:raise TimeoutError('Frozen wall-time cap; no automatic extension or retry')
def describe(counts,cache,y,pattern,h,arm,q,intervention=None,ret=None):
    pred=np.where(counts[:,0]==counts[:,1],-1,counts.argmax(1));T=len(cache);n=len(y)
    bs=np.stack([a['bs'] for a in cache],1);hs=np.stack([a['hs'] for a in cache],1);os_=np.stack([a['os'] for a in cache],1)
    prev=np.stack([a['prev'] for a in cache],1)
    recalls=[float(np.mean(pred[y==j]==j)) for j in (0,1)];ba=float(np.mean(recalls));cov=float(np.mean(pred>=0))
    cut=intervention is not None and intervention['kind']=='cut_recurrence'
    rec=prev.sum((1,2))*2*h
    if cut:rec-=prev[:,16:q].sum((1,2))*2*h
    out=prev.sum((1,2))*2
    branch=bs.sum((1,2));holding=bs[:,16:q].sum((1,2))+hs[:,16:q].sum((1,2))+os_[:,16:q].sum((1,2))
    return dict(balanced_accuracy=ba,accuracy=float(np.mean(pred==y)),coverage=cov,passed=bool(ba>=.9 and cov>=.9),
        class_recall=recalls,pattern_recall=[float(np.mean(pred[pattern==p]==y[pattern==p])) for p in range(16)],
        confusion=[[int(np.sum((y==j)&(pred==k))) for k in (0,1,-1)] for j in (0,1)],silent_fraction=float(np.mean(np.all(counts==0,1))),
        nonsilent_tie_fraction=float(np.mean((pred<0)&np.any(counts!=0,1))),
        branch_events=float(branch.mean()),hidden_events=float(hs.sum((1,2)).mean()),output_events=float(os_.sum((1,2)).mean()),hold_events=float(holding.mean()),
        hold_hidden_events=float(hs[:,16:q].sum((1,2)).mean()),recurrent_deliveries=float(rec.mean()),output_deliveries=float(out.mean()),branch_deliveries=float(branch.mean()),
        internal_deliveries=float((rec+out+branch).mean()),graded_local_transfers=2*h*T if arm=='DEND' else 0,
        parameters=model.parameter_count(h),continuous_slots=3*h+2,persistent_continuous_variables=(h+2 if ret is not None and ret[0]==0 else 3*h+2),
        state_updates=(3*h+2)*T,forward_macs=(2*h*h+14*h)*T,backward_macs=(4*h*h+16*h)*T,steps=T)
def summarize_interval(values):
    a=np.asarray(values,float);idx=np.random.default_rng(3009000001).integers(0,len(a),(20000,len(a)))
    return dict(mean=float(a.mean()),ci95=np.quantile(a[idx].mean(1),[.025,.975]).tolist(),values=a.tolist(),n=len(a))
