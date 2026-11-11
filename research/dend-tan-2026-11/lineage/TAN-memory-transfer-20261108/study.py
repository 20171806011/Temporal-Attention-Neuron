"""Frozen staged runner. Each scientific stage runs once in a fresh directory."""
from pathlib import Path
import sys,json,time,hashlib,platform,os,ctypes
from datetime import datetime,timezone
import numpy as np
import core_model as model
import data,engine

R=Path(__file__).resolve().parent
CONFIGS={'DEND_full':dict(arm='DEND',ret=[.95,.85],loss='RATE'),
         'DEND_rd0':dict(arm='DEND',ret=[0,.85],loss='RATE'),
         'POINT_full':dict(arm='POINT',ret=[.95,.85],loss='CE')}
CAPS={'old':120,'train':300,'new':180}
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def peak():
    class M(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    m=M();m.cb=ctypes.sizeof(M);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(m),m.cb);return m.PeakWorkingSetSize

def check_freeze():
    for p,h in read(R/'FREEZE.json')['files'].items():assert sha(R/p)==h, 'freeze mismatch '+p
def start(stage):
    check_freeze();out=R/stage;out.mkdir();return out,time.perf_counter(),time.process_time()
def guard(stage,tic):
    if time.perf_counter()-tic>CAPS[stage]:raise TimeoutError(stage+' stage cap')
def finish(out,tic,cpu,**extra):
    dump(out/'SUMMARY.json',dict(utc=utc(),wall_seconds=time.perf_counter()-tic,cpu_seconds=time.process_time()-cpu,peak_working_set_bytes=peak(),**extra))
def savebase(out,name,b):np.savez_compressed(out/name,**b)

def profile(out,ident,s,w,cfg,base,mapping,fitbase,probebase,stage,tic):
    rows=[];tr=[];q=40;x,q=data.assemble(base,24,'clean');c,cache=model.forward(x,w,cfg['arm'],24,ret=cfg['ret'],query=q)
    state=engine.boundary(cache,28);np.savez_compressed(out/f'{ident}_baseline.npz',**engine.trace(c,cache))
    costs=engine.cost(cache,w,cfg['arm'],q);costs['persistent_continuous']=26 if cfg['ret'][0]==0 else 74
    rows.append(dict(id=ident,seed=s,config=cfg,condition='baseline',**engine.basic(c,base['y']),**costs))
    allcounts={};construction=0.;resume_seconds=0.
    for kind,part in [('identity','all')]+[(k,p) for k in ('opposite','same') for p in engine.PARTS]:
        guard(stage,tic);tick=time.perf_counter();ids=np.arange(len(c)) if kind=='identity' else mapping[kind]
        mixed=engine.mix(state,ids,part);construction+=time.perf_counter()-tick;tick=time.perf_counter()
        result=engine.resume(x,w,cfg['arm'],cfg['ret'],mixed);resume_seconds+=time.perf_counter()-tick
        name=kind+'_'+part;np.savez_compressed(out/f'{ident}_{name}.npz',**result);allcounts[name]=result['counts']
        if part=='all':assert np.array_equal(result['counts'],c[ids]), 'whole-state donor recovery'
        rows.append(dict(id=ident,seed=s,config=cfg,condition=name,**engine.basic(result['counts'],base['y']),
                    suffix_recurrent_deliveries=float(result['rec'].mean()),suffix_output_deliveries=float(result['out'].mean())))
    if cfg['ret'][0]==0:
        for k in ('opposite','same'):
            assert np.array_equal(allcounts[k+'_d'],c)
            assert np.array_equal(allcounts[k+'_dvp'],allcounts[k+'_vp'])
    for part in engine.PARTS:tr.append(dict(id=ident,seed=s,part=part,config=cfg,**engine.transfer(allcounts['opposite_'+part],allcounts['same_'+part],c,base['y'],mapping)))
    # Reuse only caches whose input base is exactly the diagnostic base.
    tick=time.perf_counter();features=[];patterns=[]
    for b in (fitbase,probebase):
        if b is base:cc=cache
        else:
            xx,qq=data.assemble(b,24,'clean');_,cc=model.forward(xx,w,cfg['arm'],24,ret=cfg['ret'],query=qq)
        ids=np.flatnonzero(np.isin(b['pattern'],[3,12]));features.append(engine.features(cc,ids));patterns.append(b['pattern'][ids])
    pr,pa=engine.probes(*features,*patterns,s);np.savez_compressed(out/f'{ident}_probes.npz',**pa)
    for row in pr:row.update(id=ident,seed=s,config=cfg)
    overhead=dict(id=ident,transplant_construction_seconds=construction,transplant_resume_seconds=resume_seconds,probe_seconds=time.perf_counter()-tick)
    if stage=='old':assert np.array_equal(c,np.load(R/'legacy'/f's{s}_{cfg["arm"]}_baseline_counts.npy'))
    if stage=='new':
        xx,qq=data.assemble(base,0,'clean');c0,k0=model.forward(xx,w,cfg['arm'],24,ret=cfg['ret'],query=qq)
        np.savez_compressed(out/f'{ident}_d0.npz',**engine.trace(c0,k0));co=engine.cost(k0,w,cfg['arm'],qq);co['persistent_continuous']=26 if cfg['ret'][0]==0 else 74
        rows.append(dict(id=ident,seed=s,config=cfg,condition='d0',**engine.basic(c0,base['y']),**co))
    return rows,tr,pr,overhead

def old():
    out,tic,cpu=start('old');rows=[];trs=[];prs=[];over=[]
    for s in range(3):
        b=dict(np.load(R/'legacy'/f'base_s{s}.npz'));p=data.make_base(512,3110010000+10*s);savebase(out,f'probe_s{s}.npz',p)
        maps=engine.donors(b['y'],3110020000+10*s);np.savez_compressed(out/f'donors_s{s}.npz',**maps)
        for arm in ('DEND','POINT'):
            cfg=CONFIGS[arm+'_full'];ident=f's{s}_{arm}';w=np.load(R/'legacy'/f'{ident}.npy')
            a,t,pr,o=profile(out,ident,s,w,cfg,b,maps,b,p,'old',tic);rows+=a;trs+=t;prs+=pr;over.append(o)
            print('old',ident,'complete',flush=True)
        for name,obj in [('ROWS',rows),('TRANSFERS',trs),('PROBES',prs),('OVERHEAD',over)]:dump(out/(name+'.json'),obj)
    guard('old',tic);finish(out,tic,cpu,models=6,evaluations=len(rows),probe_rows=len(prs),new_task_network_fits=0)

def train():
    out,tic,cpu=start('train');rows=[]
    for s in range(5):
        groups=[]
        for g,delay in enumerate((0,24)):
            b=data.make_base(64,3111010000+100*s+10*g);savebase(out,f'base_s{s}_g{g}.npz',b);x,q=data.assemble(b,delay,'clean');groups.append((x,b['y'],q))
        for name,cfg in CONFIGS.items():
            ident=f's{s}_{name}';started=time.perf_counter();initial=model.initialize(3111030000+s,24);w=initial.copy();m=np.zeros_like(w);v=m.copy();grads=[];counts=[];hist=[]
            try:
                for k in range(200):
                    guard('train',tic);grad=np.zeros_like(w);loss=0.;cs=[]
                    for x,y,q in groups:
                        l,d,c=model.loss_grad(x,y,w,cfg['arm'],cfg['loss'],24,ret=cfg['ret'],query=q);loss+=l/2;grad+=d/2;cs.append(c)
                    norm=float(np.linalg.norm(grad));assert np.isfinite(grad).all() and np.isfinite(loss)
                    grad/=max(1,norm);grads.append(grad.copy());counts.append(np.concatenate(cs));hist.append([loss,norm])
                    m=.9*m+.1*grad;v=.999*v+.001*grad*grad
                    w=np.clip(w-.01*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
            except Exception:
                np.savez_compressed(out/f'{ident}_PARTIAL.npz',initial=initial,theta=w,gradients=grads,counts=counts,history=hist);raise
            np.savez_compressed(out/f'{ident}.npz',initial=initial,theta=w,gradients=grads,counts=counts,history=hist)
            rows.append(dict(id=ident,seed=s,name=name,config=cfg,seconds=time.perf_counter()-started,updates=200))
            dump(out/'ROWS.json',rows);print('train',ident,'complete',flush=True)
    guard('train',tic);finish(out,tic,cpu,fits=len(rows),optimizer_updates=3000,train_examples_per_fit=128,
        dense_forward_macs=200*15*64*(24+48)*1488,dense_backward_macs=200*15*64*(24+48)*2688)
    dump(out/'TRAIN_LOCK.json',dict(utc=utc(),test_generated=False,files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))

def new():
    lock=read(R/'train'/'TRAIN_LOCK.json')
    for p,h in lock['files'].items():assert sha(R/'train'/p)==h
    out,tic,cpu=start('new');rows=[];trs=[];prs=[];over=[]
    for s in range(5):
        b=data.make_base(512,3111020000+100*s);savebase(out,f'test_s{s}.npz',b)
        bb=[dict(np.load(R/'train'/f'base_s{s}_g{g}.npz')) for g in (0,1)]
        fit={k:np.concatenate([a[k] for a in bb]) for k in ('pattern','y','write','distractor')}
        maps=engine.donors(b['y'],3111040000+100*s);np.savez_compressed(out/f'donors_s{s}.npz',**maps)
        for name,cfg in CONFIGS.items():
            ident=f's{s}_{name}';w=np.load(R/'train'/f'{ident}.npz')['theta']
            a,t,pr,o=profile(out,ident,s,w,cfg,b,maps,fit,b,'new',tic);rows+=a;trs+=t;prs+=pr;over.append(o)
            print('new',ident,'complete',flush=True)
        for name,obj in [('ROWS',rows),('TRANSFERS',trs),('PROBES',prs),('OVERHEAD',over)]:dump(out/(name+'.json'),obj)
    guard('new',tic);finish(out,tic,cpu,models=15,evaluations=len(rows),probe_rows=len(prs))

def freeze():
    assert not (R/'FREEZE.json').exists()
    paths=[p for p in sorted(R.iterdir()) if p.suffix in ('.py','.md','.json') and p.name not in ('FREEZE.json','ENVIRONMENT.json')]+sorted((R/'legacy').iterdir())
    dump(R/'FREEZE.json',dict(utc=utc(),new_scientific_data_generated=False,files={str(p.relative_to(R)):sha(p) for p in paths}))
    dump(R/'ENVIRONMENT.json',dict(utc=utc(),machine=platform.node(),python=sys.version,numpy=np.__version__,executable=sys.executable,
          threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    print('FROZEN',sha(R/'FREEZE.json'))

if __name__=='__main__':
    stage=sys.argv[1]
    try: {'freeze':freeze,'old':old,'train':train,'new':new}[stage]()
    except Exception as e:
        dump(R/(stage+'_FAILURE.json'),dict(utc=utc(),exception=type(e).__name__,reason=str(e)));raise
