"""Frozen bounded execution stages; each stage is explicit and refuses overwrite."""
from pathlib import Path
import os,sys,json,time,hashlib,platform,importlib.util,csv,ctypes
import numpy as np
import train_model as model
from engine import forward,measure,specifications

R=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location(name,R/'vendor'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
V9=load('v9_model');D9=load('v9_data');D10=load('v10_data');V10=load('v10_model')
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()
def peak_memory():
    if os.name!='nt':return None
    class PMC(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    p=PMC();p.cb=ctypes.sizeof(p)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb)
    return int(p.PeakWorkingSetSize) if ok else None
def check_freeze():
    f=json.loads((R/'FREEZE.json').read_text(encoding='utf-8'))
    for name,digest in f['files'].items():assert sha(R/name)==digest,name
    return sha(R/'FREEZE.json')
def xor_twins(n,seed):
    rng=np.random.default_rng(seed);pair=np.repeat(np.arange(4),n//4);rng.shuffle(pair)
    a=pair//2;b=pair%2;amp=rng.uniform(.8,1.2,(n,2));j=rng.integers(0,3,(n,2))
    x=np.zeros((n,24,6));twin=x.copy()
    for i in range(n):
        x[i,2+j[i,0]:6+j[i,0],a[i]]=amp[i,0];twin[i,2+j[i,0]:6+j[i,0],1-a[i]]=amp[i,0]
        x[i,10+j[i,1]:14+j[i,1],2+b[i]]=amp[i,1];twin[i,10+j[i,1]:14+j[i,1],2+b[i]]=amp[i,1]
    x[:,16:,4]=1;twin[:,16:,4]=1
    nuisance=rng.uniform(0,.5,(n,3));x[:,7:10,5]=nuisance;twin[:,7:10,5]=nuisance
    noise=rng.normal(0,.03,x.shape);x=np.maximum(x+noise,0);twin=np.maximum(twin+noise,0)
    same=np.arange(n)
    for p in range(4):ix=np.flatnonzero(pair==p);same[ix]=np.roll(ix,1)
    return dict(x=x,twin=twin,y=a^b,pair=pair,same=same)
def interval(values):
    a=np.array(values,float);assert a.shape==(5,)
    idx=np.random.default_rng(2611079001).integers(0,5,(20000,5))
    return dict(mean=float(a.mean()),ci95=np.quantile(a[idx].mean(1),[.025,.975]).tolist(),values=a.tolist(),n=5)
def candidates():
    return [dict(id=i,loss=loss,lr=.01,ret=ret) for i,(loss,ret) in enumerate([('RATE',[.85,.65]),('CE',[.85,.65]),('RATE',[.95,.85]),('CE',[.95,.85])])]
def fixture():
    rng=np.random.default_rng(4917);x=rng.uniform(0,.7,(4,24,6));y=np.array([0,1,0,1]);err=0;derivatives=0
    for h in (2,6):
      w=model.initialize(731+h,h)
      for arm in ('DEND','POINT'):
       for ret in ((.85,.65),(.95,.85)):
        c,cache=model.forward(x,w,arm,h,ret=ret);native=forward(x,w,arm,h,ret)
        assert np.array_equal(c,native['counts'])
        assert set(np.unique(native['spikes']))<=set([0.,1.])
        if ret==(.85,.65):
            old,_=V10.forward(x,w,arm,h);assert np.array_equal(c,old)
            if h==6:old9,_=V9.forward(x,w,arm);assert np.array_equal(c,old9)
        for kind in ('RATE','CE'):
            _,g,_=model.loss_grad(x,y,w,arm,kind,h,ret=ret)
            for k in range(len(w)):
                plus=w.copy();minus=w.copy();plus[k]+=1e-6;minus[k]-=1e-6
                fd=(model.objective(x,y,plus,arm,kind,h,cache,ret)-model.objective(x,y,minus,arm,kind,h,cache,ret))/2e-6
                err=max(err,abs(fd-g[k]));derivatives+=1
        for spec in specifications():
            if 'sham' in spec['name']:
                v=forward(x,w,arm,h,ret,spec);assert np.array_equal(native['counts'],v['counts']);assert np.array_equal(native['states'],v['states'])
        for i in range(len(x)):
            solo=forward(x[i:i+1],w,arm,h,ret);assert np.array_equal(solo['counts'],native['counts'][i:i+1])
    assert err<1e-7,err
    z=np.zeros((1,24,6));w=np.zeros(model.parameter_count(2));model.unpack(w,2)[-1][:]=.5
    v=forward(z,w,'POINT',2,ret=(.85,0));assert np.all(v['spikes'][:,:,-2:]==0)
    model.unpack(w,2)[-1][:]=np.nextafter(.5,1);v=forward(z,w,'POINT',2,ret=(.85,0));assert np.all(v['spikes'][:,:,-2:]==1);assert np.all(v['states'][:,:,-2:]==0)
    # Synthetic fixture seeds only; rescue property is independent of fitting.
    d=xor_twins(16,4921);orig=D9.make(16,4921);assert np.array_equal(orig[0],d['x']);assert np.array_equal(d['x'][:,8:],d['twin'][:,8:])
    for arm in ('DEND','POINT'):
        w=model.initialize(733,6);t=forward(d['twin'],w,arm)
        spec=dict(kind='transplant',tick=9,component='all');r=forward(d['x'],w,arm,spec=spec,donor=t['memory'])
        assert np.array_equal(r['counts'],t['counts'])
    dump(R/'PRECHECK.json',dict(status='PASS',utc=utc(),anchored_finite_differences=derivatives,max_error=err,native_parity=True,shams=True,strict_threshold_reset=True,batch_independence=True,full_state_rescue=True))
    print('PRECHECK PASS',derivatives,err,flush=True)
def freeze():
    assert not (R/'FREEZE.json').exists()
    names=['PROTOCOL.md','AMENDMENTS.md','PRECHECK.json','INDEPENDENT_PREFLIGHT.json','SOURCE_AUDIT.json','engine.py','study.py','train_model.py','verify.py']
    files=[R/n for n in names]+sorted((R/'vendor').glob('*'))+sorted((R/'weights').glob('*'))
    dump(R/'FREEZE.json',dict(utc=utc(),source_commit='7dcddee41490d8d917936891989eeb7ddb67e0da',files={p.relative_to(R).as_posix():sha(p) for p in files}))
    dump(R/'ENVIRONMENT.json',dict(machine=platform.node(),python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform(),cwd=str(Path.cwd()),threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},utc=utc()))
    print('FROZEN',sha(R/'FREEZE.json'),flush=True)
def stage_start(name):
    freeze=check_freeze();out=R/name;out.mkdir(exist_ok=False)
    dump(out/'STARTED.json',dict(utc=utc(),freeze_sha256=freeze,cap_seconds=300));return out,time.perf_counter(),time.process_time()
def guard(start):
    if time.perf_counter()-start>300:raise TimeoutError('Frozen 300 second stage cap reached; no automatic restart')
def save_run(out,ident,v,metadata):
    np.savez_compressed(out/f'{ident}.npz',counts=v['counts'],states=v['states'][:32],spikes=v['spikes'][:32],candidates=v['candidates'][:32],memory=v['memory'][:32])
    return dict(id=ident,**metadata)
def mechanism():
    out,start,cpu=stage_start('mechanism');rows=[];specs=specifications();dump(out/'SPECS.json',specs)
    old=list(csv.DictReader((R/'vendor/v9_metrics.csv').open(encoding='utf-8')));parity=[]
    for s in range(5):
        d=xor_twins(512,2611070100+10*s);np.savez_compressed(out/f'data_s{s}.npz',**d)
        for arm,cohort in [('DEND','retrained'),('DEND','intact'),('POINT','retrained')]:
            w=np.load(R/'weights'/f's{s}_{arm}_{cohort}.npy');family=arm+'_'+cohort
            oldx,oldy,_=D9.make(512,2090010000+10*s);oldv=forward(oldx,w,arm);oldm=measure(oldv,oldy,arm=arm)
            historic=next(r for r in old if r['split']=='test' and r['id']==f's{s}_{family}')
            assert oldm['accuracy']==float(historic['accuracy']);assert oldm['coverage']==float(historic['coverage'])
            parity.append(dict(seed=s,family=family,accuracy=oldm['accuracy'],coverage=oldm['coverage']))
            base=forward(d['x'],w,arm);flip=forward(d['twin'],w,arm)
            for spec in specs:
                guard(start);donor=None
                if spec['kind']=='transplant':donor=flip['memory'] if spec['donor']=='flip' else base['memory'][d['same']]
                v=base if spec['name']=='baseline' else forward(d['x'],w,arm,spec=spec,donor=donor)
                if 'sham' in spec['name']:assert np.array_equal(v['states'],base['states']);assert np.array_equal(v['counts'],base['counts'])
                if spec['name']=='hold_flip_all':assert np.array_equal(v['counts'],flip['counts'])
                ident=f's{s}_{family}_{spec["name"]}'
                rows.append(save_run(out,ident,v,dict(seed=s,arm=arm,cohort=cohort,condition=spec['name'],**measure(v,d['y'],d['pair'],arm=arm))))
            rows.append(save_run(out,f's{s}_{family}_twin',flip,dict(seed=s,arm=arm,cohort=cohort,condition='twin',**measure(flip,1-d['y'],d['pair'],arm=arm))))
            dump(out/'ROWS.json',rows);print('MECHANISM',s,family,'baseline',measure(base,d['y'],arm=arm)['accuracy'],flush=True)
    lookup={(r['seed'],r['arm'],r['cohort'],r['condition']):r for r in rows}
    primary=[];fd=[];fv=[]
    for s in range(5):
        def q(name):return lookup[s,'DEND','retrained',name]['flipped_label_fraction']
        a=q('hold_flip_d')-q('hold_same_d');b=q('hold_flip_v')-q('hold_same_v');fd.append(a);fv.append(b);primary.append(a-b)
    ci=interval(primary)
    dump(out/'SUMMARY.json',dict(primary=ci,branch_transfer=interval(fd),soma_transfer=interval(fv),criterion_met=ci['mean']>=.2 and ci['ci95'][0]>0,
        rows=len(rows),historical_baseline_parity=parity,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_working_set_bytes=peak_memory()))
def train(out,ident,x,y,seed,h,arm,cfg,start):
    w=model.initialize(seed,h);initial=w.copy();m=np.zeros_like(w);v=np.zeros_like(w);grads=[];counts=[];history=[];tick=time.perf_counter()
    try:
        for k in range(200):
            guard(start);loss,g,c=model.loss_grad(x,y,w,arm,cfg['loss'],h,ret=cfg['ret']);norm=float(np.linalg.norm(g))
            assert np.isfinite(loss) and np.isfinite(g).all()
            g=g/max(1,norm);grads.append(g.copy());counts.append(c);history.append([loss,norm])
            m=.9*m+.1*g;v=.999*v+.001*g*g
            w=np.clip(w-cfg['lr']*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
            assert np.isfinite(w).all()
    except Exception:
        np.savez_compressed(out/f'{ident}_PARTIAL.npz',initial=initial,theta=w,gradients=np.array(grads),counts=np.array(counts),history=np.array(history));raise
    np.savez_compressed(out/f'{ident}_train.npz',initial=initial,theta=w,gradients=np.array(grads),counts=np.array(counts),history=np.array(history))
    return w,time.perf_counter()-tick
def tuning():
    out,start,cpu=stage_start('tuning');rows=[]
    for s in range(2):
        x,y,p=D10.make(128,2611071000+10*s);vx,vy,vp=D10.make(256,2611072000+10*s)
        np.savez_compressed(out/f'data_s{s}.npz',x=x,y=y,pattern=p,vx=vx,vy=vy,vpattern=vp)
        for h in (2,6):
            for arm in ('DEND','POINT'):
                for cfg in candidates():
                    ident=f's{s}_{arm}_h{h}_c{cfg["id"]}';w,seconds=train(out,ident,x,y,2611073000+100*h+s,h,arm,cfg,start)
                    nv=forward(vx,w,arm,h,cfg['ret']);np.savez_compressed(out/f'{ident}_val.npz',counts=nv['counts'])
                    rows.append(dict(id=ident,seed=s,arm=arm,h=h,candidate=cfg,seconds=seconds,**measure(nv,vy,vp,h,arm)))
                dump(out/'ROWS.json',rows);print('TUNING',s,h,arm,'completed all four candidates',flush=True)
    selected={};scores=[]
    for h in (2,6):
        for arm in ('DEND','POINT'):
            candidates_by_score=[]
            for cfg in candidates():
                a=[r for r in rows if r['h']==h and r['arm']==arm and r['candidate']['id']==cfg['id']]
                score=dict(arm=arm,h=h,candidate=cfg,ba=float(np.mean([r['balanced_accuracy'] for r in a])),coverage=float(np.mean([r['coverage'] for r in a])))
                scores.append(score);candidates_by_score.append(score)
            best=sorted(candidates_by_score,key=lambda q:(-q['ba'],-q['coverage'],q['candidate']['id']))[0]
            selected[f'{arm}_h{h}']=best['candidate']
    dump(out/'SELECTION_LOCK.json',dict(utc=utc(),selected=selected,scores=scores,files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},test_generated=False,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_working_set_bytes=peak_memory()))
def final_train():
    selection=json.loads((R/'tuning/SELECTION_LOCK.json').read_text())
    for n,digest in selection['files'].items():assert sha(R/'tuning'/n)==digest
    out,start,cpu=stage_start('final_train');rows=[]
    for s in range(5):
        x,y,p=D10.make(128,2611074000+10*s);np.savez_compressed(out/f'data_s{s}.npz',x=x,y=y,pattern=p)
        for h in (2,6):
            for arm in ('DEND','POINT'):
                cfg=selection['selected'][f'{arm}_h{h}'];ident=f's{s}_{arm}_h{h}'
                w,seconds=train(out,ident,x,y,2611075000+100*h+s,h,arm,cfg,start)
                nv=forward(x,w,arm,h,cfg['ret'])
                rows.append(dict(id=ident,seed=s,arm=arm,h=h,candidate=cfg,seconds=seconds,**measure(nv,y,p,h,arm)))
        dump(out/'ROWS.json',rows);print('FINAL TRAIN',s,'four cells complete',flush=True)
    dump(out/'TRAIN_LOCK.json',dict(utc=utc(),selection_sha256=sha(R/'tuning/SELECTION_LOCK.json'),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},test_generated=False,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_working_set_bytes=peak_memory()))
def test():
    lock=json.loads((R/'final_train/TRAIN_LOCK.json').read_text())
    for n,digest in lock['files'].items():assert sha(R/'final_train'/n)==digest
    out,start,cpu=stage_start('test');rows=[];trainrows=json.loads((R/'final_train/ROWS.json').read_text())
    for s in range(5):
        x,y,p=D10.make(512,2611076000+10*s);np.savez_compressed(out/f'data_s{s}.npz',x=x,y=y,pattern=p)
        for r in [r for r in trainrows if r['seed']==s]:
            guard(start);w=np.load(R/'final_train'/f'{r["id"]}_train.npz')['theta'];nv=forward(x,w,r['arm'],r['h'],r['candidate']['ret'])
            np.savez_compressed(out/f'{r["id"]}.npz',counts=nv['counts'],spikes=nv['spikes'],states=nv['states'],candidates=nv['candidates'])
            rows.append(dict(id=r['id'],seed=s,arm=r['arm'],h=r['h'],candidate=r['candidate'],**measure(nv,y,p,r['h'],r['arm'])))
    dump(out/'ROWS.json',rows)
    contrasts={}
    for h in (2,6):
        vals=[]
        for s in range(5):
            a=next(r for r in rows if r['seed']==s and r['h']==h and r['arm']=='DEND');b=next(r for r in rows if r['seed']==s and r['h']==h and r['arm']=='POINT')
            vals.append(a['balanced_accuracy']-b['balanced_accuracy'])
        contrasts[str(h)]=interval(vals)
    dump(out/'SUMMARY.json',dict(contrasts=contrasts,train_lock_sha256=sha(R/'final_train/TRAIN_LOCK.json'),wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_working_set_bytes=peak_memory()))
if __name__=='__main__':
    stage=sys.argv[1];out=R/stage
    try:dict(precheck=fixture,freeze=freeze,mechanism=mechanism,tuning=tuning,final_train=final_train,test=test)[stage]()
    except Exception as e:
        if out.is_dir():dump(out/'FAILURE.json',dict(utc=utc(),type=type(e).__name__,reason=str(e)))
        raise
