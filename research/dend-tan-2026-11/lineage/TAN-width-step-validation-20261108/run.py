"""One-shot tuning, fresh training and held-out evaluation stages."""
import sys,time,platform,os
import numpy as np
import model,base_data,engine
from common import *

def groups(stage,s,out):
    outgroups=[]
    for g,D in enumerate((0,24)):
        b=base_data.make_base(64,seeds(stage,s)['train'][g]);np.savez_compressed(out/f'base_s{s}_g{g}.npz',**b);x,q=assemble(b,D,'clean');outgroups.append((x,b['y'],q))
    return outgroups
def fit(B,ident,s,h,arm,lr,gs):
    tic=time.perf_counter();initial=model.initialize(seeds('train',s)['initial'],h);w=initial.copy();m=np.zeros_like(w);v=m.copy();grads=[];counts=[];history=[]
    diag={f'oc_D{D}':[] for D in (0,24)};diag['completed_updates']=[];extra_macs=0
    try:
        for k in range(200):
            B.guard()
            if k in CHECKPOINTS:
                diag['completed_updates'].append(k)
                for j,(x,y,q) in enumerate(gs):
                    cc,cache=model.forward(x[:8],w,arm,h,ret=RET,query=q)
                    diag[f'oc_D{(0,24)[j]}'].append(np.stack([a['oc'] for a in cache],1))
                    B.forward_cost(8,x.shape[1],h);extra_macs+=8*x.shape[1]*(2*h*h+14*h)
            g=np.zeros_like(w);l=0.;cs=[]
            for x,y,q in gs:
                ll,gg,cc=model.loss_grad(x,y,w,arm,'CE',h,ret=RET,query=q);g+=gg/2;l+=ll/2;cs.append(cc);B.forward_cost(len(x),x.shape[1],h)
            norm=float(np.linalg.norm(g));assert np.isfinite(g).all() and np.isfinite(l),'nonfinite update';g/=max(1,norm)
            grads.append(g.copy());counts.append(np.concatenate(cs).astype(np.uint8));history.append([l,norm]);m=.9*m+.1*g;v=.999*v+.001*g*g
            w=np.clip(w-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
    except Exception:
        np.savez_compressed(B.out/f'{ident}_PARTIAL.npz',initial=initial,theta=w,gradients=grads,counts=counts,history=history,**diag);raise
    np.savez_compressed(B.out/f'{ident}.npz',initial=initial,theta=w,gradients=grads,counts=counts,history=history,**diag);B.guard(disk=True)
    return w,dict(id=ident,seed=s,h=h,arm=arm,loss='CE',ret=list(RET),lr=lr,updates=200,seconds=time.perf_counter()-tic,forward_macs=200*64*72*(2*h*h+14*h),backward_macs=200*64*72*(4*h*h+16*h),diagnostic_forward_macs=extra_macs,diagnostic_checkpoints=list(CHECKPOINTS))

def train(B):
    B.begin('train')
    for s in range(3):
        gs=groups('train',s,B.out)
        for h in SIZES:
            for arm in ARMS:
                for rate_index,lr in enumerate(RATES):
                    ident=f's{s}_{arm}_h{h}_lr{rate_index}';_,row=fit(B,ident,s,h,arm,lr,gs);B.rows.append(row);B.flush();print('train',ident,'complete',flush=True)
    B.finish_stage(fits=24,updates=4800,training_backward_macs=sum(r['backward_macs'] for r in B.rows),diagnostic_forward_macs=sum(r['diagnostic_forward_macs'] for r in B.rows))
    dump(B.out/'TRAIN_LOCK.json',dict(utc=utc(),test_generated=False,files={p.name:sha(p) for p in sorted(B.out.iterdir()) if p.is_file()}))

def evaluate(B):
    lock=read(R/'train'/'TRAIN_LOCK.json')
    for p,h in lock['files'].items():assert sha(R/'train'/p)==h,'train lock '+p
    B.begin('evaluate');rows=B.rows;trs=B.trs;overhead=B.overhead;trainrows=read(R/'train'/'ROWS.json')
    for s in range(3):
        b=base_data.make_base(256,seeds('train',s)['test']);np.savez_compressed(B.out/f'test_s{s}.npz',**b);mapping=donors(b['y'],seeds('train',s)['donors']);np.savez_compressed(B.out/f'donors_s{s}.npz',**mapping)
        for tr in [r for r in trainrows if r['seed']==s]:
            ident=tr['id'];h=tr['h'];arm=tr['arm'];w=np.load(R/'train'/f'{ident}.npz')['theta'];baselines={};construction=0.;suffix_seconds=0.
            meta={k:tr[k] for k in ('id','seed','h','arm','loss','ret','lr')}
            for D,kind in CONDITIONS:
                B.guard();x,q=assemble(b,D,kind);tick=16+D//2 if kind=='clean' and D in (24,96) else None
                saved,cost,state=engine.evaluate(x,w,arm,h,q,tick);B.forward_cost(len(x),x.shape[1],h);np.savez_compressed(B.out/f'{ident}_d{D}_{kind}_baseline.npz',**saved);metric,_=basic(saved['counts'],b['y'])
                rows.append(dict(**meta,delay=D,kind=kind,condition='baseline',**metric,**cost))
                if tick:baselines[D]=(saved['counts'],state)
            for D in (24,96):
                x,q=assemble(b,D,'clean');tick=16+D//2;baseline,state=baselines[D];counts={}
                for kind,part in [('identity','all')]+[(k,p) for k in ('opposite','same') for p in PARTS]:
                    B.guard();t=time.perf_counter();idx=np.arange(len(x)) if kind=='identity' else mapping[kind];mixed=[a[idx].copy() if j in PARTS[part] else a.copy() for j,a in enumerate(state)];construction+=time.perf_counter()-t;t=time.perf_counter()
                    saved,cost=engine.suffix(x,w,arm,h,q,tick,mixed);suffix_seconds+=time.perf_counter()-t;B.forward_cost(len(x),x.shape[1]-tick,h)
                    name=kind+'_'+part;np.savez_compressed(B.out/f'{ident}_d{D}_clean_{name}.npz',**saved);counts[name]=saved['counts'];metric,_=basic(saved['counts'],b['y'])
                    if part=='all':assert np.array_equal(saved['counts'],baseline[idx]),'full-state donor output'
                    rows.append(dict(**meta,delay=D,kind='clean',condition=name,**metric,**cost,suffix_steps=x.shape[1]-tick,suffix_forward_macs=(x.shape[1]-tick)*(2*h*h+14*h),suffix_state_updates=(x.shape[1]-tick)*(3*h+2),suffix_graded_local_transfers=(x.shape[1]-tick)*2*h if arm=='DEND' else 0))
                for part in PARTS:trs.append(dict(**meta,delay=D,part=part,**engine.transfer(counts['opposite_'+part],counts['same_'+part],baseline,b['y'],mapping)))
            overhead.append(dict(id=ident,transplant_construction_seconds=construction,suffix_inference_seconds=suffix_seconds));B.guard(disk=True)
            for name,obj in [('ROWS',rows),('TRANSFERS',trs),('OVERHEAD',overhead)]:dump(B.out/(name+'.json'),obj)
            print('evaluate',ident,'complete',flush=True)
    B.finish_stage(models=24,evaluations=624,transfer_rows=240)
def freeze():
    assert not (R/'FREEZE.json').exists();assert read(R/'PRECHECK.json')['status']=='PASS'
    paths=[p for p in sorted(R.iterdir()) if p.suffix in ('.py','.md','.json')]
    dump(R/'FREEZE.json',dict(utc=utc(),scientific_data_generated=False,files={p.name:sha(p) for p in paths}))
    dump(R/'ENVIRONMENT.json',dict(utc=utc(),machine=platform.node(),python=sys.version,executable=sys.executable,numpy=np.__version__,threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},initial_memory=memory()))
    print('FROZEN',sha(R/'FREEZE.json'),flush=True)
def science():
    B=Budget()
    try:train(B);evaluate(B);B.finish()
    except Exception as e:B.fail(e);raise
if __name__=='__main__':
    {'freeze':freeze,'science':science}[sys.argv[1]]()
