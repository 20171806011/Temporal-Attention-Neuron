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
def fit(B,ident,s,h,arm,loss,gs):
    tic=time.perf_counter();initial=model.initialize(seeds(B.stage,s)['initial'],h);w=initial.copy();m=np.zeros_like(w);v=m.copy();grads=[];counts=[];history=[]
    try:
        for k in range(200):
            B.guard();g=np.zeros_like(w);l=0.;cs=[]
            for x,y,q in gs:
                ll,gg,cc=model.loss_grad(x,y,w,arm,loss,h,ret=RET,query=q);g+=gg/2;l+=ll/2;cs.append(cc);B.forward_cost(len(x),x.shape[1],h)
            norm=float(np.linalg.norm(g));assert np.isfinite(g).all() and np.isfinite(l),'nonfinite update';g/=max(1,norm)
            grads.append(g.copy());counts.append(np.concatenate(cs).astype(np.uint8));history.append([l,norm]);m=.9*m+.1*g;v=.999*v+.001*g*g
            w=np.clip(w-.01*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
    except Exception:
        np.savez_compressed(B.out/f'{ident}_PARTIAL.npz',initial=initial,theta=w,gradients=grads,counts=counts,history=history);raise
    np.savez_compressed(B.out/f'{ident}.npz',initial=initial,theta=w,gradients=grads,counts=counts,history=history);B.guard(disk=True)
    row=dict(id=ident,seed=s,h=h,arm=arm,loss=loss,ret=list(RET),lr=.01,updates=200,seconds=time.perf_counter()-tic,forward_macs=200*64*72*(2*h*h+14*h),backward_macs=200*64*72*(4*h*h+16*h))
    return w,row
def tune():
    B=Budget('tune');rows=[]
    for s in range(2):
        gs=groups('tune',s,B.out);b=base_data.make_base(256,seeds('tune',s)['test']);np.savez_compressed(B.out/f'val_s{s}.npz',**b)
        for h in SIZES:
            for arm in ARMS:
                for loss in LOSSES:
                    ident=f's{s}_{arm}_h{h}_{loss}';w,row=fit(B,ident,s,h,arm,loss,gs);cc=[];mm=[]
                    for D in (0,24):
                        x,q=assemble(b,D,'clean');saved,cost,_=engine.evaluate(x,w,arm,h,q);B.forward_cost(len(x),len(x[0]),h);cc.append(saved['counts']);met,_=basic(saved['counts'],b['y']);mm.append(met)
                    np.savez_compressed(B.out/f'{ident}_val.npz',counts=np.array(cc,dtype=np.uint8));row.update(score=float(np.mean([m['balanced_accuracy'] for m in mm])),coverage=float(np.mean([m['coverage'] for m in mm])),conditions=mm);rows.append(row)
                    dump(B.out/'ROWS.json',rows);print('tune',ident,'complete',flush=True)
    selected={};rankings=[]
    for h in SIZES:
        for arm in ARMS:
            candidates=[]
            for loss in LOSSES:
                rr=[r for r in rows if (r['h'],r['arm'],r['loss'])==(h,arm,loss)]
                candidates.append(dict(loss=loss,score=float(np.mean([r['score'] for r in rr])),coverage=float(np.mean([r['coverage'] for r in rr]))))
            choice=sorted(candidates,key=lambda r:(-r['score'],-r['coverage'],LOSSES.index(r['loss'])))[0];selected[f'{arm}_h{h}']=choice['loss'];rankings.append(dict(h=h,arm=arm,candidates=candidates))
    B.finish(fits=24,updates=4800,training_backward_macs=sum(r['backward_macs'] for r in rows))
    dump(B.out/'SELECTION_LOCK.json',dict(utc=utc(),selected=selected,rankings=rankings,final_training_generated=False,files={p.name:sha(p) for p in sorted(B.out.iterdir()) if p.is_file()}))
def train():
    selection=read(R/'tune'/'SELECTION_LOCK.json')
    for p,h in selection['files'].items():assert sha(R/'tune'/p)==h,'tune lock '+p
    B=Budget('train');rows=[]
    for s in range(5):
        gs=groups('train',s,B.out)
        for h in SIZES:
            for arm in ARMS:
                loss=selection['selected'][f'{arm}_h{h}'];ident=f's{s}_{arm}_h{h}';_,row=fit(B,ident,s,h,arm,loss,gs);rows.append(row);dump(B.out/'ROWS.json',rows);print('train',ident,'complete',flush=True)
    B.finish(fits=30,updates=6000,training_backward_macs=sum(r['backward_macs'] for r in rows))
    dump(B.out/'TRAIN_LOCK.json',dict(utc=utc(),selection_sha256=sha(R/'tune'/'SELECTION_LOCK.json'),test_generated=False,files={p.name:sha(p) for p in sorted(B.out.iterdir()) if p.is_file()}))
def evaluate():
    lock=read(R/'train'/'TRAIN_LOCK.json')
    for p,h in lock['files'].items():assert sha(R/'train'/p)==h,'train lock '+p
    B=Budget('evaluate');rows=[];trs=[];overhead=[];trainrows=read(R/'train'/'ROWS.json')
    for s in range(5):
        b=base_data.make_base(256,seeds('train',s)['test']);np.savez_compressed(B.out/f'test_s{s}.npz',**b);mapping=donors(b['y'],seeds('train',s)['donors']);np.savez_compressed(B.out/f'donors_s{s}.npz',**mapping)
        for tr in [r for r in trainrows if r['seed']==s]:
            ident=tr['id'];h=tr['h'];arm=tr['arm'];w=np.load(R/'train'/f'{ident}.npz')['theta'];baselines={};construction=0.;suffix_seconds=0.
            meta={k:tr[k] for k in ('id','seed','h','arm','loss','ret')}
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
    B.finish(models=30,evaluations=840,transfer_rows=300)
def freeze():
    assert not (R/'FREEZE.json').exists();assert read(R/'PRECHECK.json')['status']=='PASS'
    paths=[p for p in sorted(R.iterdir()) if p.suffix in ('.py','.md','.json')]
    dump(R/'FREEZE.json',dict(utc=utc(),scientific_data_generated=False,files={p.name:sha(p) for p in paths}))
    dump(R/'ENVIRONMENT.json',dict(utc=utc(),machine=platform.node(),python=sys.version,executable=sys.executable,numpy=np.__version__,threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},initial_memory=memory()))
    print('FROZEN',sha(R/'FREEZE.json'),flush=True)
if __name__=='__main__':
    stage=sys.argv[1]
    try:{'freeze':freeze,'tune':tune,'train':train,'evaluate':evaluate}[stage]()
    except Exception as e:
        dump(R/f'{stage}_FAILURE.json',dict(utc=utc(),reason=str(e),exception=type(e).__name__));raise
