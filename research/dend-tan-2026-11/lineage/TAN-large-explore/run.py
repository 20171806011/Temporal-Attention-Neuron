"""Explicit finite exploration/validation stages with source and selection locks."""
import sys,time,json,platform,os
from pathlib import Path
import numpy as np
import model,data
from common import *

def train_data(stage,s,out):
    groups=[]
    for g,(delay,condition) in enumerate(data.TRAIN_CONDITIONS):
        base=data.make_base(32,seeds(stage,s)['train'][g]);np.savez_compressed(out/f'train_s{s}_g{g}.npz',**base)
        x,q=data.assemble(base,delay,condition);groups.append((x,base['y'],q))
    return groups
def fit(stage,out,ident,groups,s,h,arm,cfg,tic,prior):
    w=model.initialize(seeds(stage,s,h)['initial'],h);initial=w.copy();m=np.zeros_like(w);v=np.zeros_like(w)
    grads=[];counts=[];history=[];started=time.perf_counter()
    try:
        for k in range(200):
            guard(stage,tic,prior);g=np.zeros_like(w);loss=0;cs=[]
            for x,y,q in groups:
                l,d,c=model.loss_grad(x,y,w,arm,cfg['loss'],h,ret=cfg['ret'],query=q);loss+=l/4;g+=d/4;cs.append(c)
            norm=float(np.linalg.norm(g));assert np.isfinite(g).all() and np.isfinite(loss)
            g/=max(1,norm);grads.append(g.copy());counts.append(np.concatenate(cs,0));history.append([loss,norm])
            m=.9*m+.1*g;v=.999*v+.001*g*g
            w=np.clip(w-cfg['lr']*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
    except Exception:
        np.savez_compressed(out/f'{ident}_PARTIAL.npz',initial=initial,theta=w,gradients=np.array(grads),counts=np.array(counts),history=np.array(history));raise
    np.savez_compressed(out/f'{ident}_train.npz',initial=initial,theta=w,gradients=np.array(grads),counts=np.array(counts),history=np.array(history))
    return w,time.perf_counter()-started
def evaluate(base,w,h,arm,cfg,delay,condition,intervention=None):
    x,q=data.assemble(base,delay,condition);c,cache=model.forward(x,w,arm,h,ret=cfg['ret'],query=q,intervention=intervention)
    wi,wr,db,vb,wo,ob=model.unpack(w,h)
    # Dense topology: verify trained nonzero fanout before using the simple fanout formula.
    assert np.all(wr!=0) and np.all(wo!=0)
    row=describe(c,cache,base['y'],base['pattern'],h,arm,q,intervention,cfg['ret'])
    trace=dict(counts=c,spikes=np.stack([np.concatenate([a['bs'],a['hs'],a['os']],1)[:8] for a in cache],1),states=np.stack([a['states'][:8] for a in cache],1))
    return row,trace
def tuning(stage):
    out,tic,cpu,prior=start(stage);rows=[];sizes=SIZES if stage=='explore_tune' else (6,)
    modes=('full','rd0') if stage=='confirm_tune' and read(R/'HYPOTHESIS_LOCK.json')['include_rd0'] else ('full',)
    for s in range(2):
        groups=train_data(stage,s,out);base=data.make_base(128,seeds(stage,s)['evaluation']);np.savez_compressed(out/f'eval_s{s}.npz',**base)
        for h in sizes:
            for arm in ARMS:
                for mode in modes:
                    for cfg in candidates(mode=='rd0'):
                        ident=f's{s}_{arm}_h{h}_{mode}_c{cfg["id"]}';w,sec=fit(stage,out,ident,groups,s,h,arm,cfg,tic,prior);scores=[];allcounts=[]
                        for delay,condition in data.TRAIN_CONDITIONS:
                            metric,trace=evaluate(base,w,h,arm,cfg,delay,condition);scores.append(metric);allcounts.append(trace['counts'])
                        np.savez_compressed(out/f'{ident}_val.npz',counts=np.stack(allcounts,0))
                        rows.append(dict(id=ident,seed=s,h=h,arm=arm,mode=mode,config=cfg,seconds=sec,score=float(np.mean([r['balanced_accuracy'] for r in scores])),coverage=float(np.mean([r['coverage'] for r in scores])),condition_metrics=scores))
                        dump(out/'ROWS.json',rows)
                    dump(out/'ROWS.json',rows);print(stage,s,h,arm,mode,'four candidates complete',flush=True)
    selected={};scores=[]
    for h in sizes:
        for arm in ARMS:
            for mode in modes:
                table=[]
                for cfg in candidates(mode=='rd0'):
                    rr=[r for r in rows if (r['h'],r['arm'],r['mode'],r['config']['id'])==(h,arm,mode,cfg['id'])]
                    table.append(dict(config=cfg,score=float(np.mean([r['score'] for r in rr])),coverage=float(np.mean([r['coverage'] for r in rr]))))
                best=sorted(table,key=lambda a:(-a['score'],-a['coverage'],a['config']['id']))[0];key=f'{arm}_h{h}_{mode}';selected[key]=best['config'];scores.append(dict(cell=key,candidates=table))
    dump(out/'SELECTION_LOCK.json',dict(utc=utc(),selected=selected,scores=scores,files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},scientific_profile_generated=False))
    dump(out/'SUMMARY.json',dict(wall_seconds=time.perf_counter()-tic,cpu_seconds=time.process_time()-cpu,peak_working_set=peak(),fits=len(rows)))
def profile(stage):
    tuning_stage='explore_tune' if stage=='explore' else 'confirm_tune';selection=read(R/tuning_stage/'SELECTION_LOCK.json')
    for n,hsh in selection['files'].items():assert sha(R/tuning_stage/n)==hsh
    out,tic,cpu,prior=start(stage);sizes=SIZES if stage=='explore' else (6,)
    modes=('full','rd0') if stage=='confirm' and read(R/'HYPOTHESIS_LOCK.json')['include_rd0'] else ('full',);num=3 if stage=='explore' else 5
    trained=[];rows=[]
    for s in range(num):
        groups=train_data(stage,s,out)
        for h in sizes:
            for arm in ARMS:
                for mode in modes:
                    cfg=selection['selected'][f'{arm}_h{h}_{mode}'];ident=f's{s}_{arm}_h{h}_{mode}'
                    w,sec=fit(stage,out,ident,groups,s,h,arm,cfg,tic,prior);trained.append(dict(id=ident,seed=s,h=h,arm=arm,mode=mode,config=cfg,seconds=sec))
                    dump(out/'TRAIN_ROWS.json',trained)
        dump(out/'TRAIN_ROWS.json',trained);print(stage,'training block',s,'complete',flush=True)
    dump(out/'TRAIN_LOCK.json',dict(utc=utc(),selection_sha256=sha(R/tuning_stage/'SELECTION_LOCK.json'),test_generated=False,files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    # All trained weights are locked before ANY profile or validation-test base is generated.
    for s in range(num):
        base=data.make_base(256,seeds(stage,s)['evaluation']);np.savez_compressed(out/f'eval_s{s}.npz',**base)
        for tr in [r for r in trained if r['seed']==s]:
            w=np.load(out/f'{tr["id"]}_train.npz')['theta']
            cases=[(delay,condition,dict(name='baseline',kind='none')) for delay,condition in data.CONDITIONS]
            cases += [(24,'clean',a) for a in interventions() if a['name']!='baseline']
            for delay,condition,intervention in cases:
                guard(stage,tic,prior);metric,trace=evaluate(base,w,tr['h'],tr['arm'],tr['config'],delay,condition,intervention)
                ident=tr['id']+f'_d{delay}_{condition}_{intervention["name"]}';np.savez_compressed(out/f'{ident}.npz',**trace)
                rows.append(dict(id=ident,model_id=tr['id'],seed=s,h=tr['h'],arm=tr['arm'],mode=tr['mode'],config=tr['config'],delay=delay,condition=condition,intervention=intervention,**metric))
        dump(out/'ROWS.json',rows);print(stage,'profile block',s,'complete',flush=True)
    dump(out/'SUMMARY.json',dict(wall_seconds=time.perf_counter()-tic,cpu_seconds=time.process_time()-cpu,peak_working_set=peak(),fits=len(trained),evaluations=len(rows)))
def freeze():
    assert not (R/'EXPLORE_FREEZE.json').exists()
    files=['PLAN.md','model.py','data.py','common.py','run.py','audit.py','precheck.py','PRECHECK.json','RUNTIME_PILOT.json','vendor/previous_model.py']
    dump(R/'EXPLORE_FREEZE.json',dict(utc=utc(),files={n:sha(R/n) for n in files},scientific_data_generated=False))
    dump(R/'ENVIRONMENT.json',dict(utc=utc(),machine=platform.node(),python=sys.version,numpy=np.__version__,executable=sys.executable,threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    print('FROZEN',sha(R/'EXPLORE_FREEZE.json'),flush=True)
if __name__=='__main__':
    stage=sys.argv[1]
    try:
        if stage=='freeze':freeze()
        elif stage.endswith('_tune'):tuning(stage)
        else:profile(stage)
    except Exception as e:
        out=R/stage
        if out.is_dir():dump(out/'FAILURE.json',dict(utc=utc(),exception=type(e).__name__,reason=str(e)))
        raise
