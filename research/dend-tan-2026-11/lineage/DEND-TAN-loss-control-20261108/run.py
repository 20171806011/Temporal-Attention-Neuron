import sys,platform,os,time,numpy as np
import model,population,task_data,training
from common import *
def train(B):
    B.begin('train')
    for s in range(3):
        b=task_data.make_base(64,3991010000+100*s);np.savez_compressed(B.out/f'base_s{s}.npz',**b);x,q=task_data.assemble(b)
        for arm in ARMS:
            for kind in training.LOSSES:
                lr=.0025;ident=f's{s}_{arm}_{kind}';tic=time.perf_counter();initial=population.initialize(3991030000+s,6);w=initial.copy();m=w*0;v=w*0;gradients=[];counts=[];history=[];components=[];component_gradients=[];diagnostic_steps=[]
                def save(name):np.savez_compressed(B.out/(name+'.npz'),initial=initial,theta=w,gradients=gradients,counts=counts,history=history,components=components,component_gradients=component_gradients,diagnostic_steps=diagnostic_steps)
                try:
                    for k in range(UPDATES):
                        B.guard();diagnostic=k in training.DIAGNOSTIC_STEPS;loss,g,c,terms,cg=training.loss_grad(x,b['y'],w,arm,q,kind,diagnostic);B.charge(64,24);norm=float(np.linalg.norm(g));rn=.0002*w/176;assert np.isfinite(g).all() and np.isfinite(loss)
                        history.append([loss,norm,float(np.linalg.norm(g-rn)),float(np.linalg.norm(rn))]);components.append(terms)
                        if diagnostic:component_gradients.append(cg);diagnostic_steps.append(k)
                        g/=max(1,norm);gradients.append(g.copy());counts.append(c.astype(np.uint8));m=.9*m+.1*g;v=.999*v+.001*g*g;w=np.clip(w-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
                except Exception:save(ident+'_PARTIAL');raise
                save(ident);bp=UPDATES+(len(training.DIAGNOSTIC_STEPS) if kind!='CE' else 0)
                B.rows.append(dict(id=ident,seed=s,arm=arm,h=6,lr=lr,loss=kind,updates=UPDATES,seconds=time.perf_counter()-tic,backward_passes=bp,diagnostic_backward_passes=bp-UPDATES,dense_forward_macs=UPDATES*64*24*156,dense_backward_macs=bp*64*24*240,cost_per_training_episode=structural_cost(arm,6,24),attention_backward_unit_steps=bp*64*24*12 if arm in ('FULL','NOATT') else 0,history_gradient_accumulations=bp*64*(5*24-10)*12 if arm in ('FULL','NOATT') else 0));B.flush();B.guard(disk=True);print('train',ident,'complete',flush=True)
    B.finish_stage(fits=36,updates=36*UPDATES,dense_backward_macs=sum(r['dense_backward_macs'] for r in B.rows),attention_backward_unit_steps=sum(r['attention_backward_unit_steps'] for r in B.rows),history_gradient_accumulations=sum(r['history_gradient_accumulations'] for r in B.rows));dump(B.out/'TRAIN_LOCK.json',dict(utc=utc(),test_generated=False,files={p.name:sha(p) for p in sorted(B.out.iterdir()) if p.is_file()}))
def evaluate(B):
    for n,h in read(R/'train/TRAIN_LOCK.json')['files'].items():assert sha(R/'train'/n)==h
    B.begin('evaluate');tr=read(R/'train/ROWS.json')
    for s in range(3):
        b=task_data.make_base(256,3991020000+100*s);np.savez_compressed(B.out/f'test_s{s}.npz',**b)
        for row in [r for r in tr if r['seed']==s]:
            w=np.load(R/'train'/(row['id']+'.npz'))['theta'];meta={k:row[k] for k in ('id','seed','arm','h','lr','loss')}
            for condition in task_data.CONDITIONS:
                B.guard();x,q=task_data.assemble(b,condition);z=model.evaluate(x,w,row['arm'],q);np.savez_compressed(B.out/(row['id']+'_'+condition+'.npz'),**z);B.rows.append(dict(**meta,condition=condition,**basic(z['counts'],b['y']),mean_resources=z['resources'].mean(0).tolist(),**structural_cost(row['arm'],6,24)));B.charge(256,24)
            B.flush();B.guard(disk=True);print('evaluate',row['id'],'complete',flush=True)
    B.finish_stage(models=36,evaluations=216)
def freeze():
    assert not (R/'FREEZE.json').exists()
    for name in ('PRECHECK','DATA_CHECK','LOSS_CHECK','GRADIENT_CHECK','TIMING'):assert read(R/(name+'.json'))['status']=='PASS'
    assert (R/'audit.py').exists() and (R/'PROTOCOL.md').exists()
    paths=[p for p in sorted(R.iterdir()) if p.is_file() and p.suffix in ('.py','.md','.json') and p.name not in ('summarize.py','report.py')];dump(R/'FREEZE.json',dict(utc=utc(),scientific_data_generated=False,files={p.name:sha(p) for p in paths}));print('FROZEN',flush=True)
def science():
    B=Budget();dump(R/'ENVIRONMENT.json',dict(machine=platform.node(),python=sys.version,numpy=np.__version__,threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    try:train(B);evaluate(B);B.finish()
    except Exception as e:B.fail(e);raise
if __name__=='__main__':{'freeze':freeze,'science':science}[sys.argv[1]]()
