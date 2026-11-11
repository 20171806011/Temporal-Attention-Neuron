import sys,platform,os,time,numpy as np
import model,population,base_data
from common import *
def groups(s,out):
    gs=[]
    for g,D in enumerate((0,24)):
        b=base_data.make_base(64,3971010000+100*s+10*g);np.savez_compressed(out/f'base_s{s}_g{g}.npz',**b);x,q=assemble(b,D);gs.append((x,b['y'],q))
    return gs
def fit(B,ident,s,arm,lr,gs):
    tic=time.perf_counter();initial=population.initialize(3971030000+s,H);w=initial.copy();m=np.zeros_like(w);v=np.zeros_like(w);grads=[];counts=[];history=[]
    try:
        for k in range(UPDATES):
            B.guard();g=np.zeros_like(w);loss=0.;cc=[]
            for x,y,q in gs:
                ll,gg,c=model.loss_grad(x,y,w,arm,q);g+=gg/2;loss+=ll/2;cc.append(c);B.charge(len(x),x.shape[1])
            norm=float(np.linalg.norm(g));assert np.isfinite(g).all() and np.isfinite(loss),'finite training';g/=max(1,norm);grads.append(g.copy());counts.append(np.concatenate(cc).astype(np.uint8));history.append([loss,norm,float(np.linalg.norm(g*max(1,norm)-.0002*w/len(w))),float(np.linalg.norm(.0002*w/len(w)))])
            m=.9*m+.1*g;v=.999*v+.001*g*g;w=np.clip(w-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
    except Exception:
        np.savez_compressed(B.out/f'{ident}_PARTIAL.npz',initial=initial,theta=w,gradients=grads,counts=counts,history=history);raise
    np.savez_compressed(B.out/f'{ident}.npz',initial=initial,theta=w,gradients=grads,counts=counts,history=history)
    row=dict(id=ident,seed=s,arm=arm,h=H,lr=lr,loss='CE',updates=UPDATES,seconds=time.perf_counter()-tic,dense_forward_macs=UPDATES*64*72*156,dense_backward_macs=UPDATES*64*72*240,cost_per_training_pair=structural_cost(arm,H,72),attention_backward_unit_steps=UPDATES*64*72*12 if arm in ('FULL','NOATT') else 0,history_gradient_accumulations=UPDATES*64*(5*72-20)*12 if arm in ('FULL','NOATT') else 0)
    return row
def train(B):
    B.begin('train')
    for s in range(3):
        gs=groups(s,B.out)
        for arm in ARMS:
            for j,lr in enumerate(RATES):
                ident=f's{s}_{arm}_lr{j}';row=fit(B,ident,s,arm,lr,gs);B.rows.append(row);B.flush();B.guard(disk=True);print('train',ident,'complete',flush=True)
    B.finish_stage(fits=24,updates=4800,dense_backward_macs=sum(r['dense_backward_macs'] for r in B.rows),attention_backward_unit_steps=sum(r['attention_backward_unit_steps'] for r in B.rows),history_gradient_accumulations=sum(r['history_gradient_accumulations'] for r in B.rows))
    dump(B.out/'TRAIN_LOCK.json',dict(utc=utc(),test_generated=False,files={p.name:sha(p) for p in sorted(B.out.iterdir()) if p.is_file()}))
def evaluate(B):
    for name,hs in read(R/'train/TRAIN_LOCK.json')['files'].items():assert sha(R/'train'/name)==hs,'trained model lock'
    B.begin('evaluate');training=read(R/'train/ROWS.json')
    for s in range(3):
        b=base_data.make_base(256,3971020000+100*s);np.savez_compressed(B.out/f'test_s{s}.npz',**b)
        for tr in [r for r in training if r['seed']==s]:
            ident=tr['id'];w=np.load(R/'train'/f'{ident}.npz')['theta'];meta={k:tr[k] for k in ('id','seed','arm','h','lr','loss')}
            for D in DELAYS:
                B.guard();x,q=assemble(b,D);saved=model.evaluate(x,w,tr['arm'],q);cost=dict(mean_resources=saved['resources'].mean(0).tolist(),**structural_cost(tr['arm'],H,x.shape[1]));np.savez_compressed(B.out/f'{ident}_d{D}.npz',**saved);B.rows.append(dict(**meta,delay=D,**basic(saved['counts'],b['y']),**cost));B.charge(len(x),x.shape[1])
            B.flush();B.guard(disk=True);print('evaluate',ident,'complete',flush=True)
    B.finish_stage(models=24,evaluations=96)
def freeze():
    assert not (R/'FREEZE.json').exists();assert read(R/'PRECHECK.json')['status']=='PASS';assert read(R/'TIMING.json')['forecast_with_margin']<840
    paths=[p for p in sorted(R.rglob('*')) if p.is_file() and p.suffix in ('.py','.md','.json') and p.name not in ('report.py','summarize.py')]
    dump(R/'FREEZE.json',dict(utc=utc(),scientific_data_generated=False,files={str(p.relative_to(R)):sha(p) for p in paths}))
    dump(R/'ENVIRONMENT.json',dict(machine=platform.node(),python=sys.version,numpy=np.__version__,executable=sys.executable,threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}));print('FROZEN',sha(R/'FREEZE.json'),flush=True)
def science():
    B=Budget()
    try:train(B);evaluate(B);B.finish()
    except Exception as e:B.fail(e);raise
if __name__=='__main__':{'freeze':freeze,'science':science}[sys.argv[1]]()
