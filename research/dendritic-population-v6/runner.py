"""Fixed eighty-run training stage. Never generates held-out test data."""
import json,time,platform
import numpy as np
import model
from data import make
from common import R,OUT,ARMS,RECIPES,REGISTRY,CHECKPOINTS,dump,sha,freeze_check,describe,count_metrics,save_native

def main():
    freeze_check();OUT.mkdir(exist_ok=False);start=time.perf_counter();rows=[]
    dump(OUT/'TRAIN_STARTED.json',{'freeze_sha256':sha(R/'FREEZE.json'),'seed_registry_sha256':sha(R/'SEEDS.json')})
    for seed in REGISTRY['pairs']:
        s=seed['seed'];x,y,pair=make(128,seed['train_seed']);df=f'train_s{s}.npz'
        np.savez_compressed(OUT/df,x=x,y=y,pair=pair,seed=np.array(seed['train_seed']))
        for arm in ARMS:
            for recipe,(kind,lr) in RECIPES.items():
                ident=f's{s}_{arm}_{recipe}';t=model.initialize(seed['initial_seed']);initial=ident+'_initial.npy';np.save(OUT/initial,t)
                m=np.zeros(176);v=np.zeros(176);history=[];grads=[];counts=[];weights=[];diagnostics=[];tick=time.perf_counter()
                for k in range(201):
                    if time.perf_counter()-start>300:
                        np.save(OUT/(ident+'_PARTIAL.npy'),t);np.savez_compressed(OUT/(ident+'_PARTIAL_trace.npz'),history=np.array(history),clipped_gradients=np.array(grads),train_counts=np.array(counts));dump(OUT/'FAILURE.json',{'id':ident,'updates':k,'reason':'300s training cap'});raise RuntimeError('Training cap, no automatic retry')
                    if k in CHECKPOINTS:
                        nv=model.metrics(x,y,t,arm);weights.append(t.copy());diagnostics.append(dict(updates=k,**describe(nv,y)))
                        if k==200:
                            train_file=ident+'_train_native.npz';save_native(OUT/train_file,nv);final_metrics=describe(nv,y)
                    if k==200:break
                    loss,g,c=model.loss_grad(x,y,t,arm,kind);norm=float(np.linalg.norm(g))
                    if not (np.isfinite(loss) and np.isfinite(norm) and np.isfinite(g).all()):
                        np.save(OUT/(ident+'_PARTIAL.npy'),t);np.savez_compressed(OUT/(ident+'_PARTIAL_trace.npz'),history=np.array(history),clipped_gradients=np.array(grads),train_counts=np.array(counts));dump(OUT/'FAILURE.json',{'id':ident,'updates':k,'reason':'nonfinite objective or gradient'});raise RuntimeError('Numerical failure, no automatic retry')
                    g=g/max(1,norm)
                    cm=count_metrics(c,y);grads.append(g.copy());counts.append(c.copy());m=.9*m+.1*g;v=.999*v+.001*g*g
                    new=np.clip(t-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
                    if not np.isfinite(new).all():
                        np.save(OUT/(ident+'_PARTIAL.npy'),t);np.savez_compressed(OUT/(ident+'_PARTIAL_trace.npz'),history=np.array(history),clipped_gradients=np.array(grads),train_counts=np.array(counts));dump(OUT/'FAILURE.json',{'id':ident,'updates':k,'reason':'nonfinite proposed parameters; last gradient has no completed update'});raise RuntimeError('Numerical failure, no automatic retry')
                    history.append([loss,norm,float(np.linalg.norm(new-t)),cm['accuracy'],cm['coverage'],cm['silent_fraction'],cm['mean_signed_count_margin'],cm['nonsilent_tie_fraction']]);t=new
                f=ident+'.npy';trace=ident+'_trace.npz';checkpoints=ident+'_checkpoints.npz';diag=ident+'_checkpoints.json'
                np.save(OUT/f,t);np.savez_compressed(OUT/trace,history=np.array(history),clipped_gradients=np.array(grads),train_counts=np.array(counts))
                np.savez_compressed(OUT/checkpoints,updates=np.array(CHECKPOINTS),theta=np.array(weights));dump(OUT/diag,diagnostics)
                row=dict(id=ident,seed=s,arm=arm,recipe=recipe,loss=kind,lr=lr,initial_seed=seed['initial_seed'],data_file=df,
                    initial_file=initial,file=f,trace=trace,checkpoints=checkpoints,diagnostics=diag,train_native=train_file,
                    sha256=sha(OUT/f),steps=200,seconds=time.perf_counter()-tick,**final_metrics)
                rows.append(row);dump(OUT/'TRAIN_LOG.json',rows);print(ident,'train_accuracy',row['accuracy'],'coverage',row['coverage'],flush=True)
    dump(OUT/'TRAIN_LOCK.json',dict(rows=rows,seconds=time.perf_counter()-start,files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        freeze_sha256=sha(R/'FREEZE.json'),test_generated=False,versions=dict(python=platform.python_version(),numpy=np.__version__),
        history_columns=['loss','raw_gradient_norm','update_norm','preupdate_accuracy','preupdate_coverage','preupdate_silent_fraction','preupdate_mean_signed_count_margin','preupdate_nonsilent_tie_fraction']))
if __name__=='__main__':main()
