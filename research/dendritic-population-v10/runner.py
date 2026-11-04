"""Forty fixed binding/size training runs. Does not generate any test data."""
import json,time,platform
import numpy as np
import model
from data import make
from common import R,OUT,ARMS,SIZES,REGISTRY,CHECKPOINTS,dump,sha,freeze_check,require_committed,describe,count_metrics,save_native

def main():
    freeze_check()
    frozen=json.loads((R/'FREEZE.json').read_text())['sha256']
    protocol_commit=require_committed([R/'FREEZE.json']+[R/f for f in frozen])
    OUT.mkdir(exist_ok=False);start=time.perf_counter();rows=[]
    dump(OUT/'TRAIN_STARTED.json',{'freeze_sha256':sha(R/'FREEZE.json'),'seed_registry_sha256':sha(R/'SEEDS.json'),'protocol_commit':protocol_commit})
    for seed in REGISTRY['pairs']:
        s=seed['seed'];x,y,pattern=make(128,seed['train_seed']);df=f'train_s{s}.npz'
        np.savez_compressed(OUT/df,x=x,y=y,pattern=pattern,seed=np.array(seed['train_seed']))
        for h in SIZES:
            for arm in ARMS:
                kind,lr='RATE',.01;P=model.parameter_count(h);initial_seed=seed['initial_seeds'][str(h)]
                ident=f's{s}_{arm}_h{h}';t=model.initialize(initial_seed,h);initial=ident+'_initial.npy';np.save(OUT/initial,t)
                m=np.zeros(P);v=np.zeros(P);history=[];grads=[];counts=[];weights=[];diagnostics=[];tick=time.perf_counter()
                def stop(k,reason):
                    np.save(OUT/(ident+'_PARTIAL.npy'),t)
                    np.savez_compressed(OUT/(ident+'_PARTIAL_trace.npz'),history=np.array(history),clipped_gradients=np.array(grads),train_counts=np.array(counts))
                    dump(OUT/'FAILURE.json',{'id':ident,'updates':k,'reason':reason,'completed_runs':len(rows)});raise RuntimeError(reason+'; no automatic retry')
                for k in range(201):
                    if time.perf_counter()-start>300:stop(k,'300s training cap')
                    if k in CHECKPOINTS:
                        nv=model.metrics(x,y,t,arm,h);weights.append(t.copy());diagnostics.append(dict(updates=k,**describe(nv,y,pattern,h)))
                        if k==200:
                            train_file=ident+'_train_native.npz';save_native(OUT/train_file,nv);final_metrics=describe(nv,y,pattern,h)
                    if k==200:break
                    loss,g,c=model.loss_grad(x,y,t,arm,kind,h);norm=float(np.linalg.norm(g))
                    if not (np.isfinite(loss) and np.isfinite(norm) and np.isfinite(g).all()):stop(k,'nonfinite objective or gradient')
                    g=g/max(1,norm);cm=count_metrics(c,y);grads.append(g.copy());counts.append(c.copy());m=.9*m+.1*g;v=.999*v+.001*g*g
                    new=np.clip(t-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
                    if not np.isfinite(new).all():stop(k,'nonfinite proposed parameters; last saved gradient has no completed update')
                    history.append([loss,norm,float(np.linalg.norm(new-t)),cm['accuracy'],cm['coverage'],cm['silent_fraction'],cm['mean_signed_count_margin'],cm['nonsilent_tie_fraction'],cm['balanced_accuracy']]);t=new
                f=ident+'.npy';trace=ident+'_trace.npz';checkpoints=ident+'_checkpoints.npz';diag=ident+'_checkpoints.json'
                np.save(OUT/f,t);np.savez_compressed(OUT/trace,history=np.array(history),clipped_gradients=np.array(grads),train_counts=np.array(counts))
                np.savez_compressed(OUT/checkpoints,updates=np.array(CHECKPOINTS),theta=np.array(weights));dump(OUT/diag,diagnostics)
                row=dict(id=ident,seed=s,arm=arm,h=h,loss=kind,lr=lr,initial_seed=initial_seed,data_file=df,initial_file=initial,file=f,trace=trace,
                    checkpoints=checkpoints,diagnostics=diag,train_native=train_file,sha256=sha(OUT/f),steps=200,seconds=time.perf_counter()-tick,**final_metrics)
                rows.append(row);dump(OUT/'TRAIN_LOG.json',rows)
                print(ident,'train_BA',row['balanced_accuracy'],'accuracy',row['accuracy'],'coverage',row['coverage'],flush=True)
    dump(OUT/'TRAIN_LOCK.json',dict(rows=rows,seconds=time.perf_counter()-start,files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        freeze_sha256=sha(R/'FREEZE.json'),protocol_commit=protocol_commit,test_generated=False,versions=dict(python=platform.python_version(),numpy=np.__version__),
        history_columns=['loss','raw_gradient_norm','update_norm','preupdate_accuracy','preupdate_coverage','preupdate_silent_fraction','preupdate_mean_signed_count_margin','preupdate_nonsilent_tie_fraction','preupdate_balanced_accuracy']))
if __name__=='__main__':main()
