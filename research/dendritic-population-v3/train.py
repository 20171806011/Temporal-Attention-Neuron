import hashlib,json,time,platform
from pathlib import Path
import numpy as np
from model import initialize,loss_grad,metrics,loss_counts,P
from data import make
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    freeze=json.loads((R/'FREEZE.json').read_text())
    for f,h in freeze['sha256'].items():assert sha(R/f)==h
    out=R/'results';out.mkdir(exist_ok=False);start=time.perf_counter();logs=[]
    for s in range(3):
        x,y=make(128,2026110500+10*s);vx,vy=make(128,2026110501+10*s)
        np.savez_compressed(out/f'development_{s}.npz',x=x,y=y,vx=vx,vy=vy)
        initial=initialize(2026110550+s);np.save(out/f'initial_{s}.npy',initial)
        for arm in ('DEND','POINT'):
            im=metrics(vx,vy,initial,arm)
            for kind in ('CE','RATE'):
                for lr in (.01,.03):
                    t=initial.copy();m=np.zeros(P);v=np.zeros(P);history=[];gradients=[];tick=time.perf_counter()
                    for k in range(200):
                        if time.perf_counter()-start>300:raise RuntimeError('Training wall budget exceeded; no test')
                        loss,g,c=loss_grad(x,y,t,arm,kind);norm=float(np.linalg.norm(g));g=g/max(1,norm);gradients.append(g.copy())
                        m=.9*m+.1*g;v=.999*v+.001*g*g
                        new=np.clip(t-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
                        pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1));history.append([loss,norm,float(np.linalg.norm(new-t)),float(np.mean(pred==y))]);t=new
                    name=f'candidate_{s}_{arm}_{kind}_{lr}';np.save(out/(name+'.npy'),t);np.savez_compressed(out/(name+'_trace.npz'),history=np.array(history),clipped_gradients=np.array(gradients))
                    val=metrics(vx,vy,t,arm);rate,_=loss_counts(val['counts'],vy,'RATE')
                    row=dict(seed=s,arm=arm,loss=kind,lr=lr,file=name+'.npy',trace=name+'_trace.npz',sha256=sha(out/(name+'.npy')),validation_accuracy=val['accuracy'],validation_coverage=val['coverage'],validation_rate_error=rate,validation_abstentions=val['abstentions'],validation_events_mean=val['events_mean'],initial_accuracy=im['accuracy'],initial_coverage=im['coverage'],seconds=time.perf_counter()-tick,parameter_change=float(np.linalg.norm(t-initial)))
                    logs.append(row);(out/'training_log.json').write_text(json.dumps(logs,indent=2)+'\n')
                    print(s,arm,kind,lr,'validation',val['accuracy'],flush=True)
    configs=[]
    for kind in ('CE','RATE'):
        for lr in (.01,.03):
            rows=[z for z in logs if z['loss']==kind and z['lr']==lr]
            configs.append(dict(loss=kind,lr=lr,min_accuracy=min(z['validation_accuracy'] for z in rows),mean_accuracy=float(np.mean([z['validation_accuracy'] for z in rows])),mean_rate_error=float(np.mean([z['validation_rate_error'] for z in rows]))))
    best=min(configs,key=lambda z:(-z['min_accuracy'],-z['mean_accuracy'],z['mean_rate_error'],z['lr'],0 if z['loss']=='RATE' else 1))
    selected=[z for z in logs if (z['loss'],z['lr'])==(best['loss'],best['lr'])]
    gate=all(z['validation_accuracy']>=.9 and z['validation_coverage']>=.9 for z in selected)
    lock=dict(configurations=configs,shared_configuration=best,gate_pass=gate,selected=selected,training_seconds=time.perf_counter()-start,test_generated=False,versions={'python':platform.python_version(),'numpy':np.__version__},freeze_sha256=sha(R/'FREEZE.json'))
    (out/'SELECTION.json').write_text(json.dumps(lock,indent=2)+'\n');print(json.dumps({'gate':gate,'shared_configuration':best,'seconds':lock['training_seconds']},indent=2),flush=True)
if __name__=='__main__':main()
