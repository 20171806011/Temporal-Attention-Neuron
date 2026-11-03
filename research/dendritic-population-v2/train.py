import hashlib,json,time,platform
from pathlib import Path
import numpy as np
from model import initialize,objective,forward,P
from data import make
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    freeze=json.loads((ROOT/'FREEZE.json').read_text())
    for f,h in freeze['sha256'].items():assert sha(ROOT/f)==h
    out=ROOT/'results';out.mkdir(exist_ok=False);start=time.perf_counter();selected=[];logs=[]
    for seed in range(3):
        x,y,pair=make(128,2026110400+10*seed);vx,vy,vp=make(128,2026110401+10*seed)
        np.savez_compressed(out/f'development_{seed}.npz',x=x,y=y,pair=pair,vx=vx,vy=vy,vpair=vp)
        init=initialize(2026110450+seed);np.save(out/f'initial_{seed}.npy',init)
        for arm in ('DEND','POINT'):
            candidates=[]
            for lr in (.03,.1):
                theta=init.copy();rng=np.random.default_rng(2026110460+seed);history=[];tick=time.perf_counter()
                for k in range(100):
                    if time.perf_counter()-start>300:raise RuntimeError('CPU wall budget exceeded; no test evaluation')
                    d=rng.choice([-1.,1.],P);c=.15/(k+1)**.101;a=lr/(k+1)**.602
                    plus=objective(x,y,theta+c*d,arm);minus=objective(x,y,theta-c*d,arm)
                    grad=np.clip((plus-minus)/(2*c)*d,-5,5);new=np.clip(theta-a*grad,-2,2)
                    history.append([plus,minus,float(np.linalg.norm(new-theta))]);theta=new
                pred=forward(vx,theta,arm)['prediction'];acc=float(np.mean(pred==vy));loss=objective(vx,vy,theta,arm)
                name=f'candidate_{seed}_{arm}_{lr}.npy';np.save(out/name,theta)
                record=dict(seed=seed,arm=arm,lr=lr,file=name,sha256=sha(out/name),validation_accuracy=acc,validation_objective=loss,seconds=time.perf_counter()-tick,history=history,parameter_change=float(np.linalg.norm(theta-init)))
                candidates.append(record);logs.append(record)
                (out/'training_log.json').write_text(json.dumps(logs,indent=2)+'\n')
                print(seed,arm,lr,'validation',acc,flush=True)
            selected.append(min(candidates,key=lambda z:(-z['validation_accuracy'],z['validation_objective'],z['lr'])))
    lock=dict(selected=[{k:v for k,v in r.items() if k!='history'} for r in selected],training_seconds=time.perf_counter()-start,versions={'python':platform.python_version(),'numpy':np.__version__},test_generated=False,freeze_sha256=sha(ROOT/'FREEZE.json'))
    (out/'SELECTION.json').write_text(json.dumps(lock,indent=2)+'\n')
    print('Selection frozen; test not generated.',flush=True)
if __name__=='__main__':main()
