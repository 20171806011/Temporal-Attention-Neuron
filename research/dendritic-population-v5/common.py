from pathlib import Path
import hashlib,json
import numpy as np
import model
R=Path(__file__).resolve().parent;OUT=R/'results';GRID=[(loss,lr) for loss in ('CE','RATE') for lr in (.003,.01,.03)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def freeze_check():
    for n,h in json.loads((R/'FREEZE.json').read_text())['sha256'].items():assert sha(R/n)==h,n

def brief(v,y):
    pred=v['prediction'];c=v['counts'];matrix=np.zeros((2,3),dtype=int)
    for truth,guess in zip(y,pred):matrix[int(truth),int(guess) if guess>=0 else 2]+=1
    return dict(accuracy=float(np.mean(pred==y)),coverage=float(np.mean(pred>=0)),abstentions=int(np.sum(pred<0)),abstention_rate=float(np.mean(pred<0)),silent_fraction=float(np.mean(np.all(c==0,1))),both_saturated_fraction=float(np.mean(np.all(c==8,1))),class_recall=[float(np.mean(pred[y==k]==k)) for k in (0,1)],confusion=matrix.tolist(),rate_error=model.loss_counts(c,y,'RATE')[0],events_mean=v['spikes'].sum(1).mean(0).tolist(),failed=bool(np.mean(pred==y)<.9 or np.mean(pred>=0)<.9))
def metrics(x,y,t,arm):return brief(model.metrics(x,y,t,arm),y)
def choose(rows):
    choices={};candidates={}
    for arm in ('DEND','POINT'):
        cs=[]
        for loss,lr in GRID:
            g=[z for z in rows if z['stage']=='search' and (z['arm'],z['loss'],z['lr'])==(arm,loss,lr)];assert len(g)==5
            cs.append(dict(loss=loss,lr=lr,min_accuracy=min(z['validation']['accuracy'] for z in g),mean_accuracy=float(np.mean([z['validation']['accuracy'] for z in g])),mean_rate_error=float(np.mean([z['validation']['rate_error'] for z in g]))))
        choices[arm]=min(cs,key=lambda z:(-z['min_accuracy'],-z['mean_accuracy'],z['mean_rate_error'],z['lr'],0 if z['loss']=='RATE' else 1));candidates[arm]=cs
    return {'selected':choices,'candidates':candidates}
