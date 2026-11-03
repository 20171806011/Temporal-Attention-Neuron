import argparse,json,time,platform
import numpy as np
import model
from data import make
from common import R,OUT,GRID,dump,sha,freeze_check,metrics,choose
ROWS=[];START=0;SPENT=0

def dataset(name,train_seed,val_seed=None):
    x,y,pair=make(128,train_seed);d=dict(x=x,y=y,pair=pair,train_seed=np.array(train_seed))
    if val_seed is not None:
        vx,vy,vp=make(128,val_seed);d.update(vx=vx,vy=vy,vpair=vp,val_seed=np.array(val_seed))
    file=name+'.npz';assert not (OUT/file).exists();np.savez_compressed(OUT/file,**d);return file,d

def train(stage,s,arm,loss,lr,role,datafile,d,init_seed):
    ident=f'{stage}_s{s}_{arm}_{role}_{loss}_{lr}';t=model.initialize(init_seed);initial=ident+'_initial.npy';np.save(OUT/initial,t);m=np.zeros(176);v=np.zeros(176);hist=[];grad=[];tick=time.perf_counter()
    for k in range(200):
        if SPENT+time.perf_counter()-START>600:
            np.save(OUT/(ident+'_PARTIAL.npy'),t);np.savez_compressed(OUT/(ident+'_PARTIAL_trace.npz'),history=np.array(hist),clipped_gradients=np.array(grad));dump(OUT/'BUDGET_FAILURE.json',{'id':ident,'completed_updates':k});raise RuntimeError('600 s training cap; partial candidate retained, no automatic retry')
        lossval,raw,c=model.loss_grad(d['x'],d['y'],t,arm,loss);norm=float(np.linalg.norm(raw));g=raw/max(1,norm);grad.append(g.copy());m=.9*m+.1*g;v=.999*v+.001*g*g
        new=np.clip(t-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);hist.append([lossval,norm,float(np.linalg.norm(new-t))]);t=new
    file=ident+'.npy';trace=ident+'_trace.npz';np.save(OUT/file,t);np.savez_compressed(OUT/trace,history=np.array(hist),clipped_gradients=np.array(grad))
    row=dict(id=ident,stage=stage,seed=s,arm=arm,loss=loss,lr=lr,role=role,data_file=datafile,initial_seed=init_seed,initial_file=initial,file=file,trace=trace,sha256=sha(OUT/file),steps=200,seconds=time.perf_counter()-tick,reused=False,validation=metrics(d['vx'],d['vy'],t,arm) if 'vx' in d else None)
    ROWS.append(row);dump(OUT/('FINAL_LOG.json' if stage=='B' else 'PREPARE_LOG.json'),ROWS)
    print(ident,'validation',row['validation']['accuracy'] if row['validation'] else 'not selected on validation',flush=True);return row

def prepare():
    global START
    freeze_check();OUT.mkdir(exist_ok=False);START=time.perf_counter()
    for s in range(10):
        f,d=dataset(f'A_train_s{s}',2026116000+10*s)
        for arm in ('DEND','POINT'):train('A',s,arm,'RATE',.01,'fixed',f,d,2026116200+s)
    for s in range(5):
        f,d=dataset(f'search_s{s}',2026117000+10*s,2026117001+10*s)
        for arm in ('DEND','POINT'):
            for loss,lr in GRID:train('search',s,arm,loss,lr,'candidate',f,d,2026117100+s)
    selection=choose(ROWS);dump(OUT/'SELECTION.json',selection)
    dump(OUT/'PREPARE_LOCK.json',{'rows':ROWS,'selection':selection,'seconds':time.perf_counter()-START,'files':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},'freeze_sha256':sha(R/'FREEZE.json'),'versions':{'python':platform.python_version(),'numpy':np.__version__},'test_generated':False})
    print('SELECTION',json.dumps(selection['selected']),flush=True)

def final_train():
    global START,SPENT,ROWS
    freeze_check();lock=json.loads((OUT/'PREPARE_LOCK.json').read_text())
    for f,h in lock['files'].items():assert sha(OUT/f)==h,f
    assert not (OUT/'FINAL_STARTED.json').exists();dump(OUT/'FINAL_STARTED.json',{'prepare_lock_sha256':sha(OUT/'PREPARE_LOCK.json')});START=time.perf_counter();SPENT=lock['seconds'];ROWS=[]
    for s in range(10):
        f,d=dataset(f'B_train_s{s}',2026118000+10*s)
        for arm in ('DEND','POINT'):
            fixed=train('B',s,arm,'RATE',.01,'fixed',f,d,2026118200+s);cfg=lock['selection']['selected'][arm]
            if (cfg['loss'],cfg['lr'])==('RATE',.01):
                alias=dict(fixed,id=f'B_s{s}_{arm}_selected_reuse',role='selected',reused=True,reused_from=fixed['id'],seconds=0);ROWS.append(alias);dump(OUT/'FINAL_LOG.json',ROWS)
            else:train('B',s,arm,cfg['loss'],cfg['lr'],'selected',f,d,2026118200+s)
    dump(OUT/'FINAL_LOCK.json',{'rows':ROWS,'seconds':time.perf_counter()-START,'prepare_lock_sha256':sha(OUT/'PREPARE_LOCK.json'),'files':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and (p.name.startswith('B_') or p.name=='FINAL_LOG.json')},'test_generated':False})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','final']);a=p.parse_args();prepare() if a.phase=='prepare' else final_train()
