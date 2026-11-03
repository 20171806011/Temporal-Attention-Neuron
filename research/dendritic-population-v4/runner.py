"""Development and locked-config replication training; never creates test data."""
import argparse,time,json,platform
import numpy as np
import model
from data import make,dev_seed
from common import R,OUT,dump,sha,mask,stats,parity,choice,check_freeze
LOG=[];START=0;CAP=600

def dataset(name,level,train_seed,val_seed=None):
    p=OUT/(name+'.npz');assert not p.exists()
    x,y,pair=make(level,128,train_seed);d=dict(x=x,y=y,pair=pair,level=np.array(level),train_seed=np.array(train_seed))
    if val_seed is not None:
        vx,vy,vpair=make(level,128,val_seed);d.update(vx=vx,vy=vy,vpair=vpair,val_seed=np.array(val_seed))
    np.savez_compressed(p,**d);return p.name,d

def train(stage,level,s,arm,loss,lr,method,datafile,d,init_seed,condition='intact'):
    ident=f'{stage}_L{level}_s{s}_{arm}_{method}_{loss}_{lr}_{condition}';theta=model.initialize(init_seed)
    active=mask() if condition=='retrained' else np.ones(176);theta*=active
    initialfile=ident+'_initial.npy';np.save(OUT/initialfile,theta);m=np.zeros(176);v=np.zeros(176);history=[];grads=[];directions=[];snapshots={};tick=time.perf_counter()
    steps=100 if method=='SPSA' else 200;rng=np.random.default_rng(2026111200+s)
    for k in range(steps):
        if time.perf_counter()-START>CAP:
            np.save(OUT/(ident+'_PARTIAL.npy'),theta)
            np.savez_compressed(OUT/(ident+'_PARTIAL_trace.npz'),history=np.array(history),clipped_gradients=np.array(grads),directions=np.array(directions))
            dump(OUT/'BUDGET_FAILURE.json',{'id':ident,'completed_updates':k,'status':'stopped, partial candidate preserved, no automatic retry'})
            raise RuntimeError('Training wall budget exceeded; partial outputs retained')
        if method=='ADAM':
            lossval,raw,c=model.loss_grad(d['x'],d['y'],theta,arm,loss);raw*=active;norm=float(np.linalg.norm(raw));g=raw/max(1,norm)
            m=(.9*m+.1*g)*active;v=(.999*v+.001*g*g)*active
            new=np.clip(theta-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)*active
            history.append([lossval,norm,float(np.linalg.norm(new-theta))])
        else:
            direction=rng.choice([-1.,1.],176);directions.append(direction);c=.15/(k+1)**.101;a=lr/(k+1)**.602
            plus=model.objective(d['x'],d['y'],theta+c*direction,arm,loss);minus=model.objective(d['x'],d['y'],theta-c*direction,arm,loss)
            g=np.clip((plus-minus)/(2*c)*direction,-5,5);new=np.clip(theta-a*g,-2,2)
            history.append([plus,minus,float(np.linalg.norm(new-theta))])
        grads.append(g.copy());theta=new
        if stage=='A' and k==99:snapshots[100]=theta.copy()
    snapshots[steps]=theta.copy();trace=ident+'_trace.npz'
    np.savez_compressed(OUT/trace,history=np.array(history),clipped_gradients=np.array(grads),directions=np.array(directions))
    duration=time.perf_counter()-tick
    for endpoint,t in snapshots.items():
        file=ident+f'_n{endpoint}.npy';np.save(OUT/file,t)
        row=dict(id=ident+f'_n{endpoint}',stage=stage,level=level,seed=s,arm=arm,method=method,loss=loss,lr=lr,condition=condition,steps=endpoint,trajectory_steps=steps,initial_seed=init_seed,initial_file=initialfile,data_file=datafile,file=file,trace=trace,sha256=sha(OUT/file),seconds_trajectory=duration,validation=stats(d['vx'],d['vy'],t,arm) if 'vx' in d else None)
        LOG.append(row)
    dump(OUT/('replication_log.json' if stage=='R' else 'development_log.json'),LOG)
    print(ident,'acc',LOG[-1]['validation']['accuracy'] if LOG[-1]['validation'] else 'locked replication',flush=True)

def develop():
    global START,LOG
    check_freeze();OUT.mkdir(exist_ok=False);START=time.perf_counter();parity('A')
    for s in range(5):
        f,d=dataset(f'A_data_s{s}',3,2026111000+10*s,2026111001+10*s)
        for arm in ('DEND','POINT'):
            for lr in (.01,.03):
                for loss in ('CE','RATE'):train('A',3,s,arm,loss,lr,'ADAM',f,d,2026111100+s)
                train('A',3,s,arm,'CE',lr,'SPSA',f,d,2026111100+s)
    selections=[choice(LOG,b) for b in (100,200)];cfg=selections[1]['selected'];dump(OUT/'A_SELECTION.json',selections)
    parity('B');levels=[]
    for level in range(4):
        for s in range(5):
            f,d=dataset(f'B_L{level}_data_s{s}',level,dev_seed(level,s),dev_seed(level,s)+1)
            for arm in ('DEND','POINT'):train('B',level,s,arm,cfg['loss'],cfg['lr'],'ADAM',f,d,2026112500+s)
        rows=[r for r in LOG if r['stage']=='B' and r['level']==level]
        gates={a:all(r['validation']['accuracy']>=.9 and r['validation']['coverage']>=.9 for r in rows if r['arm']==a) for a in ('DEND','POINT')}
        ceiling=all(r['validation']['accuracy']>=.99 for r in rows);levels.append(dict(level=level,gates=gates,ceiling=ceiling,test_eligible=any(gates.values()) and not ceiling))
        dump(OUT/'B_LEVELS.json',levels)
        if not any(gates.values()):break
    memory=[z['level'] for z in levels if z['level'] in (1,2) and any(z['gates'].values())];clevel=max(memory) if memory else None
    if clevel is not None:
        parity('C')
        for s in range(5):
            f=f'B_L{clevel}_data_s{s}.npz';d=dict(np.load(OUT/f,allow_pickle=False))
            for arm in ('DEND','POINT'):
                train('C',clevel,s,arm,cfg['loss'],cfg['lr'],'ADAM',f,d,2026112500+s,'retrained')
                intact=next(r for r in LOG if r['stage']=='B' and r['level']==clevel and r['seed']==s and r['arm']==arm)
                t=np.load(OUT/intact['file'],allow_pickle=False)*mask();file=f'C_L{clevel}_s{s}_{arm}_cut.npy';np.save(OUT/file,t)
                LOG.append(dict(id=file[:-4],stage='C',level=clevel,seed=s,arm=arm,method='CUT',loss=cfg['loss'],lr=cfg['lr'],condition='cut',steps=0,trajectory_steps=0,initial_seed=2026112500+s,initial_file=intact['initial_file'],data_file=f,file=file,trace=None,sha256=sha(OUT/file),seconds_trajectory=0,source_file=intact['file'],validation=stats(d['vx'],d['vy'],t,arm)))
        dump(OUT/'development_log.json',LOG)
    lock=dict(configuration=cfg,selections=selections,levels=levels,c_level=clevel,failed_comparator_exception=True,development_seconds=time.perf_counter()-START,versions={'python':platform.python_version(),'numpy':np.__version__},freeze_sha256=sha(R/'FREEZE.json'),files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},test_generated=False)
    dump(OUT/'DEVELOPMENT_LOCK.json',lock);print('DEVELOPMENT LOCK',json.dumps({'configuration':cfg,'levels':levels,'c_level':clevel}),flush=True)

def replicate():
    global START,LOG
    check_freeze();lock=json.loads((OUT/'DEVELOPMENT_LOCK.json').read_text())
    for f,h in lock['files'].items():assert sha(OUT/f)==h
    assert not (OUT/'REPLICATION_STARTED.json').exists();dump(OUT/'REPLICATION_STARTED.json',{'development_lock_sha256':sha(OUT/'DEVELOPMENT_LOCK.json')});LOG=[];START=time.perf_counter();parity('REPLICATION')
    cfg=lock['configuration'];eligible=[z['level'] for z in lock['levels'] if z['test_eligible']]
    for level in eligible:
        for s in range(5):
            f,d=dataset(f'R_L{level}_data_s{s}',level,2026114000+100*level+10*s)
            for arm in ('DEND','POINT'):
                train('R',level,s,arm,cfg['loss'],cfg['lr'],'ADAM',f,d,2026113000+100*level+s)
                if level==lock['c_level']:train('R',level,s,arm,cfg['loss'],cfg['lr'],'ADAM',f,d,2026113000+100*level+s,'retrained')
    dump(OUT/'REPLICATION_LOCK.json',dict(eligible_levels=eligible,rows=LOG,seconds=time.perf_counter()-START,development_lock_sha256=sha(OUT/'DEVELOPMENT_LOCK.json'),files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and (p.name.startswith('R_') or p.name=='replication_log.json')},test_generated=False))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['develop','replicate']);args=a.parse_args()
    develop() if args.phase=='develop' else replicate()
