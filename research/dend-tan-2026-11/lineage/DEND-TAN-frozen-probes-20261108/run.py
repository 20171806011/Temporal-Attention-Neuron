import sys,time,platform,os,numpy as np
import task_data,capture,probe
from core import *
def metadata(row):return {k:row[k] for k in ('id','seed','arm','loss')}
def fits(B,models,theta):
    B.stage='fit';B.mark('running');out=R/'fits';out.mkdir();(R/'data').mkdir();(R/'snapshots').mkdir();rows=[]
    for s in range(3):
        base=task_data.make_base(256,4001010000+100*s);base['permuted_y']=probe.permutations(base,4001030000+100*s);np.savez_compressed(R/'data'/f'fit_s{s}.npz',**base);x,_=task_data.assemble(base);labels=np.column_stack((base['y'],base['permuted_y']))
        for row in [r for r in models if r['seed']==s]:
            B.guard();z=capture.capture(x,theta[row['index']],row['arm']);np.savez_compressed(R/'snapshots'/f'{row["id"]}_fit_nominal.npz',**z);B.captured(len(x))
            for group,t in capture.configurations(row['arm']):
                B.guard();xx=capture.features(z,group,t);fit=probe.fit(xx,labels);ident=f'{row["id"]}_{group}_t{t}';np.savez_compressed(out/(ident+'.npz'),**fit);m=probe.metrics(fit['fit_scores'][:,0],base['y']);rows.append(dict(**metadata(row),probe_id=ident,group=group,time=t,dimension=xx.shape[1],active_dimension=int(fit['active'].sum()),fit_ba=m['balanced_accuracy'],fit_mse=m['mse']));B.fits+=1
            dump(out/'ROWS.json',rows);print('probes fitted',row['id'],flush=True)
    assert len(rows)==1332
    paths=list((R/'data').glob('fit_*.npz'))+list((R/'snapshots').glob('*_fit_nominal.npz'))+list(out.glob('*'))
    dump(R/'FIT_LOCK.json',dict(utc=utc(),test_generated=False,files={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in sorted(paths) if p.is_file()}))
def evaluate(B,models,theta):
    for name,h in read(R/'FIT_LOCK.json')['files'].items():assert sha(R/name)==h
    B.stage='evaluate';B.mark('running');out=R/'evaluate';out.mkdir();fr=read(R/'fits/ROWS.json');rows=[];native_rows=[]
    for s in range(3):
        base=task_data.make_base(512,4001020000+100*s);base['permuted_y']=probe.permutations(base,4001040000+100*s);np.savez_compressed(R/'data'/f'test_s{s}.npz',**base)
        for row in [r for r in models if r['seed']==s]:
            captured={}
            for cond in CONDITIONS:
                B.guard();x,_=task_data.assemble(base,cond);z=capture.capture(x,theta[row['index']],row['arm']);np.savez_compressed(R/'snapshots'/f'{row["id"]}_test_{cond}.npz',**z);captured[cond]=z;B.captured(len(x));native_rows.append(dict(**metadata(row),condition=cond,**native(z['counts'],base['y']),mean_resources=z['resources'].mean(0).tolist()))
            for fitrow in [f for f in fr if f['id']==row['id']]:
                B.guard();fit=dict(np.load(R/'fits'/(fitrow['probe_id']+'.npz')));score=np.stack([probe.predict(fit,capture.features(captured[c],fitrow['group'],fitrow['time'])) for c in CONDITIONS]);np.savez_compressed(out/(fitrow['probe_id']+'.npz'),test_scores=score)
                for ci,cond in enumerate(CONDITIONS):
                    rec=dict(**fitrow,condition=cond,**probe.metrics(score[ci,:,0],base['y']))
                    if cond=='nominal':
                        fm=[probe.metrics(score[ci,:,j+1],base['y']) for j in range(20)];tm=[probe.metrics(score[ci,:,0],base['permuted_y'][:,j]) for j in range(20)]
                        for key in ('balanced_accuracy','mse','score_rms'):rec['fit_shuffle_'+('ba' if key=='balanced_accuracy' else key)]=[v[key] for v in fm]
                        for key in ('balanced_accuracy','mse'):rec['test_shuffle_'+('ba' if key=='balanced_accuracy' else key)]=[v[key] for v in tm]
                    rows.append(rec)
            dump(out/'ROWS.json',rows);dump(out/'NATIVE_ROWS.json',native_rows);print('probes evaluated',row['id'],flush=True)
    assert len(rows)==3996 and len(native_rows)==108
def freeze():
    assert not (R/'FREEZE.json').exists()
    for name in ('PRECHECK','TIMING'):assert read(R/(name+'.json'))['status']=='PASS'
    assert (R/'audit.py').exists() and (R/'PROTOCOL.md').exists()
    files={p.name:sha(p) for p in sorted(R.iterdir()) if p.is_file() and p.suffix in ('.py','.md','.json','.npz') and p.name not in ('summarize.py','report.py')};dump(R/'FREEZE.json',dict(utc=utc(),scientific_data_generated=False,neural_weights_fixed=True,files=files));print('FROZEN',flush=True)
def science():
    B=Budget();dump(R/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,machine=platform.node(),threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    try:
        models=read(R/'MODELS.json')['rows'];theta=np.load(R/'models.npz')['theta'];fits(B,models,theta);evaluate(B,models,theta);B.finish()
    except Exception as e:B.mark('stopped',reason=str(e));dump(R/'SCIENCE_FAILURE.json',dict(stage=B.stage,reason=str(e),exception=type(e).__name__,probe_configs_completed=B.fits,wall_seconds=time.perf_counter()-B.tic));raise
if __name__=='__main__':{'freeze':freeze,'science':science}[sys.argv[1]]()
