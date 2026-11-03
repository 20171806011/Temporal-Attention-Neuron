import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,hashlib,time,platform
from pathlib import Path
import numpy as np, scipy
R=Path(__file__).resolve().parent
P=R/'upstream_probe1'
sys.path.insert(0,str(P));import run_probe1 as old
T=old.td
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'FROZEN_PROTOCOL.md')==(R/'FREEZE.sha256').read_text().split()[0]
for line in (P/'MANIFEST.sha256').read_text().splitlines():
 h,p=line.split(maxsplit=1);assert sha(P/p.strip())==h,p
out={'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'protocol_sha256':sha(R/'FROZEN_PROTOCOL.md'),'source_hashes':{p:sha(P/p) for p in ['run_probe1.py','source/task_excerpt.py','source/tan.py']},'runs':[]}
start=time.perf_counter();records=[]
for seed in [20260904,20260905,20260906]:
 p=P/'run'/f'inputs_{seed}.npz';d=np.load(p);y=d['train_y'];dist=d['train_dist'];yt=d['test_y'];mask=~d['test_dist'];groups=d['test_group']
 Ftr=T.resp_features(old.simulate(d['train_X'],'source_softmax'));Fte=T.resp_features(old.simulate(d['test_X'],'source_softmax'))
 prior=np.load(P/'run'/f'traces_{seed}.npz')['source_softmax_features'];err=float(np.max(abs(prior-Fte)));assert err<=1e-12
 perm=np.random.RandomState(2026110351+seed).permutation(len(y));clean=[];mix=[]
 for k in range(3):
  c=perm[(y[perm]==k)&~dist[perm]];m=perm[(y[perm]==k)&dist[perm]];assert len(c)>=500 and len(m)>=250
  clean.extend(c[:500]);mix.extend(c[:250]);mix.extend(m[:250])
 row={'seed':seed,'input_sha256':sha(p),'feature_max_difference':err,'arms':{}};preds={};save={'test_y':yt,'test_clean':mask,'test_group':groups}
 for arm,inds in [('clean_only',clean),('mixed',mix)]:
  ids=np.sort(inds);assert len(ids)==1500 and len(set(ids))==1500;assert np.array_equal(np.bincount(y[ids],minlength=3),[500]*3)
  mu,sd,w=T.softmax_fit(Ftr[ids],y[ids]);prob=T.predict(mu,sd,w,Fte);pred=prob.argmax(1);assert np.isfinite(prob).all()
  cm=T.confusion(pred[mask],yt[mask]);rec=cm.diagonal()/cm.sum(axis=1);score=float(rec.mean());assert abs(score-T.bal3(pred,yt,mask))<1e-15
  row['arms'][arm]={'balanced3':score,'confusion':cm.tolist(),'recalls':rec.tolist(),'optimizer':T.FIT_RECORDS[-1],'class_counts':np.bincount(y[ids],minlength=3).tolist(),'condition_counts':np.bincount(dist[ids].astype(int),minlength=2).tolist(),'selected_indices':ids.tolist(),'mean':mu.tolist(),'std':sd.tolist(),'weights':w.tolist()}
  preds[arm]=pred;save[arm+'_prediction']=pred;save[arm+'_probability']=prob;save[arm+'_indices']=ids
  print(seed,arm,score,T.FIT_RECORDS[-1],flush=True)
 row['difference']=row['arms']['clean_only']['balanced3']-row['arms']['mixed']['balanced3'];out['runs'].append(row);records.append((yt,mask,groups,preds));np.savez_compressed(R/f'predictions_{seed}.npz',**save)
assert len(T.FIT_RECORDS)==6
rng=np.random.default_rng(2026110409);boot=np.zeros(2000)
for y,mask,g,preds in records:
 ng=int(g.max()+1);draw=rng.integers(0,ng,size=(2000,ng));vals={}
 for arm,pred in preds.items():
  den=np.zeros((ng,3));num=np.zeros((ng,3))
  for k in range(3):
   z=mask&(y==k);den[:,k]=np.bincount(g[z],minlength=ng);num[:,k]=np.bincount(g[z&(pred==k)],minlength=ng)
  ds=den[draw].sum(1);assert (ds>0).all();vals[arm]=(num[draw].sum(1)/ds).mean(1)
 boot+=(vals['clean_only']-vals['mixed'])/3
out['primary_difference']=float(np.mean([x['difference'] for x in out['runs']]));out['conditional_percentile95']=np.percentile(boot,[2.5,97.5]).tolist();out['means']={a:float(np.mean([x['arms'][a]['balanced3'] for x in out['runs']])) for a in ['clean_only','mixed']};out['seconds']=time.perf_counter()-start
(R/'results.json').write_text(json.dumps(out,indent=2)+'\n');np.save(R/'conditional_bootstrap.npy',boot)
print('SUMMARY',out['means'],out['primary_difference'],out['conditional_percentile95'],out['seconds'])
