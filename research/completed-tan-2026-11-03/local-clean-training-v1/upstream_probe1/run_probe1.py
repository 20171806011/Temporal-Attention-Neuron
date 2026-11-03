#!/usr/bin/env python3
"""Fixed retrospective task-grounded C2 ablation. No new labels or hyperparameters.
Fits only the archived 33-parameter linear readout.
"""
import argparse,copy,hashlib,json,os,platform,sys,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import scipy
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'source'))
import task_excerpt as td
from tan import make_variant
METHODS=['source_softmax','C2','source_uniform']
FROZEN={20260904:{'legacy_task_softmax':(.328,.427),'source_uniform':(.333,.371)},20260905:{'legacy_task_softmax':(.329,.398),'source_uniform':(.322,.355)},20260906:{'legacy_task_softmax':(.329,.342),'source_uniform':(.470,.285)}}
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
class Capture:
    def __init__(self,r):self.r=r;self.perm=None
    def __getattr__(self,k):return getattr(self.r,k)
    def permutation(self,n):self.perm=self.r.permutation(n);return self.perm

def dataset(rng,n):
    # Extra recording does not change any random-number call or data.
    before=copy.deepcopy(rng.bit_generator.state);cap=Capture(rng)
    X,y,cT,cQ,present=td.make_trials(cap,n)
    ref=np.random.default_rng();ref.bit_generator.state=before
    direct=td.make_trials(ref,n)
    assert all(np.array_equal(a,b) for a,b in zip((X,y,cT,cQ,present),direct))
    assert ref.bit_generator.state==rng.bit_generator.state
    nc=int(round(n/3));npr=n-nc;nd=npr//2;ncl=npr-nd
    ids=np.r_[np.arange(nd),np.arange(ncl),ncl+np.arange(nc)][cap.perm]
    for g in range(nd):
        ix=np.flatnonzero(ids==g);assert len(ix)==2
        assert cT[ix[0]]==cT[ix[1]] and cQ[ix[0]]==cQ[ix[1]]
    assert np.array_equal(y,np.where(present,cT,2))
    return {'X':X,'y':y,'cT':cT,'cQ':cQ,'group':ids,'dist':(X[:,5:8].max(axis=1)>0)}

def simulate(X,method):
    B,T=X.shape;H=np.zeros((B,5));hp=np.zeros(B)
    out={k:np.zeros((B,T)) for k in ('u','h','A','C','S')}
    out['spike']=np.zeros((B,T),dtype=np.int8)
    for t in range(T):
        H=np.roll(H,-1,axis=1);H[:,-1]=X[:,t]
        mu=H.mean(axis=1);S=np.maximum(0.,X[:,t]-mu);q=2.*S
        if method in ('source_softmax','legacy_task_softmax'):
            scores=q[:,None]*H
            if method=='source_softmax':scores=scores-scores.max(axis=1,keepdims=True)
            ex=np.exp(scores);att=ex/(ex.sum(axis=1,keepdims=True)+1e-9)
            C=(att*H).sum(axis=1)
        elif method=='C2':C=np.clip(mu+q*np.mean((H-mu[:,None])**2,axis=1),H.min(axis=1),H.max(axis=1))
        elif method=='source_uniform':C=mu
        else:raise ValueError(method)
        A=np.tanh(S)*C;u=.5*hp+A;sp=u>.5;hp=np.where(sp,0.,u)
        for k,z in [('u',u),('h',hp),('A',A),('C',C),('S',S),('spike',sp)]:out[k][:,t]=z
    assert all(np.isfinite(x).all() for x in out.values())
    return out

def fidelity(X,outs):
    maximum=0.;n=0
    for method,variant in [('source_softmax','TAN-full'),('source_uniform','TAN-no-attention')]:
        for i in range(min(64,len(X))):
            m=make_variant(variant)
            for t,x in enumerate(X[i]):
                s=m.forward(float(x));n+=1
                assert s==outs[method]['spike'][i,t]
                maximum=max(maximum,abs(float(m.h)-outs[method]['h'][i,t]))
    assert maximum<=1e-12
    return {'checked_steps':n,'max_post_h_abs_difference':maximum,'archived_classes_imported':True}

def fit(Ftr,ytr,Fte):
    mu,sd,w=td.softmax_fit(Ftr,ytr)
    p=td.predict(mu,sd,w,Fte)
    assert np.isfinite(p).all()
    return p,{'mean':mu.tolist(),'std':sd.tolist(),'w':w.tolist(),'optimizer':td.FIT_RECORDS[-1]}

def scores(pred,y,dist):
    d={}
    for name,mask in [('all',np.ones(len(y),bool)),('clean',~dist),('distractor',dist)]:
        cm=td.confusion(pred[mask],y[mask]);bal=td.bal3(pred,y,mask)
        missing=[k for k in range(3) if cm[k].sum()>=20 and cm[:,k].sum()==0]
        flag=('INVALID_FOR_HEALTHY_THREE_CLASS_INTERPRETATION' if bal>.375 else 'AT_CHANCE_COLLAPSE_FLAG') if missing else 'NO_MISSING_PREDICTED_CLASS'
        d[name]={'n':int(mask.sum()),'balanced3':bal,'macro_f1':td.macro_f1(pred,y,mask),'accuracy':float(np.mean(pred[mask]==y[mask])),'class_counts':cm.sum(axis=1).tolist(),'recalls':(cm.diagonal()/cm.sum(axis=1)).tolist(),'confusion':cm.tolist(),'missing_predicted_classes':missing,'coverage_flag':flag}
    d['degradation']=d['clean']['balanced3']-d['distractor']['balanced3']
    return d

def boot_contrasts(records,spec):
    # Paired code-index clusters, stratified by fixed seed. No refitting.
    rng=np.random.default_rng(spec['bootstrap']['seed']);B=spec['bootstrap']['replicates']
    metrics={m:{s:np.zeros(B) for s in ['clean','distractor','degradation']} for m in METHODS}
    checks=[]
    for r in records:
        y=np.array(r['y']);dist=np.array(r['dist'],bool);g=np.array(r['group']);ng=int(g.max()+1)
        draws=rng.integers(0,ng,size=(B,ng))
        for m in METHODS:
            pred=np.array(r['pred'][m]);cell={}
            for name,mask in [('clean',~dist),('distractor',dist)]:
                den=np.zeros((ng,3));num=np.zeros_like(den)
                for k in range(3):
                    good=mask&(y==k)
                    den[:,k]=np.bincount(g[good],minlength=ng)
                    num[:,k]=np.bincount(g[good&(pred==k)],minlength=ng)
                ds=den[draws].sum(axis=1);ns=num[draws].sum(axis=1)
                assert (ds>0).all()
                cell[name]=(ns/ds).mean(axis=1)
                metrics[m][name]+=cell[name]/len(records)
            metrics[m]['degradation']+=(cell['clean']-cell['distractor'])/len(records)
        checks.append({'seed':r['seed'],'groups':ng,'singleton_groups':int(np.count_nonzero(np.bincount(g)==1)),'paired_groups':int(np.count_nonzero(np.bincount(g)==2))})
    out=[]
    for other in ['C2','source_uniform']:
        for c in ['clean','distractor','degradation']:
            z=metrics['source_softmax'][c]-metrics[other][c]
            if c=='degradation':val=float(np.mean([r['scores']['source_softmax'][c]-r['scores'][other][c] for r in records]))
            else:val=float(np.mean([r['scores']['source_softmax'][c]['balanced3']-r['scores'][other][c]['balanced3'] for r in records]))
            out.append({'contrast':'source_softmax-minus-'+other,'endpoint':c,'estimate':val,'conditional_cluster_percentile95':np.percentile(z,[2.5,97.5]).tolist()})
    return {'grouping':checks,'replicates':B,'seed':spec['bootstrap']['seed'],'conditional_not_new_training_uncertainty':True,'contrasts':out}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outdir',default=str(R/'run'));a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    raw=(R/'frozen_spec.json').read_bytes();s=json.loads(raw)
    assert sha(raw)==(R/'FREEZE_SHA256.txt').read_text().split()[0]
    src=(R/'source'/'tan.py').read_bytes()
    assert hashlib.sha1(b'blob '+str(len(src)).encode()+b'\0'+src).hexdigest()==s['source_blobs']['snapshot/code/model/tan.py']
    t0=time.perf_counter();res={'spec_sha256':sha(raw),'scope':s['scope'],'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'runs':[],'failures':[]};boots=[];timings=[]
    for seed in s['task']['seeds']:
        t1=time.perf_counter();rng=np.random.default_rng(seed)
        tr=dataset(rng,s['task']['train']);te=dataset(rng,s['task']['test'])
        overlap=len(set(map(tuple,tr['X']))&set(map(tuple,te['X'])));assert overlap==0
        dpath=out/f'inputs_{seed}.npz';np.savez_compressed(dpath,**{'train_'+k:v for k,v in tr.items()},**{'test_'+k:v for k,v in te.items()})
        rr={'seed':seed,'audit':{'direct_input_overlap':overlap,'hidden_labels_correct':True,'paired_label_groups_rng_unchanged':True,'input_file_sha256':sha(dpath.read_bytes())},'models':{}}
        sh_rng=np.random.default_rng(s['audit']['shuffle_seed_base']+seed)
        perms=[sh_rng.permutation(len(tr['y'])) for _ in range(s['audit']['shuffles'])]
        st={m:simulate(tr['X'],m) for m in METHODS+['legacy_task_softmax']}
        se={m:simulate(te['X'],m) for m in METHODS+['legacy_task_softmax']}
        rr['audit']['source_class_fidelity']=fidelity(te['X'],se)
        rr['audit']['native_vs_legacy']={'train_u_max':float(np.max(np.abs(st['source_softmax']['u']-st['legacy_task_softmax']['u']))),'test_u_max':float(np.max(np.abs(se['source_softmax']['u']-se['legacy_task_softmax']['u']))),'test_spike_disagreements':int(np.count_nonzero(se['source_softmax']['spike']!=se['legacy_task_softmax']['spike']))}
        preds={};arr={}
        for m in METHODS+['legacy_task_softmax']:
            Ftr=td.resp_features(st[m]);Fte=td.resp_features(se[m]);p,fitrec=fit(Ftr,tr['y'],Fte);pr=p.argmax(axis=1)
            sc=scores(pr,te['y'],te['dist']);row={'scores':sc,'readout':fitrec}
            if not fitrec['optimizer']['success']:res['failures'].append({'seed':seed,'method':m,'failure':'optimizer_nonconvergence','detail':fitrec['optimizer']})
            if m in METHODS:
                null=[];null_fit=[]
                for perm in perms:
                    ps,rf=fit(Ftr,tr['y'][perm],Fte);null.append(td.bal3(ps.argmax(1),te['y']));null_fit.append(rf['optimizer'])
                row['shuffle_diagnostic']={'bal3':null,'mean':float(np.mean(null)),'std':float(np.std(null)),'fit_diagnostics':null_fit,'not_original_amended_100_permutation_gate':True}
                preds[m]=pr.tolist()
                for cell in ('clean','distractor'):
                    if sc[cell]['missing_predicted_classes']:res['failures'].append({'seed':seed,'method':m,'cell':cell,'failure':sc[cell]['coverage_flag'],'missing':sc[cell]['missing_predicted_classes']})
            if m in FROZEN[seed]:
                rounded=FROZEN[seed][m];diff=max(abs(sc['clean']['balanced3']-rounded[0]),abs(sc['distractor']['balanced3']-rounded[1]));row['archived_rounded_check']={'reported':list(rounded),'max_absolute_difference':diff,'within_1e3':diff<=.001}
                if diff>.001:res['failures'].append({'seed':seed,'method':m,'failure':'archived_rounded_metric_mismatch','difference':diff})
            rr['models'][m]=row
            arr.update({m+'_'+k:se[m][k] for k in se[m]});arr[m+'_features']=Fte;arr[m+'_pred']=pr;arr[m+'_p']=p
            print(seed,m,'clean',sc['clean']['balanced3'],'dist',sc['distractor']['balanced3'],'fit',fitrec['optimizer']['success'],flush=True)
        for name,ztr,zte in [('oracle_target',tr['X'][:,4:5],te['X'][:,4:5]),('oracle_window',tr['X'][:,4:9],te['X'][:,4:9])]:
            p,fr=fit(ztr,tr['y'],zte);rr[name]={'scores':scores(p.argmax(1),te['y'],te['dist']),'fit':fr,'evaluator_only':True};arr[name+'_pred']=p.argmax(1)
        rr['audit']['train_distractor_larger_than_target']=float(np.mean(tr['X'][tr['dist']&(tr['cT']>=0),5:8].max(axis=1)>tr['X'][tr['dist']&(tr['cT']>=0),4]))
        for x in ('train','test'):
            d=tr if x=='train' else te
            rr['audit'][x+'_class_counts']={k:np.bincount(d['y'][mask],minlength=3).tolist() for k,mask in [('clean',~d['dist']),('distractor',d['dist'])]}
        opath=out/f'traces_{seed}.npz';np.savez_compressed(opath,**arr);rr['trace_sha256']=sha(opath.read_bytes())
        res['runs'].append(rr);boots.append({'seed':seed,'y':te['y'].tolist(),'dist':te['dist'].tolist(),'group':te['group'].tolist(),'pred':preds,'scores':{m:rr['models'][m]['scores'] for m in METHODS}})
        timings.append({'seed':seed,'seconds':time.perf_counter()-t1})
    res['paired_bootstrap']=boot_contrasts(boots,s)
    res['pooled']={m:{'equal_seed_clean_bal3':float(np.mean([r['models'][m]['scores']['clean']['balanced3'] for r in res['runs']])),'equal_seed_dist_bal3':float(np.mean([r['models'][m]['scores']['distractor']['balanced3'] for r in res['runs']])),'equal_seed_degradation':float(np.mean([r['models'][m]['scores']['degradation'] for r in res['runs']]))} for m in METHODS}
    res['counts']={'train_histories_per_seed':4000,'test_histories_total':6000,'test_distractor_trials':sum(sum(r['audit']['test_class_counts']['distractor']) for r in res['runs']),'test_clean_trials':sum(sum(r['audit']['test_class_counts']['clean']) for r in res['runs']),'test_code_reuse_groups':sum(r['groups'] for r in res['paired_bootstrap']['grouping']),'trained_linear_readouts_total':len(td.FIT_RECORDS)}
    res['not_claimed']=['new independent replication of v11','original100-permutation amendment replay','prospectively untouched test set','spike-only task accuracy','source-native and historical simulator numerical identity','universal attention benefit','general key/value binding']
    dump(out/'results.json',res);dump(out/'failures.json',res['failures']);dump(out/'timing.json',{'total_seconds':time.perf_counter()-t0,'rows':timings,'not_hardware_energy':True})
    print(json.dumps(res['pooled'],indent=2));print('results',sha((out/'results.json').read_bytes()),flush=True)
if __name__=='__main__':main()
