#!/usr/bin/env python3
"""Separate scalar/accounting checks. Does not import run_whq.py.
Same-agent verification, NOT an independent third-party reproduction.
"""
from pathlib import Path
import argparse,hashlib,json,math,sys
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'source'))
from reference_m1 import step as source_step

def g(q,s):
    a=[-.5*((q-k)/s['key_bandwidth'])**2 for k in s['keys']]
    mm=max(a);v=[math.exp(x-mm) for x in a];tot=sum(v)
    return [x/tot for x in v]

def quant(v,cond):
    if cond['fractional_bits'] is None:return list(v)
    h=2.**(-cond['fractional_bits'])
    return [float(np.rint(x/h)*h) for x in v]

def scalar_one(i,data,model,delay,cond,s,proc,meas):
    z=[0.,-1.,-1.,0.] if model=='original_M1' else [0.,0.,0.,0.]
    t=0;ys=[]
    def update(key,val,event,q,cue):
        nonlocal t,z
        u,m0,m1,r=z;dt=.1
        if model=='original_M1':
            un,An,rn,sp,uc=source_step(u,2.5*(m0+1),r,key+val if event else 0.,max(0.,abs(key)-1.4) if event else 0.,q if cue else 0.)
            mn=[2*An/5-1,m1];y=0. if uc is None else uc
        else:
            m=[m0,m1];gg=g(key,s);mn=[]
            for j in range(2):
                wg=gg[j]*float(event);err=val-m[j]
                psi=.25*math.tanh(err/.25) if model=='bounded_WHQ' else max(-.25,min(.25,err))
                mn.append(m[j]+dt*(wg*psi-(1-wg)*m[j]/2000.))
            drive=.5+.6*sum(a*b for a,b in zip(g(q,s),m)) if cue else 0.
            # Independent scalar membrane recurrence.
            if r>0.:un=0.;rn=max(0.,r-dt);y=0.
            else:
                y=u+(-u+drive)/10.*dt
                un=0. if y>=1. else y;rn=2. if y>=1. else 0.
        e=proc[t,i];t+=1
        z=quant([un+e[0],max(-1.,min(1.,mn[0]+e[1])),max(-1.,min(1.,mn[1]+e[2])),rn],cond)
        return y
    for slot in range(5):
        active=slot in (0,4);p=0 if slot==0 else 1
        j=int(data['order'][i][p]);k=s['keys'][j] if active else s['keys'][0];v=data['v'][i][j] if active else 0.
        for st in range(200):update(k,v,active and st<100,0.,False)
    for _ in range(int(delay*10)):update(s['keys'][0],0.,False,0.,False)
    for role in range(2):
        j=data['qo'][i][role];q=s['keys'][j]+data['jitter'][i][role];ss=[]
        for st in range(1,101):
            y=update(s['keys'][0],0.,False,q,True)
            if st in (10,100):ss.append(y+meas[i,role,0 if st==10 else 1])
        ys.append(ss)
    return np.array(ys)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default=str(R/'results.json'));ap.add_argument('--out',default=str(R/'verification.json'));ap.add_argument('--skip-prior-bundles',action='store_true');ap.add_argument('--prior-bundles-dir',type=Path,default=None,help='Optional directory containing the original historical archives; not required for scientific checks');a=ap.parse_args()
    s=json.loads((R/'frozen_spec.json').read_text());res=json.loads(Path(a.results).read_text());checks=0
    assert res['spec_sha256']==hashlib.sha256((R/'frozen_spec.json').read_bytes()).hexdigest();checks+=1
    nsum=0;worstmetric=0.
    for row in res['runs']:
        p=np.array(row['histories']['predictions']);t=np.array(row['histories']['targets']);er=p-t
        vals={'rmse':np.sqrt(np.mean(er**2)),'mae':np.mean(np.abs(er)),'max_abs_error':np.max(np.abs(er)),
              'history_completion_fraction':np.mean(np.max(np.abs(er),axis=1)<.05),'per_query_completion_fraction':np.mean(np.abs(er)<.05)}
        for k,v in vals.items():
            d=abs(v-row['metrics'][k]);worstmetric=max(worstmetric,d);assert d<1e-14,(k,row);checks+=1
        assert np.all(np.isfinite(p));checks+=1
        if row['model'] in ('bounded_WHQ','rate_limited_EMA'):
            samples=np.array(row['histories']['samples']);aa=.99**90;mm=1-.1/2000
            dd=.6*.01*(mm**100-aa*mm**10)/(mm-.99)
            pp=(samples[:,:,1]-aa*samples[:,:,0]-.5*(1-aa))/dd
            assert np.max(np.abs(pp-p))<1e-12;checks+=1
            assert row['metrics']['spikes_total']==0 and row['metrics']['boundary_clip_history_count']==0;checks+=1
        if row['split']=='test': nsum+=len(p)
    datasets={}
    for split,h in res['dataset_hashes'].items():
        bb=(R/f'inputs_{split}.json').read_bytes();assert hashlib.sha256(bb).hexdigest()==h;checks+=1
        datasets[split]=json.loads(bb)
    allset=[set(tuple(v) for v in datasets[name]['v']) for name in ('calibration','validation','test')]
    assert all(not allset[i]&allset[j] for i in range(3) for j in range(i));checks+=1
    data=datasets['test'];n=data['n'];maxdiff=0.;scalar_cases=0
    for cond in s['conditions']:
        for delay in s['delays_ms']:
            rng=np.random.RandomState(data['seed']+10000+int(delay*10));total=1000+int(delay*10)+200
            pb=cond['process_bound'];mb=cond['measurement_bound']
            proc=rng.uniform(-pb,pb,(total,n,3)) if pb else np.zeros((total,n,3))
            meas=rng.uniform(-mb,mb,(n,2,2)) if mb else np.zeros((n,2,2))
            for model in s['models']:
                row=next(r for r in res['runs'] if r['split']=='test' and r['condition']==cond['id'] and r['delay_ms']==delay and r['model']==model and not r['negative_control'])
                for i in [0,119,255]:
                    expected=scalar_one(i,data,model,delay,cond,s,proc,meas)
                    actual=np.array(row['histories']['samples'][i]);d=float(np.max(np.abs(expected-actual)));maxdiff=max(maxdiff,d)
                    assert d<1e-11,(cond['id'],delay,model,i,d);checks+=1;scalar_cases+=1
    pri=json.loads((R/'prior_bundle_hashes.json').read_text());unchanged={}
    for p,h in ([] if a.skip_prior_bundles or a.prior_bundles_dir is None else pri.items()):
        now=hashlib.sha256((a.prior_bundles_dir/p).read_bytes()).hexdigest();unchanged[p]=(now==h);assert now==h;checks+=1
    out={'checks_passed':checks,'scalar_episode_checks':scalar_cases,'max_scalar_vs_vector_sample_error':maxdiff,'max_recomputed_metric_discrepancy':float(worstmetric),
         'test_model_condition_histories_including_auxiliary_and_negative':nsum,'prior_bundles_unchanged':unchanged,'same_agent_not_third_party':True,
         'limitations':'No full upstream P1B replay; scalar original recurrence was transcribed from directly read source. No population guarantee or test-set optimization.'}
    Path(a.out).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
