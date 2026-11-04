#!/usr/bin/env python3
"""One frozen, local TAN-inspired write/hold/query comparison, CPU-only.
NOT P1-B/v11 reproduction; no LM weights/training/Codex. Learner step receives only
current phase, key, value, query and four state coordinates. Target values and
histories outside active events are evaluator-only. No adaptive experiment search.
"""
from __future__ import annotations
import argparse,hashlib,json,math,platform,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from reference_m1 import step as scalar_step


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')


def make_data(seed,n):
    rng=np.random.RandomState(seed)
    v=rng.uniform(-.5,.5,(n,2))
    order=np.column_stack((rng.randint(0,2,n),np.zeros(n,dtype=int)))
    order[:,1]=1-order[:,0]
    qo=np.column_stack((rng.randint(0,2,n),np.zeros(n,dtype=int)))
    qo[:,1]=1-qo[:,0]
    jit=rng.uniform(-.01,.01,(n,2))
    return {'seed':seed,'n':n,'v':v,'order':order,'qo':qo,'jitter':jit}


def gates(k,s):
    scores=-((np.asarray(k)[:,None]-np.asarray(s['keys'])[None,:])/s['key_bandwidth'])**2/2
    scores-=scores.max(axis=1,keepdims=True)
    g=np.exp(scores); return g/g.sum(axis=1,keepdims=True)


def encode(z,cond):
    bits=cond['fractional_bits']
    if bits is None:return z.copy()
    z=np.rint(z*(2.**bits))
    if not np.all(np.isfinite(z)) or np.max(np.abs(z))>=2**31:raise RuntimeError('state overflow/nonfinite')
    return z.astype(np.int32)


def decode(z,cond):
    return z.astype(np.float64)*(2.**(-cond['fractional_bits'])) if cond['fractional_bits'] is not None else z.copy()


def advance(st,model,key,value,event,query,is_query,cond,noise,s):
    """State is [u, m0=2A0/5-1, m1=2A1/5-1, r]."""
    z=decode(st,cond); u=z[:,0]; m=z[:,1:3]; r=z[:,3]
    dt=s['dt_ms']
    if model=='original_M1':
        A=2.5*(m[:,0]+1.)
        novelty=np.where(event,np.maximum(0.,np.abs(key)-1.4),0.)
        An=np.clip(A+(-A+novelty)/15.*dt,0.,5.)
        content=np.where(event,key+value,0.)
        drive=A*content+(query if is_query else 0.)
        mn=m.copy();mn[:,0]=2.*An/5.-1.
        # Inert fourth coordinate does not carry usable content in original M1.
    else:
        wg=gates(key,s)*float(event)
        err=value[:,None]-m
        cap=s['write_rate_cap_per_ms'];tauw=s['write_tau_ms']
        if model=='bounded_WHQ':psi=cap*np.tanh(err/(cap*tauw))
        elif model=='rate_limited_EMA':psi=np.clip(err/tauw,-cap,cap)
        else:raise ValueError(model)
        mn=m+dt*(wg*psi-(1.-wg)*m/s['hold_tau_ms'])
        drive=s['read_bias']+s['read_gain']*np.sum(gates(query,s)*m,axis=1) if is_query else np.zeros(len(u))
    # Original membrane/refractory threshold recurrence, old memory state in drive.
    cand=u+(-u+drive)/s['read_tau_ms']*dt
    refractory=r>0.
    spike=(~refractory)&(cand>=1.)
    obs=np.where(refractory,0.,cand) # pre-reset deadline convention
    un=np.where(refractory|spike,0.,cand)
    rn=np.where(refractory,np.maximum(0.,r-dt),np.where(spike,2.,0.))
    zn=np.column_stack((un,mn,rn))
    raw_memory=zn[:,1:3]+noise[:,1:3]
    hits=np.any((raw_memory < -1.)|(raw_memory > 1.),axis=1)
    zn[:,0]+=noise[:,0]
    zn[:,1:3]=np.clip(raw_memory,-1.,1.)
    q=encode(zn,cond)
    # Dimensionless control/action proxies, NOT energy.
    action=np.sum(np.abs(mn-m),axis=1)
    return q,obs,spike,hits,action,np.abs(drive)


def simulate(data,model,delay,cond,s,wrong=False):
    n=data['n']; ix=np.arange(n);dt=s['dt_ms'];ds=int(round(delay/dt))
    total=1000+ds+200
    nrng=np.random.RandomState(data['seed']+10000+ds)
    pb=cond['process_bound'];mb=cond['measurement_bound']
    # Evaluator pre-generates disturbances; no model can inspect future disturbances.
    proc=nrng.uniform(-pb,pb,(total,n,3)) if pb else np.zeros((total,n,3))
    meas=nrng.uniform(-mb,mb,(n,2,2)) if mb else np.zeros((n,2,2))
    z=np.zeros((n,4));
    if model=='original_M1':z[:,1:3]=-1.
    st=encode(z,cond);spikes=np.zeros(n,int);clips=np.zeros(n,int)
    action=np.zeros(n);drivecost=np.zeros(n);peakdrive=np.zeros(n)
    used=0; samples=np.zeros((n,2,2)); observed_queries=np.zeros((n,2)); cue_states=[]
    keys=np.asarray(s['keys']);zero=np.zeros(n);dkey=np.full(n,keys[0])
    def one(st,k,v,event,q,cue):
        nonlocal used
        st,o,sp,hit,a,d=advance(st,model,k,v,event,q,cue,cond,proc[used],s)
        used+=1;spikes[:]+=sp;clips[:]+=hit;action[:]+=a;drivecost[:]+=d*dt;peakdrive[:]=np.maximum(peakdrive,d)
        return st,o
    for slot in range(5):
        ev=slot in (0,4)
        pos=0 if slot==0 else 1
        target=data['order'][:,pos]
        k=keys[target] if ev else dkey
        v=data['v'][ix,target] if ev else zero
        for step in range(200):
            active=ev and step<100
            st,_=one(st,k,v,active,zero,False)
    enc=decode(st,cond)
    for _ in range(ds): st,_=one(st,dkey,zero,False,zero,False)
    delayed=decode(st,cond)
    for role in range(2):
        idx=data['qo'][:,role]
        q=keys[1-idx if wrong else idx]+data['jitter'][:,role]
        observed_queries[:,role]=q
        cue_states.append(decode(st,cond))
        for step in range(1,101):
            st,o=one(st,dkey,zero,False,q,True)
            if step in s['sample_steps']:
                at=s['sample_steps'].index(step)
                samples[:,role,at]=o+meas[:,role,at]
    assert used==total
    final=decode(st,cond)
    target=data['v'][ix[:,None],data['qo']]
    if model=='original_M1':
        cal=s['native_original_readout']
        pred=cal['gain']*samples[:,:,1]+cal['bias']
    else:
        n1,n2=s['sample_steps'];lam=1.-dt/s['read_tau_ms'];mu=1.-dt/s['hold_tau_ms'];a=lam**(n2-n1)
        D=s['read_gain']*dt/s['read_tau_ms']*(mu**n2-a*mu**n1)/(mu-lam)
        pred=(samples[:,:,1]-a*samples[:,:,0]-s['read_bias']*(1-a))/D
    out={'prediction':pred,'target':target,'samples':samples,'queries':observed_queries,
         'spikes':spikes,'memory_boundary_clip_steps':clips,'memory_total_variation':action,
         'integral_abs_membrane_drive':drivecost,'peak_abs_membrane_drive':peakdrive,
         'encoding_state':enc,'delayed_state':delayed,'final_state':final,'cue_states':np.stack(cue_states,axis=1)}
    return out


def metrics(out,model,s):
    er=out['prediction']-out['target'];t=s['endpoints']['local_history_abs_target']
    rec={'history_count':len(er),'query_count':er.size,'output_finite_fraction':float(np.mean(np.isfinite(out['prediction']))),
     'attempt_fraction':1.,'selective_rejection':False,'rmse':float(np.sqrt(np.mean(er**2))),
     'mae':float(np.mean(np.abs(er))),'max_abs_error':float(np.max(np.abs(er))),
     'history_completion_fraction':float(np.mean(np.max(np.abs(er),axis=1)<t)),
     'per_query_completion_fraction':float(np.mean(np.abs(er)<t)),
     'spikes_total':int(out['spikes'].sum()),'boundary_clip_history_count':int(np.count_nonzero(out['memory_boundary_clip_steps'])),
     'mean_memory_total_variation':float(np.mean(out['memory_total_variation'])),
     'mean_integral_abs_membrane_drive':float(np.mean(out['integral_abs_membrane_drive'])),
     'max_membrane_drive':float(np.max(out['peak_abs_membrane_drive']))}
    if model!='original_M1' and not model.startswith('original_'):
        # canonical targets recovered from query ordering outside model
        pass
    return rec


def source_fidelity(s):
    """Compare vector implementation with separately transcribed scalar recurrence.
    This does not certify original file byte identity nor rerun original P1B."""
    data=make_data(2026110191,8);c={'fractional_bits':None,'process_bound':0.,'measurement_bound':0.}
    out=simulate(data,'original_M1',5,c,s)
    errs=[];counts=[]
    for i in range(8):
        u=A=r=0.;spikes=0
        for slot in range(5):
            if slot in (0,4):
                j=int(data['order'][i,0 if slot==0 else 1]);k=s['keys'][j];v=float(data['v'][i,j])
            else:k=v=0.
            for step in range(200):
                act=slot in (0,4) and step<100
                u,A,r,sp,_=scalar_step(u,A,r,k+v if act else 0.,max(0.,abs(k)-1.4) if act else 0.,0.)
                spikes+=sp
        errs.append(np.max(np.abs(np.array([u,2*A/5-1,r])-out['encoding_state'][i,[0,1,3]])))
        for _ in range(50):u,A,r,sp,_=scalar_step(u,A,r,0.,0.,0.);spikes+=sp
        for role in range(2):
            q=s['keys'][int(data['qo'][i,role])]+float(data['jitter'][i,role])
            for stp in range(1,101):
                u,A,r,sp,uc=scalar_step(u,A,r,0.,0.,q);spikes+=sp
                if stp in s['sample_steps']:
                    expected=0. if uc is None else uc
                    errs.append(abs(expected-out['samples'][i,role,s['sample_steps'].index(stp)]))
        errs.append(np.max(np.abs(np.array([u,2*A/5-1,r])-out['final_state'][i,[0,1,3]])))
        counts.append(spikes==out['spikes'][i])
    mx=float(max(errs));assert mx<1e-12 and all(counts),(mx,counts)
    return {'histories':8,'max_abs_discrepancy':mx,'spike_counts_match':all(counts),'reference':'inspected source recurrence, copied scalar step; no complete file download'}


def analytic_decoder_check(s):
    dt=s['dt_ms'];la=1-dt/s['read_tau_ms'];mu=1-dt/s['hold_tau_ms'];n1,n2=s['sample_steps']
    a=la**(n2-n1);D=s['read_gain']*dt/s['read_tau_ms']*(mu**n2-a*mu**n1)/(mu-la)
    errors=[]
    for u0 in [0.,.3,.7]:
        for m0 in [-.5,0.,.5]:
            u=u0;m=m0;y=[]
            for n in range(1,101):
                u=u+dt/s['read_tau_ms']*(-u+s['read_bias']+s['read_gain']*m);m*=mu
                if n in s['sample_steps']:y.append(u)
            pred=(y[1]-a*y[0]-s['read_bias']*(1-a))/D
            errors.append(abs(pred-m0))
    assert max(errors)<1e-12
    return {'lambda':la,'hold_decay':mu,'elimination_factor':a,'signal_denominator':D,
            'independent_unit_sample_noise_gain':math.sqrt(1+a*a)/abs(D),
            'bounded_sample_noise_worst_gain':(1+abs(a))/abs(D),'max_identity_error':max(errors)}


def features(out):
    return np.column_stack((np.ones(out['target'].size),out['queries'].ravel(),out['samples'][:,:,0].ravel(),out['samples'][:,:,1].ravel()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',default=str(ROOT/'frozen_spec.json'));ap.add_argument('--out',default=str(ROOT/'results.json'));ap.add_argument('--timing',default=str(ROOT/'timing.json'));args=ap.parse_args()
    raw=Path(args.spec).read_bytes();s=json.loads(raw);freeze=(ROOT/'FREEZE_SHA256.txt').read_text().split()[0]
    assert hashlib.sha256(raw).hexdigest()==freeze,'frozen spec changed'
    t0=time.perf_counter();result={'scope':s['scope'],'spec_sha256':freeze,'environment':{'python':platform.python_version(),'numpy':np.__version__},'source_fidelity':source_fidelity(s),'analytic_decoder':analytic_decoder_check(s),'runs':[]}
    datas={name:make_data(v['seed'],v['n']) for name,v in s['splits'].items()}
    cal=simulate(datas['calibration'],'original_M1',0,s['conditions'][0],s)
    coef=np.linalg.lstsq(features(cal),cal['target'].ravel(),rcond=None)[0]
    result['original_auxiliary_affine']={'coefficients':coef.tolist(),'input_order':['constant','q','sample10','sample100'],'training_histories':128,'training_split_only':True,'new_model_decoders_untrained':True}
    # Validation is reported but does not choose parameters or stop failed accuracy.
    row_times=[]
    for split in ['validation','test']:
        data=datas[split]
        delays=[0] if split=='validation' else s['delays_ms']
        for cond in s['conditions']:
            for delay in delays:
                for model in s['models']:
                    t1=time.perf_counter();out=simulate(data,model,delay,cond,s)
                    record={'split':split,'condition':cond['id'],'delay_ms':delay,'model':model,'negative_control':False,'metrics':metrics(out,model,s)}
                    if model!='original_M1':
                        record['metrics']['retention_rmse']={label:float(np.sqrt(np.mean((out[state][:,1:3]-data['v'])**2))) for label,state in [('encoding_end','encoding_state'),('cue_start','delayed_state'),('after_both_queries','final_state')]}
                    record['histories']={'predictions':out['prediction'].tolist(),'targets':out['target'].tolist(),'samples':out['samples'].tolist(),'queries':out['queries'].tolist(),'spike_counts':out['spikes'].tolist(),'clip_steps':out['memory_boundary_clip_steps'].tolist()}
                    result['runs'].append(record)
                    row_times.append({'split':split,'condition':cond['id'],'delay_ms':delay,'model':model,'seconds':time.perf_counter()-t1})
                    if model=='original_M1':
                        aux=dict(out);aux['prediction']=(features(out)@coef).reshape(-1,2)
                        result['runs'].append({'split':split,'condition':cond['id'],'delay_ms':delay,'model':'original_M1_aux_affine','negative_control':False,'metrics':metrics(aux,'original_M1_aux_affine',s),'histories':{'predictions':aux['prediction'].tolist(),'targets':aux['target'].tolist()}})
        # Zero uses same keyed targets, independent of query order.
        target=data['v'][np.arange(data['n'])[:,None],data['qo']]
        result.setdefault('zero_predictor',{})[split]={'rmse':float(np.sqrt(np.mean(target**2))),'history_completion_fraction':float(np.mean(np.max(np.abs(target),axis=1)<.05))}
    nc=s['negative_control'];data=datas['test'];cond=next(c for c in s['conditions'] if c['id']==nc['condition'])
    for model in nc['models']:
        out=simulate(data,model,nc['delay_ms'],cond,s,True)
        result['runs'].append({'split':'test','condition':cond['id'],'delay_ms':nc['delay_ms'],'model':model,'negative_control':True,'metrics':metrics(out,model,s),'histories':{'predictions':out['prediction'].tolist(),'targets':out['target'].tolist(),'samples':out['samples'].tolist(),'queries':out['queries'].tolist()}})
    # Pairing is over histories, never over individual role observations.
    test={ (r['condition'],r['delay_ms'],r['model']):r for r in result['runs'] if r['split']=='test' and not r['negative_control']}
    br=np.random.RandomState(s['statistics']['seed']);n=data['n'];idx=br.randint(n,size=(s['statistics']['bootstrap'],n))
    contrasts=[]
    for delay in s['delays_ms']:
        a=test[('finite_bounded',delay,'bounded_WHQ')];b=test[('finite_bounded',delay,'rate_limited_EMA')]
        target=np.array(a['histories']['targets']);pa=np.array(a['histories']['predictions']);pb=np.array(b['histories']['predictions'])
        ea=np.mean((pa-target)**2,axis=1);eb=np.mean((pb-target)**2,axis=1);ez=np.mean(target**2,axis=1)
        for name,eother in [('WHQ_minus_EMA',eb),('WHQ_minus_zero',ez)]:
            boot=np.sqrt(ea[idx].mean(axis=1))-np.sqrt(eother[idx].mean(axis=1))
            contrasts.append({'delay_ms':delay,'contrast':name,'rmse_difference':float(np.sqrt(ea.mean())-np.sqrt(eother.mean())),'paired_history_percentile_95':np.percentile(boot,[2.5,97.5]).tolist(),'bootstrap_replicates':len(idx),'postselection':False})
    result['paired_contrasts']=contrasts
    result['decision']={'local_success':all(test[('finite_bounded',d,'bounded_WHQ')]['metrics']['rmse']<=.03 and test[('finite_bounded',d,'bounded_WHQ')]['metrics']['history_completion_fraction']>=.90 and test[('finite_bounded',d,'bounded_WHQ')]['metrics']['rmse']<result['zero_predictor']['test']['rmse'] for d in s['primary_delays_ms']),
        'subthreshold_interpretation_valid':all(r['metrics']['spikes_total']==0 and r['metrics']['boundary_clip_history_count']==0 for r in result['runs'] if r['model'] in ('bounded_WHQ','rate_limited_EMA')),
        'no_original_P1B_label_changed':True,'no_general_online_binding_claim':True}
    result['dataset_hashes']={}
    for split,d in datas.items():
        exported={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items()}
        b=(json.dumps(exported,sort_keys=True,indent=2)+'\n').encode();(ROOT/f'inputs_{split}.json').write_bytes(b)
        result['dataset_hashes'][split]=hashlib.sha256(b).hexdigest()
    dump(args.out,result);dump(args.timing,{'total_seconds':time.perf_counter()-t0,'rows':row_times,'not_energy':True})
    for row in result['runs']:
        if row['split']=='test':
            m=row['metrics'];print(row['condition'],row['delay_ms'],row['model'],'negative',row['negative_control'],'rmse',round(m['rmse'],6),'complete',m['history_completion_fraction'])
    print('decision',result['decision']);print('sha256',hashlib.sha256(Path(args.out).read_bytes()).hexdigest())
if __name__=='__main__':main()
