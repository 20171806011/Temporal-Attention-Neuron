#!/usr/bin/env python3
"""Independent v10 reconstruction; no primary simulator/statistic helpers imported.

The explicit neuron recurrence is reused from this reviewer's independently
written and validated v9 implementation. Information, kernels, confidence
certificate, and bootstrap are independently implemented below.
"""
from __future__ import annotations
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import numpy as np
from scipy.special import betainc
from scipy.stats import beta, binom

HERE=Path(__file__).resolve().parent
V9=HERE.parent/'noise_preservation_v9'
REPO=Path(os.environ.get('TAN_REPO_ROOT',str(HERE.parents[1]/'Temporal-Attention-Neuron')))
PROTOCOL_HASH='14ef73c1418b4dcea2a5a55de20391e3a8c82925bef70741ee3033ebcbfca179'
COMMIT='85213f0d417d54e4f48b301ed1c0e4200ef3c421'
KEEP=529/1013
N=2048
TOL=1e-12
ARMS=('TAN','LIF','LIF_THINNED','ZERO_OUTPUT')
CONDITIONS=('step','catch')
METRIC_NAMES=('C_H','C_H_given_S','C_I_signal','C_U_deletion','C_I_path',
              'D_H','D_H_given_S','D_I_signal','D_U_deletion','D_I_path',
              'Delta_signal','Delta_path','Delta_deletion')
RESOURCE_FIELDS=('sum_abs_current','sum_current_squared','sum_candidate_squared','peak_abs_current',
                 'peak_candidate','peak_abs_candidate','peak_membrane','peak_abs_membrane')
LEDGER_FIELDS=('history_id','condition','S','arm','underlying_model','underlying_onset_count','count_type',
               'P_C0','P_C1','P_C2','P_C3','P_C4','P_C5','P_D1','observed_total_count',
               'observed_onset_count','observed_late_count','R','generated_total_count',
               'generated_onset_count','generated_late_count')+RESOURCE_FIELDS
CHECKS=[]
COMPARISONS=[]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check(name, condition, detail=None):
    record = {'check': name, 'passed': bool(condition)}
    if detail is not None:
        record['detail'] = detail
    CHECKS.append(record)
    if not condition:
        raise AssertionError(f'{name}: {detail}')

def compare(name, actual, expected, exact=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    check(name + ': shape', actual.shape == expected.shape,
          {'actual': list(actual.shape), 'expected': list(expected.shape)})
    if actual.dtype.kind in 'biufc' and expected.dtype.kind in 'biufc':
        check(name + ': finite', bool(np.isfinite(actual).all() and np.isfinite(expected).all()))
        max_error = float(np.max(np.abs(actual.astype(float) - expected.astype(float)))) if actual.size else 0.
        equal = np.array_equal(actual, expected) if exact else max_error <= TOL
    else:
        max_error = None
        equal = np.array_equal(actual, expected)
    COMPARISONS.append({'name': name, 'elements': int(actual.size),
                        'exact': bool(exact), 'max_abs_error': max_error})
    check(name + ': values', equal, {'max_abs_error': max_error, 'exact': exact})

def git_status():
    if not (REPO / '.git').exists():
        return {'available': False, 'commit': None, 'status': None,
                'note': 'Source subset has no .git; source hashes remain mandatory.'}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    check('live repository exact commit', commit == COMMIT, commit)
    check('live repository status clean', status == '', status)
    return {'available': True, 'commit': commit, 'status': status}

def independent_native(delivered, model):
    """One full waveform; scalar membrane and explicit finite history.

    Appending a sample, computing a new mean, and then the softmax are explicit;
    no AST source model or primary implementation is invoked.
    """
    delivered = np.asarray(delivered, dtype=float)
    membrane = 0.
    past = np.zeros(10)
    result = {name: [] for name in ('spike', 'candidate', 'membrane', 'current')}
    if model == 'TAN':
        result.update({name: [] for name in ('surprise', 'gate', 'context', 'history', 'attention')})
    for value in delivered:
        if model == 'LIF':
            effective = value
            threshold = 2.
        else:
            past = np.concatenate((past[1:], [value]))
            deviation = value - np.mean(past)
            similarity = (2. * deviation) * past
            unnormalized = np.exp(similarity - np.max(similarity))
            weights = unnormalized / (np.sum(unnormalized) + 1e-9)
            context = np.sum(weights * past)
            gate = np.tanh(abs(deviation))
            effective = gate * context
            threshold = 1.
            result['surprise'].append(deviation)
            result['gate'].append(gate)
            result['context'].append(context)
            result['history'].append(past.copy())
            result['attention'].append(weights.copy())
        candidate = .8 * membrane + effective
        fired = int(candidate > threshold)
        membrane = 0. if fired else candidate
        result['spike'].append(fired)
        result['candidate'].append(candidate)
        result['membrane'].append(membrane)
        result['current'].append(effective)
    return {name: np.array(values, dtype=np.uint8 if name == 'spike' else float)
            for name, values in result.items()}

def batch_native(waveforms, model):
    traces = [independent_native(w, model) for w in waveforms]
    return {key: np.stack([t[key] for t in traces]) for key in traces[0]}

def resource_vectors(trace):
    current, candidate = trace['current'], trace['candidate']
    return {'sum_abs_current': abs(current).sum(axis=1),
            'sum_current_squared': (current*current).sum(axis=1),
            'sum_candidate_squared': (candidate*candidate).sum(axis=1),
            'peak_abs_current': abs(current).max(axis=1),
            'peak_candidate': candidate.max(axis=1),
            'peak_abs_candidate': abs(candidate).max(axis=1),
            'peak_membrane': trace['membrane'].max(axis=1),
            'peak_abs_membrane': abs(trace['membrane']).max(axis=1),
            'generated_spikes': trace['spike'].sum(axis=1)}

def describe(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'min': float(values.min()), 'max': float(values.max()),
            'sample_sd': float(values.std(ddof=1)) if values.size > 1 else None}

def compare_json_tree(label, actual, expected):
    if isinstance(expected, dict):
        check(label + ': expected fields present', isinstance(actual, dict) and set(expected) <= set(actual))
        for key, value in expected.items():
            compare_json_tree(label+'/'+key, actual[key], value)
    elif expected is None:
        check(label, actual is None)
    elif isinstance(expected, (str, bool)):
        check(label, actual == expected, {'actual': actual, 'expected': expected})
    else:
        compare(label, actual, expected)

def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sources():
    check('v10 frozen protocol SHA256',sha(HERE/'PREREGISTRATION.md')==PROTOCOL_HASH)
    manifest=json.loads((V9/'NOISE_SOURCE_MANIFEST.json').read_text())
    for relative,spec in manifest['source_files'].items():
        check('original mandatory source SHA256 '+relative,sha(REPO/relative)==spec['sha256'])
    declarations=json.loads((HERE/'SOURCE_PATHS.json').read_text())['sha256']
    for relative,expected in declarations.items():
        if relative.startswith('Temporal-Attention-Neuron/'):
            path=REPO/relative.split('/',1)[1]
        else:
            path=HERE.parents[1]/relative
        check('v10 source-context SHA256 '+relative,sha(path)==expected)
    extracted=V9/'taskb_original_extracted.py'
    check('verified source extraction SHA256',sha(extracted)==manifest['extracted_sha256'])
    original_text=(REPO/'snapshot/code/experiments/statistical_validation.py').read_text()
    original_nodes={node.name:node for node in ast.parse(original_text).body
                    if isinstance(node,(ast.ClassDef,ast.FunctionDef))}
    extracted_nodes={node.name:node for node in ast.parse(extracted.read_text()).body
                     if isinstance(node,(ast.ClassDef,ast.FunctionDef))}
    for entry in manifest['selected_source_nodes']:
        name=entry['name']
        raw=ast.get_source_segment(original_text,original_nodes[name])
        check('source exact segment '+name,hashlib.sha256(raw.encode()).hexdigest()==entry['exact_segment_sha256'])
        check('source extracted AST identity '+name,ast.dump(original_nodes[name],include_attributes=False)==
              ast.dump(extracted_nodes[name],include_attributes=False))
    prior=json.loads((V9/'PRIMARY_MANIFEST.json').read_text())
    check('v9 original100-record provenance',prior['phase_A_records']==100 and prior['status']=='COMPLETED')
    for filename,spec in prior['files'].items():
        check('unchanged v9 primary artifact '+filename,sha(V9/filename)==spec['sha256'])
    check('v9 independent validation PASS',json.loads((V9/'INDEPENDENT_REVIEW_CHECKS.json').read_text())['status']=='PASS')
    return {'original_manifest':sha(V9/'NOISE_SOURCE_MANIFEST.json'),
            'v10_sources':sha(HERE/'SOURCE_PATHS.json'),
            'v9_primary_manifest':sha(V9/'PRIMARY_MANIFEST.json')}


def make_design():
    base=np.zeros((2,60));base[0,15:]=1.5
    noise=np.random.RandomState(2026110240).normal(0.,.4,size=(N,60))
    return {'fresh':{'noise':noise,'base':base,'waveforms':np.maximum(noise[:,None,:]+base[None,:,:],0.),
                      'history_ids':np.arange(N),'conditions':np.array(CONDITIONS),'S':np.array([1,0])},
            'clean':{'noise':np.zeros((1,60)),'base':base,'waveforms':base[None,:,:].copy(),
                      'history_ids':np.arange(1),'conditions':np.array(CONDITIONS),'S':np.array([1,0])}}


def all_native(design):
    traces={}
    for stage,inputs in design.items():
        for model in ('TAN','LIF'):
            for c,condition in enumerate(CONDITIONS):
                traces[(stage,model,condition)]=batch_native(inputs['waveforms'][:,c,:],model)
    return traces


def entropy(probabilities,axis=-1):
    p=np.asarray(probabilities,dtype=float)
    logs=np.zeros_like(p)
    np.log2(p,out=logs,where=p>0)
    return -(p*logs).sum(axis=axis)


def binary_entropy(probability):
    q=np.asarray(probability,dtype=float)
    return entropy(np.stack([1.-q,q],axis=-1))


def count_channel(retention):
    matrix=np.zeros((6,6))
    for k in range(6):
        for y in range(k+1):
            matrix[k,y]=math.comb(k,y)*retention**y*(1.-retention)**(k-y)
    return matrix


def make_laws(traces,stage,kernel):
    count={model:np.stack([traces[(stage,model,c)]['spike'][:,15:20].sum(axis=1)
                           for c in CONDITIONS],axis=1).astype(int) for model in ('TAN','LIF')}
    laws={}
    for arm in ARMS:
        k=count['LIF'] if arm=='LIF_THINNED' else count.get(arm)
        if arm=='ZERO_OUTPUT':
            k=np.full(count['LIF'].shape,-1,dtype=int)
            cprob=np.zeros((*k.shape,6));cprob[:,:,0]=1.
            dprob=np.zeros((*k.shape,2));dprob[:,:,0]=1.
        elif arm=='LIF_THINNED':
            cprob=kernel[k]
            positive=-np.expm1(k*np.log1p(-KEEP))
            dprob=np.stack([1.-positive,positive],axis=-1)
        else:
            cprob=np.eye(6)[k]
            dprob=np.eye(2)[(k>=1).astype(int)]
        laws[arm]={'count_pmf':cprob,'decision_pmf':dprob,'underlying_count':k}
    return laws,count


def independent_metrics(native_hist,kernel):
    """Thirteen terms per arm from two native conditional count histograms."""
    information={};pmfs={}
    q=1.-(1.-KEEP)**np.arange(6)
    decision_kernel=np.column_stack([1.-q,q])
    for arm in ARMS:
        if arm=='ZERO_OUTPUT':
            c=np.zeros((2,6));c[:,0]=1.
            d=np.zeros((2,2));d[:,0]=1.
            uc=ud=0.
        elif arm=='LIF_THINNED':
            source=native_hist['LIF']
            c=source@kernel;d=source@decision_kernel
            uc=float((source@entropy(kernel)).mean())
            ud=float((source@entropy(decision_kernel)).mean())
        else:
            c=native_hist[arm]
            d=np.column_stack([c[:,0],c[:,1:].sum(axis=1)])
            uc=ud=0.
        ch=float(entropy(c.mean(axis=0)));cc=float(entropy(c).mean())
        dh=float(entropy(d.mean(axis=0)));dc=float(entropy(d).mean())
        terms=np.array([ch,cc,ch-cc,uc,cc-uc,dh,dc,dh-dc,ud,dc-ud,
                        (ch-cc)-(dh-dc),(cc-uc)-(dc-ud),uc-ud])
        information[arm]=terms
        pmfs[arm]={'C_conditional':c,'C_marginal':c.mean(axis=0),
                   'D_conditional':d,'D_marginal':d.mean(axis=0)}
    return information,pmfs


def count_histograms(count,indices=None):
    hist={}
    for model in ('TAN','LIF'):
        a=count[model] if indices is None else count[model][indices]
        hist[model]=np.stack([np.bincount(a[:,c],minlength=6) for c in range(2)])/len(a)
    return hist


def all_bootstraps(count,kernel):
    draws=np.random.RandomState(2026110241).randint(0,N,size=(2000,N))
    metrics={arm:np.empty((2000,13)) for arm in ARMS}
    pmfs={arm:{'C_conditional':np.empty((2000,2,6)),'C_marginal':np.empty((2000,6)),
               'D_conditional':np.empty((2000,2,2)),'D_marginal':np.empty((2000,2))} for arm in ARMS}
    for r,indices in enumerate(draws):
        terms,distributions=independent_metrics(count_histograms(count,indices),kernel)
        for arm in ARMS:
            metrics[arm][r]=terms[arm]
            for name,value in distributions[arm].items():pmfs[arm][name][r]=value
    return draws,metrics,pmfs


def cp_upper_bisection(k,n,alpha):
    """Independent inversion: beta-CDF evaluation plus monotone bisection.

    The primary uses a beta quantile; this implementation does not call ppf.
    """
    if k==0:return -math.expm1(math.log(alpha)/n)
    if k==n:return 1.
    lo,hi=0.,1.
    for _ in range(90):
        mid=(lo+hi)/2.
        tail=float(betainc(n-k,k+1,1.-mid))
        if tail>alpha:lo=mid
        else:hi=mid
    return (lo+hi)/2.


def independent_certificate(count,information):
    delta=math.sqrt(math.log(4*(2**6-2)/.025)/(2*N))
    continuity=float(binary_entropy(delta)+delta*math.log2(5))
    result={'n':N,'alpha_TV':.025,'alpha_CP':.025,'alpha_cell':.00625,
            'delta':delta,'F6_delta':continuity,'arms':{}}
    check('entropy modulus monotonic-range condition',delta<=5/6)
    for model in ('TAN','LIF'):
        misses=int((count[model][:,0]==0).sum())
        false=int((count[model][:,1]>=1).sum())
        mu=cp_upper_bisection(misses,N,.00625)
        fu=cp_upper_bisection(false,N,.00625)
        for label,k,upper in [('miss',misses,mu),('FPR',false,fu)]:
            if k<N:
                compare('CP independent binomial-CDF inversion '+model+'/'+label,binom.cdf(k,N,upper),.00625)
        e=(mu+fu)/2.
        b=float(binary_entropy(e)) if e<=.5 else 1.
        hc=information[model][1]
        result['arms'][model]={'miss_count':misses,'FPR_count':false,'miss_upper':mu,'FPR_upper':fu,
                              'error_upper':e,'b':b,'I_signal_D_lower':1.-b,
                              'Delta_signal_upper':b,'Delta_path_lower':max(0.,hc-continuity-b),
                              'H_hat_C_given_S':float(hc)}
    # Formula controls do not run new models or add empirical arms.
    compare('CP k0 exact formula',cp_upper_bisection(0,N,.00625),-math.expm1(math.log(.00625)/N),exact=True)
    compare('CP kN exact upper1',cp_upper_bisection(N,N,.00625),1.,exact=True)
    for k in (1,N//2,N-1):
        upper=cp_upper_bisection(k,N,.00625)
        compare('CP helper fixed interior control '+str(k),binom.cdf(k,N,upper),.00625)
    return result


def mathematical_invariants(kernel,information,bootstrap,laws,traces,design):
    compare('analytic thinning channel normalized',kernel.sum(axis=1),np.ones(6))
    check('analytic thinning impossible counts exactly zero',all((kernel[k,k+1:]==0).all() for k in range(6)))
    compare('analytic p1 channel identity',count_channel(1.),np.eye(6),exact=True)
    zero_kernel=np.zeros((6,6));zero_kernel[:,0]=1.
    compare('analytic p0 channel silence',count_channel(0.),zero_kernel,exact=True)
    for stage,stage_laws in laws.items():
        for arm,arrays in stage_laws.items():
            for name in ('count_pmf','decision_pmf'):
                p=arrays[name]
                check(stage+'/'+arm+'/'+name+' nonnegative',bool((p>=0).all()))
                compare(stage+'/'+arm+'/'+name+' normalized',p.sum(axis=-1),np.ones(p.shape[:-1]))
            c,d=arrays['count_pmf'],arrays['decision_pmf']
            compare(stage+'/'+arm+' observed coarsening D0',d[:,:,0],c[:,:,0])
            compare(stage+'/'+arm+' observed coarsening D1',d[:,:,1],c[:,:,1:].sum(axis=-1))
    oracle=np.zeros((2,6));oracle[0,1]=1.;oracle[1,0]=1.
    oracle_terms,_=independent_metrics({'TAN':oracle,'LIF':oracle},kernel)
    compare('label oracle signal information C=1bit',oracle_terms['TAN'][2],1.,exact=True)
    compare('label oracle conditional path detail zero',oracle_terms['TAN'][4],0.,exact=True)
    for stage,values in [('point',{arm:x[None,:] for arm,x in information.items()}),('bootstrap',bootstrap)]:
        for arm,v in values.items():
            check(stage+'/'+arm+' all information nonnegative within tolerance',bool((v>=-TOL).all()))
            check(stage+'/'+arm+' binary-label information at most1bit',bool((v[:,[2,7]]<=1.+TOL).all()))
            compare(stage+'/'+arm+' count entropy closure',v[:,0],v[:,2]+v[:,3]+v[:,4])
            compare(stage+'/'+arm+' decision entropy closure',v[:,5],v[:,7]+v[:,8]+v[:,9])
            compare(stage+'/'+arm+' signal detail definition',v[:,10],v[:,2]-v[:,7])
            compare(stage+'/'+arm+' path detail definition',v[:,11],v[:,4]-v[:,9])
            compare(stage+'/'+arm+' deletion detail definition',v[:,12],v[:,3]-v[:,8])
            check(stage+'/'+arm+' count-to-decision data processing',bool((v[:,10:]>=-TOL).all()))
            if arm in ('TAN','LIF','ZERO_OUTPUT'):
                compare(stage+'/'+arm+' deterministic residual deletion0',v[:,[3,8]],np.zeros_like(v[:,[3,8]]),exact=True)
        check(stage+' thinning count cannot increase label information',
              bool((values['LIF_THINNED'][:,2]<=values['LIF'][:,2]+TOL).all()))
        check(stage+' thinning count cannot increase conditional Gaussian-path information',
              bool((values['LIF_THINNED'][:,4]<=values['LIF'][:,4]+TOL).all()))
        compare(stage+' all zero-output terms vanish',values['ZERO_OUTPUT'],np.zeros_like(values['ZERO_OUTPUT']),exact=True)
    for stage in ('fresh','clean'):
        for model in ('TAN','LIF'):
            a,b=[traces[(stage,model,c)] for c in CONDITIONS]
            for field in a:
                compare(stage+'/'+model+' identical paired pre-onset '+field,a[field][:,:15],b[field][:,:15],exact=True)
            for c,condition in enumerate(CONDITIONS):
                t=traces[(stage,model,condition)]
                label=f'{stage}/{model}/{condition}'
                threshold=2. if model=='LIF' else 1.
                check(label+' strict native threshold',np.array_equal(t['spike'],t['candidate']>threshold))
                compare(label+' reset identity',t['membrane'],np.where(t['spike'],0.,t['candidate']),exact=True)
                old=np.pad(t['membrane'][:,:-1],((0,0),(1,0)))
                compare(label+' explicit old-state recurrence',t['candidate'],.8*old+t['current'],exact=True)
                if model=='TAN':
                    compare(label+' input-inclusive history',t['history'][:,:,-1],design[stage]['waveforms'][:,c,:],exact=True)
                    compare(label+' absolute surprise gate',t['gate'],np.tanh(abs(t['surprise'])),exact=True)
    expected_spikes=np.zeros(60);expected_spikes[16:59:2]=1.
    compare('clean LIF full exact spike certificate',traces[('clean','LIF','step')]['spike'][0],expected_spikes,exact=True)
    compare('clean LIF late rate exact.5',traces[('clean','LIF','step')]['spike'][0,20:].mean(),.5,exact=True)


def validate_inputs(design):
    for stage,expected in design.items():
        with np.load(HERE/(stage+'_inputs.npz'),allow_pickle=False) as saved:
            check(stage+' input fields complete',set(saved.files)==set(expected),saved.files)
            for name,value in expected.items():compare(stage+' input '+name,saved[name],value,exact=True)


def validate_traces(traces):
    fields={'spikes':'spike','current':'current','candidate':'candidate','membrane':'membrane',
            'exact_class_spikes':'spike','exact_class_membrane':'membrane'}
    tan={'history':'history','exact_class_history':'history','attention':'attention',
         'surprise':'surprise','gate':'gate','context':'context'}
    for stage in ('fresh','clean'):
        expected_fields={'history_ids','conditions'}
        with np.load(HERE/(stage+'_trajectories.npz'),allow_pickle=False) as saved:
            compare(stage+' trace condition order',saved['conditions'],CONDITIONS,exact=True)
            compare(stage+' trace IDs',saved['history_ids'],np.arange(N if stage=='fresh' else 1),exact=True)
            for model in ('TAN','LIF'):
                mapping={**fields,**(tan if model=='TAN' else {})}
                for field,own in mapping.items():
                    key=model+'__'+field;expected_fields.add(key)
                    expected=np.stack([traces[(stage,model,c)][own] for c in CONDITIONS],axis=1)
                    compare(stage+' full native '+key,saved[key],expected,exact=own in ('spike','history'))
                if model=='TAN':
                    key='TAN__query';expected_fields.add(key)
                    compare(stage+' full native query',saved[key],2.*np.stack([traces[(stage,model,c)]['surprise'] for c in CONDITIONS],axis=1))
            check(stage+' complete trace fields',set(saved.files)==expected_fields,saved.files)
    # Clean replay must match the already independently audited v9 clean states.
    with np.load(V9/'clean_trajectories.npz',allow_pickle=False) as saved:
        for model in ('TAN','LIF'):
            for field,own in {**fields,**(tan if model=='TAN' else {})}.items():
                expected=np.stack([traces[('clean',model,c)][own] for c in CONDITIONS],axis=1)
                compare('v9 clean exact replay '+model+'__'+field,saved[model+'__'+field],expected,
                        exact=own in ('spike','history'))


def validate_laws(laws):
    for stage,stage_laws in laws.items():
        expected={arm+'__'+key:value for arm,fields in stage_laws.items() for key,value in fields.items()}
        with np.load(HERE/(stage+'_laws.npz'),allow_pickle=False) as saved:
            check(stage+' analytic law fields',set(saved.files)==set(expected),saved.files)
            for key,value in expected.items():compare(stage+' all analytic output laws '+key,saved[key],value,exact=key.endswith('underlying_count'))


def own_ledger(stage,traces,laws):
    size=N if stage=='fresh' else 1
    rows={}
    resources={(m,c):resource_vectors(traces[(stage,m,c)]) for m in ('TAN','LIF') for c in CONDITIONS}
    for arm in ARMS:
        for c,condition in enumerate(CONDITIONS):
            for i in range(size):
                law=laws[arm]
                row={field:None for field in LEDGER_FIELDS}
                row.update(history_id=i,condition=condition,S=1-c,arm=arm,
                           underlying_model=None if arm=='ZERO_OUTPUT' else ('LIF' if arm=='LIF_THINNED' else arm),
                           underlying_onset_count=int(law['underlying_count'][i,c]),
                           count_type='expectation' if arm=='LIF_THINNED' else 'actual',
                           P_D1=float(law['decision_pmf'][i,c,1]))
                row.update({'P_C'+str(k):float(law['count_pmf'][i,c,k]) for k in range(6)})
                if arm=='ZERO_OUTPUT':
                    row.update(observed_total_count=0.,observed_onset_count=0.,observed_late_count=0.,R=0.)
                else:
                    model=row['underlying_model'];spikes=traces[(stage,model,condition)]['spike'][i]
                    total,onset,late=int(spikes.sum()),int(spikes[15:20].sum()),int(spikes[20:].sum())
                    p=KEEP if arm=='LIF_THINNED' else 1.
                    row.update(observed_total_count=p*total,observed_onset_count=p*onset,
                               observed_late_count=p*late,R=p*late/40.,generated_total_count=total,
                               generated_onset_count=onset,generated_late_count=late)
                    row.update({key:float(resources[(model,condition)][key][i]) for key in RESOURCE_FIELDS})
                rows[(i,condition,arm)]=row
    return rows


def validate_ledgers(traces,laws):
    all_rows={}
    for stage in ('fresh','clean'):
        expected=own_ledger(stage,traces,laws[stage]);all_rows[stage]=expected
        with (HERE/(stage+'_ledger.csv')).open(newline='') as stream:
            reader=csv.DictReader(stream)
            check(stage+' exact ledger columns',reader.fieldnames==list(LEDGER_FIELDS),reader.fieldnames)
            rows=list(reader)
        check(stage+' ledger all rows retained',len(rows)==len(expected),len(rows))
        actual={(int(r['history_id']),r['condition'],r['arm']):r for r in rows}
        check(stage+' ledger no duplicate IDs',len(actual)==len(rows))
        check(stage+' ledger entire Cartesian product retained',set(actual)==set(expected))
        order=sorted(expected)
        for field in LEDGER_FIELDS:
            numeric=field not in ('condition','arm','underlying_model','count_type')
            present=[key for key in order if expected[key][field] is not None]
            check(stage+' nullable physical resources '+field,
                  all(actual[key][field]=='' for key in order if expected[key][field] is None))
            a=[actual[key][field] for key in present];e=[expected[key][field] for key in present]
            if numeric:a=[float(x) for x in a]
            exact=not numeric or field in ('history_id','S','underlying_onset_count') or field.startswith('generated_')
            compare(stage+' full ledger '+field,a,e,exact=exact)
    return all_rows


def validate_kernel(kernel):
    q=1.-(1.-KEEP)**np.arange(6)
    expected={'count_kernel':kernel,'decision_success':q,'count_kernel_entropy':entropy(kernel),
              'decision_kernel_entropy':entropy(np.column_stack([1.-q,q]))}
    with np.load(HERE/'thinning_kernel.npz',allow_pickle=False) as saved:
        check('kernel saved fields',set(saved.files)==set(expected))
        for field,value in expected.items():compare('analytic kernel '+field,saved[field],value)


def validate_pmfs(filename,pmfs,conditional_only=False):
    expected={arm+'__'+key:value for arm,fields in pmfs.items() for key,value in fields.items()
              if not conditional_only or key.endswith('conditional')}
    with np.load(HERE/filename,allow_pickle=False) as saved:
        check(filename+' all distribution fields',set(saved.files)==set(expected),saved.files)
        for key,value in expected.items():compare(filename+' '+key,saved[key],value)


def validate_point_information(information,clean_information,pmfs,clean_pmfs):
    validate_pmfs('point_pmfs.npz',pmfs)
    validate_pmfs('clean_pmfs.npz',clean_pmfs)
    with (HERE/'point_information.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        check('point information exact columns',reader.fieldnames==['arm',*METRIC_NAMES])
        rows=list(reader)
    keyed={row['arm']:row for row in rows}
    check('point information all arms once',len(rows)==4 and set(keyed)==set(ARMS))
    for arm in ARMS:compare('point all13 terms '+arm,[float(keyed[arm][k]) for k in METRIC_NAMES],information[arm])
    clean=json.loads((HERE/'clean_results.json').read_text())
    compare_json_tree('clean information results',clean,{'native_trajectories':4,'no_replicate_interpretation':True,
        'comparison_with_v9_max_abs_error':0.,'information':{arm:dict(zip(METRIC_NAMES,map(float,clean_information[arm]))) for arm in ARMS}})
    for arm in ARMS:
        t,f=pmfs[arm]['D_conditional'][:,1]
        formula=float(binary_entropy((t+f)/2.)-(binary_entropy(t)+binary_entropy(f))/2.)
        compare('binary MI formula with explicit step/catch '+arm,information[arm][7],formula)


def validate_bootstraps(draws,metrics,pmfs,information):
    compare('all2000 whole-history bootstrap draws',np.load(HERE/'bootstrap_draws.npy',allow_pickle=False),draws,exact=True)
    with np.load(HERE/'bootstrap_metrics.npz',allow_pickle=False) as saved:
        check('bootstrap metric archive fields',set(saved.files)==set(ARMS)|{'metric_names'})
        compare('bootstrap metric ordering',saved['metric_names'],METRIC_NAMES,exact=True)
        for arm in ARMS:compare('all2000x13 bootstrap terms '+arm,saved[arm],metrics[arm])
    validate_pmfs('bootstrap_pmfs.npz',pmfs,conditional_only=True)
    with (HERE/'bootstrap_metrics.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        check('bootstrap CSV exact columns',reader.fieldnames==['replicate','arm',*METRIC_NAMES])
        rows=list(reader)
    keyed={(int(row['replicate']),row['arm']):row for row in rows}
    check('bootstrap CSV all8000rows',len(rows)==8000 and len(keyed)==8000)
    check('bootstrap CSV allarm-replicate combinations',set(keyed)=={(i,a) for i in range(2000) for a in ARMS})
    expected={}
    for arm in ARMS:
        actual=[[float(keyed[(i,arm)][key]) for key in METRIC_NAMES] for i in range(2000)]
        compare('all bootstrap CSV terms '+arm,actual,metrics[arm])
        expected[arm]={}
        for j,key in enumerate(METRIC_NAMES):
            x=metrics[arm][:,j];point=information[arm][j]
            expected[arm][key]={'estimate':float(point),'bootstrap_mean':float(x.mean()),
                'bootstrap_bias_diagnostic':float(x.mean()-point),'percentile_95_ci':np.quantile(x,[.025,.975]).tolist()}
    compare_json_tree('all52bootstrap summaries',json.loads((HERE/'bootstrap_summary.json').read_text()),expected)
    return expected


def certificate_schema(own):
    expected={'n_per_native_condition':N,'alpha_TV':.025,'alpha_CP':.025,'alpha_CP_cell':.00625,
        'joint_confidence_at_least':.95,'TV_radius':own['delta'],'F6_entropy_modulus_bits':own['F6_delta'],
        'subsets_per_distribution':62,'native_conditional_distributions':4,'native_arms':{}}
    for arm,values in own['arms'].items():
        cells={}
        for label,count_key,upper_key in [('miss','miss_count','miss_upper'),('FPR','FPR_count','FPR_upper')]:
            k,u=values[count_key],values[upper_key]
            cdf=float(binom.cdf(k,N,u))
            cells[label]={'errors':k,'n':N,'upper_probability':u,
                'method':'exact_zero_error_expm1' if k==0 else 'all_errors_boundary' if k==N else 'one_sided_beta_quantile',
                'binomial_cdf_at_upper':cdf,'cdf_minus_alpha_cell':cdf-.00625 if k<N else None}
        raw=values['H_hat_C_given_S']-own['F6_delta']-values['b']
        expected['native_arms'][arm]={'error_cells':cells,'error_upper_balanced':values['error_upper'],
            'b_bits':values['b'],'Hhat_C_given_S':values['H_hat_C_given_S'],
            'I_signal_D_lower_bound':values['I_signal_D_lower'],'Delta_signal_lower_bound':0.,
            'Delta_signal_upper_bound':values['Delta_signal_upper'],'Delta_path_raw_lower_bound':raw,
            'Delta_path_lower_bound':values['Delta_path_lower']}
    return expected


def validate_population_certificate(certificate):
    expected=certificate_schema(certificate)
    compare_json_tree('simultaneous native population certificate',json.loads((HERE/'population_certificate.json').read_text()),expected)
    return expected


def validate_summary(information,clean_information,pmfs,count,ledger,certificate):
    expected={'protocol_sha256':PROTOCOL_HASH,'n_histories':N,'n_native_fresh_trajectories':8192,
        'n_native_clean_trajectories':4,'n_ledger_rows':16384,'no_exclusions':True,
        'noise_seed':2026110240,'bootstrap_seed':2026110241,'bootstrap_replicates':2000,
        'keep_probability':KEEP,'new_deletion_masks_drawn':False,'condition_order':list(CONDITIONS),
        'condition_S_values':[1,0],'balanced_S_prior':[.5,.5],'onset_time':15,'onset_time_entropy':0.,
        'count_alphabet':list(range(6)),'metric_order':list(METRIC_NAMES),
        'information':{arm:dict(zip(METRIC_NAMES,map(float,information[arm]))) for arm in ARMS},
        'population_certificate':certificate,'native_count_frequencies':{},'activity_and_resource_context':{},
        'detection_context':{},'clean_v9_max_abs_error':0.}
    for model in ('TAN','LIF'):
        expected['native_count_frequencies'][model]={}
        for c,condition in enumerate(CONDITIONS):
            frequencies=np.bincount(count[model][:,c],minlength=6)
            expected['native_count_frequencies'][model][condition]={'frequencies':frequencies.tolist(),
                'alphabet':list(range(6)),'empty_bins':np.flatnonzero(frequencies==0).tolist(),
                'singleton_bins':np.flatnonzero(frequencies==1).tolist()}
    for arm in ARMS:
        t,f=pmfs[arm]['D_conditional'][:,1]
        expected['detection_context'][arm]={'TPR':float(t),'FPR':float(f),'BA':float(.5*(t+1.-f))}
        activity={'conditions':{}}
        for condition in CONDITIONS:
            rows=[ledger['fresh'][(i,condition,arm)] for i in range(N)]
            cell={field:describe([r[field] for r in rows]) for field in
                  ('observed_total_count','observed_onset_count','observed_late_count','R')}
            cell['generated_total_count']=None if arm=='ZERO_OUTPUT' else describe([r['generated_total_count'] for r in rows])
            cell['resource_proxies']=None if arm=='ZERO_OUTPUT' else {field:describe([r[field] for r in rows]) for field in RESOURCE_FIELDS}
            activity['conditions'][condition]=cell
        clean_r=ledger['clean'][(0,'step',arm)]['R']
        activity['clean_step_R']=clean_r
        activity['noisy_minus_clean_step_R']=activity['conditions']['step']['R']['mean']-clean_r
        expected['activity_and_resource_context'][arm]=activity
    activity=expected['activity_and_resource_context']
    expected['fresh_late_budget_mismatch_TAN_minus_thinned_LIF']=activity['TAN']['conditions']['step']['R']['mean']-activity['LIF_THINNED']['conditions']['step']['R']['mean']
    compare_json_tree('primary full scientific summary',json.loads((HERE/'primary_results.json').read_text()),expected)
    return expected


def primary_provenance():
    manifest=json.loads((HERE/'PRIMARY_MANIFEST.json').read_text())
    check('primary completed before independent validation',manifest['status']=='COMPLETED')
    check('primary protocol SHA256',manifest['protocol_sha256']==PROTOCOL_HASH)
    check('primary executing script SHA256',manifest['primary_script_sha256']==sha(HERE/'run_noise_transfer.py'))
    check('primary condition labels',manifest['conditions']==list(CONDITIONS) and manifest['S']==[1,0])
    check('primary allarms and metric order',manifest['arms']==list(ARMS) and manifest['metric_order']==list(METRIC_NAMES))
    check('primary ledger schema',manifest['ledger_fields']==list(LEDGER_FIELDS))
    for name,spec in manifest['files'].items():
        check('primary declared SHA256 '+name,sha(HERE/name)==spec['sha256'])
        check('primary declared byte count '+name,(HERE/name).stat().st_size==spec['bytes'])
    for name,fields in manifest['npz_schema'].items():
        with np.load(HERE/name,allow_pickle=False) as saved:
            check('primary declared all NPZ fields '+name,set(saved.files)==set(fields))
            for field,spec in fields.items():
                value=saved[field]
                check('primary declared NPZ shape '+name+'/'+field,list(value.shape)==spec['shape'])
                check('primary declared NPZ dtype '+name+'/'+field,str(value.dtype)==spec['dtype'])
    frozen=json.loads((HERE/'INPUTS_FROZEN.json').read_text())
    check('pre-evaluation input marker',frozen['native_models_evaluated_yet'] is False)
    check('input marker precedes completion',frozen['time_utc']<manifest['completed_utc'])
    for stage in ('fresh','clean'):
        check(stage+' frozen archive SHA256',frozen[stage+'_inputs_sha256']==sha(HERE/(stage+'_inputs.npz')))
        with np.load(HERE/(stage+'_inputs.npz'),allow_pickle=False) as saved:
            declarations=frozen[stage+'_array_manifest']
            check(stage+' frozen all input arrays',set(saved.files)==set(declarations))
            for name,spec in declarations.items():
                value=saved[name]
                check(stage+' frozen raw input SHA256 '+name,hashlib.sha256(value.tobytes(order='C')).hexdigest()==spec['raw_c_order_sha256'])
                check(stage+' frozen raw input shape/dtype '+name,list(value.shape)==spec['shape'] and str(value.dtype)==spec['dtype'])
    guard=json.loads((HERE/'SOURCE_GUARD.json').read_text())
    check('primary sourceguard protocol',guard['v10_protocol_sha256']==PROTOCOL_HASH)
    check('primary original100 previously reproduced not new',guard['original_100_records_previously_reproduced'] is True and
          guard['original_100_records_rerun_in_v10'] is False)
    for relative,digest in guard['source_files_sha256'].items():check('sourceguard original source '+relative,sha(REPO/relative)==digest)
    for filename,digest in guard['v9_fixed_artifact_sha256'].items():check('sourceguard fixedv9 '+filename,sha(V9/filename)==digest)
    names=set(manifest['files'])|{'PRIMARY_MANIFEST.json','run_noise_transfer.py','PREREGISTRATION.md','SOURCE_PATHS.json'}
    return {name:sha(HERE/name) for name in sorted(names)}


def write_report(result):
    text=['# Independent v10 runtime review','','Status: **PASS**.', '',
          'The reviewer reused its own independently written v9 neuron recurrence and independently implemented the finite-output laws, information calculations, whole-history bootstrap, and population certificate. No primary simulator or metric helper was imported. Primary source was consulted only for output schema after the independent core was written.', '',
          f"Protocol SHA256: `{PROTOCOL_HASH}`.",
          f"Independent reviewer SHA256: `{result['independent_script_sha256']}`.",'',
          'Verified all8,192 fresh and4clean native trajectories, including every spike, candidate, reset membrane, current, history, surprise, attention and query; all16,384 fresh and8clean output-law/resource ledger rows; every finite pmf; all104,000 bootstrap metric values and4,096,000 bootstrap indices. Clean native traces also match the independently audited v9 clean traces. Spikes, inputs and bootstrap indices agree exactly.', '',
          f"Maximum absolute numerical comparison residual: `{result['max_abs_float_residual']:.17g}`.",
          f"Maximum full native float trajectory residual: `{result['max_native_float_residual']:.17g}`.",'',
          'The axis check is explicit: condition0 is step/S=1, condition1 is catch/S=0. CP miss counts use zero step counts; FPR counts use positive catch counts. Symmetric entropy agreement alone was not used to verify this mapping.', '',
          '| Native arm | Fixed-detector signal lower bound (bits) | Extra signal detail upper bound (bits) | Conditional path detail lower bound (bits) |',
          '|---|---:|---:|---:|']
    for model,values in result['independent_population_certificate']['native_arms'].items():
        text.append(f"| {model} | {values['I_signal_D_lower_bound']:.9f} | {values['Delta_signal_upper_bound']:.9f} | {values['Delta_path_lower_bound']:.9f} |")
    text+=['',
        'These three statements hold jointly for both native arms with confidence at least95% under the frozen iid history design: .025 failure budget for four six-category histogram TV bounds and .025 for four one-sided exact binomial error limits. The reviewer calculated interior binomial upper limits by monotone tail inversion rather than the primary beta-quantile call, and checked tail residuals separately. The entropy-modulus proof and zero-error treatment are in POPULATION_CERTIFICATE_PROOF.md.', '',
        'The positive path-detail lower bound, if present, concerns input-path information in the onset count discarded by the external fixed binary decision. It does not identify individual noise-caused spikes or prove a native filtering mechanism. The extra signal-information upper bound describes the same count-to-decision compression. No timing entropy, energy, novel information theory, or population-perfect-performance claim follows.', '',
        'All52 reported information quantities have their complete2,000-replicate descriptive paired bootstrap distributions and pointwise intervals verified. The native population certificate is separate from those intervals; it remains nondegenerate for zero observed errors. CP is not applied to fractionally integrated thinning outputs. Analytic thinning agrees with Binomial(K,529/1013); no new thinning masks or fake thinned spike trains were introduced.', '',
        'Every arm satisfies the finite-output chain rules and its count-to-decision information inequalities within1e-12. Valid thinning comparisons use native LIF count→thinned count→thinned decision. No generally invalid native-binary-decision→thinned-binary-decision information ordering is assumed.', '',
        f"Primary artifacts unchanged: `{result['primary_artifacts_unchanged']}`. Repository unchanged: `{result['repository_status_unchanged']}`.",
        f"Repository provenance: `{result['repository_after']}`.",'',
        'Completed command:', '',
        '`PYTHONDONTWRITEBYTECODE=1 python tan_audit/noise_transfer_v10/independent_review.py`', '',
        'All validation attempts retain their exact script snapshot, check JSON and log. No original source, prior result, primary artifact or protocol was changed by this review. No remaining blocker.']
    return '\n'.join(text)+'\n'


def main():
    started=datetime.now(timezone.utc).isoformat()
    attempt=1
    while (HERE/f'review_attempt_{attempt:02d}.json').exists():attempt+=1
    prefix=HERE/f'review_attempt_{attempt:02d}'
    script_sha=sha(__file__)
    prefix.with_suffix('.script.txt').write_text(Path(__file__).read_text())
    result={'status':'RUNNING','attempt':attempt,'started_utc':started,
        'independent_script_sha256':script_sha,'protocol_sha256':PROTOCOL_HASH,
        'execution_command':f'PYTHONDONTWRITEBYTECODE=1 python {Path(__file__).resolve()}',
        'python':platform.python_version(),'numpy':np.__version__,
        'no_primary_helpers_imported':True,'own_v9_independent_recurrence_reused':True,
        'CP_method':'Independent monotone binomial-tail inversion; no beta.ppf call.'}
    try:
        print('Independent v10 review started',started,flush=True)
        result['source_provenance']=sources()
        result['repository_before']=git_status()
        result['primary_sha256']=primary_provenance()
        print('Source/input/protocol provenance verified; independently reconstructing native trajectories.',flush=True)
        design=make_design();validate_inputs(design)
        traces=all_native(design);validate_traces(traces)
        kernel=count_channel(KEEP);validate_kernel(kernel)
        laws={};counts={}
        for stage in ('fresh','clean'):laws[stage],counts[stage]=make_laws(traces,stage,kernel)
        validate_laws(laws)
        information,pmfs=independent_metrics(count_histograms(counts['fresh']),kernel)
        clean_information,clean_pmfs=independent_metrics(count_histograms(counts['clean']),kernel)
        ledger=validate_ledgers(traces,laws)
        validate_point_information(information,clean_information,pmfs,clean_pmfs)
        print('Full native/law/ledger comparisons passed; independently reconstructing2000 bootstrap replicates.',flush=True)
        draws,boot,boot_pmfs=all_bootstraps(counts['fresh'],kernel)
        validate_bootstraps(draws,boot,boot_pmfs,information)
        mathematical_invariants(kernel,information,boot,laws,traces,design)
        certificate=independent_certificate(counts['fresh'],information)
        certificate_expected=validate_population_certificate(certificate)
        summary=validate_summary(information,clean_information,pmfs,counts['fresh'],ledger,certificate_expected)
        result['repository_after']=git_status()
        result['repository_status_unchanged']=(result['repository_before']==result['repository_after']
                                              if result['repository_before']['available'] else None)
        sources()
        result['primary_artifacts_unchanged']=result['primary_sha256']=={name:sha(HERE/name) for name in result['primary_sha256']}
        check('all primary files unchanged',result['primary_artifacts_unchanged'])
        check('independent reviewer script unchanged',sha(__file__)==script_sha)
        result['independent_information']={arm:dict(zip(METRIC_NAMES,map(float,information[arm]))) for arm in ARMS}
        result['independent_population_certificate']=certificate_expected
        result['independent_detection_context']=summary['detection_context']
        result['checked_counts']={'fresh_noise_paths':2048,'conditions_per_history':2,
            'fresh_native_trajectories':8192,'clean_native_trajectories':4,'all_native_trajectories':8196,
            'all_native_update_steps':491760,'TAN_history_vectors':245880,'TAN_history_entries':2458800,
            'fresh_output_law_rows':16384,'clean_output_law_rows':8,'fresh_ledger_rows':16384,'clean_ledger_rows':8,
            'point_information_terms':52,'clean_information_terms':52,'bootstrap_replicates':2000,
            'bootstrap_index_entries':4096000,'bootstrap_metric_entries':104000,
            'bootstrap_conditional_pmf_entries':128000,'native_population_histograms':4,'native_CP_error_cells':4,
            'native_models_joint_certificate':2,'population_certificate_inequalities':6}
        result['status']='PASS'
        print('PASS:8196 native trajectories;16384 fresh law/ledger rows;104000 bootstrap metric values;two native joint population certificates.',flush=True)
    except Exception as exc:
        result['status']='FAIL';result['exception']=repr(exc);result['traceback']=traceback.format_exc()
        print(result['traceback'],flush=True)
    finally:
        result['completed_utc']=datetime.now(timezone.utc).isoformat()
        result['checks']=CHECKS;result['comparisons']=COMPARISONS
        result['check_count']=len(CHECKS);result['comparison_count']=len(COMPARISONS)
        result['max_abs_float_residual']=max((r['max_abs_error'] or 0. for r in COMPARISONS),default=0.)
        result['max_native_float_residual']=max((r['max_abs_error'] or 0. for r in COMPARISONS if 'full native' in r['name']),default=0.)
        result['prior_attempt_files']=[p.name for p in sorted(HERE.glob('review_attempt_*.json'))]
        json_write(prefix.with_suffix('.json'),result)
        json_write(HERE/'INDEPENDENT_REVIEW_CHECKS.json',result)
        prefix.with_suffix('.log').write_text('\n'.join([f"status={result['status']}",f'attempt={attempt}',
            f'started={started}',f"completed={result['completed_utc']}",f"checks={len(CHECKS)}",
            f'script_sha256={script_sha}',result.get('traceback','')])+'\n')
        if result['status']=='PASS':
            (HERE/'REVIEW.md').write_text(write_report(result))
            print('Maximum absolute comparison residual',result['max_abs_float_residual'],flush=True)
        else:
            (HERE/'REVIEW.md').write_text('# Independent runtime review pending\n\n'
                f"Attempt{attempt}: `{result.get('exception')}`. Exact script, traceback and all completed checks are retained in `{prefix.name}` files. No primary outputs or protocol were edited.\n")
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
