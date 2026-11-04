#!/usr/bin/env python3
"""Independent final v11 validation. No primary simulator or metric imports.

Uses this reviewer's earlier independently written TAN recurrence, plus a
separately implemented vectorized EMA/current/spike baseline and diagnostics.
No new experiment or parameter search. Stop after this frozen review.
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

HERE=Path(__file__).resolve().parent
V9=HERE.parent/'noise_preservation_v9'
V10=HERE.parent/'noise_transfer_v10'
REPO=Path(os.environ.get('TAN_REPO_ROOT',str(HERE.parents[1]/'Temporal-Attention-Neuron')))
PROTOCOL_HASH='1d4a98c9c4b26583b3ab7d388659c4144c585756deebb29214137fdab600d362'
COMMIT='85213f0d417d54e4f48b301ed1c0e4200ef3c421'
NTRAIN=512
NTEST=1024
TOL=1e-12
MODELS=('TAN','ADAPT')
CONDITIONS=('step','catch')
REGIMES=('iid','ar09')
STAGE_REGIMES={'train':('iid',),'test':REGIMES,'clean':('clean',)}
PRIMARY_CONTRASTS=('Delta_AR','Interaction')
RESOURCE_FIELDS=('sum_abs_current','sum_current_squared','sum_candidate_squared','peak_abs_current',
                 'peak_candidate','peak_abs_candidate','peak_membrane','peak_abs_membrane','aux_state_peak_abs')
LEDGER_FIELDS=('history_id','regime','condition','S','arm','detector_positive','preonset_alarm',
               'first_onset_delay_capped','total_count','onset_count','late_count','R',
               'sum_abs_input','sum_input_squared','sum_noise_squared')+RESOURCE_FIELDS
PAIR_FIELDS=('history_id','regime','arm','D_step','D_catch','BA_contribution','preonset_alarm',
             'step_capped_latency','step_hit','conditional_hit_latency')
CLASS_METRICS=('total_count','onset_count','late_count','R','noisy_minus_clean_R')+RESOURCE_FIELDS
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


def source_validation():
    check('v11 frozen protocol hash',sha(HERE/'PREREGISTRATION.md')==PROTOCOL_HASH)
    original=json.loads((V9/'NOISE_SOURCE_MANIFEST.json').read_text())
    for relative,spec in original['source_files'].items():
        check('mandatory original source SHA256 '+relative,sha(REPO/relative)==spec['sha256'])
    context=json.loads((HERE/'SOURCE_HASHES.json').read_text())
    for relative,digest in context['sha256'].items():
        path=REPO/relative.split('/',1)[1] if relative.startswith('Temporal-Attention-Neuron/') else HERE.parents[1]/relative
        check('v11 source-context SHA256 '+relative,sha(path)==digest)
    native=(REPO/'snapshot/code/experiments/statistical_validation.py').read_text()
    nodes={n.name:n for n in ast.parse(native).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    copied=V9/'taskb_original_extracted.py'
    check('verified source extraction SHA256',sha(copied)==original['extracted_sha256'])
    extracted={n.name:n for n in ast.parse(copied.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    for info in original['selected_source_nodes']:
        name=info['name'];body=ast.get_source_segment(native,nodes[name])
        check('exact native source segment '+name,hashlib.sha256(body.encode()).hexdigest()==info['exact_segment_sha256'])
        check('exact extracted native AST '+name,ast.dump(nodes[name],include_attributes=False)==ast.dump(extracted[name],include_attributes=False))
    prior_hashes={}
    for label,folder in [('v9',V9),('v10',V10)]:
        manifest=json.loads((folder/'PRIMARY_MANIFEST.json').read_text())
        check(label+' completed preserved primary',manifest['status']=='COMPLETED')
        for name,spec in manifest['files'].items():
            value=sha(folder/name);prior_hashes[label+'/'+name]=value
            check(label+' prior primary artifact unchanged '+name,value==spec['sha256'])
        check(label+' independently validated',json.loads((folder/'INDEPENDENT_REVIEW_CHECKS.json').read_text())['status']=='PASS')
    return prior_hashes


def inputs_independent():
    base=np.zeros((2,60));base[0,15:]=1.5
    train_noise=np.random.RandomState(2026110250).normal(0.,.4,size=(NTRAIN,60))
    return {'base':base,'train_noise':train_noise[:,None,:],
            'train_waveforms':np.maximum(train_noise[:,None,None,:]+base[None,None,:,:],0.),
            'clean_waveforms':base[None,None,:,:].copy()}


def add_test_inputs(inputs):
    z=np.random.RandomState(2026110251).standard_normal((NTEST,60))
    ar=np.empty_like(z);ar[:,0]=.4*z[:,0]
    scale=.4*math.sqrt(1-.9**2)
    for t in range(1,60):ar[:,t]=.9*ar[:,t-1]+scale*z[:,t]
    noise=np.stack([.4*z,ar],axis=1)
    inputs.update(innovations=z,test_noise=noise,
                  test_waveforms=np.maximum(noise[:,:,None,:]+inputs['base'][None,None,:,:],0.))


def native_tan(waveforms):
    shape=waveforms.shape
    raw=batch_native(waveforms.reshape((-1,60)),'TAN')
    return {name:value.reshape((*shape[:-1],*value.shape[1:])) for name,value in raw.items()}


def adaptation_reference(waveforms):
    mean=np.zeros(waveforms.shape[:-1]);means=np.empty_like(waveforms);deviation=np.empty_like(waveforms)
    for t in range(60):
        mean=.8*mean+.2*waveforms[...,t]
        means[...,t]=mean
        deviation[...,t]=(9/8)*(waveforms[...,t]-mean)
    return {'reference':means,'deviation':deviation,'unscaled_current':abs(deviation)}


def adaptation_native(waveforms,kappa,reference=None):
    ref=adaptation_reference(waveforms) if reference is None else reference
    current=kappa*ref['unscaled_current']
    membrane=np.zeros(waveforms.shape[:-1]);candidates=np.empty_like(waveforms);states=np.empty_like(waveforms)
    spikes=np.empty(waveforms.shape,dtype=np.uint8)
    for t in range(60):
        candidate=.8*membrane+current[...,t]
        fired=candidate>1.
        membrane=np.where(fired,0.,candidate)
        candidates[...,t]=candidate;states[...,t]=membrane;spikes[...,t]=fired
    return {**ref,'current':current,'candidate':candidates,'membrane':states,'spike':spikes}


def reconstruct_models(inputs):
    train_tan=native_tan(inputs['train_waveforms'])
    reference=adaptation_reference(inputs['train_waveforms'])
    numerator=math.fsum(abs(train_tan['current']).ravel(order='C'))
    denominator=math.fsum(reference['unscaled_current'].ravel(order='C'))
    check('calibration numerator and denominator finite positive',np.isfinite([numerator,denominator]).all() and numerator>0 and denominator>0)
    kappa=numerator/denominator
    check('calibration scalar finite positive',np.isfinite(kappa) and kappa>0)
    train_adapt=adaptation_native(inputs['train_waveforms'],kappa,reference)
    traces={'train':{'TAN':train_tan,'ADAPT':train_adapt}}
    for stage in ('clean',):
        traces[stage]={'TAN':native_tan(inputs[stage+'_waveforms']),
                       'ADAPT':adaptation_native(inputs[stage+'_waveforms'],kappa)}
    return traces,{'kappa':kappa,'numerator':numerator,'denominator':denominator,
                   'ADAPT_training_L1':math.fsum(abs(train_adapt['current']).ravel(order='C')),
                   'budget_residual':math.fsum(abs(train_adapt['current']).ravel(order='C'))-numerator}


def metric_arrays(trace):
    spike=trace['spike'];window=spike[...,15:20];detector=(window.sum(axis=-1)>0)
    delay=np.where(detector,np.argmax(window,axis=-1),5)
    return {'D':detector.astype(np.uint8),'latency_capped':delay,
        'total_count':spike.sum(axis=-1),'onset_count':window.sum(axis=-1),'late_count':spike[...,20:].sum(axis=-1),
        'R':spike[...,20:].mean(axis=-1),'preonset_count':spike[...,:15].sum(axis=-1),
        'preonset_alarm':spike[...,:15].any(axis=-1).astype(np.uint8),
        'sum_abs_current':abs(trace['current']).sum(axis=-1),'sum_current_squared':np.square(trace['current']).sum(axis=-1),
        'sum_candidate_squared':np.square(trace['candidate']).sum(axis=-1),
        'peak_abs_current':abs(trace['current']).max(axis=-1),'peak_candidate':trace['candidate'].max(axis=-1),
        'peak_abs_candidate':abs(trace['candidate']).max(axis=-1),'peak_membrane':trace['membrane'].max(axis=-1),
        'peak_abs_membrane':abs(trace['membrane']).max(axis=-1),
        'aux_state_peak_abs':abs(trace['history']).max(axis=(-2,-1)) if 'history' in trace else abs(trace['reference']).max(axis=-1)}


def invariants(inputs,traces,calibration,metrics):
    compare('AR stationary initial draw equals iid first draw',inputs['test_noise'][:,1,0],inputs['test_noise'][:,0,0],exact=True)
    compare('AR all recurrences',inputs['test_noise'][:,1,1:],.9*inputs['test_noise'][:,1,:-1]+.4*math.sqrt(1-.9**2)*inputs['innovations'][:,1:],exact=True)
    compare('fixed signal absolute sum67.5',inputs['base'][0].sum(),67.5,exact=True)
    compare('fixed signal squared sum101.25',np.square(inputs['base'][0]).sum(),101.25,exact=True)
    compare('AR theoretical stationary marginal variance',.9**2*.16+(.4*math.sqrt(1-.9**2))**2,.16)
    compare('training calibrated currentL1 match',calibration['ADAPT_training_L1'],calibration['numerator'])
    for stage,models in traces.items():
        wave=inputs[stage+'_waveforms']
        for model,t in models.items():
            label=stage+'/'+model
            check(label+' strict threshold',np.array_equal(t['spike'],t['candidate']>1.))
            compare(label+' reset state',t['membrane'],np.where(t['spike'],0.,t['candidate']),exact=True)
            previous=np.concatenate([np.zeros_like(t['membrane'][...,:1]),t['membrane'][...,:-1]],axis=-1)
            compare(label+' old membrane recurrence',t['candidate'],.8*previous+t['current'],exact=True)
            if model=='TAN':
                compare(label+' native current-inclusive history',t['history'][...,-1],wave,exact=True)
                compare(label+' absolute-surprise gate',t['gate'],np.tanh(abs(t['surprise'])),exact=True)
            else:
                previous=np.concatenate([np.zeros_like(t['reference'][...,:1]),t['reference'][...,:-1]],axis=-1)
                compare(label+' leaky mean recurrence not reset on spikes',t['reference'],.8*previous+.2*wave,exact=True)
                compare(label+' updated mean in contrast',t['deviation'],(9/8)*(wave-t['reference']),exact=True)
                compare(label+' frozen calibrated full-wave current',t['current'],calibration['kappa']*abs(t['deviation']),exact=True)
            # condition is always the penultimate waveform axis, before time.
            for field,value in t.items():
                time_axis=-2 if field in ('history','attention') else -1
                if time_axis==-2:
                    a=value[...,0,:15,:];b=value[...,1,:15,:]
                else:
                    a=value[...,0,:15];b=value[...,1,:15]
                compare(label+' paired pre-onset identity '+field,a,b,exact=True)
            m=metrics[stage][model]
            compare(label+' pre-onset alarm counted once',m['preonset_alarm'][...,0],m['preonset_alarm'][...,1],exact=True)
            check(label+' capped latency misses retained',bool((m['latency_capped'][m['D']==0]==5).all()))
            check(label+' delay within fixed window',bool(((m['latency_capped']>=0)&(m['latency_capped']<=5)).all()))
    d=traces['clean']['ADAPT']['deviation'][0,0,0]
    expected=np.zeros(60);expected[15:]=1.35*.8**np.arange(45)
    compare('clean ADAPT exact geometric contrast',d,expected)
    check('clean ADAPT adaptation retained through spikes',bool((traces['clean']['ADAPT']['reference'][0,0,0,15:]>0).all()))
    for model in MODELS:check('clean catch silent '+model,not traces['clean'][model]['spike'][0,0,1].any())


def own_paired_arrays(metrics):
    paired={}
    for model in MODELS:
        m=metrics['test'][model]
        step=m['D'][:,:,0].astype(float);catch=m['D'][:,:,1].astype(float)
        latency=m['latency_capped'][:,:,0]
        paired[model]={'TPR':step,'FPR':catch,'specificity':1.-catch,'BA':.5*(step+1.-catch),
            'preonset_alarm':m['preonset_alarm'][:,:,0],
            'latency_capped':latency,'latency_hit_numerator':latency*step,'latency_hit_denominator':step,
            'latency_distribution':np.stack([latency==d for d in range(6)],axis=-1).astype(float),
            'latency_cdf':np.stack([latency<=d for d in range(5)],axis=-1).astype(float)}
    differences=paired['TAN']['BA']-paired['ADAPT']['BA']
    return paired,{'iid_difference':differences[:,0],'AR_difference':differences[:,1],
                   'interaction':differences[:,1]-differences[:,0]}


def validate_native_traces(stage,traces,regimes):
    expected_fields={'conditions','history_ids','regimes','S'}
    tan_fields={'spikes':'spike','candidate':'candidate','membrane':'membrane','current':'current',
        'history':'history','surprise':'surprise','gate':'gate','context':'context','attention':'attention',
        'exact_class_spikes':'spike','exact_class_membrane':'membrane','exact_class_history':'history'}
    adapt_fields={'spikes':'spike','candidate':'candidate','membrane':'membrane','current':'current',
        'reference':'reference','unscaled_signed_contrast':'deviation','unscaled_magnitude':'unscaled_current'}
    with np.load(HERE/(stage+'_trajectories.npz'),allow_pickle=False) as saved:
        compare(stage+' explicit trace class order',saved['conditions'],CONDITIONS,exact=True)
        compare(stage+' explicit trace regimes',saved['regimes'],regimes,exact=True)
        compare(stage+' explicit trace S labels',saved['S'],[1,0],exact=True)
        count=traces['TAN']['spike'].shape[0]
        compare(stage+' explicit trajectory IDs',saved['history_ids'],np.arange(count),exact=True)
        for model,mapping in [('TAN',tan_fields),('ADAPT',adapt_fields)]:
            for field,key in mapping.items():
                name=model+'__'+field;expected_fields.add(name)
                compare(stage+' full trajectory '+name,saved[name],traces[model][key],exact=key in ('spike','history'))
        expected_fields.add('TAN__query')
        compare(stage+' full trajectory TAN__query',saved['TAN__query'],2.*traces['TAN']['surprise'])
        check(stage+' exact trajectory schema',set(saved.files)==expected_fields,saved.files)


def compare_npz_fields(filename,expected,exact=False):
    with np.load(HERE/filename,allow_pickle=False) as saved:
        check(filename+' exact fields',set(saved.files)==set(expected),saved.files)
        for name,value in expected.items():compare(filename+'/'+name,saved[name],value,exact=exact)


def validate_clean_against_v9(traces):
    names={'spikes':'spike','candidate':'candidate','membrane':'membrane','current':'current','history':'history',
           'surprise':'surprise','gate':'gate','context':'context','attention':'attention'}
    with np.load(V9/'clean_trajectories.npz',allow_pickle=False) as saved:
        for suffix,key in names.items():
            compare('clean nativeTAN unchanged fromv9 '+suffix,saved['TAN__'+suffix],traces['TAN'][key][:,0],
                    exact=key in ('spike','history'))


def input_bundles(inputs):
    bundles={}
    for stage in ('train','test','clean'):
        if stage+'_waveforms' not in inputs:continue
        wave=inputs[stage+'_waveforms']
        noise=inputs[stage+'_noise'] if stage!='clean' else np.zeros((1,1,60))
        b={'noise':noise,'base':inputs['base'],'waveforms':wave,
           'history_ids':np.arange(len(wave)),'regimes':np.array(STAGE_REGIMES[stage]),
           'conditions':np.array(CONDITIONS),'S':np.array([1,0])}
        if stage=='test':b['Z']=inputs['innovations']
        bundles[stage]=b
    return bundles


def validate_calibration(calibration):
    expected={'training_seed':2026110250,'n_training_histories':512,'balanced_classes':True,
        'current_samples_per_model':512*2*60,'gain_kappa':calibration['kappa'],
        'numerator_TAN_current_L1':calibration['numerator'],
        'denominator_unscaled_ADAPT_current_L1':calibration['denominator'],
        'calibrated_ADAPT_current_L1':calibration['ADAPT_training_L1'],
        'raw_budget_residual':calibration['budget_residual'],
        'raw_budget_residual_per_current_sample':calibration['budget_residual']/(512*2*60),
        'sum_convention':'math.fsum over C-order float64 entries','gain_finite_positive':True,
        'gain_fit_uses_spikes_accuracy_latency_or_test':False,'test_inputs_generated_yet':False,
        'training_inputs_sha256':sha(HERE/'train_inputs.npz'),
        'training_trajectories_sha256':sha(HERE/'train_trajectories.npz')}
    compare_json_tree('training-only fixed gain and budget',json.loads((HERE/'CALIBRATION_FROZEN.json').read_text()),expected)
    return expected


def reconstructed_ledgers(stage,bundle,metrics):
    rows={};paired={}
    n=len(bundle['history_ids'])
    for r,regime in enumerate(bundle['regimes']):
        for model in MODELS:
            m=metrics[model]
            for i in range(n):
                ds,dc=[int(m['D'][i,r,c]) for c in range(2)]
                lat=int(m['latency_capped'][i,r,0]);pre=int(m['preonset_alarm'][i,r,0])
                paired[(i,str(regime),model)]={'history_id':i,'regime':str(regime),'arm':model,
                    'D_step':ds,'D_catch':dc,'BA_contribution':.5*(ds+1-dc),'preonset_alarm':pre,
                    'step_capped_latency':lat,'step_hit':ds,'conditional_hit_latency':lat if ds else None}
                for c,condition in enumerate(CONDITIONS):
                    wave=bundle['waveforms'][i,r,c];noise=bundle['noise'][i,r]
                    row={'history_id':i,'regime':str(regime),'condition':condition,'S':1-c,'arm':model,
                        'detector_positive':int(m['D'][i,r,c]),'preonset_alarm':int(m['preonset_alarm'][i,r,c]),
                        'first_onset_delay_capped':int(m['latency_capped'][i,r,c]),
                        'total_count':int(m['total_count'][i,r,c]),'onset_count':int(m['onset_count'][i,r,c]),
                        'late_count':int(m['late_count'][i,r,c]),'R':float(m['R'][i,r,c]),
                        'sum_abs_input':float(abs(wave).sum()),'sum_input_squared':float(np.square(wave).sum()),
                        'sum_noise_squared':float(np.square(noise).sum())}
                    row.update({field:float(m[field][i,r,c]) for field in RESOURCE_FIELDS})
                    rows[(i,str(regime),condition,model)]=row
    return rows,paired


def compare_ledger_csv(filename,fields,expected,key_fields):
    with (HERE/filename).open(newline='') as stream:
        reader=csv.DictReader(stream)
        check(filename+' exact columns',reader.fieldnames==list(fields),reader.fieldnames)
        actual_rows=list(reader)
    def key(row):return tuple(int(row[f]) if f=='history_id' else row[f] for f in key_fields)
    actual={key(row):row for row in actual_rows}
    check(filename+' no exclusions or duplicate rows',len(actual_rows)==len(expected) and len(actual)==len(expected),len(actual_rows))
    check(filename+' all experimental groups',set(actual)==set(expected))
    order=sorted(expected)
    for field in fields:
        notnull=[k for k in order if expected[k][field] is not None]
        check(filename+' preserve undefined '+field,all(actual[k][field]=='' for k in order if expected[k][field] is None))
        numeric=field not in ('regime','condition','arm')
        a=[actual[k][field] for k in notnull]
        if numeric:a=[float(x) for x in a]
        e=[expected[k][field] for k in notnull]
        exact=(not numeric) or field in ('history_id','S','D_step','D_catch','step_hit','detector_positive',
            'preonset_alarm','first_onset_delay_capped','step_capped_latency','conditional_hit_latency',
            'total_count','onset_count','late_count','BA_contribution')
        compare(filename+'/'+field,a,e,exact=exact)


def validate_ledgers(bundles,metrics):
    tables={}
    for stage,bundle in bundles.items():
        rows,pairs=reconstructed_ledgers(stage,bundle,metrics[stage]);tables[stage]=(rows,pairs)
        compare_ledger_csv(stage+'_ledger.csv',LEDGER_FIELDS,rows,('history_id','regime','condition','arm'))
        compare_ledger_csv(stage+'_paired_history_metrics.csv',PAIR_FIELDS,pairs,('history_id','regime','arm'))
    return tables


def build_samples_independent(inputs,metrics,paired,differences):
    samples={};ratios={}
    for r,regime in enumerate(REGIMES):
        for model in MODELS:
            prefix=regime+'__'+model+'__';p=paired[model]
            for name in ('TPR','FPR','specificity','BA','preonset_alarm'):
                samples[prefix+name]=p[name][:,r].astype(float)
            samples[prefix+'capped_latency']=p['latency_capped'][:,r].astype(float)
            for delay in range(6):samples[prefix+'delay_p'+str(delay)]=p['latency_distribution'][:,r,delay]
            for delay in range(5):samples[prefix+'hit_by_'+str(delay)]=p['latency_cdf'][:,r,delay]
            ratios[prefix+'conditional_hit_latency']=(p['latency_hit_numerator'][:,r],p['latency_hit_denominator'][:,r])
            for c,condition in enumerate(CONDITIONS):
                for field in CLASS_METRICS:
                    if field=='noisy_minus_clean_R':
                        value=metrics['test'][model]['R'][:,r,c]-metrics['clean'][model]['R'][0,0,c]
                    else:value=metrics['test'][model][field][:,r,c]
                    samples[prefix+condition+'__'+field]=value.astype(float)
        for c,condition in enumerate(CONDITIONS):
            wave=inputs['test_waveforms'][:,r,c]
            samples['input__'+regime+'__'+condition+'__sum_abs_input']=abs(wave).sum(axis=-1)
            samples['input__'+regime+'__'+condition+'__sum_input_squared']=np.square(wave).sum(axis=-1)
        samples['input__'+regime+'__sum_noise_squared']=np.square(inputs['test_noise'][:,r]).sum(axis=-1)
    samples['contrast__Delta_AR']=differences['AR_difference']
    samples['contrast__Interaction']=differences['interaction']
    samples['contrast__Delta_iid']=differences['iid_difference']
    for r,regime in enumerate(REGIMES):
        for field in ('TPR','FPR','capped_latency','preonset_alarm'):
            samples['contrast__'+regime+'__'+field+'_gap']=samples[regime+'__TAN__'+field]-samples[regime+'__ADAPT__'+field]
        samples['contrast__'+regime+'__step_R_gap']=metrics['test']['TAN']['R'][:,r,0]-metrics['test']['ADAPT']['R'][:,r,0]
        samples['contrast__'+regime+'__balanced_current_L1_gap']=.5*sum(
            metrics['test']['TAN']['sum_abs_current'][:,r,c]-metrics['test']['ADAPT']['sum_abs_current'][:,r,c] for c in range(2))
    samples['contrast__input_AR_minus_iid_noise_squared']=np.square(inputs['test_noise'][:,1]).sum(axis=-1)-np.square(inputs['test_noise'][:,0]).sum(axis=-1)
    for c,condition in enumerate(CONDITIONS):
        samples['contrast__input_AR_minus_iid_'+condition+'_squared']=np.square(inputs['test_waveforms'][:,1,c]).sum(axis=-1)-np.square(inputs['test_waveforms'][:,0,c]).sum(axis=-1)
    return samples,ratios


def independent_bootstrap(samples,ratios):
    names=list(samples)+list(ratios)
    means=np.column_stack(list(samples.values()))
    numerators=np.column_stack([x for x,y in ratios.values()])
    denominators=np.column_stack([y for x,y in ratios.values()])
    draws=np.random.RandomState(2026110252).randint(0,NTEST,size=(2000,NTEST))
    point=np.r_[means.mean(axis=0),np.divide(numerators.sum(axis=0),denominators.sum(axis=0),
                    out=np.full(len(ratios),np.nan),where=denominators.sum(axis=0)>0)]
    values=np.empty((2000,len(names)))
    # Bounded chunks use the same fixed draw matrix; no stochastic refit of gain.
    for r,idx in enumerate(draws):
        values[r,:len(samples)]=means[idx].mean(axis=0)
        num=numerators[idx].sum(axis=0);den=denominators[idx].sum(axis=0)
        values[r,len(samples):]=np.divide(num,den,out=np.full(len(ratios),np.nan),where=den>0)
    summary={}
    for j,name in enumerate(names):
        finite=np.isfinite(values[:,j]);primary=name in ('contrast__Delta_AR','contrast__Interaction')
        summary[name]={'estimate':float(point[j]) if np.isfinite(point[j]) else None,
            'interval_level':.975 if primary else .95,
            'percentile_interval':np.quantile(values[finite,j],[.0125,.9875] if primary else [.025,.975]).tolist() if finite.any() else None,
            'defined_bootstrap_replicates':int(finite.sum()),'undefined_bootstrap_replicates':int((~finite).sum())}
    return {'names':names,'history_means':means,'conditional_names':list(ratios),'numerators':numerators,
            'denominators':denominators,'draws':draws,'values':values,'point':point,'summary':summary}


def compare_allow_undefined(name,actual,expected):
    actual=np.asarray(actual,dtype=float);expected=np.asarray(expected,dtype=float)
    compare(name+' undefined mask',np.isnan(actual),np.isnan(expected),exact=True)
    check(name+' no infinite values',not np.isinf(actual).any() and not np.isinf(expected).any())
    valid=np.isfinite(expected)
    compare(name+' finite values',actual[valid],expected[valid])


def validate_statistics(samples,boot):
    expected={'metric_names':np.array(list(samples)),'values':boot['history_means'],
        'conditional_names':np.array(boot['conditional_names']),'conditional_numerators':boot['numerators'],
        'conditional_denominators':boot['denominators']}
    compare_npz_fields('history_metric_samples.npz',expected)
    compare('all2000 paired innovation bootstrap draws',np.load(HERE/'bootstrap_draws.npy',allow_pickle=False),boot['draws'],exact=True)
    with np.load(HERE/'bootstrap_metrics.npz',allow_pickle=False) as saved:
        check('bootstrap NPZ exact fields',set(saved.files)=={'metric_names','values'})
        compare('bootstrap metric exact ordering',saved['metric_names'],boot['names'],exact=True)
        compare_allow_undefined('all2000x212 fixed bootstrap metrics',saved['values'],boot['values'])
    with (HERE/'bootstrap_metrics.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        check('bootstrap CSV exact columns',reader.fieldnames==['replicate',*boot['names']])
        rows=list(reader)
    compare('bootstrap CSV complete replicateIDs',[int(r['replicate']) for r in rows],np.arange(2000),exact=True)
    actual=np.array([[float(row[name]) if row[name] else np.nan for name in boot['names']] for row in rows])
    compare_allow_undefined('full bootstrap CSV matches independent',actual,boot['values'])
    compare_json_tree('all212 primary/secondary intervals and undefined counts',
                      json.loads((HERE/'metric_summary.json').read_text()),boot['summary'])
    with (HERE/'bootstrap_primary_contrasts.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        check('primary contrast CSV exact columns',reader.fieldnames==['replicate',*PRIMARY_CONTRASTS])
        rows=list(reader)
    compare('primary contrast complete replicateIDs',[int(r['replicate']) for r in rows],np.arange(2000),exact=True)
    for name in PRIMARY_CONTRASTS:
        j=boot['names'].index('contrast__'+name)
        compare('all2000 primary replicate '+name,[float(r[name]) for r in rows],boot['values'][:,j])
    check('only two97.5percent primary intervals',
          {name for name,s in boot['summary'].items() if s['interval_level']==.975}=={'contrast__Delta_AR','contrast__Interaction'})


def independent_clean_algebra(traces):
    tan=traces['TAN'];ad=traces['ADAPT'];delta=1.5
    box_expected=np.zeros(60);box_expected[15:24]=delta*np.arange(9,0,-1)/10.
    ad_expected=np.zeros(60);ad_expected[15:]=.9*delta*.8**np.arange(45)
    coeff=np.array([.9]+[-.1]*9)
    covariance=.9**abs(np.arange(10)[:,None]-np.arange(10)[None,:])
    expected={'boxcar_clean_surprise_max_abs_error':float(abs(tan['surprise'][0,0,0]-box_expected).max()),
        'ADAPT_clean_contrast_max_abs_error':float(abs(ad['deviation'][0,0,0]-ad_expected).max()),
        'clean_native_v9_max_abs_error':0.,'first_signed_transient_target':.9*delta,
        'infinite_signed_transient_area_target':4.5*delta,
        'ADAPT_finite_signed_transient_sum':math.fsum(ad['deviation'][0,0,0]),
        'ADAPT_finite_signed_transient_formula':4.5*delta*(1-.8**45),
        'stationary_unclipped_theory':{'white_noise_variance_factor_TAN':float(coeff@coeff),
            'white_noise_variance_factor_ADAPT':.9,'ar09_variance_factor_TAN':float(coeff@covariance@coeff),
            'ar09_variance_factor_ADAPT':.9*(1-.9)/(1-.8*.9)}}
    compare_json_tree('clean/algebraic stationary filter checks',json.loads((HERE/'clean_checks.json').read_text()),expected)
    return expected


def validate_summary(calibration,cal_schema,clean_checks,boot):
    primary={name:boot['summary']['contrast__'+name] for name in PRIMARY_CONTRASTS}
    supported=all(value['percentile_interval'][0]>0 for value in primary.values())
    expected={'protocol_sha256':PROTOCOL_HASH,'gain_kappa':calibration['kappa'],
        'n_training_histories':512,'n_test_innovation_histories':1024,
        'test_seed':2026110251,'training_seed':2026110250,'bootstrap_seed':2026110252,'bootstrap_replicates':2000,
        'regime_order':list(REGIMES),'condition_order':list(CONDITIONS),'S':[1,0],'arms':list(MODELS),
        'model_parameters':{'ADAPT':{'EMA_old_weight':.8,'EMA_input_weight':.2,'signed_contrast_scale':9/8,
            'rectifier':'absolute value','membrane_leak':.8,'threshold':1.,'strict_greater_than':True,
            'initial_reference':0.,'initial_membrane':0.}},
        'calibration':cal_schema,'no_exclusions':True,'clean_algebra_checks':clean_checks,
        'test_ledger_rows':8192,'paired_history_rows':4096,'fixed_metric_count':len(boot['names']),
        'primary_contrasts':primary,'both_primary_lower_bounds_above_zero':supported,
        'primary_direction_disfavored':{name:value['percentile_interval'][1]<=0 for name,value in primary.items()},
        'nominal_signal_sum':67.5,'nominal_signal_squared_sum':101.25,'expected_noise_squared_sum':9.6,
        'deterministic_clean_control_count':4,'onset_known_externally':True,'latency_miss_penalty':5,
        'inference_conditional_on_frozen_calibration':True}
    compare_json_tree('primary entire fixed scientific summary',json.loads((HERE/'primary_results.json').read_text()),expected)
    return expected


def primary_provenance():
    manifest=json.loads((HERE/'PRIMARY_MANIFEST.json').read_text())
    check('primary execution completed',manifest['status']=='COMPLETED')
    check('primary frozen protocol SHA256',manifest['protocol_sha256']==PROTOCOL_HASH)
    check('primary executing script SHA256',manifest['primary_script_sha256']==sha(HERE/'run_gate_mechanism.py'))
    check('primary exact axes/labels',manifest['condition_order']==list(CONDITIONS) and manifest['S']==[1,0]
          and manifest['regime_order']==list(REGIMES))
    check('primary ledger schema',manifest['ledger_fields']==list(LEDGER_FIELDS) and manifest['paired_ledger_fields']==list(PAIR_FIELDS))
    check('primary only two named contrasts',manifest['primary_contrasts']==list(PRIMARY_CONTRASTS))
    for name,record in manifest['files'].items():
        check('primary declared SHA256 '+name,sha(HERE/name)==record['sha256'])
        check('primary declared byte count '+name,(HERE/name).stat().st_size==record['bytes'])
    for filename,fields in manifest['npz_schema'].items():
        with np.load(HERE/filename,allow_pickle=False) as saved:
            check(filename+' declared schema keys',set(saved.files)==set(fields))
            for name,spec in fields.items():
                a=saved[name]
                check(filename+'/'+name+' declared shape',list(a.shape)==spec['shape'])
                check(filename+'/'+name+' declared dtype',str(a.dtype)==spec['dtype'])
    calibration=json.loads((HERE/'CALIBRATION_FROZEN.json').read_text())
    times={}
    for stage in ('train','test','clean'):
        frozen=json.loads((HERE/(stage.upper()+'_INPUTS_FROZEN.json')).read_text())
        check(stage+' inputs marked before model evaluation',frozen['models_evaluated_yet'] is False)
        check(stage+' frozen input SHA256',frozen['inputs_sha256']==sha(HERE/frozen['input_file']))
        with np.load(HERE/frozen['input_file'],allow_pickle=False) as saved:
            declarations=frozen['array_manifest']
            check(stage+' frozen raw-array keys',set(saved.files)==set(declarations))
            for field,spec in declarations.items():
                value=saved[field]
                check(stage+'/'+field+' frozen raw SHA256',hashlib.sha256(value.tobytes(order='C')).hexdigest()==spec['raw_c_order_sha256'])
                check(stage+'/'+field+' frozen array shape/dtype',list(value.shape)==spec['shape'] and str(value.dtype)==spec['dtype'])
        times[stage]=frozen['time_utc']
        if stage=='test':check('test freezes reference already frozen gain',frozen['calibration_frozen_sha256']==sha(HERE/'CALIBRATION_FROZEN.json'))
    check('training input then calibration then test input ordering',times['train']<calibration['time_utc']<times['test']<manifest['completed_utc'])
    check('clean input marker precedes completion',times['clean']<manifest['completed_utc'])
    guard=json.loads((HERE/'SOURCE_GUARD.json').read_text())
    check('sourceguard only authorized native model',guard['selected_native_models']==['TAN'])
    check('sourceguard protocol',guard['protocol_sha256']==PROTOCOL_HASH)
    for relative,digest in guard['source_files_sha256'].items():check('sourceguard source SHA256 '+relative,sha(REPO/relative)==digest)
    for name,digest in guard['prior_artifact_sha256'].items():check('sourceguard prior artifact SHA256 '+name,sha(V9/name)==digest)
    names=set(manifest['files'])|{'PRIMARY_MANIFEST.json','run_gate_mechanism.py','PREREGISTRATION.md','SOURCE_HASHES.json'}
    return {name:sha(HERE/name) for name in sorted(names)}


def write_review(result):
    p=result['independent_primary_contrasts']
    lines=['# Independent review of final v11 round','','Status: **PASS**.', '',
        'Reconstructed original TAN using the reviewer\'s own earlier independent recurrence and implemented the frozen ADAPT baseline separately. No primary simulation or metric helper was imported. Used full-wave magnitude, pole.8, contrast scale9/8 and common leak.8/threshold1; no earlier alternative baseline was substituted.', '',
        f"Protocol SHA256: `{PROTOCOL_HASH}`.",f"Reviewer SHA256: `{result['independent_script_sha256']}`.",'',
        'Verified512 paired training histories,1024 held-out innovation histories in both noise regimes/classes/models, all four clean controls, all full trajectories, all per-trial/paired ledgers, all212 metric definitions, and all2,000 bootstrap rows. The gain was independently reconstructed from training currents only with math.fsum before test innovations were generated. AR initial noise is stationary, not zero. Baseline adaptation state is retained through spikes.', '',
        f"Independent gain: `{result['independent_calibration']['kappa']:.17g}`. Training current-L1 residual: `{result['independent_calibration']['budget_residual']:.17g}`.",
        f"Largest native/adaptive trajectory residual: `{result['max_trajectory_float_residual']:.17g}`; largest numerical comparison residual overall: `{result['max_abs_float_residual']:.17g}`.",'',
        '| Frozen primary contrast | Estimate | 97.5% paired percentile interval |',
        '|---|---:|---|']
    for name in PRIMARY_CONTRASTS:
        v=p[name];lo,hi=v['percentile_interval']
        lines.append(f"| {name} | {v['estimate']:.9f} | [{lo:.9f}, {hi:.9f}] |")
    lines+=['',f"Both required lower endpoints exceed zero: `{result['both_primary_lower_bounds_above_zero']}`.",'',
        'The two primary intervals use quantiles.0125/.9875; all other intervals use.025/.975. These remain paired-bootstrap reporting criteria, not an exact finite-sample simultaneous theorem. The frozen calibration is not refit in any bootstrap. The reviewer retained misses with capped latency5 and checked conditional-on-hit latency with its hit coverage separately. Pre-onset alarms are counted once per innovation/model/regime; catch-window false alarms remain separate.', '',
        'A disfavored correlation-advantage direction must be reported with the TPR/FPR, latency and current-budget tradeoffs. This comparison does not isolate TAN\'s gate causally, show universal inferiority/superiority, or match biological energy. Training current-L1 equality does not assert test-current equality. Exact input marginals/nominal step are preserved across regimes; realized finite input energies are allowed to differ.', '',
        f"Primary artifacts unchanged: `{result['primary_artifacts_unchanged']}`. Repository unchanged: `{result['repository_status_unchanged']}`.",
        f"Repository provenance: `{result['repository_after']}`.",'',
        'Completed command:', '',
        '`PYTHONDONTWRITEBYTECODE=1 python tan_audit/gate_mechanism_v11/independent_review.py`', '',
        'Numbered attempts preserve reviewer source snapshots, logs and every completed check. No original source, prior result, protocol or primary output was modified. No remaining blocker.', '',
        'V11 separate computational review completed; this is AI-assisted verification, not external human certification.']
    return '\n'.join(lines)+'\n'


def main():
    attempt=1
    while (HERE/f'review_attempt_{attempt:02d}.json').exists():attempt+=1
    prefix=HERE/f'review_attempt_{attempt:02d}'
    prefix.with_suffix('.script.txt').write_text(Path(__file__).read_text())
    started=datetime.now(timezone.utc).isoformat();script_sha=sha(__file__)
    result={'status':'RUNNING','attempt':attempt,'started_utc':started,
        'protocol_sha256':PROTOCOL_HASH,'independent_script_sha256':script_sha,
        'execution_command':f'PYTHONDONTWRITEBYTECODE=1 python {Path(__file__).resolve()}',
        'python':platform.python_version(),'numpy':np.__version__,
        'no_primary_helper_imports':True,'calibration_reconstructed_before_test_innovations':True,
        'scope':'Final frozen v11 only; stop after independent verification.'}
    try:
        print('Independent finalv11 review started',started,flush=True)
        result['prior_artifact_sha256']=source_validation()
        result['repository_before']=git_status()
        result['primary_sha256']=primary_provenance()
        print('Provenance verified; independently rebuilding training currents and gain.',flush=True)
        inputs=inputs_independent()
        bundles=input_bundles(inputs)
        for stage in ('train','clean'):compare_npz_fields(stage+'_inputs.npz',bundles[stage],exact=True)
        traces,calibration=reconstruct_models(inputs)
        cal_schema=validate_calibration(calibration)
        for stage in ('train','clean'):validate_native_traces(stage,traces[stage],STAGE_REGIMES[stage])
        validate_clean_against_v9(traces['clean'])
        result['independent_gain_fixed_before_test_generation']=dict(calibration)
        print('Training gain reconstructed and frozen; regenerating held-out AR/iid inputs.',flush=True)
        add_test_inputs(inputs)
        bundles=input_bundles(inputs)
        compare_npz_fields('test_inputs.npz',bundles['test'],exact=True)
        traces['test']={'TAN':native_tan(inputs['test_waveforms']),
                        'ADAPT':adaptation_native(inputs['test_waveforms'],calibration['kappa'])}
        validate_native_traces('test',traces['test'],REGIMES)
        metrics={stage:{model:metric_arrays(t) for model,t in stage_traces.items()} for stage,stage_traces in traces.items()}
        invariants(inputs,traces,calibration,metrics)
        tables=validate_ledgers(bundles,metrics)
        clean_checks=independent_clean_algebra(traces['clean'])
        paired,differences=own_paired_arrays(metrics)
        samples,ratios=build_samples_independent(inputs,metrics,paired,differences)
        print('Full traces/gain/ledgers verified; independently rebuilding212 metrics and2000 paired bootstraps.',flush=True)
        boot=independent_bootstrap(samples,ratios)
        validate_statistics(samples,boot)
        summary=validate_summary(calibration,cal_schema,clean_checks,boot)
        result['repository_after']=git_status()
        result['repository_status_unchanged']=(result['repository_before']==result['repository_after'] if result['repository_before']['available'] else None)
        new_prior=source_validation()
        check('all earlier experiment artifacts unchanged throughout review',new_prior==result['prior_artifact_sha256'])
        result['primary_artifacts_unchanged']=result['primary_sha256']=={name:sha(HERE/name) for name in result['primary_sha256']}
        check('all primary artifacts unchanged throughout review',result['primary_artifacts_unchanged'])
        check('reviewer script unchanged throughout run',sha(__file__)==script_sha)
        result['independent_calibration']=calibration
        result['independent_primary_contrasts']=summary['primary_contrasts']
        result['both_primary_lower_bounds_above_zero']=summary['both_primary_lower_bounds_above_zero']
        result['primary_direction_disfavored']=summary['primary_direction_disfavored']
        result['independent_fixed_metric_summary']=boot['summary']
        result['checked_counts']={'training_noise_histories':512,'test_innovation_histories':1024,
            'training_model_trajectories':2048,'test_model_trajectories':8192,'clean_model_trajectories':4,
            'all_model_trajectories':10244,'all_model_update_steps':614640,
            'TAN_history_vectors':307320,'TAN_history_entries':3073200,
            'training_ledger_rows':2048,'test_ledger_rows':8192,'clean_ledger_rows':4,
            'training_paired_rows':1024,'test_paired_rows':4096,'clean_paired_rows':2,
            'fixed_metrics':len(boot['names']),'mean_metric_samples':len(samples)*NTEST,
            'conditional_latency_ratio_series':len(ratios),'bootstrap_replicates':2000,
            'bootstrap_index_entries':2048000,'bootstrap_metric_entries':boot['values'].size,
            'primary_contrasts':2,'primary_bootstrap_entries':4000}
        result['status']='PASS'
        result['stop_status']='Final v11 separate computational review complete.'
        print('PASS:10244 full model trajectories;gain/AR/ledger/212metrics/all2000bootstrap rows verified. STOP aftercurrentround.',flush=True)
    except Exception as exc:
        result['status']='FAIL';result['exception']=repr(exc);result['traceback']=traceback.format_exc()
        print(result['traceback'],flush=True)
    finally:
        result['completed_utc']=datetime.now(timezone.utc).isoformat()
        result['checks']=CHECKS;result['comparisons']=COMPARISONS
        result['check_count']=len(CHECKS);result['comparison_count']=len(COMPARISONS)
        result['max_abs_float_residual']=max((r['max_abs_error'] or 0. for r in COMPARISONS),default=0.)
        result['max_trajectory_float_residual']=max((r['max_abs_error'] or 0. for r in COMPARISONS if 'full trajectory' in r['name']),default=0.)
        result['prior_attempt_files']=[p.name for p in sorted(HERE.glob('review_attempt_*.json'))]
        json_write(prefix.with_suffix('.json'),result);json_write(HERE/'INDEPENDENT_REVIEW_CHECKS.json',result)
        prefix.with_suffix('.log').write_text('\n'.join([f"status={result['status']}",f'attempt={attempt}',f'started={started}',
            f"completed={result['completed_utc']}",f"checks={len(CHECKS)}",f'script_sha256={script_sha}',result.get('traceback','')])+'\n')
        if result['status']=='PASS':
            (HERE/'REVIEW.md').write_text(write_review(result))
            print('Maximum numerical residual',result['max_abs_float_residual'],flush=True)
        else:
            (HERE/'REVIEW.md').write_text('# Finalv11 independent review pending\n\n'
                f"Attempt{attempt}: `{result.get('exception')}`. Exact reviewer source, full traceback and all completed checks remain in `{prefix.name}` files. No primary output or protocol was edited.\n")
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
