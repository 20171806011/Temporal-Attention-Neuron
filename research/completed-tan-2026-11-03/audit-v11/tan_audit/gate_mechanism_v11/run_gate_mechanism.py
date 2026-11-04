#!/usr/bin/env python3
"""Final frozen v11 round. A fresh --output-dir permits artifact replay only.

TAN_REPO_ROOT optionally selects the unchanged source checkout. The default
uses the sibling repository. Source-only copies explicitly lack Git provenance.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import numpy as np

BASE = Path(__file__).resolve().parent
V9 = BASE.parent/'noise_preservation_v9'
PROTOCOL = '1d4a98c9c4b26583b3ab7d388659c4144c585756deebb29214137fdab600d362'
COMMIT = '85213f0d417d54e4f48b301ed1c0e4200ef3c421'
HELPER_HASH = 'bd08b8aeef0ebdef4f69bba1e4efb8d9d4cb916800c0660f60c2bfd59c0e017f'
PRIOR_MANIFEST_HASH = '2eccf54e95bf4fc4b27dc8e5bda506edafb3df0543185f6f2c738c1118f67e93'
ARMS = ('TAN', 'ADAPT')
REGIMES = ('iid', 'ar09')
CONDITIONS = ('step', 'catch')
PRIMARY_CONTRASTS = ('Delta_AR', 'Interaction')
TOL = 1e-12
RESOURCE_FIELDS = ('sum_abs_current', 'sum_current_squared', 'sum_candidate_squared',
    'peak_abs_current', 'peak_candidate', 'peak_abs_candidate', 'peak_membrane',
    'peak_abs_membrane', 'aux_state_peak_abs')
LEDGER_FIELDS = ('history_id', 'regime', 'condition', 'S', 'arm', 'detector_positive',
    'preonset_alarm', 'first_onset_delay_capped', 'total_count', 'onset_count', 'late_count',
    'R', 'sum_abs_input', 'sum_input_squared', 'sum_noise_squared')+RESOURCE_FIELDS
PAIR_FIELDS = ('history_id', 'regime', 'arm', 'D_step', 'D_catch', 'BA_contribution',
    'preonset_alarm', 'step_capped_latency', 'step_hit', 'conditional_hit_latency')
CLASS_METRICS = ('total_count', 'onset_count', 'late_count', 'R', 'noisy_minus_clean_R')+RESOURCE_FIELDS
PAIR_METRICS = ('TPR', 'FPR', 'specificity', 'BA', 'preonset_alarm', 'capped_latency')+tuple(
    'delay_p'+str(j) for j in range(6))+tuple('hit_by_'+str(j) for j in range(5))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def jsave(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def csave(path, fields, rows):
    with Path(path).open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
def total(a): return math.fsum(map(float, np.asarray(a).ravel(order='C')))


def git_provenance(repo):
    if not (repo/'.git').exists(): return {'available': False, 'commit': None, 'status': None}
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    state=subprocess.check_output(['git','-C',str(repo),'status','--porcelain=v1'],text=True)
    assert commit==COMMIT and state=='', (commit,state)
    return {'available': True, 'commit': commit, 'status': state}


def sources(repo, out):
    assert sha(BASE/'PREREGISTRATION.md')==PROTOCOL
    assert sha(V9/'run_noise_preservation.py')==HELPER_HASH
    assert sha(V9/'PRIMARY_MANIFEST.json')==PRIOR_MANIFEST_HASH
    prior=json.loads((V9/'PRIMARY_MANIFEST.json').read_text())
    prior_hashes={}
    for name, record in prior['files'].items():
        prior_hashes[name]=sha(V9/name)
        assert prior_hashes[name]==record['sha256'], name
    assert json.loads((V9/'INDEPENDENT_REVIEW_CHECKS.json').read_text())['status']=='PASS'
    original=json.loads((V9/'original_results.json').read_text())
    assert original['phase_A_completion']['exact_raw_record_matches']==100
    v9manifest=json.loads((V9/'NOISE_SOURCE_MANIFEST.json').read_text())
    source_hashes={}
    for name, record in v9manifest['source_files'].items():
        source_hashes[name]=sha(repo/name)
        assert source_hashes[name]==record['sha256']
    declarations=json.loads((BASE/'SOURCE_HASHES.json').read_text())
    for relative, expected in declarations['sha256'].items():
        assert relative.startswith('Temporal-Attention-Neuron/')
        name=relative.split('/',1)[1]
        source_hashes[name]=sha(repo/name); assert source_hashes[name]==expected
    raw=(repo/'snapshot/code/experiments/statistical_validation.py').read_text()
    extracted=(V9/'taskb_original_extracted.py').read_text()
    assert sha(V9/'taskb_original_extracted.py')==v9manifest['extracted_sha256']
    nodes={n.name:n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    copied={n.name:n for n in ast.parse(extracted).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    for record in v9manifest['selected_source_nodes']:
        name=record['name']; segment=ast.get_source_segment(raw,nodes[name])
        assert segment==ast.get_source_segment(extracted,copied[name])
        assert hashlib.sha256(segment.encode()).hexdigest()==record['exact_segment_sha256']
    namespace={'np':np}
    exec(compile(ast.Module(body=[copied['TANNeuron']],type_ignores=[]),'<verified-TAN>','exec'),namespace)
    spec=importlib.util.spec_from_file_location('audited_v9_for_v11',V9/'run_noise_preservation.py')
    helper=importlib.util.module_from_spec(spec)
    old_policy=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True; spec.loader.exec_module(helper)
    finally: sys.dont_write_bytecode=old_policy
    helper.NATIVE=('TAN',)  # Select only the authorized model; source file unchanged.
    guard={'time_utc':utc(),'protocol_sha256':PROTOCOL,'helper_sha256':HELPER_HASH,
        'prior_manifest_sha256':PRIOR_MANIFEST_HASH,'prior_artifact_sha256':prior_hashes,
        'source_files_sha256':source_hashes,'selected_native_models':['TAN'],
        'prior_100_record_reproduction_verified':True,'all_guards_passed':True,
        'repository_provenance':git_provenance(repo)}
    jsave(out/'SOURCE_GUARD.json',guard)
    return namespace,helper,guard


def input_bundle(noise, regimes, ids, innovations=None):
    base=np.stack((np.r_[np.zeros(15),np.full(45,1.5)],np.zeros(60)))
    result={'noise':noise,'base':base,'waveforms':np.maximum(0.,noise[:,:,None,:]+base[None,None,:,:]),
        'history_ids':np.asarray(ids),'regimes':np.asarray(regimes),
        'conditions':np.asarray(CONDITIONS),'S':np.array([1,0])}
    if innovations is not None: result['Z']=innovations
    return result


def save_inputs(stage, bundle, out, helper, extra=None):
    filename=stage+'_inputs.npz'; np.savez_compressed(out/filename,**bundle)
    record={'time_utc':utc(),'models_evaluated_yet':False,'input_file':filename,
        'inputs_sha256':sha(out/filename),'array_manifest':helper.array_manifest(bundle)}
    if extra: record.update(extra)
    jsave(out/(stage.upper()+'_INPUTS_FROZEN.json'),record)
    return record


def tan_traces(bundle, namespace, helper):
    waves=bundle['waveforms']; n,r,c,t=waves.shape
    labels=[str(regime)+'_'+condition for regime in bundle['regimes'] for condition in CONDITIONS]
    result,checks=helper.simulate_native(waves.reshape(n,r*c,t),labels,bundle['history_ids'],namespace)
    traces={key:value.reshape((n,r,c)+value.shape[2:]) for key,value in result.items() if key.startswith('TAN__')}
    return traces,checks['TAN']


def adaptation_filter(waves):
    reference=np.zeros_like(waves); signed=np.zeros_like(waves)
    m=np.zeros(waves.shape[:-1])
    for t in range(60):
        m=.8*m+.2*waves[...,t]
        reference[...,t]=m
        signed[...,t]=(9/8)*(waves[...,t]-m)
    return reference,signed


def adapt_traces(reference,signed,kappa):
    magnitude=abs(signed); current=kappa*magnitude
    candidate=np.zeros_like(current); membrane=np.zeros_like(current); spikes=np.zeros(current.shape,dtype=np.int8)
    u=np.zeros(current.shape[:-1])
    for t in range(60):
        cand=.8*u+current[...,t]; fired=cand>1.
        u=np.where(fired,0.,cand)
        candidate[...,t]=cand; membrane[...,t]=u; spikes[...,t]=fired
    return {'ADAPT__reference':reference,'ADAPT__unscaled_signed_contrast':signed,
        'ADAPT__unscaled_magnitude':magnitude,'ADAPT__current':current,
        'ADAPT__candidate':candidate,'ADAPT__membrane':membrane,'ADAPT__spikes':spikes}


def save_traces(stage,bundle,traces,out):
    metadata={name:bundle[name] for name in ('history_ids','regimes','conditions','S')}
    np.savez_compressed(out/(stage+'_trajectories.npz'),**metadata,**traces)


def prefix_checks(bundle,traces):
    assert np.array_equal(bundle['waveforms'][:,:,0,:15],bundle['waveforms'][:,:,1,:15])
    for arm in ARMS:
        for field in ('current','candidate','membrane','spikes'):
            a=traces[arm+'__'+field]
            assert np.array_equal(a[:,:,0,:15],a[:,:,1,:15]), (arm,field)
    return True


def ledgers(bundle,traces):
    n,nr,_,_=bundle['waveforms'].shape
    rows=[]; paired=[]
    for i in range(n):
        for r,regime in enumerate(bundle['regimes']):
            for arm in ARMS:
                all_spikes=traces[arm+'__spikes'][i,r]
                step=all_spikes[0,15:20]; hit=int(step.any())
                latency=int(np.flatnonzero(step)[0]) if hit else 5
                catch=int(all_spikes[1,15:20].any())
                paired.append({'history_id':int(bundle['history_ids'][i]),'regime':str(regime),'arm':arm,
                    'D_step':hit,'D_catch':catch,'BA_contribution':.5*(hit+1-catch),
                    'preonset_alarm':int(all_spikes[0,:15].any()),'step_capped_latency':latency,
                    'step_hit':hit,'conditional_hit_latency':latency if hit else None})
                for c,condition in enumerate(CONDITIONS):
                    spike=all_spikes[c]; window=spike[15:20]; observed=int(window.any())
                    current=traces[arm+'__current'][i,r,c]
                    candidate=traces[arm+'__candidate'][i,r,c]
                    membrane=traces[arm+'__membrane'][i,r,c]
                    aux=traces['TAN__history'][i,r,c] if arm=='TAN' else traces['ADAPT__reference'][i,r,c]
                    delivered=bundle['waveforms'][i,r,c]; noise=bundle['noise'][i,r]
                    rows.append({'history_id':int(bundle['history_ids'][i]),'regime':str(regime),
                        'condition':condition,'S':int(c==0),'arm':arm,'detector_positive':observed,
                        'preonset_alarm':int(spike[:15].any()),
                        'first_onset_delay_capped':int(np.flatnonzero(window)[0]) if observed else 5,
                        'total_count':int(spike.sum()),'onset_count':int(window.sum()),
                        'late_count':int(spike[20:60].sum()),'R':float(spike[20:60].mean()),
                        'sum_abs_input':float(abs(delivered).sum()),'sum_input_squared':float(np.square(delivered).sum()),
                        'sum_noise_squared':float(np.square(noise).sum()),
                        'sum_abs_current':float(abs(current).sum()),'sum_current_squared':float(np.square(current).sum()),
                        'sum_candidate_squared':float(np.square(candidate).sum()),
                        'peak_abs_current':float(abs(current).max()),'peak_candidate':float(candidate.max()),
                        'peak_abs_candidate':float(abs(candidate).max()),'peak_membrane':float(membrane.max()),
                        'peak_abs_membrane':float(abs(membrane).max()),'aux_state_peak_abs':float(abs(aux).max())})
    return rows,paired


def record_ledgers(stage,bundle,traces,out):
    prefix_checks(bundle,traces)
    rows,paired=ledgers(bundle,traces)
    csave(out/(stage+'_ledger.csv'),LEDGER_FIELDS,rows)
    csave(out/(stage+'_paired_history_metrics.csv'),PAIR_FIELDS,paired)
    return rows,paired


def calibration(namespace,helper,out):
    noise=np.random.RandomState(2026110250).normal(0,.4,size=(512,60))[:,None,:]
    bundle=input_bundle(noise,('iid',),range(512))
    save_inputs('train',bundle,out,helper,{'seed':2026110250})
    traces,checks=tan_traces(bundle,namespace,helper)
    reference,signed=adaptation_filter(bundle['waveforms'])
    numerator=total(abs(traces['TAN__current'])); denominator=total(abs(signed))
    kappa=numerator/denominator
    assert np.isfinite([numerator,denominator,kappa]).all() and min(numerator,denominator,kappa)>0
    traces.update(adapt_traces(reference,signed,kappa))
    actual=total(abs(traces['ADAPT__current']))
    save_traces('train',bundle,traces,out)
    record_ledgers('train',bundle,traces,out)
    record={'time_utc':utc(),'training_seed':2026110250,'n_training_histories':512,
        'balanced_classes':True,'current_samples_per_model':512*2*60,
        'gain_kappa':kappa,'numerator_TAN_current_L1':numerator,
        'denominator_unscaled_ADAPT_current_L1':denominator,
        'calibrated_ADAPT_current_L1':actual,'raw_budget_residual':actual-numerator,
        'raw_budget_residual_per_current_sample':(actual-numerator)/(512*2*60),
        'sum_convention':'math.fsum over C-order float64 entries',
        'gain_finite_positive':True,'gain_fit_uses_spikes_accuracy_latency_or_test':False,
        'test_inputs_generated_yet':False,'native_trace_checks':checks,
        'training_inputs_sha256':sha(out/'train_inputs.npz'),
        'training_trajectories_sha256':sha(out/'train_trajectories.npz')}
    jsave(out/'CALIBRATION_FROZEN.json',record)
    print('CALIBRATION FROZEN',json.dumps(record),flush=True)
    return kappa,record


def clean_checks(bundle,traces):
    delta=1.5
    expected_box=np.r_[np.zeros(15),delta*np.arange(9,0,-1)/10,np.zeros(36)]
    expected_ad=np.r_[np.zeros(15),.9*delta*.8**np.arange(45)]
    box=traces['TAN__surprise'][0,0,0]
    ad=traces['ADAPT__unscaled_signed_contrast'][0,0,0]
    box_res=float(np.max(abs(box-expected_box))); ad_res=float(np.max(abs(ad-expected_ad)))
    assert max(box_res,ad_res)<=TOL
    assert np.array_equal(traces['TAN__spikes'][0,0,1],np.zeros(60,dtype=np.int8))
    assert np.array_equal(traces['ADAPT__spikes'][0,0,1],np.zeros(60,dtype=np.int8))
    with np.load(V9/'clean_trajectories.npz',allow_pickle=False) as old:
        prior_res=0.
        for key,value in traces.items():
            if not key.startswith('TAN__'): continue
            expected=old[key][:,None,...]
            error=float(np.max(abs(value.astype(float)-expected.astype(float))))
            prior_res=max(prior_res,error); assert error<=TOL
            if 'spike' in key: assert np.array_equal(value,expected)
    c=np.r_[.9,np.full(9,-.1)]
    rho=.9; cov=rho**abs(np.arange(10)[:,None]-np.arange(10)[None,:])
    return {'boxcar_clean_surprise_max_abs_error':box_res,'ADAPT_clean_contrast_max_abs_error':ad_res,
        'clean_native_v9_max_abs_error':prior_res,'first_signed_transient_target':.9*delta,
        'infinite_signed_transient_area_target':4.5*delta,
        'ADAPT_finite_signed_transient_sum':total(ad),
        'ADAPT_finite_signed_transient_formula':4.5*delta*(1-.8**45),
        'stationary_unclipped_theory':{'white_noise_variance_factor_TAN':float(c@c),
            'white_noise_variance_factor_ADAPT':.9,
            'ar09_variance_factor_TAN':float(c@cov@c),
            'ar09_variance_factor_ADAPT':.9*(1-.9)/(1-.8*.9)},
        'theory_scope':'Linear contrast before clipping/rectification/context; not empirical spike predictions.'}


def make_samples(rows,paired,clean_rows):
    samples={}; ratios={}
    for regime in REGIMES:
        for arm in ARMS:
            prefix=regime+'__'+arm+'__'
            p=[x for x in paired if x['regime']==regime and x['arm']==arm]
            hit=np.array([x['D_step'] for x in p],float); catch=np.array([x['D_catch'] for x in p],float)
            latency=np.array([x['step_capped_latency'] for x in p],float)
            samples.update({prefix+'TPR':hit,prefix+'FPR':catch,prefix+'specificity':1-catch,
                prefix+'BA':.5*(hit+1-catch),prefix+'preonset_alarm':np.array([x['preonset_alarm'] for x in p],float),
                prefix+'capped_latency':latency})
            for j in range(6): samples[prefix+'delay_p'+str(j)]=(latency==j).astype(float)
            for j in range(5): samples[prefix+'hit_by_'+str(j)]=(latency<=j).astype(float)
            ratios[prefix+'conditional_hit_latency']=(hit*latency,hit)
            for condition in CONDITIONS:
                selected=[x for x in rows if x['regime']==regime and x['arm']==arm and x['condition']==condition]
                clean_R=next(x['R'] for x in clean_rows if x['arm']==arm and x['condition']==condition)
                for field in CLASS_METRICS:
                    values=([x['R']-clean_R for x in selected] if field=='noisy_minus_clean_R'
                            else [x[field] for x in selected])
                    samples[prefix+condition+'__'+field]=np.asarray(values,float)
        for condition in CONDITIONS:
            selected=[x for x in rows if x['regime']==regime and x['arm']=='TAN' and x['condition']==condition]
            for field in ('sum_abs_input','sum_input_squared'):
                samples['input__'+regime+'__'+condition+'__'+field]=np.array([x[field] for x in selected])
        selected=[x for x in rows if x['regime']==regime and x['arm']=='TAN' and x['condition']=='step']
        samples['input__'+regime+'__sum_noise_squared']=np.array([x['sum_noise_squared'] for x in selected])
    delta={regime:samples[regime+'__TAN__BA']-samples[regime+'__ADAPT__BA'] for regime in REGIMES}
    samples['contrast__Delta_AR']=delta['ar09']
    samples['contrast__Interaction']=delta['ar09']-delta['iid']
    samples['contrast__Delta_iid']=delta['iid']
    for regime in REGIMES:
        for metric in ('TPR','FPR','capped_latency','preonset_alarm'):
            samples['contrast__'+regime+'__'+metric+'_gap']=samples[regime+'__TAN__'+metric]-samples[regime+'__ADAPT__'+metric]
        samples['contrast__'+regime+'__step_R_gap']=samples[regime+'__TAN__step__R']-samples[regime+'__ADAPT__step__R']
        samples['contrast__'+regime+'__balanced_current_L1_gap']=.5*sum(
            samples[regime+'__TAN__'+condition+'__sum_abs_current']-samples[regime+'__ADAPT__'+condition+'__sum_abs_current']
            for condition in CONDITIONS)
    samples['contrast__input_AR_minus_iid_noise_squared']=samples['input__ar09__sum_noise_squared']-samples['input__iid__sum_noise_squared']
    for condition in CONDITIONS:
        samples['contrast__input_AR_minus_iid_'+condition+'_squared']=samples['input__ar09__'+condition+'__sum_input_squared']-samples['input__iid__'+condition+'__sum_input_squared']
    return samples,ratios


def uncertainty(samples,ratios,out):
    names=list(samples)+list(ratios)
    matrix=np.column_stack([samples[name] for name in samples])
    n=len(matrix); draws=np.random.RandomState(2026110252).randint(0,n,size=(2000,n))
    np.save(out/'bootstrap_draws.npy',draws)
    metrics=np.empty((2000,len(names))); point=list(matrix.mean(axis=0))
    for numerator,denominator in ratios.values():
        point.append(float(numerator.sum()/denominator.sum()) if denominator.sum() else np.nan)
    for b,indices in enumerate(draws):
        metrics[b,:len(samples)]=matrix[indices].mean(axis=0)
        for j,(numerator,denominator) in enumerate(ratios.values(),start=len(samples)):
            hits=denominator[indices].sum()
            metrics[b,j]=numerator[indices].sum()/hits if hits else np.nan
    np.savez_compressed(out/'history_metric_samples.npz',metric_names=np.asarray(list(samples)),
        values=matrix,conditional_names=np.asarray(list(ratios)),
        conditional_numerators=np.column_stack([a for a,b in ratios.values()]),
        conditional_denominators=np.column_stack([b for a,b in ratios.values()]))
    np.savez_compressed(out/'bootstrap_metrics.npz',metric_names=np.asarray(names),values=metrics)
    csave(out/'bootstrap_metrics.csv',('replicate',)+tuple(names),
        [{'replicate':i,**{name:float(metrics[i,j]) if np.isfinite(metrics[i,j]) else None
            for j,name in enumerate(names)}} for i in range(2000)])
    summaries={}
    for j,name in enumerate(names):
        finite=np.isfinite(metrics[:,j]); quantiles=[.0125,.9875] if name in ['contrast__'+s for s in PRIMARY_CONTRASTS] else [.025,.975]
        summaries[name]={'estimate':float(point[j]) if np.isfinite(point[j]) else None,
            'interval_level':.975 if quantiles[0]==.0125 else .95,
            'percentile_interval':np.quantile(metrics[finite,j],quantiles).tolist() if finite.any() else None,
            'defined_bootstrap_replicates':int(finite.sum()),
            'undefined_bootstrap_replicates':int((~finite).sum())}
    jsave(out/'metric_summary.json',summaries)
    primary={name:summaries['contrast__'+name] for name in PRIMARY_CONTRASTS}
    csave(out/'bootstrap_primary_contrasts.csv',('replicate',)+PRIMARY_CONTRASTS,
        [{'replicate':i,**{name:float(metrics[i,names.index('contrast__'+name)]) for name in PRIMARY_CONTRASTS}}
            for i in range(2000)])
    return summaries,primary,names


def run(namespace,helper,out):
    kappa,cal=calibration(namespace,helper,out)
    assert (out/'CALIBRATION_FROZEN.json').is_file()
    Z=np.random.RandomState(2026110251).standard_normal((1024,60))
    noise=np.empty((1024,2,60)); noise[:,0]=.4*Z; noise[:,1,0]=.4*Z[:,0]
    for t in range(1,60): noise[:,1,t]=.9*noise[:,1,t-1]+.4*math.sqrt(1-.9**2)*Z[:,t]
    bundle=input_bundle(noise,REGIMES,range(1024),Z)
    frozen=save_inputs('test',bundle,out,helper,{'seed':2026110251,
        'calibration_frozen_sha256':sha(out/'CALIBRATION_FROZEN.json')})
    print('TEST INPUTS FROZEN',frozen['inputs_sha256'],flush=True)
    clean=input_bundle(np.zeros((1,1,60)),('clean',),[0])
    save_inputs('clean',clean,out,helper)
    clean_traces,clean_native_checks=tan_traces(clean,namespace,helper)
    reference,signed=adaptation_filter(clean['waveforms']); clean_traces.update(adapt_traces(reference,signed,kappa))
    save_traces('clean',clean,clean_traces,out)
    clean_rows,clean_paired=record_ledgers('clean',clean,clean_traces,out)
    clean_proofs=clean_checks(clean,clean_traces)
    jsave(out/'clean_checks.json',clean_proofs)
    traces,native_checks=tan_traces(bundle,namespace,helper)
    reference,signed=adaptation_filter(bundle['waveforms']); traces.update(adapt_traces(reference,signed,kappa))
    save_traces('test',bundle,traces,out)
    rows,paired=record_ledgers('test',bundle,traces,out)
    assert len(rows)==8192 and len(paired)==4096
    samples,ratios=make_samples(rows,paired,clean_rows)
    summary,primary,names=uncertainty(samples,ratios,out)
    support=all(primary[name]['percentile_interval'][0]>0 for name in PRIMARY_CONTRASTS)
    results={'time_utc':utc(),'protocol_sha256':PROTOCOL,'gain_kappa':kappa,
        'n_training_histories':512,'n_test_innovation_histories':1024,'test_seed':2026110251,
        'training_seed':2026110250,'bootstrap_seed':2026110252,'bootstrap_replicates':2000,
        'regime_order':list(REGIMES),'condition_order':list(CONDITIONS),'S':[1,0],
        'arms':list(ARMS),'model_parameters':{'ADAPT':{'EMA_old_weight':.8,'EMA_input_weight':.2,
            'signed_contrast_scale':9/8,'rectifier':'absolute value','membrane_leak':.8,'threshold':1.,
            'strict_greater_than':True,'initial_reference':0.,'initial_membrane':0.}},
        'calibration':cal,'no_exclusions':True,'test_native_trajectory_checks':native_checks,
        'clean_native_trajectory_checks':clean_native_checks,'clean_algebra_checks':clean_proofs,
        'test_ledger_rows':len(rows),'paired_history_rows':len(paired),'fixed_metric_count':len(names),
        'primary_contrasts':primary,'both_primary_lower_bounds_above_zero':support,
        'primary_direction_disfavored':{name:primary[name]['percentile_interval'][1]<=0 for name in PRIMARY_CONTRASTS},
        'metric_summary_file':'metric_summary.json','all_per_history_metrics_file':'history_metric_samples.npz',
        'nominal_signal_sum':67.5,'nominal_signal_squared_sum':101.25,'expected_noise_squared_sum':9.6,
        'deterministic_clean_control_count':4,'onset_known_externally':True,
        'latency_miss_penalty':5,'inference_conditional_on_frozen_calibration':True,
        'interpretation_limits':['New nuisance distribution; no unique gate-causal identification.',
            'Calibration matches iid training total current L1 only, not spike counts or physical energy.',
            'Primary percentile intervals are a reporting rule, not exact simultaneous coverage.',
            'Degenerate empirical intervals do not establish population perfection.',
            'Undefined conditional latency replicates, if any, remain recorded; no histories removed.'],
        'round_status':'Experiment complete; separate computational review recorded.'}
    jsave(out/'primary_results.json',results)
    for regime in REGIMES:
        for arm in ARMS:
            prefix=regime+'__'+arm+'__'
            print(regime,arm,json.dumps({metric:summary[prefix+metric]['estimate']
                for metric in ('TPR','FPR','BA','preonset_alarm','capped_latency','conditional_hit_latency')}),flush=True)
    print('PRIMARY CONTRASTS',json.dumps(primary),'support_rule',support,flush=True)
    return names


class Tee:
    def __init__(self,stream,file): self.stream,self.file=stream,file
    def write(self,value): self.stream.write(value); self.file.write(value); self.file.flush()
    def flush(self): self.stream.flush(); self.file.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,default=BASE)
    out=parser.parse_args().output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'TRAIN_INPUTS_FROZEN.json').exists() or (out/'PRIMARY_MANIFEST.json').exists():
        raise RuntimeError('Preserving prior outputs; a reproduction requires a fresh --output-dir.')
    repo=Path(os.environ.get('TAN_REPO_ROOT',BASE.parents[1]/'Temporal-Attention-Neuron')).resolve()
    oldout,olderr=sys.stdout,sys.stderr; started=utc()
    with (out/'run.log').open('w') as log:
        sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try:
            before=git_provenance(repo); jsave(out/'repo_status_before.json',before)
            print('START',started,sys.argv,'repository',before,flush=True)
            namespace,helper,guard=sources(repo,out); metric_names=run(namespace,helper,out)
            after=git_provenance(repo); jsave(out/'repo_status_after.json',after); assert before==after
            for name,digest in guard['prior_artifact_sha256'].items(): assert sha(V9/name)==digest
            for name,digest in guard['source_files_sha256'].items(): assert sha(repo/name)==digest
            assert sha(V9/'run_noise_preservation.py')==HELPER_HASH
            completed=utc(); print('COMPLETED',completed,'repository',after,flush=True)
            files=['SOURCE_GUARD.json','CALIBRATION_FROZEN.json','clean_checks.json',
                'history_metric_samples.npz','bootstrap_draws.npy','bootstrap_metrics.npz','bootstrap_metrics.csv',
                'bootstrap_primary_contrasts.csv','metric_summary.json','primary_results.json',
                'repo_status_before.json','repo_status_after.json','run.log']
            for stage in ('train','test','clean'):
                files.extend([stage+'_inputs.npz',stage.upper()+'_INPUTS_FROZEN.json',stage+'_trajectories.npz',
                    stage+'_ledger.csv',stage+'_paired_history_metrics.csv'])
            manifest={'status':'COMPLETED','started_utc':started,'completed_utc':completed,
                'execution_command':sys.argv,'python':platform.python_version(),'numpy':np.__version__,
                'protocol_sha256':PROTOCOL,'primary_script_sha256':sha(__file__),
                'audited_v9_helper_sha256':HELPER_HASH,'selected_native_models':['TAN'],
                'repository':str(repo),'repository_provenance':after,
                'repository_status_unchanged':True if after['available'] else None,
                'ledger_fields':list(LEDGER_FIELDS),'paired_ledger_fields':list(PAIR_FIELDS),
                'metric_names':metric_names,'primary_contrasts':list(PRIMARY_CONTRASTS),
                'trajectory_axes':'history, regime, class, time[, history_slot]',
                'condition_order':list(CONDITIONS),'S':[1,0],'regime_order':list(REGIMES),
                'float_tolerance':TOL,'files':{},'npz_schema':{}}
            for name in files:
                path=out/name; manifest['files'][name]={'sha256':sha(path),'bytes':path.stat().st_size}
                if name.endswith('.npz'):
                    with np.load(path,allow_pickle=False) as arrays:
                        manifest['npz_schema'][name]={key:{'shape':list(arrays[key].shape),'dtype':str(arrays[key].dtype)} for key in arrays.files}
            jsave(out/'PRIMARY_MANIFEST.json',manifest)
        except Exception as exc:
            jsave(out/'PRIMARY_RUN_FAILURE.json',{'time_utc':utc(),'exception':repr(exc),
                'traceback':traceback.format_exc(),'calibration_frozen':(out/'CALIBRATION_FROZEN.json').exists(),
                'test_inputs_frozen':(out/'TEST_INPUTS_FROZEN.json').exists(),'repository_after':git_provenance(repo)})
            traceback.print_exc(); raise
        finally: sys.stdout,sys.stderr=oldout,olderr


if __name__=='__main__': main()
