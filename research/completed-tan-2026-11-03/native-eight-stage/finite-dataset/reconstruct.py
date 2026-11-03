#!/usr/bin/env python3
"""Targets-only reconstruction. Executes only separately reviewed task_generator.py."""
from pathlib import Path
import csv, hashlib, importlib.util, json, platform, sys
import numpy as np
BASE=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
SOURCE=BASE/'task_generator.py'
SUMMARY=BASE/'summary_p1b.json'
SEEDS={2:2026091920,3:2026091930}

def independent_trial(rng,n):
    rejection=0
    while True:
        keys=np.sort(rng.uniform(.3,2.5,size=n))
        if np.all(np.diff(keys)>=.05):break
        rejection+=1
        if rejection>20000:raise RuntimeError('rejection limit')
    keys=rng.permutation(keys)
    vals=rng.uniform(-3.,3.,size=n)
    positions=sorted(rng.choice(5,size=n,replace=False).tolist())
    fullkeys=np.zeros(5); fullvals=np.zeros(5)
    fullkeys[positions]=keys; fullvals[positions]=vals
    targets=rng.choice(positions,size=2,replace=False)
    return fullkeys,fullvals,positions,targets,rejection

def same_rng(a,b):
    a,b=a.get_state(),b.get_state()
    return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]

def bootstrap(loss,seed):
    rng=np.random.RandomState(seed)
    means=[]
    for _ in range(100):
        means.extend(loss[rng.randint(0,len(loss),size=(100,len(loss)))].mean(axis=1).tolist())
    return np.quantile(means,[.025,.975]).tolist()

def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '674cf9f49f5cd7f4975b87d29cd413e801fe40bd10fc56b966e663c9e6fc989d', 'Generator differs from reviewed source; do not execute'
    assert hashlib.sha256(SUMMARY.read_bytes()).hexdigest() == '5a19a5b20225fdad51641990b61e189f9b00569abdaf08207c5e2d4941136ff9', 'Summary input changed'
    spec=importlib.util.spec_from_file_location('reviewed_task_generator',SOURCE)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    gen=module.TaskGenerator(window_size=5)
    summary=json.loads(SUMMARY.read_text())
    result={'source_commit':'85213f0d417d54e4f48b301ed1c0e4200ef3c421','python':platform.python_version(),'numpy':np.__version__,
      'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE,SUMMARY]},
      'status':'deterministic reconstruction; no original target ledger or historical runtime available for byte-match verification',
      'rng':'separate continuous numpy.random.RandomState stream for each N; per-trial seed field metadata only',
      'bootstrap':'10000 percentile replicates of independent histories per N; seed 20261102+N; no paired model-minus-oracle CI',
      'datasets':{},'aggregate_excess':[]}
    codes={'M4_Count':np.array([-2.,-1.,0.,1.,2.]),'M4_Latency':np.sort(np.append(-2.85+np.arange(20)*.30,0.))}
    for n,seed in SEEDS.items():
        rng=np.random.RandomState(seed);ind=np.random.RandomState(seed)
        records=[];rejects=0
        for tid in range(2000):
            trial=gen.generate_single_trial(rng,trial_id=tid,seed=seed+tid,N=n)
            k,v,p,t,r=independent_trial(ind,n)
            assert np.array_equal(k,trial.keys) and np.array_equal(v,trial.values)
            assert p==trial.event_positions and tuple(map(int,t))==trial.target_idx and r==trial.rejection_count
            assert same_rng(rng,ind)
            rejects+=r
            records.append([n,tid,seed+tid,*trial.target_idx,*trial.target_values])
        targets=np.array([r[-2:] for r in records])
        filename=OUT/f'reconstructed_targets_N{n}.csv'
        with filename.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['N','trial_id','seed_metadata','target_idx_A','target_idx_B','target_A','target_B']);writer.writerows(records)
        info={'base_seed':seed,'n_histories':len(records),'total_key_rejections':rejects,
          'independent_implementation_all_records_and_rng_states_match':True,
          'target_array_sha256_little_endian_float64':hashlib.sha256(targets.astype('<f8').tobytes()).hexdigest(),
          'ledger_sha256':hashlib.sha256(filename.read_bytes()).hexdigest(),
          'first_three_records':records[:3], 'oracle':{}}
        for name,code in codes.items():
            distance=np.abs(targets[:,:,None]-code[None,None,:]).min(axis=2)
            loss=distance.max(axis=1)
            info['oracle'][name]={'mean_e_comp':float(loss.mean()),'scalar_mae':float(distance.mean()),'sample_sd':float(loss.std(ddof=1)),
              'bootstrap_percentile_95_ci':bootstrap(loss,20261102+n), 'min_loss':float(loss.min()),'max_loss':float(loss.max())}
            for cell,entry in summary['cell_results'].items():
                if entry['N']!=n:continue
                assert entry['n_trials']==len(records)
                observed=entry['model_summaries'][name]['mean_e_comp']
                assert observed==entry['spike_channel'][name]['mean_e_comp']
                result['aggregate_excess'].append({'cell':cell,'model':name,'reported_model_mean':observed,'reconstructed_oracle_mean':float(loss.mean()),'descriptive_mean_excess':observed-float(loss.mean())})
        result['datasets'][f'N{n}']=info
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['oracle'] for k,v in result['datasets'].items()},indent=2))
    print('aggregate excess rows:',len(result['aggregate_excess']))
if __name__=='__main__':main()
