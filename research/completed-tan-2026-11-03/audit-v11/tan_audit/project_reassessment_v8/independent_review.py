"""Independent scalar reconstruction of the frozen Probe1 swap experiment.

Does not import either the primary scripts or the repository model/generator.
"""
from pathlib import Path
from fractions import Fraction as Q
import hashlib, itertools, json, math, platform, resource, subprocess, time
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent/'Temporal-Attention-Neuron'
PLAN='d8f0b97300f7fd1d3e3b0c07cd5ca17c3cfb4b40b20efa25792e6ab3dd78df23'
SOURCES={
 'snapshot/code/experiments/natural_collision.py':'8c7b4e75ec8284b671bca6280ee848498ddd9c6109a6dba31254f6d407ec4d7f',
 'snapshot/code/experiments/temporal_distractor.py':'6177df497f9735d9fdb434ff20e7751a8dbb4af7a32d836bb3c4898c06ad4f23',
}
MODELS=['B1_lif','B2_buf','B3_noattn','B4_tan','B4_no_reset_t8']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def exp_interval(x,n=40):
    assert x>=0 and x<n+2
    term=total=Q(1)
    for k in range(1,n+1):term=term*x/k;total+=term
    next_term=term*x/(n+1)
    return total,total+next_term/(1-x/Q(n+2))

def theorem():
    el,eu=exp_interval(Q(557,500))
    hi_term=exp_interval(Q(557,500)*Q(273,200))[0]
    lo_term=exp_interval(Q(593,500)*Q(127,200))[1]
    sign=Q(73,100)*hi_term-Q(77,100)*lo_term-1-Q(1,10**9)
    assert el>3 and sign>0
    s8=(Q(4)*Q(273,200)-2*Q(127,200)-Q(277,200))/5
    s8hi=(Q(4)*Q(277,200)-2*Q(123,200)-Q(273,200))/5
    s9=(Q(4)*Q(273,200)-2*Q(127,200)-2*Q(277,200))/5
    s9hi=(Q(4)*Q(277,200)-2*Q(123,200)-2*Q(273,200))/5
    assert (s8,s8hi)==(Q(561,1000),Q(589,1000))
    assert (s9,s9hi)==(Q(71,250),Q(79,250))
    assert Q(8,25)*Q(277,200)<Q(1,2)
    mass=Q(1,50)**5
    assert mass==Q(1,312500000)
    return {'exp_1_114_lower_minus_3':str(el-3),'context_C8_minus_1_positive_numerator_lower':str(sign),
      'exp_degree':40,'exact_S8_range':[str(s8),str(s8hi)],'exact_S9_range':[str(s9),str(s9hi)],
      'u9_conservative_upper':'277/625','conditional_orientation_mass':str(mass),
      'conditional_union_mass':str(2*mass),'conditional_population_error_lower':str(mass),
      'clarification':'The certified numerator equals (sum(exp)+delta)*(C8-1), not sum(exp)*(C8-1). The frozen explicit bound has the correct -delta term.'}

def regenerate():
    lo=np.array([.615,1.365,.615,1.365,1.365]);hi=np.array([.635,1.385,.635,1.385,1.385])
    anchor=np.array([[.625,1.375,.625,1.375,1.375]])
    corners=np.array(list(itertools.product(*zip(lo,hi))))
    random=np.random.RandomState(2026110225).uniform(lo,hi,size=(512,5))
    pars=np.vstack([anchor,corners,random])
    X=np.zeros((545,2,14));X[:,0,4:9]=pars;X[:,1,4:9]=pars[:,[1,0,2,3,4]]
    return pars,X

def scalar_simulate(X,model):
    shape=X.shape
    arrays={k:np.zeros(shape) for k in ['u','h','j','spikes']}
    if model in ['B3_noattn','B4_tan','B4_no_reset_t8']:arrays['s']=np.zeros(shape)
    if model.startswith('B4'):arrays['alpha']=np.zeros((*shape,5))
    windows=np.zeros((*shape,5))
    for pair in range(shape[0]):
      for side in range(2):
        h=0.;win=[0.]*5
        for t,x in enumerate(X[pair,side]):
          win=win[1:]+[float(x)];windows[pair,side,t]=win;mu=sum(win)/5
          if model=='B1_lif':j=float(x)
          elif model=='B2_buf':j=sum(float(i+1)/15*w for i,w in enumerate(win))
          else:
            s=max(0.,float(x)-mu);arrays['s'][pair,side,t]=s
            if model=='B3_noattn':context=mu
            else:
              ee=[math.exp(2*s*w) for w in win];den=sum(ee)+1e-9
              aa=[v/den for v in ee];arrays['alpha'][pair,side,t]=aa
              context=sum(a*w for a,w in zip(aa,win))
            j=math.tanh(s)*context
          u=.5*h+j;spike=int(u>.5)
          h=u if model=='B4_no_reset_t8' and t==7 else (0. if spike else u)
          for k,v in [('u',u),('h',h),('j',j),('spikes',spike)]:arrays[k][pair,side,t]=v
    arrays['features']=np.concatenate([arrays['u'][:,:,8:13],arrays['spikes'][:,:,8:13]],axis=2)
    return arrays,windows

def git_status():
    if not (REPO/'.git').exists():return None
    return subprocess.run(['git','status','--porcelain'],cwd=REPO,check=True,capture_output=True,text=True).stdout

def stats(x):
    x=np.asarray(x,dtype=float)
    return {'min':float(np.min(x)),'max':float(np.max(x)),'mean':float(np.mean(x))}

def compare(actual,expected,tol=2e-13,path='root'):
    if isinstance(expected,dict):
      assert set(actual)==set(expected),(path,set(actual)^set(expected))
      return max([compare(actual[k],v,tol,path+'.'+k) for k,v in expected.items()]+[0.])
    if isinstance(expected,list):
      assert len(actual)==len(expected),(path,len(actual),len(expected))
      return max([compare(a,e,tol,path+f'[{i}]') for i,(a,e) in enumerate(zip(actual,expected))]+[0.])
    if isinstance(expected,(str,bool)) or expected is None:
      assert actual==expected,(path,actual,expected);return 0.
    d=abs(float(actual)-float(expected));assert d<=tol,(path,actual,expected,d)
    return d

def main():
    started=time.perf_counter()
    assert sha(HERE/'PREREGISTRATION.md')==PLAN
    for rel,want in SOURCES.items():assert sha(REPO/rel)==want,rel
    names=['PREREGISTRATION.md','run_probe1_collision.py','theory_certificates.py',
      'theory_certificates.json','probe1_inputs.jsonl','PROBE1_INPUTS_SHA256',
      'probe1_trajectories.npz','probe1_pair_ledger.jsonl','probe1_results.json','probe1_run.log']
    # The primary run log may use a different documented filename.
    names=[n for n in names if (HERE/n).is_file()]
    before={n:sha(HERE/n) for n in names};git_before=git_status()
    proof=theorem();pars,X=regenerate()
    rows=[json.loads(s) for s in (HERE/'probe1_inputs.jsonl').read_text().splitlines()]
    assert len(rows)==545
    for i,row in enumerate(rows):
      expected={'pair_id':i,'kind':'anchor' if i==0 else 'corner' if i<=32 else 'random_box',
        'parameters':pars[i].tolist(),'X':X[i].tolist(),'labels':[0,1],
        'latent_classes':[[0,1,0,1,1],[1,0,0,1,1]],
        'target_values':X[i,:,4].tolist(),'query':float(pars[i,4])}
      assert row==expected,('input',i)
      assert all(.55<X[i,0,t]<.95 for t in [4,6])
      assert all(1.05<X[i,0,t]<1.45 for t in [5,7,8])
      assert X[i,0,8]==X[i,1,8]
      assert not np.any(X[i,:,:4]) and not np.any(X[i,:,9:])
    source_result=json.loads((HERE/'probe1_results.json').read_text())
    assert source_result['status']=='PASS'
    assert source_result['protocol_sha256']==PLAN and source_result['source_sha256']==SOURCES
    assert source_result['inputs_sha256']==sha(HERE/'probe1_inputs.jsonl')
    assert source_result['script_sha256']==sha(HERE/'run_probe1_collision.py')
    assert source_result['trajectories_sha256']==sha(HERE/'probe1_trajectories.npz')
    assert source_result['pair_ledger_sha256']==sha(HERE/'probe1_pair_ledger.jsonl')
    assert source_result['cases']=={'anchor':1,'corners':32,'random_box':512,'total_pairs':545,'histories_per_arm':1090,'exclusions':0}
    assert source_result['model_trajectory_counts']=={'primary_native':4360,'primary_single_reset_intervention':1090,'native_vector_reconstruction_validation':4360,'primary_euler_steps':76300,'validation_euler_steps':61040}
    assert all(source_result['checks'].values())
    z=np.load(HERE/'probe1_trajectories.npz')
    assert np.array_equal(z['inputs'],X) and np.array_equal(z['parameters'],pars)
    assert np.array_equal(z['labels'],np.tile([0,1],(545,1)))
    independently={};residuals={};expected_keys={'inputs','parameters','windows','labels','target_tap_oracle'}
    for model in MODELS:
      arrays,windows=scalar_simulate(X,model);independently[model]=arrays
      assert np.array_equal(z['windows'],windows)
      residuals[model]={}
      for k,v in arrays.items():
        key=model+'__'+k;expected_keys.add(key)
        assert v.shape==z[key].shape,(key,v.shape,z[key].shape)
        if k=='spikes':assert np.array_equal(v,z[key]),key
        d=float(np.max(abs(v-z[key])));assert d<2e-14,(key,d)
        residuals[model][k]=d
    assert set(z.files)==expected_keys,set(z.files)^expected_keys
    oracle=np.where(windows[:,:,8,0]<.95,0,np.where(windows[:,:,8,0]>1.05,1,-1))
    assert np.array_equal(oracle,z['target_tap_oracle']) and np.array_equal(oracle,z['labels'])
    assert np.all(np.max(abs(windows[:,0,8:10]-windows[:,1,8:10]),axis=2)>0)
    assert np.array_equal(windows[:,0,10:],windows[:,1,10:])
    native=independently['B4_tan'];lesion=independently['B4_no_reset_t8']
    assert np.all(native['h'][:,:,7]==0) and np.all(native['spikes'][:,:,7]==1)
    assert np.all(native['j'][:,:,7]>.5)
    assert np.all(native['s'][:,:,8]>0) and np.all((native['u'][:,:,8]>0)&(native['u'][:,:,8]<.5))
    assert np.all(native['s'][:,:,9:]==0)
    # Every permitted coordinate is determined by query candidate and reset.
    scalar_only_residual={}
    for model in ['B1_lif','B3_noattn','B4_tan','B4_no_reset_t8']:
      a=independently[model];q=a['u'][:,:,8];r=q>.5;retained=np.where(r,0.,q)
      expected=np.stack([q,retained/2,retained/4,retained/8,retained/16,r,
                         np.zeros_like(q),np.zeros_like(q),np.zeros_like(q),np.zeros_like(q)],axis=2)
      err=float(np.max(abs(a['features']-expected)));assert err==0.;scalar_only_residual[model]=err
    assert np.all(np.max(abs(native['features'][:,0]-native['features'][:,1]),axis=1)<2e-14)
    assert np.array_equal(native['u'][:,:,:8],lesion['u'][:,:,:8])
    assert np.array_equal(native['h'][:,:,:7],lesion['h'][:,:,:7])
    assert np.array_equal(native['spikes'][:,:,7],lesion['spikes'][:,:,7])
    assert np.array_equal(lesion['h'][:,:,7],lesion['u'][:,:,7])
    for k in ['j','s','alpha']:assert np.array_equal(native[k],lesion[k]),k
    for step in range(14):
      if step!=7:assert np.array_equal(lesion['h'][:,:,step],np.where(lesion['spikes'][:,:,step],0.,lesion['u'][:,:,step]))
    assert np.max(abs(z['B2_buf__u'][0,:,8]-[47/40,9/8]))<1e-14
    b2gap=z['B2_buf__u'][:,0,8]-z['B2_buf__u'][:,1,8]
    assert np.max(abs(b2gap-(pars[:,1]-pars[:,0])/15))<1e-14
    assert np.all(z['B2_buf__h'][:,:,7]==0)
    records=[json.loads(s) for s in (HERE/'probe1_pair_ledger.jsonl').read_text().splitlines()]
    assert len(records)==2725
    byid={(r['arm'],r['pair_id']):r for r in records};assert len(byid)==2725
    summaries={};ledger_error=summary_error=0.;spike_only={}
    for model in MODELS:
      F=z[model+'__features'];U=z[model+'__u'];H=z[model+'__h'];J=z[model+'__j'];sp=z[model+'__spikes']
      S=z[model+'__s'] if model+'__s' in z else None
      dif=np.abs(F[:,0]-F[:,1]);maxdif=np.max(dif,axis=1);l2=np.sqrt(np.sum(dif*dif,axis=1));equal=np.all(dif==0,axis=1)
      for i,row in enumerate(rows):
        want={'pair_id':i,'kind':row['kind'],'arm':model,'labels':[0,1],
          'target_values':row['target_values'],'query':row['query'],'features':F[i].tolist(),
          'literal_equal_features':bool(equal[i]),'max_abs_feature_gap':float(maxdif[i]),'l2_feature_gap':float(l2[i]),
          'target_amplitude_gap':float(abs(X[i,0,4]-X[i,1,4])),
          'window_t9_linf_gap':float(np.max(abs(windows[i,0,8]-windows[i,1,8]))),
          't8_u':U[i,:,7].tolist(),'t8_h':H[i,:,7].tolist(),'t8_spikes':sp[i,:,7].tolist(),
          't8_threshold_margin':(U[i,:,7]-.5).tolist(),'t9_u':U[i,:,8].tolist(),
          't9_threshold_headroom':(.5-U[i,:,8]).tolist(),'t9_surprise':S[i,:,8].tolist() if S is not None else None,
          'oracle_predictions':oracle[i].tolist(),'both_oracle_correct':True}
        ledger_error=max(ledger_error,compare(byid[model,i],want))
      summaries[model]={'pairs':545,'histories':1090,'literal_equal_response_pairs':int(equal.sum()),
        'near_equal_response_pairs_at_1e_13':int(np.sum(maxdif<=1e-13)),
        'response_linf_gap':stats(maxdif),'response_l2_gap':stats(l2),
        'analog_feature_linf_gap':stats(np.max(dif[:,:5],axis=1)),
        'differing_spike_feature_pairs':int(np.any(dif[:,5:]!=0,axis=1).sum()),
        'query_spike_disagreement_pairs':int(np.sum(sp[:,0,8]!=sp[:,1,8])),
        'query_u_abs_gap':stats(abs(U[:,0,8]-U[:,1,8])),
        't8_spike_histories':int(sp[:,:,7].sum()),'t8_zero_h_histories':int(np.sum(H[:,:,7]==0)),
        't8_current':stats(J[:,:,7]),'t8_candidate_margin':stats(U[:,:,7]-.5),
        't9_positive_surprise_histories':int(np.sum(S[:,:,8]>0)) if S is not None else None,
        't9_surprise':stats(S[:,:,8]) if S is not None else None,
        't9_candidate':stats(U[:,:,8]),'t9_threshold_headroom':stats(.5-U[:,:,8]),
        'anchor_features':F[0].tolist(),'anchor_t8_u':U[0,:,7].tolist(),'anchor_t8_h':H[0,:,7].tolist()}
      summary_error=max(summary_error,compare(source_result['arms'][model],summaries[model]))
      spike_only[model]={'equal_pairs':int(np.all(F[:,0,5:]==F[:,1,5:],axis=1).sum()),'total_pairs':545}
    control_expected={'target_tap_oracle_correct':1090,'target_tap_oracle_total':1090,
      'target_amplitude_gap':stats(abs(X[:,0,4]-X[:,1,4])),
      'full_window_t9_linf_gap':stats(np.max(abs(windows[:,0,8]-windows[:,1,8]),axis=1))}
    summary_error=max(summary_error,compare(source_result['controls'],control_expected))
    certificate=json.loads((HERE/'theory_certificates.json').read_text())
    # Independent proof precedes reading the primary certificate; check its
    # exact-fraction leaves for internally consistent decimal displays.
    rational_leaves=[]
    def fractions_check(o):
      if isinstance(o,dict):
        if 'rational' in o and 'decimal_display' in o:
          v=Q(o['rational']);assert abs(float(v)-o['decimal_display'])<1e-14;rational_leaves.append(v)
        for value in o.values():fractions_check(value)
      elif isinstance(o,list):
        for value in o:fractions_check(value)
    fractions_check(certificate)
    assert len(rational_leaves)>20
    after={n:sha(HERE/n) for n in names};assert before==after
    git_after=git_status();assert git_after==git_before
    result={'status':'PASS','primary_artifacts_unchanged':True,
      'repository_status_verified':git_before is not None,
      'repository_status_unchanged':True if git_before is not None else None,
      'repository_status_availability':'checked git checkout' if git_before is not None else 'unavailable in source-only snapshot',
      'repository_porcelain':git_after,'source_sha256':SOURCES,'primary_sha256':before,
      'independent_script_sha256':sha(Path(__file__)),'protocol_sha256':PLAN,
      'independently_regenerated_pairs':545,'histories_per_arm':1090,'arms':5,
      'independent_scalar_trajectories':5450,'independent_scalar_updates':76300,
      'paired_ledger_rows_checked':2725,'npz_arrays_checked':len(z.files),
      'trajectory_max_residuals':residuals,'pair_ledger_max_residual':ledger_error,
      'summary_max_residual':summary_error,'all_original_response_coordinates_checked':True,
      'scalar_only_zero_tail_feature_residuals':scalar_only_residual,
      'spike_only_response_pairs':spike_only,'independent_exact_theory':proof,
      'primary_certificate_rational_leaf_checks':len(rational_leaves),
      'reviewer_correction':'Initial checker stopped on three additional primary summary fields; its log is preserved as independent_review_initial_schema.log. The checker was extended to independently verify those fields, without changing primary files or scientific parameters.',
      'python':platform.python_version(),'numpy':np.__version__,
      'wall_seconds':time.perf_counter()-started,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'limits':['Ideal response equality is distinct from floating precision and noise robustness.',
        'The 3.2e-9 risk bound concerns the ideal continuous present-distractor distribution, not discrete PRNG prevalence or balanced-three-class accuracy.',
        'Constructed-pair separation does not demonstrate trained classification accuracy.',
        'B2 and reset-suppressed B4 can separate analog observations without separating spike-only observations.',
        'Native B3/B4 are not a single-step matched attention intervention.',
        'Saved source flow/hashes are not an external timestamp attestation.']}
    save(HERE/'INDEPENDENT_REVIEW_CHECKS.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

