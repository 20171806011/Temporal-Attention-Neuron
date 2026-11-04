"""Independent source-contract reanalysis checker. Does not import experiment/model/generator."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,itertools,math
import numpy as np
P=Path(__file__).resolve().parent;REPO=P.parent.parent/'Temporal-Attention-Neuron'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-12):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    assert a.shape==b.shape
    assert np.allclose(a,b,rtol=0,atol=tol,equal_nan=True),(np.max(abs(a-b)),a,b)

def reproduce_inputs():
    generated=[];rejection_total={}
    for N,seed in ((2,2026091920),(3,2026091930)):
        rng=np.random.RandomState(seed);rejection_total[N]=0
        for index in range(2000):
            count=0
            while True:
                k=np.sort(rng.uniform(.3,2.5,N))
                if np.all(np.diff(k)>=.05):break
                count+=1
            k=rng.permutation(k);v=rng.uniform(-3.,3.,N);pos=sorted(rng.choice(5,N,replace=False).tolist());targetpos=rng.choice(pos,2,replace=False).tolist()
            K=np.zeros(5);V=np.zeros(5);mask=np.zeros(5,dtype=bool);K[pos]=k;V[pos]=v;mask[pos]=True
            q=[float(K[i]) for i in targetpos];t=[float(V[i]) for i in targetpos];pat=''.join('1' if x>1.4 else '0' for x in k);z=pat.count('0');part='ALL_POSITIVE' if z==0 else 'EXACTLY_ONE_ZERO' if z==1 else 'AT_LEAST_TWO_ZERO'
            generated.append(dict(id=f'N{N}_{index}',N=N,trial_id=index,seed_metadata=seed+index,keys=K.tolist(),values=V.tolist(),mask=mask.tolist(),positions=pos,queries=q,target_indices=targetpos,targets=t,gt_add=t[0]+t[1],gt_sub=t[0]-t[1],rejection_count=count,novelty_pattern=pat,zero_count=z,partition=part))
            rejection_total[N]+=count
    return generated,rejection_total

def independent_step(state,content,novelty,query):
    # Separately derived registered recurrence; no imported implementation.
    u,A,r=state.T;r_before=r.copy()
    next_A=np.clip(A+(-A+novelty)/15*.1,0.,5.)
    free=r<=0;candidate=u+(-u+A*content+query)/10*.1
    spike=free&(candidate>=1.)
    next_u=np.where(free&~spike,candidate,0.)
    next_r=np.where(free,np.where(spike,2.,0.),np.maximum(0.,r-.1))
    return np.column_stack((next_u,next_A,next_r)),np.where(free,candidate,np.nan),spike,r_before

def all_predictions(inputs,cal):
    K=np.array([x['keys'] for x in inputs]);V=np.array([x['values'] for x in inputs]);mask=np.array([x['mask'] for x in inputs]);state=np.zeros((len(inputs),3))
    for step in range(1000):
        slot=step//200;active=mask[:,slot]&(step%200<100)
        content=np.where(active,K[:,slot]+V[:,slot],0.);novelty=np.where(active,np.maximum(0.,K[:,slot]-1.4),0.)
        state,_,_,_=independent_step(state,content,novelty,0.)
    encoding=state.copy();delayed={0.:state.copy()}
    for step in range(1,101):
        state,_,_,_=independent_step(state,0.,0.,0.)
        if step in (50,100):delayed[step/10]=state.copy()
    queries=np.array([r['queries'] for r in inputs]).reshape(-1);out={}
    for delay,cue in delayed.items():
        st=np.repeat(cue,2,axis=0)
        for _ in range(100):st,cand,spike,rb=independent_step(st,0.,0.,queries)
        uread=np.where(rb>0,0.,cand)
        out[delay]=dict(cue=cue,prediction=(cal['g_dec']*uread+cal['b_dec']).reshape(-1,2),uread=uread.reshape(-1,2),rb=rb.reshape(-1,2))
    return encoding,out

def check_strata():
    r=json.loads((P/'stratified_results.json').read_text());inputs=[json.loads(x) for x in (P/'stratified_inputs.jsonl').read_text().splitlines()];rows=[json.loads(x) for x in (P/'stratified_predictions.jsonl').read_text().splitlines()]
    independent,rejections=reproduce_inputs();assert independent==inputs
    assert len(inputs)==4000 and len(rows)==12000
    assert digest(P/'stratified_inputs.jsonl')==r['inputs_sha256'];assert digest(P/'run_stratified_m1.py')==r['script_sha256']
    assert digest(P/'ANALYSIS_PLAN.md')==r['analysis_plan_sha256'];assert digest(P/'PLAN_AMENDMENT_1.md')==r['amendment_sha256']
    frozen=json.loads((REPO/'p1_experiments/p1b/summary_p1b.json').read_text());assert r['calibration']==frozen['calibration']
    source_fields={'model_sha256':'p1_experiments/p1b/models_p1b_v2.py','runner_sha256':'p1_experiments/p1b/runner_p1b.py','generator_sha256':'p1_experiments/task_generator.py','frozen_summary_sha256':'p1_experiments/p1b/summary_p1b.json'}
    for key,path in source_fields.items():assert digest(REPO/path)==r[key]
    encoding,actual=all_predictions(inputs,r['calibration']);byid={x['id']:(i,x) for i,x in enumerate(inputs)}
    oracle_histories=set();oracle_max=0.;prediction_max=0.;state_max=0.;sterbenz_checks=0
    for item in inputs:
        for p in item['positions']:
            k=item['keys'][p]
            if k>1.4:
                assert F(k)-F(1.4)==F(k-1.4) and (k-1.4)+1.4==k
                sterbenz_checks+=1
    for row in rows:
        i,item=byid[row['input_id']];d=row['delay_ms'];got=actual[d]
        assert row['N']==item['N'] and row['trial_id']==item['trial_id'] and row['novelty_pattern']==item['novelty_pattern'] and row['zero_count']==item['zero_count'] and row['partition']==item['partition']
        assert row['queries']==item['queries'] and row['target']==item['targets']
        close(row['encoding_state'],encoding[i]);close(row['cue_state'],got['cue'][i]);close(row['prediction'],got['prediction'][i]);close(row['u_readout'],got['uread'][i]);close(row['r_deadline_minus'],got['rb'][i])
        prediction_max=max(prediction_max,float(max(abs(np.array(row['prediction'])-got['prediction'][i]))));state_max=max(state_max,float(max(abs(np.array(row['encoding_state'])-encoding[i]))))
        err=np.abs(got['prediction'][i]-item['targets']);close(row['absolute_errors'],err);close(row['e_comp'],max(err));assert row['joint_success']==bool(np.all(err<.5))
        assert row['sufficient_waveform_query_identifiable']==(item['zero_count']<=1) and row['oracle_is_model_access'] is False
        if item['zero_count']<=1:
            oracle_histories.add(item['id']);ports=[(item['keys'][p]+item['values'][p],max(0.,item['keys'][p]-1.4)) for p in item['positions']]
            pred=[];methods=[]
            for q in item['queries']:
                positive=[(c,n) for c,n in ports if n>0 and n+1.4==q]
                if positive:
                    assert len(positive)==1;c,n=positive[0];pred.append(c-(n+1.4));methods.append('POSITIVE_KEY_INVERSE')
                else:
                    low=[c for c,n in ports if n==0];assert len(low)==1;pred.append(low[0]-q);methods.append('UNIQUE_ZERO_EVENT_BY_VALID_QUERY')
            assert pred==row['offline_waveform_oracle_prediction'] and methods==row['offline_waveform_oracle_method']
            oracle_max=max(oracle_max,float(max(abs(np.array(pred)-item['targets']))));assert oracle_max<1e-12
        else:assert row['offline_waveform_oracle_prediction'] is None and row['offline_waveform_oracle_method'] is None
    assert len(r['cells'])==66
    for c in r['cells']:
        rr=[x for x in rows if x['N']==c['N'] and x['delay_ms']==c['delay_ms']]
        if c['group_kind']=='PATTERN':rr=[x for x in rr if x['novelty_pattern']==c['group']]
        elif c['group_kind']=='PARTITION':rr=[x for x in rr if x['partition']==c['group']]
        elif c['group_kind']=='SUFFICIENT_INFORMATION':rr=[x for x in rr if x['zero_count']<=1]
        else:assert c['group_kind']=='WHOLE' and c['group']=='ALL'
        assert len(rr)==c['n'];close(c['fraction_of_full_N'],len(rr)/2000)
        if not rr:continue
        err=np.abs(np.array([x['prediction'] for x in rr])-np.array([x['target'] for x in rr]));count=int(np.all(err<.5,axis=1).sum())
        assert count==c['joint_count'];close(c['joint_accuracy'],count/len(rr));close(c['role_mae'],err.mean(0));close(c['role_rmse'],np.sqrt(np.mean(err**2,0)));close(c['mean_ecomp'],err.max(1).mean());close(c['max_error'],err.max())
        if c['group_kind']=='WHOLE':
            s=frozen['cell_results'][f"N{c['N']}_delay{int(c['delay_ms'])}ms"]['model_summaries']['M1'];assert c['joint_accuracy']==s['joint_delivery_acc'];close(c['mean_ecomp'],s['mean_e_comp'])
    return dict(inputs=4000,paired_rows=12000,role_outputs=24000,strata_cells=66,independent_generator_rejections=rejections,independent_full_native_recurrence=True,max_prediction_difference=prediction_max,max_encoding_state_difference=state_max,oracle_unique_histories=len(oracle_histories),oracle_max_error=oracle_max,whole_frozen_cells_matched=6,exact_binary64_positive_key_subtractions=sterbenz_checks)

def check_boundary():
    result=json.loads((P/'BOUNDARY_RESULTS.json').read_text());exact=json.loads((P/'exact_witnesses.json').read_text());meta=json.loads((P/'boundary_trace_metadata.json').read_text());z=np.load(P/'actual_calls.npz');outputs=[json.loads(l) for l in (P/'boundary_outputs.jsonl').read_text().splitlines()]
    assert result['status']=='PASS' and digest(P/'boundary_check.py')==result['script_sha256']
    for rel,h in result['source_sha256'].items():assert digest(REPO/rel)==h
    cal=result['calibration'];col={name:i for i,name in enumerate(meta['columns'])};carriers=('M1','M2a','M2b','M2c','M2d')
    specs={
      'low_A':([0,4],['.8','1'],['1','-1'],['.8','1']),
      'low_B':([0,4],['1','.8'],['.8','-.8'],['.8','1']),
      'primed_A':([0,2,4],['2','.8','1'],['0','1.5','-.5'],['.8','1']),
      'primed_B':([0,2,4],['2','1','.8'],['0','1.3','-.3'],['.8','1']),
      'positive_A':([0,4],['1.6','2'],['1','-1'],['1.6','2']),
      'positive_B':([0,4],['2','1.6'],['.6','-.6'],['1.6','2'])}
    exact_by={r['id']:r for r in exact['fixtures']};data={};calls_checked=0;transitions_checked=0;max_output_error=0.
    for name,(pos,ks,vs,qs) in specs.items():
        k=list(map(F,ks));v=list(map(F,vs));q=list(map(F,qs));t=[v[k.index(a)] for a in q];rec=exact_by[name]
        assert list(map(F,rec['keys']))==k and list(map(F,rec['values']))==v and list(map(F,rec['queries']))==q and list(map(F,rec['targets']))==t and rec['positions']==pos
        assert all(F(3,10)<=a<F(5,2) for a in k) and all(-3<=a<3 for a in v) and min(abs(a-b) for i,a in enumerate(k) for b in k[i+1:])>=F(1,20)
        I=[a+b for a,b in zip(k,v)];nu=[max(F(0),a-F(7,5)) for a in k]
        assert len(rec['encoding_segments'])==10
        ports=[];schedule=[]
        for step in range(1000):
            slot=step//200;pres=step%200<100;active=slot in pos;j=pos.index(slot) if active else None
            cp=float(k[j])+float(v[j]) if active and pres else 0.;npv=max(0.,float(k[j])-1.4) if active and pres else 0.
            ports.append([cp,npv,0.]);schedule.append([slot,int(active),int(pres),step*.1])
        ports=np.array(ports);close(z[name+'_schedule'],schedule,tol=2e-14)
        for seg in rec['encoding_segments']:
            j=pos.index(seg['slot']) if seg['slot'] in pos else None;active=seg['phase']=='presentation' and j is not None
            assert F(seg['content'])==(I[j] if active else 0) and F(seg['novelty'])==(nu[j] if active else 0) and seg['mask']==(j is not None)
        data[name]=dict(pos=pos,k=k,v=v,q=q,t=t,I=I,nu=nu,ports=ports)
        for carrier in carriers:
            enc=z[f'{name}_{carrier}_encoding'];assert enc.shape==(1000,23)
            close(enc[:,[col['I_content'],col['novelty'],col['I_query']]],ports,tol=0.)
            initial=[0.,cal['A_0'] if carrier in ('M2b','M2d') else 0.,0.];close(enc[0,:3],initial,tol=0.)
            prior=enc[:-1,[col['u_next'],col['A_next'],col['r_next']]].copy()
            for slot in pos:
                j=slot*200+99
                if carrier in ('M2a','M2d'):prior[j,[0,2]]=0.
                elif carrier=='M2c':prior[j,:]=0.
            close(enc[1:,:3],prior,tol=0.);transitions_checked+=999
            for delay in (0,5,10):
                darr=z[f'{name}_{carrier}_d{delay}_delay'];assert darr.shape==(delay*10,23)
                end=enc[-1,[col['u_next'],col['A_next'],col['r_next']]]
                if delay:
                    close(darr[0,:3],end,tol=0.);close(darr[1:,:3],darr[:-1,[col['u_next'],col['A_next'],col['r_next']]],tol=0.);close(darr[:,[col['I_content'],col['novelty'],col['I_query']]],np.zeros((delay*10,3)),tol=0.)
                    end=darr[-1,[col['u_next'],col['A_next'],col['r_next']]]
                for role in (0,1):
                    qa=z[f'{name}_{carrier}_d{delay}_q{role}'];assert qa.shape==(100,23)
                    close(qa[0,:3],end,tol=0.);close(qa[1:,:3],qa[:-1,[col['u_next'],col['A_next'],col['r_next']]],tol=0.);close(qa[:,[col['I_content'],col['novelty']]],np.zeros((100,2)),tol=0.);close(qa[:,col['I_query']],np.full(100,float(q[role])),tol=0.)
    # Verify every actual numerical recurrent call from declared inputs and states.
    for name in z.files:
        if name.endswith('_schedule'):continue
        a=z[name];calls_checked+=len(a)
        if not len(a):continue
        for p,expected in [('h',.1),('tau_m',10.),('tau_A',15.),('tau_ref',2.),('g_A',1.),('alpha',1.),('A_max',5.),('theta_th',1.),('u_reset',0.),('u_rest',0.),('I_bias',0.)]:close(a[:,col[p]],np.full(len(a),expected),tol=0.)
        carrier=next(c for c in carriers if '_'+c+'_' in name)
        clamp=a[:,col['clamp_A_const']]
        if carrier in ('M2b','M2d'):close(clamp,np.full(len(a),cal['A_0']),tol=0.)
        else:assert np.isnan(clamp).all()
        state,cand,spike,rb=independent_step(a[:,:3],a[:,col['I_content']],a[:,col['novelty']],a[:,col['I_query']])
        if carrier in ('M2b','M2d'):state[:,1]=cal['A_0']
        observed=a[:,[col['u_next'],col['A_next'],col['r_next']]];close(state,observed,tol=0.);close(cand,a[:,col['u_candidate']],tol=0.);close(spike,a[:,col['spike']],tol=0.)
    assert calls_checked==52500==result['physical_recorded_euler_calls'] and len(z.files)==306
    by={(r['history'],r['model'],r['delay']):r for r in outputs};assert len(by)==144
    for row in outputs:
        name=row['history'];d=int(row['delay']);model=row['model'];target=[float(x) for x in data[name]['t']];assert row['targets']==target
        pred=[];abst=[]
        if model=='M0':pred=target;abst=[False,False]
        else:
            carrier='M1' if model.startswith('M4_') else model
            for role in (0,1):
                a=z[f'{name}_{carrier}_d{d}_q{role}'];eligible=a[:99,col['spike']].astype(int);count=int(eligible.sum())
                if not model.startswith('M4_'):
                    u=0. if a[-1,col['r']]>0 else a[-1,col['u_candidate']];pred.append(cal['g_dec']*u+cal['b_dec']);abst.append(False)
                elif model=='M4_Count':pred.append({0:0.,1:-2.,2:-1.,3:0.,4:1.,5:2.}[min(count,5)]);abst.append(count==0)
                else:
                    if count==0:pred.append(0.);abst.append(True)
                    else:
                        first=int(np.flatnonzero(eligible)[0])+1;pred.append(-2.85+min(19,max(0,int((first*.1)/.5)))*.3);abst.append(False)
        close([row['v_hat_A'],row['v_hat_B']],pred,tol=0.);assert [row['abstained_A'],row['abstained_B']]==abst
        err=np.abs(np.array(pred)-target);success=[err[j]<.5 and not abst[j] for j in (0,1)]
        close([row['err_A'],row['err_B']],err,tol=0.);assert [row['succ_A'],row['succ_B']]==[bool(x) for x in success] and row['joint_succ']==bool(all(success));close(row['e_comp'],err.max());close(row['y_hat_add'],pred[0]+pred[1],tol=0.);close(row['y_hat_sub'],pred[0]-pred[1],tol=0.)
    for rec in meta['runner_call_hashes']:
        if rec['model']=='M0':a=np.empty((0,23))
        else:
            name=rec['history'];c='M1' if rec['model'].startswith('M4_') else rec['model'];d=int(rec['delay'])
            a=np.concatenate([z[f'{name}_{c}_encoding'],z[f'{name}_{c}_d{d}_delay'],z[f'{name}_{c}_d{d}_q0'],z[f'{name}_{c}_d{d}_q1']])
        assert len(a)==rec['euler_calls'] and hashlib.sha256(a.tobytes()).hexdigest()==rec['sha256']
    for left,right in [('low_A','low_B'),('primed_A','primed_B'),('positive_A','positive_B')]:
        expect=not left.startswith('positive');a=data[left];b=data[right]
        assert a['I']==b['I'] and (a['nu']==b['nu'])==expect and a['q']==b['q'] and a['pos']==b['pos']
        gap=max(abs(x-y) for x,y in zip(a['t'],b['t']));assert gap==(F(9,5) if expect else F(8,5))
        for carrier in carriers:
            xa=z[f'{left}_{carrier}_encoding'];xb=z[f'{right}_{carrier}_encoding']
            assert np.array_equal(xa[:,col['I_content']],xb[:,col['I_content']])
            assert np.array_equal(xa[:,col['novelty']],xb[:,col['novelty']])==expect
            if expect:
                for suffix in ['encoding']+[f'd{d}_{p}' for d in (0,5,10) for p in ('delay','q0','q1')]:assert np.array_equal(z[f'{left}_{carrier}_{suffix}'],z[f'{right}_{carrier}_{suffix}'],equal_nan=True)
        if expect:
            for model in result['runner_models']:
                for d in (0,5,10):
                    p=[by[(left,model,d)]['v_hat_A'],by[(left,model,d)]['v_hat_B']];q=[by[(right,model,d)]['v_hat_A'],by[(right,model,d)]['v_hat_B']]
                    assert (p==q)==(model!='M0')
    b=F(149,150);primed=z['primed_A_M1_encoding'];gates=[]
    for slot,power in ((2,300),(4,700)):
        exact_gate=F(3,5)*(1-b**100)*b**power;assert exact_gate>0
        active=primed[slot*200:slot*200+100];assert np.all(active[:,col['A']]>0) and np.all(active[:,col['I_content']]>0)
        close(active[0,col['A']],float(exact_gate),tol=1e-14);gates.append(float(active[0,col['A']]))
    return dict(fixture_histories=6,exact_pairs=3,physical_actual_calls_checked=calls_checked,stored_arrays=len(z.files),runner_episodes_checked=144,runner_call_hashes_verified=len(meta['runner_call_hashes']),primed_positive_gate_onsets=gates,all_positive_full_input_collision=False,constructed_pair_minimax_bound='9/10',M0_raw_cache_exception_verified=True,registered_initial_states_masks_interventions_and_independent_queries_verified=True)

def main():
    started=time.perf_counter();tracked=[P/x for x in ('ANALYSIS_PLAN.md','PLAN_AMENDMENT_1.md','run_stratified_m1.py','stratified_inputs.jsonl','stratified_predictions.jsonl','stratified_results.json')]+[REPO/x for x in ('p1_experiments/task_generator.py','p1_experiments/p1b/models_p1b_v2.py','p1_experiments/p1b/runner_p1b.py','p1_experiments/p1b/summary_p1b.json')]
    tracked += [P/x for x in ('boundary_check.py','BOUNDARY_RESULTS.json','exact_witnesses.json','actual_calls.npz','boundary_outputs.jsonl','boundary_trace_metadata.json')]
    before={str(p):digest(p) for p in tracked};strata=check_strata();boundary=check_boundary()
    result=dict(status='PASS',stratified=strata,boundary=boundary,sources_preserved=all(digest(Path(p))==h for p,h in before.items()),seconds=time.perf_counter()-started,
        scope='Offline oracle receives full event ports/mask and fixed-N or at-most-one-zero promise; no such historical buffer is given to M1. Input collision at registered transduction, not raw-array Python API. No population claim from fixtures.')
    assert result['sources_preserved'];(P/'INDEPENDENT_REVIEW_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
