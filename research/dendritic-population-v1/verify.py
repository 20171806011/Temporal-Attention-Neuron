"""Separate scalar implementation and saved-evidence audit; no model/run imports."""
from pathlib import Path
import argparse,hashlib,json,math,time
import numpy as np
P=Path(__file__).resolve().parent
PAIR_LIST=((0,1),(2,3),(0,2),(1,3),(0,3),(1,2))
BRANCHES=(((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2)))
ARMS=('DEND','POOLED','POINT_NETWORK')
REGIMES=(('synchronous_clean',0,0.),('jitter4_noise03',4,.03),('jitter8_noise10',8,.10))
checks=0

def ensure(ok,label):
    global checks
    checks+=1
    if not ok:raise AssertionError(label)

def independent_episode(wave,arm):
    d=[[0.,0.] for _ in range(3)];h=[0.]*3;o=[0.]*3;last=[0]*3
    state=[];candidate=[];spikes=[];cost=[0.]*6
    for row in wave:
        oldlast=last[:];ds=[];dc=[];hidden_current=[];hidden_spikes=[];hc=[]
        for j,pair in enumerate(BRANCHES):
            bj=[];cc=[]
            for b,channels in enumerate(pair):
                drive=float(row[channels[0]])+float(row[channels[1]])
                cost[0]+=abs(drive);cost[1]+=drive**2
                val=.8*d[j][b]+.2*drive;cc.append(val)
                event=int(arm=='POINT_NETWORK' and val>1.1)
                bj.append(event);d[j][b]=0. if event else val
            ds.extend(bj);dc.extend(cc)
            if arm=='POINT_NETWORK':current=float(sum(bj))
            elif arm=='DEND':current=sum(max(v-1.1,0.) for v in d[j])
            else:current=max(sum(d[j])-1.1,0.)
            cost[2]+=abs(current);cost[3]+=current**2
            val=.5*h[j]+current;event=int(val>.5)
            hc.append(val);hidden_spikes.append(event);h[j]=0. if event else val
        output_spikes=[];oc=[]
        for j in range(3):
            current=.6*oldlast[j];cost[4]+=abs(current);cost[5]+=current**2
            val=.5*o[j]+current;event=int(val>.5)
            oc.append(val);output_spikes.append(event);o[j]=0. if event else val
        last=hidden_spikes[:]
        state.append([v for pair in d for v in pair]+h[:]+o[:])
        candidate.append(dc+hc+oc);spikes.append(ds+hidden_spikes+output_spikes)
    return np.array(state),np.array(candidate),np.array(spikes,dtype=np.uint8),np.array(cost)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default=str(P/'results'));args=ap.parse_args()
    folder=Path(args.results);started=time.perf_counter()
    report=json.loads((folder/'results.json').read_text());freeze=json.loads((P/'FREEZE.json').read_text())
    ensure(hashlib.sha256((P/'FREEZE.json').read_bytes()).hexdigest()==report['freeze_sha256'],'freeze identity')
    for file,digest in freeze['sha256'].items():ensure(hashlib.sha256((P/file).read_bytes()).hexdigest()==digest,'frozen source '+file)
    for file,digest in report['data_hashes'].items():ensure(hashlib.sha256((folder/file).read_bytes()).hexdigest()==digest,'saved data '+file)
    comparisons={};max_state=0.;max_candidate=0.;max_cost=0.;updates=0;episodes=0;sample_hashes=set()
    for split,n,seed in [('development',120,2026110311),('validation',120,2026110312),('test',600,2026110313)]:
        data=np.load(folder/(split+'_inputs.npz'),allow_pickle=False)
        # Independently regenerate the frozen draws; no primary generator import.
        rng=np.random.default_rng(seed);ids=rng.permutation(np.repeat(np.arange(6),n//6))
        ensure(np.array_equal(ids,data['pair_id']),'pair RNG '+split)
        ensure(np.array_equal(rng.uniform(.8,1.2,(n,2)),data['amplitudes']),'amplitude RNG '+split)
        ensure(np.array_equal(rng.random(n),data['jitter_uniform']),'jitter RNG '+split)
        ensure(np.array_equal(rng.standard_normal((n,32,4)),data['noise_innovations']),'noise RNG '+split)
        for i in range(n):
            digest=hashlib.sha256(data['noise_innovations'][i].tobytes()).hexdigest()
            ensure(digest not in sample_hashes,'split/sample uniqueness');sample_hashes.add(digest)
            ensure(tuple(data['pairs'][i])==PAIR_LIST[int(ids[i])],'declared pair')
        ensure(np.array_equal(data['label'],ids//2),'labels and pair balance '+split)
        for ri,(regime,J,sigma) in enumerate(REGIMES):
            for i in range(n):
                stim=np.zeros((32,4));a,b=data['pairs'][i];on=4+int(data['jitter_uniform'][i]*(J+1))
                stim[4:14,a]=data['amplitudes'][i,0];stim[on:on+10,b]=data['amplitudes'][i,1]
                ensure(np.array_equal(np.maximum(stim+sigma*data['noise_innovations'][i],0.),data['inputs'][ri,i]),'waveform and pairing')
            for arm in ARMS:
                with np.load(folder/f'{split}__{regime}__{arm}.npz',allow_pickle=False) as archive:
                    saved={key:archive[key] for key in archive.files}
                for i,wave in enumerate(data['inputs'][ri]):
                    state,candidate,spikes,cost=independent_episode(wave,arm);episodes+=1;updates+=32
                    e=float(np.max(np.abs(state-saved['states'][i])));max_state=max(max_state,e);ensure(e<1e-12,'full scalar state')
                    e=float(np.max(np.abs(candidate-saved['candidates'][i])));max_candidate=max(max_candidate,e);ensure(e<1e-12,'full pre-reset candidates')
                    e=float(np.max(np.abs(cost-saved['costs'][i])));max_cost=max(max_cost,e);ensure(e<1e-10,'per-episode current costs')
                    ensure(np.array_equal(spikes,saved['spikes'][i]),'every native spike')
                    ensure(np.array_equal(spikes[1:,9:12],spikes[:-1,6:9]),'one-step native relay')
                counts=saved['spikes'][:,:,9:12].sum(axis=1);ensure(np.array_equal(counts,saved['counts']),'spike-count observation')
                pred=[]
                for c in counts:
                    wins=np.flatnonzero(c==max(c));pred.append(int(wins[0]) if len(wins)==1 and max(c)>0 else -1)
                pred=np.array(pred);ensure(np.array_equal(pred,saved['predictions']),'unique-winner decoder')
                truth=data['label'];correct=pred==truth;covered=pred>=0;m=report['splits'][split][regime][arm]
                ensure(float(correct.mean())==m['accuracy'],'unconditional accuracy')
                ensure(float(covered.mean())==m['coverage'],'coverage')
                expected=float(correct[covered].mean()) if covered.any() else None
                ensure(expected==m['conditional_accuracy'],'conditional accuracy')
                cm=[[0]*4 for _ in range(3)]
                for y,prediction in zip(truth,pred):cm[int(y)][int(prediction) if prediction>=0 else 3]+=1
                ensure(cm==m['confusion'],'confusion including abstention')
                for name,lo,hi in [('branch_spikes_mean',0,6),('hidden_spikes_mean',6,9),('output_spikes_mean',9,12)]:
                    ensure(float(saved['spikes'][:,:,lo:hi].sum(axis=(1,2)).mean())==m[name],'spike resource '+name)
                hit_channel=saved['spikes'][np.arange(n),:,9+truth];hit=hit_channel.any(axis=1);first=hit_channel.argmax(axis=1)
                ensure(float(hit.mean())==m['correct_channel_firing_coverage'],'correct-channel coverage')
                ensure((float(first[hit].mean()) if hit.any() else None)==m['correct_channel_first_step_conditional'],'conditional delay')
                ensure(float(np.where(hit,first,32).mean())==m['correct_channel_first_step_capped'],'capped delay')
                ensure(np.allclose(saved['costs'].mean(axis=0),m['mean_costs'],rtol=0,atol=1e-10),'mean costs')
                ensure(float(np.abs(saved['states']).max())==m['max_absolute_state'],'state maximum')
                if split=='test':comparisons[regime,arm]=correct
        if split=='test':test_ids=ids
    draws=np.load(folder/'bootstrap_indices.npy',allow_pickle=False)
    rng=np.random.default_rng(2026110314);expected=np.concatenate([rng.choice(np.flatnonzero(test_ids==i),size=(2000,100),replace=True) for i in range(6)],axis=1)
    ensure(np.array_equal(draws,expected),'stratified paired bootstrap draw identity')
    for regime,_,_ in REGIMES:
        for control in ['POOLED','POINT_NETWORK']:
            delta=comparisons[regime,'DEND'].astype(float)-comparisons[regime,control]
            ci=np.quantile(delta[draws].mean(axis=1),[.025,.975]);saved=report['test_comparisons'][regime][control]
            ensure(delta.mean()==saved['difference'],'paired point estimate')
            ensure(np.array_equal(ci,saved['percentile95']),'paired interval')
    p=report['test_comparisons']['jitter4_noise03']['POOLED']
    ensure(report['H1_pass']==bool(p['difference']>.1 and p['percentile95'][0]>0),'primary decision')
    result={'status':'PASS','checks':checks,'model_episodes':episodes,'global_updates':updates,'scalar_state_updates':updates*12,'max_state_residual':max_state,'max_candidate_residual':max_candidate,'max_cost_residual':max_cost,'seconds':time.perf_counter()-started,'scope':'Separately implemented complete scalar replay, source/data hashes, stimulus generation/pairing, all predictions/costs and bootstrap. Same AI-assisted workflow, not external human replication.'}
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
