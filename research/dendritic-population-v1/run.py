"""One bounded prospective run; refuses to overwrite an output directory."""
from pathlib import Path
import argparse,hashlib,json,time,platform
import numpy as np
from model import ARMS,simulate
P=Path(__file__).resolve().parent
PAIRS=np.array([[0,1],[2,3],[0,2],[1,3],[0,3],[1,2]])
REGIMES=(('synchronous_clean',0,0.),('jitter4_noise03',4,.03),('jitter8_noise10',8,.10))
SPLITS=(('development',120,2026110311),('validation',120,2026110312),('test',600,2026110313))
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dataset(n,seed):
    rng=np.random.default_rng(seed)
    pair_id=rng.permutation(np.repeat(np.arange(6),n//6))
    pairs=PAIRS[pair_id];label=pair_id//2
    amps=rng.uniform(.8,1.2,(n,2));jitter=rng.random(n);noise=rng.standard_normal((n,32,4))
    all_x=[]
    for _,J,sigma in REGIMES:
        x=np.zeros((n,32,4));delays=(jitter*(J+1)).astype(int)
        for i in range(n):
            x[i,4:14,pairs[i,0]]=amps[i,0]
            start=4+delays[i];x[i,start:start+10,pairs[i,1]]=amps[i,1]
        all_x.append(np.maximum(x+sigma*noise,0.))
    return dict(inputs=np.array(all_x),pair_id=pair_id,pairs=pairs,label=label,amplitudes=amps,jitter_uniform=jitter,noise_innovations=noise)
def metrics(out,label):
    pred=out['predictions'];spikes=out['spikes'];n=len(label)
    correct=pred==label;covered=pred>=0
    confusion=np.zeros((3,4),int)
    for a,b in zip(label,pred):confusion[a,b if b>=0 else 3]+=1
    correct_channel=spikes[np.arange(n),:,9+label]
    hit=correct_channel.any(axis=1);first=correct_channel.argmax(axis=1)
    return dict(n=n,accuracy=float(correct.mean()),coverage=float(covered.mean()),
      conditional_accuracy=float(correct[covered].mean()) if covered.any() else None,
      confusion_columns=['class0','class1','class2','abstain'],confusion=confusion.tolist(),
      branch_spikes_mean=float(spikes[:,:,:6].sum(axis=(1,2)).mean()),
      hidden_spikes_mean=float(spikes[:,:,6:9].sum(axis=(1,2)).mean()),
      output_spikes_mean=float(spikes[:,:,9:12].sum(axis=(1,2)).mean()),
      correct_channel_firing_coverage=float(hit.mean()),
      correct_channel_first_step_conditional=float(first[hit].mean()) if hit.any() else None,
      correct_channel_first_step_capped=float(np.where(hit,first,32).mean()),
      mean_costs=out['costs'].mean(axis=0).tolist(),max_absolute_state=float(np.abs(out['states']).max()))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=False)
    freeze=json.loads((P/'FREEZE.json').read_text())
    for name,digest in freeze['sha256'].items():
        assert sha(P/name)==digest,'Frozen file changed: '+name
    started=time.perf_counter();summary={};test_outputs={};hashes={}
    for split,n,seed in SPLITS:
        data=dataset(n,seed);file=outdir/(split+'_inputs.npz');np.savez_compressed(file,**data);hashes[file.name]=sha(file)
        summary[split]={}
        for ri,(regime,_,_) in enumerate(REGIMES):
            summary[split][regime]={}
            for arm in ARMS:
                out=simulate(data['inputs'][ri],arm)
                file=outdir/f'{split}__{regime}__{arm}.npz';np.savez_compressed(file,**out);hashes[file.name]=sha(file)
                summary[split][regime][arm]=metrics(out,data['label'])
                if split=='test':test_outputs[(regime,arm)]=out['predictions']==data['label']
        if split=='test':test=data
    rng=np.random.default_rng(2026110314)
    groups=[np.flatnonzero(test['pair_id']==i) for i in range(6)]
    draws=np.concatenate([rng.choice(g,size=(2000,len(g)),replace=True) for g in groups],axis=1)
    file=outdir/'bootstrap_indices.npy';np.save(file,draws);hashes[file.name]=sha(file)
    comparisons={}
    for regime,_,_ in REGIMES:
        comparisons[regime]={}
        for control in ('POOLED','POINT_NETWORK'):
            delta=test_outputs[(regime,'DEND')].astype(float)-test_outputs[(regime,control)]
            boot=delta[draws].mean(axis=1)
            comparisons[regime][control]=dict(difference=float(delta.mean()),percentile95=np.quantile(boot,[.025,.975]).tolist())
    primary=comparisons['jitter4_noise03']['POOLED']
    report=dict(protocol_sha256=freeze['sha256']['PROTOCOL.md'],freeze_sha256=sha(P/'FREEZE.json'),
      versions={'python':platform.python_version(),'numpy':np.__version__},splits=summary,
      test_comparisons=comparisons,H1_pass=bool(primary['difference']>.10 and primary['percentile95'][0]>0),
      no_learning=True,no_parameter_selection=True,execution_seconds=time.perf_counter()-started,
      data_hashes=hashes,analysis_scope='One frozen synthetic task; state-matched not equal biological resources. No external output pulse generator. No universal TAN advantage.')
    dump(outdir/'results.json',report)
    print(json.dumps({'H1_pass':report['H1_pass'],'test_accuracy':{r:{a:summary['test'][r][a]['accuracy'] for a in ARMS} for r,_,_ in REGIMES},'comparisons':comparisons,'seconds':report['execution_seconds']},indent=2))
if __name__=='__main__':main()
