"""Independent hybrid equations, actual-source injection, Adam and ledger audit.

This audit never imports experimental model/common/population. Gradient records
are replayed through Adam; scientific gradients are not independently rederived.
"""
from pathlib import Path
from datetime import datetime
import itertools,json,hashlib,time
import numpy as np
import reference
import population_independent as legacy

R=Path(__file__).resolve().parent
START=time.perf_counter();CHECKS=0;MAXERR=0.
ARMS=('DEND','FULL','NOATT','GAIN2');RATES=(.01,.0025)

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(condition,message):
    global CHECKS
    CHECKS+=1
    if not bool(condition):raise AssertionError(message)
    if time.perf_counter()-START>300:raise TimeoutError('300-second independent audit cap')
def close(actual,expected,message):
    global MAXERR
    a=np.asarray(actual);b=np.asarray(expected)
    check(a.shape==b.shape,message+' shape')
    check(np.isfinite(a).all() and np.isfinite(b).all(),message+' finite')
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    MAXERR=max(MAXERR,error);check(error<1e-8,message+f' error {error}')
def exact(a,b,message):check(np.array_equal(a,b),message)
def base(path,n,seed):
    actual=dict(np.load(path));expected=legacy.make_base(n,seed)
    check(set(actual)==set(expected),'all dataset fields')
    for key in expected:exact(actual[key],expected[key],'regenerated data '+key)
    return actual
def objective(c,y,w):
    logits=.5*c.astype(float);logits-=logits.max(1,keepdims=True)
    probability=np.exp(logits);probability/=probability.sum(1,keepdims=True)
    return float(-np.log(probability[np.arange(len(y)),y]).mean()+.0001*np.mean(w*w))
def metrics(c,y):
    pred=np.full(len(y),-1);pred[c[:,0]>c[:,1]]=0;pred[c[:,1]>c[:,0]]=1
    rec=[float(np.mean(pred[y==j]==j)) for j in (0,1)]
    return dict(balanced_accuracy=sum(rec)/2,accuracy=float(np.mean(pred==y)),coverage=float(np.mean(pred>=0)),silence=float(np.mean(c.sum(1)==0)),recalls=rec,prediction_counts=[int(np.sum(pred==j)) for j in (-1,0,1)])
def structural(arm,T):
    att=arm in ('FULL','NOATT');full=arm=='FULL';N=12
    cost=dict(parameters=176,spiking_cells=8,graded_branches=12,membrane_or_branch_slots=20,history_slots=60 if att else 0,persistent_continuous_slots=80 if att else 20,persistent_array_bytes_per_episode=704 if att else 224,dense_forward_macs=T*156,state_candidate_updates=T*20,graded_local_transfers=T*12,fixed_pair_multiplications=T*6,injection_additions=T*12)
    attention_ops={'history_roll_copy_and_overwrite_writes':6,'attention_unit_steps':1,'qkv_multiplications':11,'surprise_mean_additions':4,'surprise_mean_divisions':1,'gate_tanh_evaluations':1,'gate_context_multiplications':1,'context_additions':4}
    full_ops={'score_multiplications':5,'beta_multiplications':5,'max_and_cached_argmax_comparisons':8,'score_centering_subtractions':5,'exponentials':5,'normalization_divisions':5,'denominator_additions':5,'context_multiplications':5}
    cost.update({name:value*N*T if att else 0 for name,value in attention_ops.items()})
    cost.update({name:value*N*T if full else 0 for name,value in full_ops.items()})
    cost['noatt_context_divisions']=N*T if arm=='NOATT' else 0
    return cost
def compare_raw(saved,ref,label):
    check(set(saved)==set(ref),label+' raw fields')
    for name in ('counts','spikes'):exact(saved[name],ref[name],label+' '+name)
    for name in set(ref)-{'counts','spikes'}:close(saved[name],ref[name],label+' '+name)
    check(np.all((saved['counts']>=0)&(saved['counts']<=8)) and np.array_equal(saved['counts'],np.round(saved['counts'])),label+' native query count range')

def provenance():
    freeze=read(R/'FREEZE.json')
    check(not freeze['scientific_data_generated'],'pre-data freeze')
    for name,h in freeze['files'].items():check(sha(R/name)==h,'frozen source '+name)
    source=read(R/'SOURCE_LOCK.json');old=(R/source['previous_directory']).resolve()
    check(sha(old/'MANIFEST.json')==source['previous_manifest_sha256'],'previous sealed milestone')
    for name,h in source['sources'].items():check(sha(R/name)==h,'unchanged source '+name)
    check(read(R/'PRECHECK.json')['status']=='PASS','preflight passed')
    lock=read(R/'train/TRAIN_LOCK.json');check(not lock['test_generated'],'test generated after training')
    for name,h in lock['files'].items():check(sha(R/'train'/name)==h,'training lock '+name)
    check(datetime.fromisoformat(freeze['utc'])<datetime.fromisoformat(lock['utc']),'freeze before training completion lock')
    for s in range(3):check((R/'evaluate'/f'test_s{s}.npz').stat().st_mtime>datetime.fromisoformat(lock['utc']).timestamp(),'test file created after lock')

def training():
    rows=read(R/'train/ROWS.json');check(len(rows)==24,'24 fits')
    expected={(s,a,lr) for s,a,lr in itertools.product(range(3),ARMS,RATES)}
    check({(r['seed'],r['arm'],r['lr']) for r in rows}==expected,'complete factorial fits')
    total=0
    for s in range(3):
        bases=[base(R/'train'/f'base_s{s}_g{g}.npz',64,3971010000+100*s+10*g) for g in (0,1)]
        for row in [r for r in rows if r['seed']==s]:
            arm=row['arm'];lr=row['lr'];ident=row['id'];z=np.load(R/'train'/f'{ident}.npz')
            check(ident==f's{s}_{arm}_lr{RATES.index(lr)}','fit identifier')
            check(row['h']==6 and row['loss']=='CE' and row['updates']==200,'fixed recipe')
            w=legacy.initial(3971030000+s,6);exact(w,z['initial'],'paired initialization')
            gradients=z['gradients'];counts=z['counts'];history=z['history']
            check(gradients.shape==(200,176) and counts.shape==(200,128,2) and history.shape==(200,4),'all training records retained')
            check(np.isfinite(gradients).all() and np.isfinite(history).all(),'finite training')
            m=np.zeros(176);v=np.zeros(176)
            for k in range(200):
                close(sum(objective(counts[k,g*64:(g+1)*64],bases[g]['y'],w)/2 for g in (0,1)),history[k,0],'stored training objective')
                g=gradients[k];norm=history[k,1];close(np.linalg.norm(g),min(1,norm),'gradient clipping')
                regularizer=.0002*w/176
                close(np.linalg.norm(g*max(1,norm)-regularizer),history[k,2],'task gradient norm decomposition')
                close(np.linalg.norm(regularizer),history[k,3],'regularizer gradient norm')
                if k in (0,199):
                    for j,D in enumerate((0,24)):
                        x,q=reference.assemble(bases[j],D);ref=reference.replay(x,w,arm,q)
                        exact(ref['counts'],counts[k,j*64:(j+1)*64],'first/last independent training counts')
                m=.9*m+.1*g;v=.999*v+.001*g*g
                w=np.clip(w-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);total+=1
            exact(w,z['theta'],'all Adam updates reconstructed')
            check(row['dense_forward_macs']==200*64*72*156,'training forward MACs')
            check(row['dense_backward_macs']==200*64*72*240,'training backward MACs')
            check(row['cost_per_training_pair']==structural(arm,72),'training cell cost ledger')
            check(row['attention_backward_unit_steps']==(200*64*72*12 if arm in ('FULL','NOATT') else 0),'attention backward steps')
            check(row['history_gradient_accumulations']==(200*64*(360-20)*12 if arm in ('FULL','NOATT') else 0),'history backward accumulations')
        print('audit training block',s,'complete',flush=True)
    return total

def evaluation():
    rows=read(R/'evaluate/ROWS.json');train=read(R/'train/ROWS.json')
    check(len(rows)==96,'96 final evaluation conditions');total=0;source_models=0;legacy_conditions=0
    for s in range(3):
        b=base(R/'evaluate'/f'test_s{s}.npz',256,3971020000+100*s)
        for tr in [r for r in train if r['seed']==s]:
            ident=tr['id'];arm=tr['arm'];w=np.load(R/'train'/f'{ident}.npz')['theta']
            rr=[r for r in rows if r['id']==ident];check([r['delay'] for r in rr]==[0,24,48,96],'fixed test delays')
            for row in rr:
                check(all(row[k]==tr[k] for k in ('seed','arm','h','lr','loss')),'eval model identity')
                D=row['delay'];x,q=reference.assemble(b,D)
                check(np.all(x[:,16:q]==0),'strict zero external-input hold')
                ref=reference.replay(x,w,arm,q);saved=dict(np.load(R/'evaluate'/f'{ident}_d{D}.npz'))
                compare_raw(saved,ref,'test '+ident+f' D{D}')
                for key,value in metrics(ref['counts'],b['y']).items():close(value,row[key],'test metric '+key)
                for key,value in structural(arm,q+8).items():check(row[key]==value,'test structural cost '+key)
                close(ref['resources'].mean(0),row['mean_resources'],'full native event and fanout ledger')
                if arm=='DEND':
                    baseline=legacy.replay(x,w,'DEND',6,q)
                    exact(ref['counts'],baseline['counts'],'DEND original population counts')
                    exact(ref['spikes'],baseline['spikes'][:,:,12:],'DEND original population spikes')
                    close(ref['states'],baseline['states'],'DEND original population states')
                    close(ref['resources'],baseline['resources'][:,[1,2,3,4,6,7]],'DEND original population events')
                    legacy_conditions+=1
                if D==0 and arm in ('FULL','NOATT'):
                    original=reference.scalar_oracle(x[:2],w,arm,q)
                    for key in ('counts','spikes'):exact(saved[key][:2],original[key],'actual-source scalar '+key)
                    for key in ('states','histories','injections'):close(saved[key][:2],original[key],'actual-source scalar '+key)
                    source_models+=1
                total+=1
            print('audit evaluation',ident,'complete',flush=True)
    return total,source_models,legacy_conditions

def main():
    provenance();updates=training();n,source,baseline=evaluation();science=read(R/'SCIENCE_SUMMARY.json')
    check((updates,n,source,baseline)==(4800,96,12,24),'all planned verification scopes')
    check(science['status']=='COMPLETE' and science['wall_seconds']<840,'bounded scientific completion')
    check((science['fits'],science['updates'],science['evaluations'])==(24,4800,96),'scientific totals')
    train=read(R/'train/ROWS.json');rows=read(R/'evaluate/ROWS.json')
    check(science['dense_forward_macs']==sum(r['dense_forward_macs'] for r in train)+256*sum(r['dense_forward_macs'] for r in rows),'aggregate forward MACs')
    check(science['sample_steps']==24*200*64*72+24*256*sum(D+24 for D in (0,24,48,96)),'all population sample steps')
    result=dict(status='PASS',checks=CHECKS,optimizer_updates=updates,evaluations=n,source_injection_oracle_models=source,unchanged_DEND_conditions=baseline,max_numeric_error=MAXERR,wall_seconds=time.perf_counter()-START,scope='Independent hybrid equations on all native evaluations and first/last training counts; actual original TAN premembrane injection captured on first two D0 episodes of every final FULL/NOATT model; original DEND replay across all DEND test conditions; all stored objectives, Adam updates and resource ledgers. No independently rederived scientific gradients; gradient contract checked separately on isolated anchored-surrogate fixtures.')
    (R/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (R/'AUDIT_FAILURE.json').write_text(json.dumps(dict(reason=str(e),exception=type(e).__name__,checks=CHECKS,wall_seconds=time.perf_counter()-START),indent=2)+'\n',encoding='utf-8');raise
