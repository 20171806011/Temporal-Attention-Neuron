"""Independent temporal data, count objectives, native hybrid and optimizer audit.

No experimental model/common/task_data imports. The copied independent neural
reference uses the original TAN source solely for its premembrane injection.
"""
from pathlib import Path
from datetime import datetime
import itertools,json,hashlib,time
import numpy as np
import reference
import independent_data as data
import independent_loss
import population_independent as legacy

R=Path(__file__).resolve().parent
START=time.perf_counter();CHECKS=0;MAXERR=0.
ARMS=('DEND','FULL','NOATT','GAIN2');OBJECTIVES=('CE','COUNT','CECOM')

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
    actual=dict(np.load(path));expected=data.make_base(n,seed)
    check(set(actual)==set(expected),'complete temporal dataset fields')
    for key in expected:exact(actual[key],expected[key],'regenerated temporal data '+key)
    return actual
def objective(c,y,w,kind):
    return independent_loss.loss_grad(c,y,kind)[0]+.0001*np.mean(w*w)
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
    freeze=read(R/'FREEZE.json');check(not freeze['scientific_data_generated'],'pre-data freeze')
    for name,h in freeze['files'].items():check(sha(R/name)==h,'frozen source '+name)
    source=read(R/'SOURCE_LOCK.json');old=(R/source['previous_directory']).resolve()
    check(sha(old/'MANIFEST.json')==source['previous_manifest_sha256'],'previous sealed hybrid milestone')
    for name,h in source['sources'].items():check(sha(R/name)==h,'unchanged source '+name)
    check(read(R/'PRECHECK.json')['status']=='PASS','neural preflight passed')
    check(read(R/'DATA_CHECK.json')['status']=='PASS','data preflight passed')
    check(read(R/'LOSS_CHECK.json')['status']=='PASS','independent count-loss fixture passed')
    check(read(R/'GRADIENT_CHECK.json')['status']=='PASS','anchored objective and component BPTT fixtures passed')
    task=read(R/'TASK_LOCK.json')
    for name,h in task['files'].items():check(sha(R/name)==h,'unchanged temporal task '+name)
    check(sha((R/task['previous_directory']).resolve()/'MANIFEST.json')==task['previous_manifest_sha256'],'sealed temporal milestone')
    lock=read(R/'train/TRAIN_LOCK.json');check(not lock['test_generated'],'test generated after training')
    for name,h in lock['files'].items():check(sha(R/'train'/name)==h,'training lock '+name)
    check(datetime.fromisoformat(freeze['utc'])<datetime.fromisoformat(lock['utc']),'freeze before training completion lock')
    for s in range(3):check((R/'evaluate'/f'test_s{s}.npz').stat().st_mtime>datetime.fromisoformat(lock['utc']).timestamp(),'test file created after lock')

def training():
    rows=read(R/'train/ROWS.json');check(len(rows)==36,'36 fits')
    expected={(s,a,kind) for s,a,kind in itertools.product(range(3),ARMS,OBJECTIVES)}
    check({(r['seed'],r['arm'],r['loss']) for r in rows}==expected,'complete factorial fits')
    total=0
    for s in range(3):
        b=base(R/'train'/f'base_s{s}.npz',64,3991010000+100*s);x,q=data.assemble(b,'nominal')
        check(data.check(b,x,'nominal')['balanced_labels'],'training temporal invariants')
        for row in [r for r in rows if r['seed']==s]:
            arm=row['arm'];lr=row['lr'];kind=row['loss'];ident=row['id'];z=np.load(R/'train'/f'{ident}.npz')
            check(ident==f's{s}_{arm}_{kind}','fit identifier')
            check(row['h']==6 and kind in OBJECTIVES and lr==.0025 and row['updates']==400,'fixed recipe')
            w=legacy.initial(3991030000+s,6);exact(w,z['initial'],'paired initialization')
            gradients=z['gradients'];counts=z['counts'];history=z['history']
            check(gradients.shape==(400,176) and counts.shape==(400,64,2) and history.shape==(400,4),'all training records retained')
            check(np.isfinite(gradients).all() and np.isfinite(history).all(),'finite training')
            components=z['components'];component_gradients=z['component_gradients'];ds=z['diagnostic_steps'].tolist()
            check(components.shape==(400,7) and component_gradients.shape==(21,3,176),'component records retained')
            check(ds==list(range(0,400,20))+[399],'fixed diagnostic update indices')
            check(np.isfinite(components).all() and np.isfinite(component_gradients).all(),'finite component records')
            m=np.zeros(176);v=np.zeros(176)
            for k in range(400):
                close(objective(counts[k],b['y'],w,kind),history[k,0],'stored training objective')
                g=gradients[k];norm=history[k,1];close(np.linalg.norm(g),min(1,norm),'gradient clipping')
                regularizer=.0002*w/176
                close(np.linalg.norm(g*max(1,norm)-regularizer),history[k,2],'task gradient norm decomposition')
                close(np.linalg.norm(regularizer),history[k,3],'regularizer gradient norm')
                terms=independent_loss.decomposition(counts[k],b['y'],kind);l2=.0001*np.mean(w*w)
                expected_components=[terms['discriminative_loss'],terms['common_loss'],l2,terms['task_loss']+l2,np.linalg.norm(terms['discriminative_gradient']),np.linalg.norm(terms['common_gradient']),np.linalg.norm(terms['total_gradient'])]
                close(components[k],expected_components,'independent count-space component losses and norms')
                if k in ds:
                    cg=component_gradients[ds.index(k)]
                    close(cg.sum(0),g*max(1,norm),'stored parameter component sum equals unclipped update gradient')
                    close(cg[0]+cg[1],g*max(1,norm)-regularizer,'discriminative plus common equals task gradient')
                    close(cg[2],regularizer,'sampled exact L2 gradient')
                    if kind=='CE':exact(cg[1],np.zeros(176),'CE common component is absent')
                if k in (0,399):
                    ref=reference.replay(x,w,arm,q)
                    exact(ref['counts'],counts[k],'first/last independent training counts')
                m=.9*m+.1*g;v=.999*v+.001*g*g
                w=np.clip(w-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);total+=1
            exact(w,z['theta'],'all Adam updates reconstructed')
            check(row['dense_forward_macs']==400*64*24*156,'training forward MACs')
            bp=400+(21 if kind!='CE' else 0)
            check(row['backward_passes']==bp and row['diagnostic_backward_passes']==bp-400,'diagnostic backward passes counted')
            check(row['dense_backward_macs']==bp*64*24*240,'training plus diagnostic backward MACs')
            check(row['cost_per_training_episode']==structural(arm,24),'training cell cost ledger')
            check(row['attention_backward_unit_steps']==(bp*64*24*12 if arm in ('FULL','NOATT') else 0),'attention backward steps')
            check(row['history_gradient_accumulations']==(bp*64*(120-10)*12 if arm in ('FULL','NOATT') else 0),'history backward accumulations')
        print('audit training block',s,'complete',flush=True)
    return total

def evaluation():
    rows=read(R/'evaluate/ROWS.json');train=read(R/'train/ROWS.json')
    check(len(rows)==216,'216 final evaluation conditions');total=0;source_models=0;legacy_conditions=0;negative_conditions=0
    for s in range(3):
        b=base(R/'evaluate'/f'test_s{s}.npz',256,3991020000+100*s)
        inputs={condition:data.assemble(b,condition) for condition in data.CONDITIONS}
        for condition,(x,q) in inputs.items():
            check(q==16 and data.check(b,x,condition)['balanced_labels'],'held-out temporal invariants '+condition)
        pairs=data.pairs(b)
        for tr in [r for r in train if r['seed']==s]:
            ident=tr['id'];arm=tr['arm'];w=np.load(R/'train'/f'{ident}.npz')['theta']
            rr=[r for r in rows if r['id']==ident];check([r['condition'] for r in rr]==list(data.CONDITIONS),'all six fixed test conditions')
            for row in rr:
                check(all(row[k]==tr[k] for k in ('seed','arm','h','lr','loss')),'eval model identity')
                condition=row['condition'];x,q=inputs[condition]
                ref=reference.replay(x,w,arm,q);saved=dict(np.load(R/'evaluate'/f'{ident}_{condition}.npz'))
                compare_raw(saved,ref,'test '+ident+' '+condition)
                metric=metrics(ref['counts'],b['y'])
                for key,value in metric.items():close(value,row[key],'test metric '+key)
                for key,value in structural(arm,24).items():check(row[key]==value,'test structural cost '+key)
                close(ref['resources'].mean(0),row['mean_resources'],'full native event and fanout ledger')
                if condition=='tag_removed':
                    for i,j in pairs:exact(ref['counts'][i],ref['counts'][j],'identical input opposite-label pair has identical native counts')
                    close(metric['balanced_accuracy'],.5*metric['coverage'],'tag_removed mathematical BA/coverage identity')
                    close(metric['accuracy'],.5*metric['coverage'],'tag_removed mathematical accuracy/coverage identity');negative_conditions+=1
                elif condition=='query_only':
                    exact(ref['counts'],np.broadcast_to(ref['counts'][0],ref['counts'].shape),'query-only identical native counts')
                    close(metric['balanced_accuracy'],.5*metric['coverage'],'query-only mathematical BA/coverage identity')
                    close(metric['accuracy'],.5*metric['coverage'],'query-only mathematical accuracy/coverage identity');negative_conditions+=1
                if arm=='DEND':
                    baseline=legacy.replay(x,w,'DEND',6,q)
                    exact(ref['counts'],baseline['counts'],'DEND original population counts')
                    exact(ref['spikes'],baseline['spikes'][:,:,12:],'DEND original population spikes')
                    close(ref['states'],baseline['states'],'DEND original population states')
                    close(ref['resources'],baseline['resources'][:,[1,2,3,4,6,7]],'DEND original population events');legacy_conditions+=1
                if condition=='nominal' and arm in ('FULL','NOATT'):
                    original=reference.scalar_oracle(x[:2],w,arm,q)
                    for key in ('counts','spikes'):exact(saved[key][:2],original[key],'actual-source scalar '+key)
                    for key in ('states','histories','injections'):close(saved[key][:2],original[key],'actual-source scalar '+key)
                    source_models+=1
                total+=1
            print('audit evaluation',ident,'complete',flush=True)
    return total,source_models,legacy_conditions,negative_conditions

def main():
    provenance();updates=training();n,source,baseline,negatives=evaluation();science=read(R/'SCIENCE_SUMMARY.json')
    check((updates,n,source,baseline,negatives)==(14400,216,18,54,72),'all planned verification scopes')
    check(science['status']=='COMPLETE' and science['wall_seconds']<840,'bounded scientific completion')
    check((science['fits'],science['updates'],science['evaluations'])==(36,14400,216),'scientific totals')
    train=read(R/'train/ROWS.json');rows=read(R/'evaluate/ROWS.json')
    check(science['dense_forward_macs']==sum(r['dense_forward_macs'] for r in train)+256*sum(r['dense_forward_macs'] for r in rows),'aggregate forward MACs')
    check(science['sample_steps']==36*400*64*24+216*256*24,'all population sample steps')
    result=dict(status='PASS',checks=CHECKS,optimizer_updates=updates,evaluations=n,source_injection_oracle_models=source,unchanged_DEND_conditions=baseline,mathematical_negative_control_conditions=negatives,sampled_parameter_gradient_decompositions=756,max_numeric_error=MAXERR,wall_seconds=time.perf_counter()-START,scope='Independent temporal data and input reconstruction, native hybrid equations on all evaluations and first/last training counts; exact duplicate-input/opposite-label controls; actual original TAN premembrane injection on first two nominal episodes of every final FULL/NOATT model; original DEND replay for all DEND tests; all independent count-space objective components/norms, sampled parameter-gradient algebraic decompositions, Adam updates and ledgers including extra diagnostic backward passes. Scientific parameter gradients not independently rederived; anchored-surrogate contract checked separately.')
    (R/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (R/'AUDIT_FAILURE.json').write_text(json.dumps(dict(reason=str(e),exception=type(e).__name__,checks=CHECKS,wall_seconds=time.perf_counter()-START),indent=2)+'\n',encoding='utf-8');raise
