"""Independent native replay, source-class oracle and optimizer reconstruction."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import independent,oracle
R=Path(__file__).resolve().parent;START=time.perf_counter();CHECKS=0;MAXERR=0.
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(v,msg):
    global CHECKS
    CHECKS+=1
    if not bool(v):raise AssertionError(msg)
    if time.perf_counter()-START>300:raise TimeoutError('300 second independent audit cap')
def close(a,b,msg):
    global MAXERR
    err=float(np.max(abs(np.asarray(a)-np.asarray(b)))) if np.size(a) else 0.;MAXERR=max(MAXERR,err);check(err<1e-8,msg+f': {err}')
def base(p,n,seed):
    saved=dict(np.load(p));expected=independent.make_base(n,seed)
    for k,v in expected.items():check(np.array_equal(saved[k],v),'regenerated data '+k)
    return saved
def objective(c,y,w):
    z=.5*c.astype(float);z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
    return -np.log(p[np.arange(len(y)),y]).mean()+.0001*np.mean(w*w)
def metric(c,y,row):
    pred=np.full(len(y),-1);pred[c[:,0]>c[:,1]]=0;pred[c[:,1]>c[:,0]]=1;rec=[np.mean(pred[y==j]==j) for j in (0,1)]
    vals=dict(balanced_accuracy=sum(rec)/2,accuracy=np.mean(pred==y),coverage=np.mean(pred>=0),silence=np.mean(c.sum(1)==0),recalls=rec,prediction_counts=[sum(pred==j) for j in (-1,0,1)])
    for k,v in vals.items():close(v,row[k],'metric '+k)
def structural(arm,h,T):
    N=3*h+2;tan=arm in ('TAN_FULL','TAN_NOATT');full=arm=='TAN_FULL'
    out=dict(parameters=2*h*h+17*h+2,spiking_cells=h+2 if arm=='DEND' else N,graded_branches=2*h if arm=='DEND' else 0,
      membrane_or_branch_slots=N,history_slots=5*N if tan else 0,persistent_continuous_slots=6*N if tan else N,previous_spike_registers=h,output_counters=2,persistent_array_bytes_per_episode=((6*N if tan else N)+h+2)*8,
      dense_forward_macs=T*(2*h*h+14*h),state_candidate_updates=T*N,graded_local_transfers=T*2*h if arm=='DEND' else 0,fixed_pair_multiplications=T*h,
      history_logical_shifts_and_writes=T*5*N if tan else 0,history_roll_copy_and_overwrite_writes=T*6*N if tan else 0)
    common={'tan_unit_steps':1,'tan_qkv_multiplications':11,'tan_surprise_window_additions':4,'tan_surprise_mean_divisions':1,'tan_surprise_current_subtractions':1,'tan_surprise_rectifications':1,'tan_gate_membrane_reset_multiplications':3,'tan_membrane_additions':1,'tan_threshold_comparisons':1,'tan_reset_subtractions':1,'tan_gate_tanh_evaluations':1}
    fullops={'attention_score_multiplications':5,'attention_beta_multiplications':5,'attention_max_comparisons':4,'attention_cached_argmax_comparisons':4,'attention_centering_subtractions':5,'attention_exp_evaluations':5,'attention_normalization_divisions':5,'attention_denominator_additions':5,'attention_context_multiplications':5,'attention_context_additions':4}
    out.update({k:T*N*v if tan else 0 for k,v in common.items()});out.update({k:T*N*v if full else 0 for k,v in fullops.items()});out.update(noatt_context_mean_additions=T*N*4 if arm=='TAN_NOATT' else 0,noatt_context_mean_divisions=T*N if arm=='TAN_NOATT' else 0);return out
def training():
    rows=read(R/'train/ROWS.json');check(len(rows)==24,'24 fixed training cells');updates=0
    for s in range(3):
        bases=[base(R/'train'/f'base_s{s}_g{g}.npz',64,3951010000+100*s+10*g) for g in (0,1)]
        for row in [r for r in rows if r['seed']==s]:
            h=row['h'];lr=row['lr'];arm=row['arm'];z=np.load(R/'train'/f'{row["id"]}.npz');w=independent.initial(3951030000+s,h);g=z['gradients'];c=z['counts'];hist=z['history'];check(np.array_equal(w,z['initial']),'paired initial weights');check(h==6 and lr in (.01,.0025) and row['loss']=='CE','fixed pilot recipe');check(g.shape==(200,176) and c.shape==(200,128,2),'all gradients and training counts retained');m=np.zeros_like(w);v=np.zeros_like(w)
            for k in range(200):
                close(sum(objective(c[k,64*j:64*(j+1)],b['y'],w)/2 for j,b in enumerate(bases)),hist[k,0],'stored CE objective');close(np.linalg.norm(g[k]),min(1,hist[k,1]),'gradient clipping')
                if k in (0,199):
                    for j,D in enumerate((0,24)):
                        x,q=independent.assemble(bases[j],D);ref=independent.replay(x,w,arm,h,q);check(np.array_equal(ref['counts'],c[k,64*j:64*(j+1)]),'first/last independent training counts')
                m=.9*m+.1*g[k];v=.999*v+.001*g[k]*g[k];w=np.clip(w-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);updates+=1
            check(np.array_equal(w,z['theta']),'all Adam updates reconstructed');check(row['dense_forward_macs']==200*64*72*156 and row['dense_backward_macs']==200*64*72*240,'dense training MAC ledger')
            expected=structural(arm,h,72)
            for name,value in row['forward_cell_operations'].items():check(value==200*64*expected[name],'training cell operations '+name)
            check(row['tan_backward_unit_steps']==(200*64*72*20 if arm.startswith('TAN_') else 0),'TAN backward step count');check(row['attention_backward_vjp_unit_steps']==(200*64*72*20 if arm=='TAN_FULL' else 0),'attention backward step count');check(row['history_gradient_accumulations']==(200*64*340*20 if arm.startswith('TAN_') else 0),'window-gradient accumulation count')
        print('audit training block',s,'complete',flush=True)
    return updates
def evaluation():
    rows=read(R/'evaluate/ROWS.json');training=read(R/'train/ROWS.json');check(len(rows)==96,'all 96 evaluations');n=0;source_oracle_models=0
    for s in range(3):
        b=base(R/'evaluate'/f'test_s{s}.npz',256,3951020000+100*s)
        for tr in [r for r in training if r['seed']==s]:
            arm=tr['arm'];h=tr['h'];ident=tr['id'];w=np.load(R/'train'/f'{ident}.npz')['theta'];rr=[r for r in rows if r['id']==ident];check([r['delay'] for r in rr]==[0,24,48,96],'all fixed delays')
            for row in rr:
                D=row['delay'];x,q=independent.assemble(b,D);check(np.all(x[:,16:q]==0),'strict zero-input hold');ref=independent.replay(x,w,arm,h,q);saved=dict(np.load(R/'evaluate'/f'{ident}_d{D}.npz'))
                for name in ('counts','spikes'):check(np.array_equal(ref[name],saved[name]),'all test '+name)
                for name in ('states','candidates','resources')+(('histories','surprises','contexts') if arm.startswith('TAN_') else ()):close(ref[name],saved[name],'all test '+name)
                metric(ref['counts'],b['y'],row);cost=structural(arm,h,q+8)
                for name,value in cost.items():check(row[name]==value,'structural resources '+name)
                keys=('first_layer_events','hidden_events','output_events','recurrent_deliveries','output_deliveries','hold_first_layer_events','hold_hidden_events','hold_output_events')
                for j,name in enumerate(keys):close(ref['resources'][:,j].mean(),row[name],'native events/fanout '+name)
                close(row['first_layer_events'],row['first_layer_local_deliveries'],'local pulse deliveries');close(sum(row[k] for k in ('first_layer_events','recurrent_deliveries','output_deliveries')),row['internal_deliveries'],'total pulse deliveries')
                if D==0 and arm.startswith('TAN_'):
                    original=oracle.replay(x[:8],w,arm,h,q)
                    for name in ('counts','spikes'):check(np.array_equal(original[name],saved[name][:8]),'final fitted model original-class oracle '+name)
                    for name in ('states','histories'):close(original[name],saved[name],'final fitted model original-class oracle '+name)
                    source_oracle_models+=1
                n+=1
            print('audit evaluation',ident,'complete',flush=True)
    return n,source_oracle_models
def main():
    for name,h in read(R/'FREEZE.json')['files'].items():check(sha(R/name)==h,'source freeze '+name)
    source=read(R/'SOURCE_LOCK.json');old=(R/source['source_recovery_directory']).resolve();check(sha(old/'MANIFEST.json')==source['source_manifest_sha256'],'previous milestone sealed source')
    for name,m in source['source_files'].items():check(sha(R/name)==m['sha256'],'unchanged source/original unit '+name)
    lock=read(R/'train/TRAIN_LOCK.json');check(not lock['test_generated'],'weights locked before test generation')
    for name,h in lock['files'].items():check(sha(R/'train'/name)==h,'trained model lock')
    updates=training();n,original=evaluation();ss=read(R/'SCIENCE_SUMMARY.json');tr=read(R/'train/ROWS.json');ev=read(R/'evaluate/ROWS.json');check(updates==4800 and n==96 and original==12,'all preplanned scientific records')
    check(ss['wall_seconds']<840 and ss['status']=='COMPLETE','bounded one-shot science');check(ss['dense_forward_macs']==sum(r['dense_forward_macs'] for r in tr)+256*sum(r['dense_forward_macs'] for r in ev),'aggregate dense MAC ledger')
    summary=dict(status='PASS',checks=CHECKS,optimizer_updates=updates,evaluations=n,source_oracle_final_models=original,max_numeric_error=MAXERR,wall_seconds=time.perf_counter()-START,scope='Independent vector equations for all evaluations and first/last training counts; original scalar TAN class on eight D0 test episodes per final TAN model; all Adam updates/objectives and resource ledgers. Training gradients checked on isolated anchored-surrogate fixtures, not rederived on all scientific updates.')
    (R/'AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (R/'AUDIT_FAILURE.json').write_text(json.dumps(dict(reason=str(e),checks=CHECKS,wall_seconds=time.perf_counter()-START),indent=2)+'\n');raise
