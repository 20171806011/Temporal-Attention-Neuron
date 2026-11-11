"""Independent native replay, optimizer reconstruction and selection/accounting audit."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from independent import replay,make_base,assemble,initial
R=Path(__file__).resolve().parent;H=R.parent/'TAN-width-study-20261108';START=time.perf_counter();CHECKS=0;MAXERR=0.;RECOVERED={}
PARTS={'d':(0,),'vp':(1,3),'dvp':(0,1,3),'o':(2,),'all':(0,1,2,3)}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(v,msg):
    global CHECKS
    CHECKS+=1
    if not bool(v):raise AssertionError(msg)
    if time.perf_counter()-START>240:raise TimeoutError('240 second independent audit cap')
def close(a,b,msg):
    global MAXERR
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) if np.size(a) else 0.;MAXERR=max(MAXERR,err);check(err<1e-8,msg+f' {err}')
def verify_base(p,n,seed):
    got=dict(np.load(p));b=make_base(n,seed)
    for k,v in b.items():check(np.array_equal(got[k],v),'data regeneration '+k)
    return got
def prediction(c):
    p=np.full(len(c),-1);p[c[:,0]>c[:,1]]=0;p[c[:,1]>c[:,0]]=1;return p
def metric(c,y,row):
    p=prediction(c);rec=[np.mean(p[y==j]==j) for j in (0,1)]
    vals=dict(balanced_accuracy=sum(rec)/2,accuracy=np.mean(p==y),coverage=np.mean(p>=0),silence=np.mean(c.sum(1)==0),recalls=rec,prediction_counts=[sum(p==j) for j in (-1,0,1)])
    for k,v in vals.items():close(v,row[k],k)
    return vals
def objective(c,y,loss,w):
    c=c.astype(float)
    if loss=='RATE':
        target=np.full(c.shape,.125);target[np.arange(len(y)),y]=.75;value=np.mean((c/8-target)**2)
    else:
        z=.5*c;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);value=-np.log(p[np.arange(len(y)),y]).mean()
    return value+.0001*np.mean(w*w)
def training(stage):
    folder=H/'tune' if stage=='tune' else R/stage;rows=read(folder/'ROWS.json');ns=2 if stage=='tune' else 5;b0=3910010000 if stage=='tune' else 3921010000;updates=0
    check(len(rows)==(17 if stage=='tune' else 30),'fixed fit count '+stage)
    for s in range(ns):
        bases=[verify_base(folder/f'base_s{s}_g{g}.npz',64,b0+100*s+10*g) for g in (0,1)]
        for row in [r for r in rows if r['seed']==s]:
            h=row['h'];arm=row['arm'];loss=row['loss'];z=dict(np.load(folder/f'{row["id"]}.npz'));w=initial(b0+20000+s,h);m=np.zeros_like(w);v=m.copy();grad=z['gradients'];cs=z['counts'];hist=z['history']
            check(np.array_equal(w,z['initial']),'paired initialization');check(row['ret']==[.95,.85] and row['lr']==.01,'fixed time constants and rate')
            check(grad.shape==(200,2*h*h+17*h+2) and cs.shape==(200,128,2),'full updates retained')
            for k in range(200):
                lossvalue=sum(objective(cs[k,64*g:64*(g+1)],b['y'],loss,w)/2 for g,b in enumerate(bases));close(lossvalue,hist[k,0],'stored objective');close(np.linalg.norm(grad[k]),min(1,hist[k,1]),'gradient clipping norm')
                if k in (0,199):
                    for g,D in enumerate((0,24)):
                        x,q=assemble(bases[g],D,'clean');rf=replay(x,w,arm,h,q);check(np.array_equal(rf['counts'],cs[k,g*64:(g+1)*64]),'first/last training forward')
                m=.9*m+.1*grad[k];v=.999*v+.001*grad[k]*grad[k];w=np.clip(w-.01*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);updates+=1
            check(np.array_equal(w,z['theta']),'all Adam final parameters')
            check(row['forward_macs']==200*64*72*(2*h*h+14*h) and row['backward_macs']==200*64*72*(4*h*h+16*h),'training MAC accounting')
        print('audit',stage,'block',s,'complete',flush=True)
    return updates
def selection():
    folder=H/'tune';rows=read(folder/'ROWS.json');lock=read(R/'selection'/'SELECTION_LOCK.json')
    for s in (0,1):
        b=verify_base(folder/f'val_s{s}.npz',256,3910020000+100*s)
        for r in [r for r in rows if r['seed']==s]:
            w=np.load(folder/f'{r["id"]}.npz')['theta'];saved=np.load(folder/f'{r["id"]}_val.npz')['counts'];scores=[];cov=[]
            for g,D in enumerate((0,24)):
                x,q=assemble(b,D,'clean');v=replay(x,w,r['arm'],r['h'],q);check(np.array_equal(v['counts'],saved[g]),'all tuning validation counts');m=metric(v['counts'],b['y'],r['conditions'][g]);scores.append(m['balanced_accuracy']);cov.append(m['coverage'])
            close(np.mean(scores),r['score'],'candidate score');close(np.mean(cov),r['coverage'],'candidate coverage')
    for h in (24,48,96):
        for arm in ('DEND','POINT'):
            rank=[]
            for li,loss in enumerate(('RATE','CE')):
                rr=[r for r in rows if (r['h'],r['arm'],r['loss'])==(h,arm,loss) and r['seed']==0];check(len(rr)==1,'one complete historical block each candidate');rank.append((np.mean([r['score'] for r in rr]),np.mean([r['coverage'] for r in rr]),-li,loss))
            check(max(rank)[3]==lock['selected'][f'{arm}_h{h}'],'independent winner')
    check(not lock['final_training_generated'],'selection before final training')
    for r in read(R/'train'/'ROWS.json'):check(r['loss']==lock['selected'][f'{r["arm"]}_h{r["h"]}'],'selected recipe used')
def maps(y,seed):
    rng=np.random.default_rng(seed);o=np.zeros(len(y),int);same=o.copy()
    for j in (0,1):
        ids=np.where(y==j)[0];pool=rng.permutation(np.where(y!=j)[0])
        for k,idx in enumerate(ids):o[idx]=pool[k%len(pool)]
        order=rng.permutation(ids)
        for k,idx in enumerate(order):same[idx]=order[k-1]
    return dict(opposite=o,same=same)
def accounting(v,row,h,arm,T,suffix=False):
    r=v['resources'];keys=('branch_events','hidden_events','output_events','recurrent_deliveries','output_deliveries','hold_branch_events','hold_hidden_events','hold_output_events')
    vals={('suffix_' if suffix else '')+k:r[:,j].mean() for j,k in enumerate(keys[:5] if suffix else keys)}
    if suffix:vals.update(suffix_steps=T,suffix_forward_macs=T*(2*h*h+14*h),suffix_state_updates=T*(3*h+2),suffix_graded_local_transfers=T*2*h if arm=='DEND' else 0)
    else:
        vals.update(branch_deliveries=vals['branch_events'],internal_deliveries=vals['branch_events']+vals['recurrent_deliveries']+vals['output_deliveries'],
          state_updates=T*(3*h+2),graded_local_transfers=T*2*h if arm=='DEND' else 0,forward_macs=T*(2*h*h+14*h),parameters=2*h*h+17*h+2,
          continuous_slots=3*h+2,persistent_continuous=3*h+2,previous_spike_registers=h,output_counters=2,spiking_cells=h+2 if arm=='DEND' else 3*h+2,graded_branches=2*h if arm=='DEND' else 0)
    for k,value in vals.items():
        if row[k] is None:
            check(row.get('origin')=='original_unflushed' and row['id']=='s4_POINT_h48' and row['delay'] in (0,24,48) and row['kind']=='clean' and row['condition']=='baseline','allowed unflushed resource origin')
            key=f"{row['id']}_d{row['delay']}_{row['kind']}_{row['condition']}"
            RECOVERED.setdefault(key,{})[k]=float(value)
        else:close(value,row[k],'resource '+k)
def evaluate():
    out=R/'evaluate';rows=read(out/'ROWS.json');trs=read(out/'TRANSFERS.json');done=0
    for s in range(5):
        b=verify_base(out/f'test_s{s}.npz',256,3921020000+100*s);mapping=maps(b['y'],3921040000+100*s);got=dict(np.load(out/f'donors_s{s}.npz'))
        for k,v in mapping.items():check(np.array_equal(v,got[k]),'donor mapping')
        check(np.all(b['y'][mapping['opposite']]!=b['y']) and np.all(b['y'][mapping['same']]==b['y']) and np.all(mapping['same']!=np.arange(256)),'donor classes and no self')
        for tr in [r for r in read(R/'train'/'ROWS.json') if r['seed']==s]:
            h=tr['h'];arm=tr['arm'];ident=tr['id'];w=np.load(R/'train'/f'{ident}.npz')['theta'];rr=[r for r in rows if r['id']==ident];check(len(rr)==28,'all planned cases');baseline={};counts={}
            for row in rr:
                D=row['delay'];kind=row['kind'];condition=row['condition'];x,q=assemble(b,D,kind);tick=16+D//2
                if kind=='clean':check(np.all(x[:,16:q]==0),'strict zero external hold')
                if condition=='baseline':
                    v=replay(x,w,arm,h,q,boundary_tick=tick);accounting(v,row,h,arm,len(x[0]));T=len(x[0])
                    if kind=='clean' and D in (24,96):baseline[D]=v
                else:
                    typ,part=condition.split('_');idx=np.arange(256) if typ=='identity' else mapping[typ];state=baseline[D]['boundary'];mixed=[a[idx].copy() if j in PARTS[part] else a.copy() for j,a in enumerate(state)]
                    v=replay(x,w,arm,h,q,start=tick,initial=mixed);T=x.shape[1]-tick;accounting(v,row,h,arm,T,True)
                    if part=='all':check(np.array_equal(v['counts'],baseline[D]['counts'][idx]),'whole donor recovery')
                saved=dict(np.load(out/f'{ident}_d{D}_{kind}_{condition}.npz'));check(np.array_equal(v['counts'],saved['counts']),'all eval counts');check(np.array_equal(v['spikes'],saved['spikes']),'all retained spikes');close(v['states'],saved['states'],'all retained states');metric(v['counts'],b['y'],row)
                if condition!='baseline':close(v['resources'][:,:5],saved['resources'],'all suffix resources')
                counts[(D,kind,condition)]=v['counts'];done+=1
            for D in (24,96):
                pb=prediction(baseline[D]['counts']);y=b['y'];don=mapping['opposite']
                for part in PARTS:
                    record=next(r for r in trs if r['id']==ident and r['delay']==D and r['part']==part);op=prediction(counts[D,'clean','opposite_'+part]);sp=prediction(counts[D,'clean','same_'+part]);contrast=[]
                    for j,row in enumerate(record['directions']):
                        ids=y==j;good=ids&(pb==y)&(pb[don]==y[don]);n=int(sum(good));a=np.mean(op[ids]==1-j);bb=np.mean(sp[ids]==1-j);contrast.append(a-bb)
                        vals=dict(n=sum(ids),opposite_target_fraction=a,same_target_fraction=bb,contrast=a-bb,donor_prediction_agreement=np.mean(op[ids]==pb[don][ids]),both_correct_n=n)
                        for k,v in vals.items():close(v,row[k],'transfer '+k)
                        if n:close(np.mean(op[good]==1-j),row['conditional_flip'],'conditional flip')
                        else:check(row['conditional_flip'] is None,'empty conditional denominator')
                    close(np.mean(contrast),record['macro_contrast'],'macro transfer')
            print('audit evaluate',ident,'complete',flush=True)
    return done
def partial_history():
    z=dict(np.load(H/'tune'/'s1_DEND_h48_CE_PARTIAL.npz'));w=initial(3910030001,48);check(np.array_equal(w,z['initial']),'partial initial')
    bases=[verify_base(H/'tune'/f'base_s1_g{g}.npz',64,3910010100+10*g) for g in (0,1)]
    g=z['gradients'];c=z['counts'];hist=z['history'];check(len(g)==199,'preserved 199-update partial');m=np.zeros_like(w);v=m.copy()
    for k in range(len(g)):
        close(sum(objective(c[k,64*j:64*(j+1)],b['y'],'CE',w)/2 for j,b in enumerate(bases)),hist[k,0],'partial objective')
        close(np.linalg.norm(g[k]),min(1,hist[k,1]),'partial clipping')
        if k in (0,len(g)-1):
            for j,D in enumerate((0,24)):
                x,q=assemble(bases[j],D,'clean');vref=replay(x,w,'DEND',48,q);check(np.array_equal(vref['counts'],c[k,j*64:(j+1)*64]),'partial initial/final native counts')
        m=.9*m+.1*g[k];v=.999*v+.001*g[k]*g[k];w=np.clip(w-.01*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
    check(np.array_equal(w,z['theta']),'partial last parameters');return len(g)

def main():
    for p,h in read(R/'FREEZE.json')['files'].items():check(sha(R/p)==h,'source freeze')
    completion=read(R/'COMPLETION_FREEZE.json')
    for p,h in completion['files'].items():check(sha(R/p)==h,'completion source freeze')
    for p,m in completion['original_raw_files'].items():check(sha(R/p)==m['sha256'],'preserved original raw data')
    original=read(R/'completion'/'original_json'/'ROWS.json');extended=read(R/'evaluate'/'ROWS.json')
    check(extended[:756]==original,'all original metadata preserved')
    check(read(R/'completion'/'SUMMARY.json')['new_evaluations']==81,'only 81 missing scientific evaluations')
    for stage,lockname in [('selection','SELECTION_LOCK.json'),('train','TRAIN_LOCK.json')]:
        for p,h in read(R/stage/lockname)['files'].items():check(sha(R/stage/p)==h,'artifact lock '+stage)
    lock=read(R/'train'/'TRAIN_LOCK.json');check(not lock['test_generated'] and lock['selection_sha256']==sha(R/'selection'/'SELECTION_LOCK.json'),'pretest weights and selection lock')
    historical=read(R/'SOURCE_LOCK.json')
    check(sha(H/'MANIFEST.json')==historical['manifest_sha256'],'historical manifest')
    for p,m in historical['files'].items():check(sha(H/p)==m['sha256'],'historical frozen artifact')
    updates=training('tune')+partial_history()+training('train');selection();n=evaluate()
    summary=dict(status='PASS',checks=CHECKS,optimizer_updates=updates,new_optimizer_updates=6000,historical_optimizer_updates=3599,evaluations=n,max_numeric_error=MAXERR,wall_seconds=time.perf_counter()-START,
      scope='Independent einsum equations validated against scalar-per-unit fixtures. All tuning validation and final counts, saved eight-example traces, all Adam updates/objectives, first/last training forwards, donor effects, selection and resource accounting. Not all training gradients independently derived; internal verification, not external replication.')
    check(len(RECOVERED)==3,'exactly three resource recoveries')
    current=read(R/'evaluate'/'ROWS.json')
    for row in current:
        key=f"{row['id']}_d{row['delay']}_{row['kind']}_{row['condition']}"
        if key in RECOVERED:
            row.update(RECOVERED[key]);row['resource_source']='independent_audit_recovery'
    (R/'completion'/'RESOURCE_RECOVERY.json').write_text(json.dumps(RECOVERED,indent=2)+'\n')
    (R/'evaluate'/'ROWS.json').write_text(json.dumps(current,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    summary.update(unflushed_resource_rows_recovered=3,execution_deviation='Original evaluation hit 120s cap; 81 missing cases completed under separately frozen 90s cap. All original raw files unchanged.')
    (R/'AUDIT_COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (R/'AUDIT_COMPLETE_FAILURE.json').write_text(json.dumps(dict(reason=str(e),checks=CHECKS,wall_seconds=time.perf_counter()-START),indent=2)+'\n');raise
