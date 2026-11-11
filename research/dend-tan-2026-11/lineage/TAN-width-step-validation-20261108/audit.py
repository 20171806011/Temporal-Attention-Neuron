"""Independent native replay, optimizer reconstruction and selection/accounting audit."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from independent import replay,make_base,assemble,initial
R=Path(__file__).resolve().parent;START=time.perf_counter();CHECKS=0;MAXERR=0.
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
def candidate(v,w,h):
    wo=w[-2-2*h:-2].reshape(h,2);ob=w[-2:];n,T,_=v['states'].shape
    prev_o=np.concatenate((np.zeros((n,1,2)),v['states'][:,:-1,3*h:]),1)
    prev_p=np.concatenate((np.zeros((n,1,h)),v['spikes'][:,:-1,2*h:3*h]),1)
    return .85*prev_o+np.einsum('nti,ij->ntj',prev_p,wo,optimize=False)+ob
def training():
    folder=R/'train';rows=read(folder/'ROWS.json');updates=0;cp=[0,1,2,5,10,25,50,100,199]
    check(len(rows)==24,'exactly 24 fits')
    for s in range(3):
        bases=[verify_base(folder/f'base_s{s}_g{g}.npz',64,3931010000+100*s+10*g) for g in (0,1)]
        for row in [r for r in rows if r['seed']==s]:
            h=row['h'];arm=row['arm'];lr=row['lr'];z=dict(np.load(folder/f'{row["id"]}.npz'));w=initial(3931030000+s,h);m=np.zeros_like(w);v=m.copy();grad=z['gradients'];cs=z['counts'];hist=z['history']
            check(np.array_equal(w,z['initial']),'paired initialization');check(row['ret']==[.95,.85] and row['loss']=='CE' and lr in (.01,.0025),'fixed factorial recipe')
            check(grad.shape==(200,2*h*h+17*h+2) and cs.shape==(200,128,2),'full updates retained');check(z['completed_updates'].tolist()==cp,'fixed diagnostic checkpoints')
            for k in range(200):
                close(sum(objective(cs[k,64*g:64*(g+1)],b['y'],'CE',w)/2 for g,b in enumerate(bases)),hist[k,0],'stored objective');close(np.linalg.norm(grad[k]),min(1,hist[k,1]),'gradient clipping norm')
                if k in cp:
                    for g,D in enumerate((0,24)):
                        x,q=assemble(bases[g],D,'clean');n=64 if k in (0,199) else 8;rf=replay(x[:n],w,arm,h,q)
                        check(np.array_equal(rf['counts'],cs[k,g*64:g*64+n]),'training native checkpoint counts')
                        close(candidate(rf,w,h),z[f'oc_D{D}'][cp.index(k)],'output candidates from independently replayed states')
                m=.9*m+.1*grad[k];v=.999*v+.001*grad[k]*grad[k];w=np.clip(w-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);updates+=1
            check(np.array_equal(w,z['theta']),'all Adam final parameters')
            check(row['forward_macs']==200*64*72*(2*h*h+14*h) and row['backward_macs']==200*64*72*(4*h*h+16*h),'training MAC accounting')
            check(row['diagnostic_forward_macs']==9*8*72*(2*h*h+14*h),'additional diagnostic MAC accounting')
        print('audit training block',s,'complete',flush=True)
    return updates

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
    for k,value in vals.items():close(value,row[k],'resource '+k)
def evaluate():
    out=R/'evaluate';rows=read(out/'ROWS.json');trs=read(out/'TRANSFERS.json');done=0
    for s in range(3):
        b=verify_base(out/f'test_s{s}.npz',256,3931020000+100*s);mapping=maps(b['y'],3931040000+100*s);got=dict(np.load(out/f'donors_s{s}.npz'))
        for k,v in mapping.items():check(np.array_equal(v,got[k]),'donor mapping')
        check(np.all(b['y'][mapping['opposite']]!=b['y']) and np.all(b['y'][mapping['same']]==b['y']) and np.all(mapping['same']!=np.arange(256)),'donor classes and no self')
        for tr in [r for r in read(R/'train'/'ROWS.json') if r['seed']==s]:
            h=tr['h'];arm=tr['arm'];ident=tr['id'];w=np.load(R/'train'/f'{ident}.npz')['theta'];rr=[r for r in rows if r['id']==ident];check(len(rr)==26,'all planned cases');baseline={};counts={}
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
def main():
    for p,h in read(R/'FREEZE.json')['files'].items():check(sha(R/p)==h,'source freeze')
    source=read(R/'SOURCE_LOCK.json');old=(R/source['directory']).resolve()
    check(sha(old/'MANIFEST.json')==source['manifest_sha256'],'sealed historical source')
    for p,h in source['files'].items():check(sha(R/p)==h and sha(old/p)==h,'unchanged native and independent sources')
    lock=read(R/'train/TRAIN_LOCK.json');check(not lock['test_generated'],'weights locked before test generation')
    for p,h in lock['files'].items():check(sha(R/'train'/p)==h,'training lock')
    updates=training();n=evaluate()
    ts=read(R/'train/SUMMARY.json');es=read(R/'evaluate/SUMMARY.json');ss=read(R/'SCIENCE_SUMMARY.json');rows=read(R/'train/ROWS.json')
    check(updates==4800 and n==624,'all preplanned observations')
    check(ss['status']=='COMPLETE' and ss['wall_seconds']<840,'shared scientific wall budget')
    check(ts['forward_macs']==sum(r['forward_macs']+r['diagnostic_forward_macs'] for r in rows),'training total MACs')
    er=read(R/'evaluate/ROWS.json');expected=sum(256*(r['forward_macs'] if r['condition']=='baseline' else r['suffix_forward_macs']) for r in er)
    check(es['forward_macs']==expected and ss['forward_macs']==expected+ts['forward_macs'],'evaluation and aggregate MACs')
    summary=dict(status='PASS',checks=CHECKS,optimizer_updates=updates,evaluations=n,max_numeric_error=MAXERR,wall_seconds=time.perf_counter()-START,
      scope='Independent einsum native equations: all final conditions/counts, retained eight-example traces, all Adam updates/objectives, first/last full training forwards, nine diagnostic native checkpoints per fit and output candidate recovery, donor effects and resources. All training gradients are not independently rederived; internal verification, not external replication.')
    (R/'AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (R/'AUDIT_FAILURE.json').write_text(json.dumps(dict(reason=str(e),checks=CHECKS,wall_seconds=time.perf_counter()-START),indent=2)+'\n');raise
