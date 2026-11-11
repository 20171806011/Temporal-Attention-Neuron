"""Independent replay; does not import core_model, data, engine, or study."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from reference import native
R=Path(__file__).resolve().parent
START=time.perf_counter();PRIOR=0.;CHECKS=0;MAXERR=0.
PARTS={'d':(0,),'vp':(1,3),'dvp':(0,1,3),'o':(2,),'all':(0,1,2,3)}
COLS={'d':np.arange(48),'vp':np.r_[48:72,74:98],'dvp':np.r_[0:72,74:98],'o':np.arange(72,74),'all':np.arange(98)}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(v,msg):
    global CHECKS
    CHECKS+=1
    if not bool(v):raise AssertionError(msg)
    if time.perf_counter()-START+PRIOR>180:raise TimeoutError('cumulative independent audit cap')
def close(a,b,msg):
    global MAXERR
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) if np.size(a) else 0.;MAXERR=max(MAXERR,err);check(err<1e-8,msg+f' {err}')
def base(n,s):
    r=np.random.default_rng(s);p=np.repeat(np.arange(16),n//16);r.shuffle(p)
    bits=np.array([[(int(v)//2**j)%2 for j in range(4)] for v in p]);y=np.array([(v[0] and v[1]) or (v[2] and v[3]) for v in bits],int)
    hi=r.uniform(.8,1.2,(n,4));lo=r.uniform(0,.1,(n,4));a=np.where(bits==1,hi,lo)
    return dict(pattern=p,y=y,write=np.clip(a[:,None,:]+r.normal(0,.03,(n,8,4)),0,None),distractor=r.uniform(0,.5,(n,4)),seed=np.array(s))
def verifybase(path,n,s):
    got=dict(np.load(path));reg=base(n,s)
    for k in reg:check(np.array_equal(got[k],reg[k]),'base '+k)
    return got
def assemble(b,delay):
    q=16+delay;x=np.zeros((len(b['y']),q+8,6))
    for t in range(8):x[:,8+t,:4]=b['write'][:,t]
    x[:,q:,4]=1
    check(np.all(x[:,16:q]==0),'strict zero external hold')
    return x,q
def pred(c):
    p=np.full(len(c),-1);p[c[:,0]>c[:,1]]=0;p[c[:,0]<c[:,1]]=1;return p
def metric(c,y,row):
    p=pred(c);rec=[float(np.mean(p[y==j]==j)) for j in (0,1)]
    for k,v in dict(balanced_accuracy=sum(rec)/2,accuracy=np.mean(p==y),coverage=np.mean(p>=0),silence=np.mean(c.sum(1)==0),recalls=rec,prediction_counts=[np.sum(p==j) for j in (-1,0,1)]).items():close(v,row[k],k)
def maps(y,s):
    rng=np.random.default_rng(s);o=np.zeros(len(y),int);same=o.copy()
    for j in (0,1):
        ids=np.where(y==j)[0];pool=rng.permutation(np.where(y!=j)[0])
        for k,idx in enumerate(ids):o[idx]=pool[k%len(pool)]
        order=rng.permutation(ids)
        for k,idx in enumerate(order):same[idx]=order[k-1]
    return {'opposite':o,'same':same}
def feature(v,ids):
    return np.array([np.concatenate((v['states'][ids,t-1],v['spikes'][ids,t-1,48:72]),axis=1) for t in range(28,41)])
def probes(stage,ident,s,w,cfg,fit,test,fitv,testv):
    saved=dict(np.load(R/stage/f'{ident}_probes.npz'));rows=[a for a in read(R/stage/'PROBES.json') if a['id']==ident]
    fi=np.flatnonzero((fit['pattern']==3)|(fit['pattern']==12));ti=np.flatnonzero((test['pattern']==3)|(test['pattern']==12))
    a=feature(fitv,fi);b=feature(testv,ti);close(a,saved['fit_features'],'fit native features');close(b,saved['test_features'],'test native features')
    check(np.array_equal(fit['pattern'][fi],saved['fit_pattern']),'fit AB patterns');check(np.array_equal(test['pattern'][ti],saved['test_pattern']),'test AB patterns')
    fy=(fit['pattern'][fi]==12).astype(int);ty=(test['pattern'][ti]==12).astype(int)
    check(len(fi)==(32 if stage=='old' else 16) and len(ti)==64,'fixed probe sizes')
    for t in range(28,41):
        labels=np.random.default_rng(3111910000+100*s+t).permutation(fy);check(np.array_equal(labels,saved[f'shuffled_{t}']),'shuffled labels')
        raw=b[t-28,:,:48];ca=raw[ty==0].mean(0);cb=raw[ty==1].mean(0);den=np.linalg.norm(ca)*np.linalg.norm(cb)
        geom={'d_centroid_cosine':float(ca@cb/den) if den else None,'d_centroid_distance':np.linalg.norm(ca-cb),
              'd_within_rms':np.sqrt(np.mean(np.r_[np.sum((raw[ty==0]-ca)**2,1),np.sum((raw[ty==1]-cb)**2,1)]))}
        for part,cols in COLS.items():
            row=next(r for r in rows if r['time']==t and r['part']==part)
            for k,v in geom.items():
                if v is None:check(row[k] is None,'undefined cosine')
                else:close(v,row[k],k)
            x=a[t-28][:,cols];mu=x.mean(0);sd=np.maximum(np.sqrt(np.mean((x-mu)**2,axis=0)),1e-8);z=(x-mu)/sd;ztest=(b[t-28][:,cols]-mu)/sd
            u,d,vt=np.linalg.svd(z,full_matrices=False)
            for kind,lab in [('real',fy),('shuffled',labels)]:
                weight=vt.T@((d/(d*d+1))*(u.T@(2*lab-1)));scores=ztest@weight;p=(scores>=0).astype(int);prefix=f'{t}_{part}_{kind}_'
                for k,v in [('mean',mu),('std',sd),('weight',weight),('scores',scores)]:close(v,saved[prefix+k],'independent ridge '+k)
                check(np.array_equal(p,saved[prefix+'prediction']),'ridge labels');close(np.mean(p==ty),row[kind+'_accuracy'],'heldout probe score')

def costs(ref,row,arm,q,ret):
    z=ref['spikes'];T=z.shape[1];branch=z[:,:,:48].sum((1,2)).mean()
    vals=dict(branch_events=branch,hidden_events=z[:,:,48:72].sum((1,2)).mean(),output_events=z[:,:,72:].sum((1,2)).mean(),
        hold_hidden_events=z[:,16:q,48:72].sum((1,2)).mean(),hold_branch_events=z[:,16:q,:48].sum((1,2)).mean(),
        recurrent_deliveries=ref['rec'].mean(),output_deliveries=ref['out'].mean(),branch_deliveries=branch,
        internal_deliveries=(ref['rec']+ref['out']).mean()+branch,graded_local_transfers=48*T if arm=='DEND' else 0,
        state_updates=74*T,forward_macs=1488*T,parameters=1562,continuous_stored=74,persistent_continuous=26 if ret[0]==0 else 74,
        previous_spike_registers=24,output_counters=2)
    for k,v in vals.items():close(v,row[k],'accounting '+k)

def profile(stage):
    out=R/stage;rows=read(out/'ROWS.json');trs=read(out/'TRANSFERS.json');ids=list(dict.fromkeys(r['id'] for r in rows));done=0
    for ident in ids:
        rr=[r for r in rows if r['id']==ident];brow=next(r for r in rr if r['condition']=='baseline');s=brow['seed'];cfg=brow['config'];arm=cfg['arm'];ret=cfg['ret']
        if stage=='old':
            b=dict(np.load(R/'legacy'/f'base_s{s}.npz'));test=verifybase(out/f'probe_s{s}.npz',512,3110010000+10*s);fit=b
            w=np.load(R/'legacy'/f'{ident}.npy');mapseed=3110020000+10*s
        else:
            b=verifybase(out/f'test_s{s}.npz',512,3111020000+100*s);test=b
            bb=[dict(np.load(R/'train'/f'base_s{s}_g{g}.npz')) for g in (0,1)]
            fit={k:np.concatenate([a[k] for a in bb]) for k in ('pattern','y','write','distractor')}
            w=np.load(R/'train'/f'{ident}.npz')['theta'];mapseed=3111040000+100*s
        mapping=maps(b['y'],mapseed);savedmaps=dict(np.load(out/f'donors_s{s}.npz'))
        for k,v in mapping.items():check(np.array_equal(v,savedmaps[k]),'donor regeneration')
        check(np.all(b['y'][mapping['opposite']]!=b['y']) and np.all(b['y'][mapping['same']]==b['y']) and np.all(mapping['same']!=np.arange(len(b['y']))),'donor classes and no self')
        x,q=assemble(b,24);ref=native(x,w,arm,24,ret,q);c=ref['counts'];state=[ref['states'][:,27,:48],ref['states'][:,27,48:72],ref['states'][:,27,72:],ref['spikes'][:,27,48:72]]
        counts={}
        check(len(rr)==(12 if stage=='old' else 13),'all planned evaluations')
        for row in rr:
            name=row['condition'];saved=dict(np.load(out/f'{ident}_{name}.npz'))
            if name=='baseline':v=ref;sl=slice(None);costs(v,row,arm,q,ret)
            elif name=='d0':
                xx,qq=assemble(b,0);v=native(xx,w,arm,24,ret,qq);sl=slice(None);costs(v,row,arm,qq,ret)
            else:
                kind,part=name.split('_');idx=np.arange(len(c)) if kind=='identity' else mapping[kind]
                mixed=[a[idx].copy() if j in PARTS[part] else a.copy() for j,a in enumerate(state)]
                v=native(x,w,arm,24,ret,q,start=28,initial_state=mixed);sl=slice(28,None)
                close(v['rec'],saved['rec'],'suffix rec deliveries');close(v['out'],saved['out'],'suffix out deliveries')
                close(v['rec'].mean(),row['suffix_recurrent_deliveries'],'suffix mean rec');close(v['out'].mean(),row['suffix_output_deliveries'],'suffix mean out')
                if part=='all':check(np.array_equal(v['counts'],c[idx]),'whole donor counts')
            check(np.array_equal(v['counts'],saved['counts']),'all native counts');check(np.array_equal(v['spikes'][:8,sl],saved['spikes']),'all saved native events')
            close(v['states'][:8,sl],saved['states'],'all saved states');metric(v['counts'],b['y'],row);counts[name]=v['counts'];done+=1
        if ret[0]==0:
            for kind in ('opposite','same'):
                check(np.array_equal(counts[kind+'_d'],c),'rd0 d no-op');check(np.array_equal(counts[kind+'_vp'],counts[kind+'_dvp']),'rd0 vp=dvp')
        for part in PARTS:
            record=next(r for r in trs if r['id']==ident and r['part']==part);op=pred(counts['opposite_'+part]);same=pred(counts['same_'+part]);pb=pred(c);y=b['y'];don=mapping['opposite'];contrasts=[]
            for j,row in enumerate(record['directions']):
                sel=y==j;good=sel&(pb==y)&(pb[don]==y[don]);n=int(sum(good));a=np.mean(op[sel]==1-j);bb=np.mean(same[sel]==1-j);contrasts.append(a-bb)
                vals=dict(n=sum(sel),opposite_target_fraction=a,same_target_fraction=bb,contrast=a-bb,donor_prediction_agreement=np.mean(op[sel]==pb[don][sel]),both_baseline_correct_n=n)
                for k,v in vals.items():close(v,row[k],'transfer '+k)
                if n:close(np.mean(op[good]==1-j),row['conditional_flip'],'conditional flip')
                else:check(row['conditional_flip'] is None,'empty conditional denominator')
            close(np.mean(contrasts),record['macro_contrast'],'macro transfer')
        if fit is b:fitv=ref
        else:xx,qq=assemble(fit,24);fitv=native(xx,w,arm,24,ret,qq)
        if test is b:testv=ref
        else:xx,qq=assemble(test,24);testv=native(xx,w,arm,24,ret,qq)
        probes(stage,ident,s,w,cfg,fit,test,fitv,testv)
        print('audit',stage,ident,'complete',flush=True)
    return done

def training():
    out=R/'train';lock=read(out/'TRAIN_LOCK.json');check(not lock['test_generated'],'test generated after training lock')
    for p,h in lock['files'].items():check(sha(out/p)==h,'training lock '+p)
    rows=read(out/'ROWS.json');check(len(rows)==15,'unconditional 15 fits');updates=0
    for s in range(5):
        groups=[verifybase(out/f'base_s{s}_g{g}.npz',64,3111010000+100*s+10*g) for g in (0,1)]
        for row in [a for a in rows if a['seed']==s]:
            saved=dict(np.load(out/f'{row["id"]}.npz'));rng=np.random.default_rng(3111030000+s);w=np.r_[rng.normal(0,.45,288),rng.normal(0,.25,1152)*.5,np.full(48,.15),np.full(24,.05),rng.normal(0,.5,48)*.5,np.full(2,.1)]
            check(np.array_equal(w,saved['initial']),'shared independent initialization');cfg=row['config'];g=saved['gradients'];c=saved['counts'];h=saved['history']
            check(g.shape==(200,1562) and c.shape==(200,128,2),'all 200 updates');m=np.zeros_like(w);v=m.copy()
            for k in range(200):
                loss=0.
                for j,b in enumerate(groups):
                    cc=c[k,64*j:64*(j+1)];y=b['y']
                    if cfg['loss']=='RATE':
                        target=np.full_like(cc,.125);target[np.arange(64),y]=.75;loss+=np.mean((cc/8-target)**2)/2
                    else:
                        z=.5*cc;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);loss-=np.log(p[np.arange(64),y]).mean()/2
                    if k in (0,199):
                        xx,qq=assemble(b,(0,24)[j]);vv=native(xx,w,cfg['arm'],24,cfg['ret'],qq);check(np.array_equal(vv['counts'],cc),'initial/final training native counts')
                loss+=.0001*np.mean(w*w);close(loss,h[k,0],'every stored objective');close(np.linalg.norm(g[k]),min(1,h[k,1]),'every clipping norm')
                m=.9*m+.1*g[k];v=.999*v+.001*g[k]*g[k];w=np.clip(w-.01*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);updates+=1
            check(np.array_equal(w,saved['theta']),'exact Adam final')
        print('audit training block',s,'complete',flush=True)
    return updates

def main(stage):
    global PRIOR
    if stage=='new':PRIOR=read(R/'AUDIT_old.json')['wall_seconds']
    for p,h in read(R/'FREEZE.json')['files'].items():check(sha(R/p)==h,'frozen source '+p)
    updates=training() if stage=='new' else 0;n=profile(stage)
    result=dict(status='PASS',stage=stage,checks=CHECKS,optimizer_updates=updates,evaluations=n,max_numeric_error=MAXERR,
        wall_seconds=time.perf_counter()-START,prior_audit_seconds=PRIOR,
        scope='Independent scalar native replay of all intervention counts and retained traces, donor maps, metrics, state features, SVD ridge probes, data generation; all Adam updates and stored objectives, first/last training forwards. Not an independent derivation of every training gradient.')
    (R/f'AUDIT_{stage}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    try:main(sys.argv[1])
    except Exception as e:
        (R/f'AUDIT_{sys.argv[1]}_FAILURE.json').write_text(json.dumps(dict(reason=str(e),checks=CHECKS,wall_seconds=time.perf_counter()-START),indent=2)+'\n');raise
