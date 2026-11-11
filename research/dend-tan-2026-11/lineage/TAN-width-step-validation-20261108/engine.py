"""Batched native evaluation and midpoint transplantation; no label in dynamics."""
import numpy as np
import model
from common import RET,PARTS,basic

def boundary(cache,t,h):
    a=cache[t-1];z=a['states'];return [z[:,:2*h].copy(),z[:,2*h:3*h].copy(),z[:,3*h:].copy(),a['hs'].copy()]
def snapshot(c,k):
    return dict(counts=c,states=np.stack([a['states'][:8] for a in k],1),spikes=np.stack([np.concatenate((a['bs'],a['hs'],a['os']),1)[:8] for a in k],1))
def costs(k,w,h,arm,q):
    _,wr,_,_,wo,_=model.unpack(w,h);n=len(k[0]['hs']);T=len(k)
    vals={a:np.zeros(n) for a in ('branch_events','hidden_events','output_events','hold_branch_events','hold_hidden_events','hold_output_events','recurrent_deliveries','output_deliveries')}
    for t,a in enumerate(k):
        for name,key in [('branch_events','bs'),('hidden_events','hs'),('output_events','os')]:
            vals[name]+=a[key].sum(1)
            if 16<=t<q:vals['hold_'+name]+=a[key].sum(1)
        vals['recurrent_deliveries']+=a['prev']@(wr!=0).sum(1);vals['output_deliveries']+=a['prev']@(wo!=0).sum(1)
    return vals
def evaluate(x,w,arm,h,q,boundary_tick=None):
    cs=[];state=[[],[],[],[]];resources={};saved=None
    for b in range(0,len(x),64):
        c,k=model.forward(x[b:b+64],w,arm,h,ret=RET,query=q);cs.append(c)
        if b==0:saved=snapshot(c,k)
        if boundary_tick is not None:
            for j,v in enumerate(boundary(k,boundary_tick,h)):state[j].append(v)
        for name,v in costs(k,w,h,arm,q).items():resources.setdefault(name,[]).append(v)
    saved['counts']=np.concatenate(cs);r={name:float(np.concatenate(v).mean()) for name,v in resources.items()}
    T=x.shape[1];r.update(branch_deliveries=r['branch_events'],internal_deliveries=r['branch_events']+r['recurrent_deliveries']+r['output_deliveries'],
      state_updates=(3*h+2)*T,graded_local_transfers=2*h*T if arm=='DEND' else 0,forward_macs=(2*h*h+14*h)*T,parameters=model.parameter_count(h),
      continuous_slots=3*h+2,persistent_continuous=3*h+2,previous_spike_registers=h,output_counters=2,spiking_cells=h+2 if arm=='DEND' else 3*h+2,graded_branches=2*h if arm=='DEND' else 0)
    return saved,r,[np.concatenate(v) for v in state] if boundary_tick is not None else None
def suffix(x,w,arm,h,q,start,initial):
    wi,wr,db,vb,wo,ob=model.unpack(w,h);cs=[];savedstates=None;savedspikes=None;resources=[]
    for b in range(0,len(x),64):
        xx=x[b:b+64];d,v,o,p=[a[b:b+64].copy() for a in initial];n=len(xx);c=np.zeros((n,2));states=[];spikes=[]
        vals=np.zeros((n,5))
        for t in range(start,xx.shape[1]):
            dc=RET[0]*d+.25*(xx[:,t]@wi+p@wr+db)
            if arm=='DEND':bs=np.zeros_like(dc);local=np.maximum(dc,0);d=dc
            else:bs=(dc>.5).astype(float);local=bs;d=dc*(1-bs)
            vc=RET[1]*v+.5*local.reshape(n,h,2).sum(2)+vb;hs=(vc>.5).astype(float);v=vc*(1-hs)
            oc=RET[1]*o+p@wo+ob;os=(oc>.5).astype(float);o=oc*(1-os)
            if t>=q:c+=os
            vals[:,0]+=bs.sum(1);vals[:,1]+=hs.sum(1);vals[:,2]+=os.sum(1);vals[:,3]+=p@(wr!=0).sum(1);vals[:,4]+=p@(wo!=0).sum(1)
            if b==0:states.append(np.concatenate((d,v,o),1)[:8]);spikes.append(np.concatenate((bs,hs,os),1)[:8])
            p=hs
        cs.append(c);resources.append(vals)
        if b==0:savedstates=np.stack(states,1);savedspikes=np.stack(spikes,1)
    r=np.concatenate(resources);keys=('branch_events','hidden_events','output_events','recurrent_deliveries','output_deliveries')
    return dict(counts=np.concatenate(cs),states=savedstates,spikes=savedspikes,resources=r),{('suffix_'+k):float(r[:,j].mean()) for j,k in enumerate(keys)}
def transfer(op,same,baseline,y,mapping):
    _,po=basic(op,y);_,ps=basic(same,y);_,pb=basic(baseline,y);don=mapping['opposite'];rows=[]
    for j in (0,1):
        ids=y==j;good=ids&(pb==y)&(pb[don]==y[don]);n=int(sum(good));a=float(np.mean(po[ids]==1-j));b=float(np.mean(ps[ids]==1-j))
        rows.append(dict(direction=f'{j}->{1-j}',n=int(sum(ids)),opposite_target_fraction=a,same_target_fraction=b,contrast=a-b,donor_prediction_agreement=float(np.mean(po[ids]==pb[don][ids])),both_correct_n=n,conditional_flip=float(np.mean(po[good]==1-j)) if n else None))
    return dict(directions=rows,macro_contrast=float(np.mean([r['contrast'] for r in rows])))
