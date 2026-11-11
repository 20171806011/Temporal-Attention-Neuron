"""Independent native TAN equations and actual source-class oracle.

No experimental model/core/population imports. Full traces are retained for the
first eight episodes; counts and native event resources cover every episode.
"""
import numpy as np
from original_tan import make_variant

def split(w):
    parts=[];pos=0
    for shape in ((6,12),(6,12),(12,),(6,),(6,2),(2,)):
        size=int(np.prod(shape));parts.append(w[pos:pos+size].reshape(shape));pos+=size
    assert pos==len(w)==176
    return parts

def node(current,mem,history,attention):
    history=np.concatenate((history[:,:,1:],current[:,:,None]),axis=2)
    surprise=np.maximum(current-np.sum(history,axis=2)/5,0)
    q=surprise[:,:,None]*2.;k=history*1.;v=history*1.
    if attention:
        scores=q*k;shifted=scores-np.max(scores,axis=2,keepdims=True)
        exp=np.exp(shifted*1.);weights=exp/(np.sum(exp,axis=2,keepdims=True)+1e-9)
        context=np.sum(weights*v,axis=2)
    else:context=np.sum(v,axis=2)/5
    candidate=mem*.5+np.tanh(surprise)*context
    spike=np.greater(candidate,.5).astype(float);mem=candidate*np.logical_not(spike)
    return mem,history,spike,candidate,surprise,context

def replay(x,w,arm,G,L):
    assert x.shape[1:]==(24,6) and arm in ('TAN_FULL','TAN_NOATT')
    wi,wr,db,vb,wo,ob=split(w);wi=wi*G;wr=wr*G;wo=wo*G
    n=len(x);mem=[np.zeros((n,k)) for k in (12,6,2)]
    history=[np.zeros((n,k,5)) for k in (12,6,2)]
    previous=np.zeros((n,6));counts=np.zeros((n,2));resources=np.zeros((n,8))
    traces={k:[] for k in ('states','spikes','histories','candidates','surprises','contexts','currents')}
    for t in range(24):
        ib=np.einsum('ni,ij->nj',x[:,t],wi,optimize=False)+np.einsum('ni,ij->nj',previous,wr,optimize=False)+db
        mem[0],history[0],first,cb,sb,vb_context=node(ib,mem[0],history[0],arm=='TAN_FULL')
        ih=L*(first[:,::2]+first[:,1::2])+vb
        mem[1],history[1],middle,ch,sh,vh_context=node(ih,mem[1],history[1],arm=='TAN_FULL')
        io=np.einsum('ni,ij->nj',previous,wo,optimize=False)+ob
        mem[2],history[2],out,co,so,vo_context=node(io,mem[2],history[2],arm=='TAN_FULL')
        if t>=16:counts+=out
        resources[:,0]+=first.sum(1);resources[:,1]+=middle.sum(1);resources[:,2]+=out.sum(1)
        resources[:,3]+=np.einsum('ni,i->n',previous,np.count_nonzero(wr,axis=1))
        resources[:,4]+=np.einsum('ni,i->n',previous,np.count_nonzero(wo,axis=1))
        if t>=16:
            resources[:,5]+=first.sum(1);resources[:,6]+=middle.sum(1);resources[:,7]+=out.sum(1)
        for key,value in (
            ('states',np.concatenate(mem,axis=1)),
            ('histories',np.concatenate(history,axis=1)),
            ('spikes',np.concatenate((first,middle,out),axis=1)),
            ('candidates',np.concatenate((cb,ch,co),axis=1)),
            ('surprises',np.concatenate((sb,sh,so),axis=1)),
            ('contexts',np.concatenate((vb_context,vh_context,vo_context),axis=1)),
            ('currents',np.concatenate((ib,ih,io),axis=1))):
            traces[key].append(value[:8].copy())
        previous=middle
    return dict(counts=counts,resources=resources,**{k:np.stack(v,axis=1) for k,v in traces.items()})

def scalar_oracle(x,w,arm,G,L):
    """Actual immutable source-class calls; independent Python-sum wiring."""
    wi,wr,db,vb,wo,ob=split(w);wi=wi*G;wr=wr*G;wo=wo*G
    variant='TAN-full' if arm=='TAN_FULL' else 'TAN-no-attention'
    result={k:[] for k in ('counts','states','histories','spikes')}
    for episode in x:
        cells=[make_variant(variant) for _ in range(20)];previous=np.zeros(6);counts=np.zeros(2)
        states=[];histories=[];spikes=[]
        for t,external in enumerate(episode):
            first=np.array([cells[j].forward(sum(external[k]*wi[k,j] for k in range(6))+sum(previous[k]*wr[k,j] for k in range(6))+db[j]) for j in range(12)],float)
            middle=np.array([cells[12+j].forward(L*(first[2*j]+first[2*j+1])+vb[j]) for j in range(6)],float)
            out=np.array([cells[18+j].forward(sum(previous[k]*wo[k,j] for k in range(6))+ob[j]) for j in range(2)],float)
            if t>=16:counts+=out
            states.append([float(c.h) for c in cells]);histories.append(np.stack([c.history.copy() for c in cells]));spikes.append(np.r_[first,middle,out]);previous=middle
        result['counts'].append(counts);result['states'].append(states);result['histories'].append(histories);result['spikes'].append(spikes)
    return {k:np.asarray(v) for k,v in result.items()}

def assemble(base):
    x=np.zeros((len(base['y']),24,6))
    for t in range(8):x[:,8+t,:4]=base['write'][:,t]
    x[:,16:,4]=1
    return x

def probes():
    """Regenerate label-free calibration bank, without classification function."""
    r=np.random.default_rng(3964010000);n=16;amplitude=np.linspace(.8,1.2,n)
    zero=np.zeros((n,24,6));out={'external_zero':zero.copy()}
    x=zero.copy();x[:,:,:4]=amplitude.reshape(n,1,1);out['external_constant']=x
    x=zero.copy();x[:,8:16,:4]=amplitude.reshape(n,1,1);out['external_step']=x
    x=zero.copy();x[:,8:,:4]=np.tile((np.arange(16)%2).reshape(1,16,1),(n,1,4))*amplitude.reshape(n,1,1);out['external_alternating']=x
    x=zero.copy();x[:,8:,:4]=r.uniform(0,1.2,(n,16,4));out['external_irregular']=x
    bits=r.integers(0,2,(n,4));high=r.uniform(.8,1.2,(n,4));low=r.uniform(0,.1,(n,4));amp=np.where(bits,high,low)
    x=zero.copy();x[:,8:16,:4]=np.maximum(0,amp[:,None,:]+r.normal(0,.03,(n,8,4)));x[:,16:,4]=1;out['task_shaped_unlabelled']=x
    return out
