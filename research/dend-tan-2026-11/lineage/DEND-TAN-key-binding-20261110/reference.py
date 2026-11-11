"""Independent continuous-DEND hybrid replay and actual TAN injection oracle.

The original source class is used only to retrieve its premembrane A_t. Its own
membrane and output spikes never enter this hybrid network.
"""
import sys
import numpy as np
from original_tan import make_variant

def split(w):
    parts=[];pos=0
    for shape in ((6,12),(6,12),(12,),(6,),(6,2),(2,)):
        n=int(np.prod(shape));parts.append(w[pos:pos+n].reshape(shape));pos+=n
    assert pos==len(w)==176
    return parts

FIELDS=('S_raw','S_t','Q','K','V','attention_scores','attention_weights','C_t','gating','A_t')
def capture_forward(cell,current):
    """Capture return-event locals from unmodified actual original forward."""
    output={};target=type(cell).forward.__code__;previous=sys.gettrace()
    def tracer(frame,event,arg):
        if frame.f_code is target:
            frame.f_trace_lines=False
            if event=='return':
                for key in FIELDS:
                    if key in frame.f_locals:
                        value=frame.f_locals[key];output[key]=value.copy() if isinstance(value,np.ndarray) else value
            return tracer
        return None
    try:
        sys.settrace(tracer);cell.forward(float(current))
    finally:sys.settrace(previous)
    assert 'A_t' in output
    return output

def source_attention(I_sequence,full):
    cell=make_variant('TAN-full' if full else 'TAN-no-attention')
    return [capture_forward(cell,current) for current in I_sequence]

def attention(current,history,full):
    history=np.concatenate((history[:,:,1:],current[:,:,None]),axis=2)
    raw=current-np.sum(history,axis=2)/5;S=np.maximum(raw,0)
    Q=S[:,:,None]*2.;K=history*1.;V=history*1.
    if full:
        score=Q*K;shift=score-np.max(score,axis=2,keepdims=True);exponent=np.exp(shift)
        alpha=exponent/(np.sum(exponent,axis=2,keepdims=True)+1e-9);C=np.sum(alpha*V,axis=2)
    else:score=None;alpha=None;C=np.sum(V,axis=2)/5
    gate=np.tanh(S);A=gate*C
    return A,history,dict(raw=raw,S=S,Q=Q,K=K,V=V,score=score,alpha=alpha,C=C,gate=gate,A=A)

def replay(x,w,arm,q):
    assert arm in ('DEND','FULL','NOATT','GAIN2') and x.shape[1]==q+8
    wi,wr,db,vb,wo,ob=split(w);n,T,_=x.shape
    d=np.zeros((n,12));v=np.zeros((n,6));o=np.zeros((n,2));previous=np.zeros((n,6))
    H=np.zeros((n,12,5));counts=np.zeros((n,2));resources=np.zeros((n,6))
    names=('states','spikes','candidates','currents','injections')+(('histories','surprises','contexts') if arm in ('FULL','NOATT') else ())
    traces={name:[] for name in names}
    for t in range(T):
        I=np.einsum('ni,ij->nj',x[:,t],wi,optimize=False)+np.einsum('ni,ij->nj',previous,wr,optimize=False)+db
        if arm in ('FULL','NOATT'):A,H,a=attention(I,H,arm=='FULL')
        else:A=I.copy() if arm=='GAIN2' else np.zeros_like(I)
        dc=.95*d+.25*(I+A);d=dc;local=np.where(dc>0,dc,0)
        vc=.85*v+.5*(local[:,::2]+local[:,1::2])+vb;hidden=np.greater(vc,.5).astype(float);v=vc*np.logical_not(hidden)
        oc=.85*o+np.einsum('ni,ij->nj',previous,wo,optimize=False)+ob;out=np.greater(oc,.5).astype(float);o=oc*np.logical_not(out)
        if t>=q:counts+=out
        resources[:,0]+=hidden.sum(1);resources[:,1]+=out.sum(1)
        resources[:,2]+=np.einsum('ni,i->n',previous,np.count_nonzero(wr,axis=1))
        resources[:,3]+=np.einsum('ni,i->n',previous,np.count_nonzero(wo,axis=1))
        if 16<=t<q:resources[:,4]+=hidden.sum(1);resources[:,5]+=out.sum(1)
        values=dict(states=np.concatenate((d,v,o),axis=1),spikes=np.concatenate((hidden,out),axis=1),candidates=np.concatenate((dc,vc,oc),axis=1),currents=I,injections=A)
        if arm in ('FULL','NOATT'):values.update(histories=H,surprises=a['S'],contexts=a['C'])
        for name in names:traces[name].append(values[name][:8].copy())
        previous=hidden
    return dict(counts=counts,resources=resources,**{k:np.stack(v,1) for k,v in traces.items()})

def scalar_oracle(x,w,arm,q):
    assert arm in ('FULL','NOATT')
    wi,wr,db,vb,wo,ob=split(w);result={k:[] for k in ('counts','states','spikes','histories','injections')}
    for episode in x:
        cells=[make_variant('TAN-full' if arm=='FULL' else 'TAN-no-attention') for _ in range(12)]
        d=np.zeros(12);v=np.zeros(6);o=np.zeros(2);previous=np.zeros(6);counts=np.zeros(2)
        traces={k:[] for k in ('states','spikes','histories','injections')}
        for t,external in enumerate(episode):
            I=np.array([sum(external[k]*wi[k,j] for k in range(6))+sum(previous[k]*wr[k,j] for k in range(6))+db[j] for j in range(12)])
            A=np.array([float(capture_forward(cells[j],I[j])['A_t']) for j in range(12)])
            d=.95*d+.25*(I+A);local=np.maximum(d,0)
            vc=.85*v+.5*(local[::2]+local[1::2])+vb;hidden=(vc>.5).astype(float);v=vc*(1-hidden)
            oc=.85*o+np.array([sum(previous[k]*wo[k,j] for k in range(6)) for j in range(2)])+ob;out=(oc>.5).astype(float);o=oc*(1-out)
            if t>=q:counts+=out
            traces['states'].append(np.r_[d,v,o]);traces['spikes'].append(np.r_[hidden,out]);traces['histories'].append(np.stack([c.history.copy() for c in cells]));traces['injections'].append(A);previous=hidden
        result['counts'].append(counts)
        for name in traces:result[name].append(traces[name])
    return {k:np.asarray(v) for k,v in result.items()}

def assemble(base,D):
    q=16+D;x=np.zeros((len(base['y']),q+8,6))
    for t in range(8):x[:,8+t,:4]=base['write'][:,t]
    x[:,q:,4]=1
    return x,q
