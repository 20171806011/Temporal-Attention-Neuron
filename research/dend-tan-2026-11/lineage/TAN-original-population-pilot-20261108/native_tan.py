"""Original scalar TAN forward, vectorized into a declared spiking population.

Fixed W=5, lambda=.5, theta=.5, tau=0, Wq=2, Wk=Wv=beta=1,
and denominator epsilon=1e-9. The learned parameters are spatial synapses only.
The scalar original_tan.py is the unit oracle; population wiring is an adaptation.
"""
import numpy as np
import population

WINDOW=5
EPS=1e-9
ARMS=('TAN_FULL','TAN_NOATT','DEND','POINT')

def cell(current,mem,history,attention=True,anchor=None):
    H=np.roll(history,-1,axis=-1);H[...,-1]=current
    raw=current-H.mean(-1)
    S=np.maximum(raw,0) if anchor is None else np.maximum(anchor['raw'],0)+(anchor['raw']>0)*(raw-anchor['raw'])
    Q=2*S[...,None];K=H*1.;V=H*1.
    if attention:
        score=Q*K
        max_index=np.argmax(score,axis=-1) if anchor is None else anchor['max_index']
        maximum=np.max(score,axis=-1,keepdims=True) if anchor is None else np.take_along_axis(score,max_index[...,None],axis=-1)
        e=np.exp(1.*(score-maximum));denom=e.sum(-1,keepdims=True)+EPS;alpha=e/denom;C=np.sum(alpha*V,axis=-1)
    else:alpha=None;denom=None;max_index=None;C=V.mean(-1)
    gate=np.tanh(S);mc=.5*mem+gate*C
    z=(mc>.5).astype(float) if anchor is None else anchor['spike']+population.slope(anchor['candidate'])*(mc-anchor['candidate'])
    newmem=mc*(1-z)
    cache=dict(history=H,raw=raw,surprise=S,context=C,gate=gate,candidate=mc,spike=z,membrane=newmem,alpha=alpha,denom=denom,max_index=max_index)
    return newmem,H,z,cache

def forward(x,w,arm,h=6,query=16,anchor=None):
    assert arm in ARMS
    if arm in ('DEND','POINT'):return population.forward(x,w,arm,h,anchor=anchor,ret=(.95,.85),query=query)
    assert x.shape[1]==query+8 and query>=16
    wi,wr,db,vb,wo,ob=population.unpack(w,h);n,T,_=x.shape
    mb=np.zeros((n,2*h));mh=np.zeros((n,h));mo=np.zeros((n,2));hb=np.zeros((n,2*h,5));hh=np.zeros((n,h,5));ho=np.zeros((n,2,5));p=np.zeros((n,h));counts=np.zeros((n,2));cache=[];att=arm=='TAN_FULL'
    for t in range(T):
        a=None if anchor is None else anchor[t]
        ib=x[:,t]@wi+p@wr+db
        mb,hb,bs,b=cell(ib,mb,hb,att,None if a is None else a['b'])
        ih=.5*bs.reshape(n,h,2).sum(2)+vb
        mh,hh,hs,hidden=cell(ih,mh,hh,att,None if a is None else a['h'])
        io=p@wo+ob
        mo,ho,os,out=cell(io,mo,ho,att,None if a is None else a['o'])
        if t>=query:counts+=os
        cache.append(dict(b=b,h=hidden,o=out,bs=bs,hs=hs,os=os,prev=p,states=np.concatenate((mb,mh,mo),1)))
        p=hs
    return counts,cache

def cell_backward(a,gm,gz,attention,zero_surrogate=False):
    slope=np.zeros_like(a['candidate']) if zero_surrogate else population.slope(a['candidate'])
    gmc=gm*(1-a['spike'])+(gz-gm*a['candidate'])*slope
    gC=gmc*a['gate'];gS=gmc*a['context']*(1-a['gate']**2);H=a['history']
    if attention:
        alpha=a['alpha'];ga=gC[...,None]*H;dot=np.sum(ga*alpha,axis=-1,keepdims=True)
        gs=alpha*(ga-dot)
        gs-=np.eye(5)[a['max_index']]*(EPS/a['denom'])*dot
        gS+=np.sum(gs*2*H,axis=-1)
        gH=gC[...,None]*alpha+gs*2*a['surprise'][...,None]
    else:gH=np.broadcast_to(gC[...,None]/5,H.shape).copy()
    gr=gS*(a['raw']>0);gH-=gr[...,None]/5;gH[...,-1]+=gr
    return .5*gmc,gH

def distribute(pending,t,gH):
    for lag in range(min(t+1,5)):pending[t-lag]+=gH[...,4-lag]

def loss_grad(x,y,w,arm,h=6,query=16,zero_surrogate=False):
    if arm in ('DEND','POINT'):return population.loss_grad(x,y,w,arm,'CE',h,zero_surrogate=zero_surrogate,ret=(.95,.85),query=query)
    wi,wr,db,vb,wo,ob=population.unpack(w,h);c,k=forward(x,w,arm,h,query);loss,gc=population.loss_counts(c,y,'CE');P=len(w);grad=np.zeros(P);gwi,gwr,gdb,gvb,gwo,gob=population.unpack(grad,h)
    n,T,_=x.shape;gb=np.zeros((n,2*h));gh=np.zeros((n,h));go=np.zeros((n,2));gp=np.zeros((n,h));pb=np.zeros((T,n,2*h));ph=np.zeros((T,n,h));po=np.zeros((T,n,2));att=arm=='TAN_FULL'
    for t in reversed(range(T)):
        a=k[t];go,dH=cell_backward(a['o'],go,gc if t>=query else np.zeros_like(gc),att,zero_surrogate);distribute(po,t,dH);g=po[t]
        gwo+=a['prev'].T@g;gob+=g.sum(0);oldprev=g@wo.T
        gh,dH=cell_backward(a['h'],gh,gp,att,zero_surrogate);distribute(ph,t,dH);g=ph[t];gvb+=g.sum(0);gbs=.5*np.repeat(g,2,axis=1)
        gb,dH=cell_backward(a['b'],gb,gbs,att,zero_surrogate);distribute(pb,t,dH);g=pb[t]
        gwi+=x[:,t].T@g;gwr+=a['prev'].T@g;gdb+=g.sum(0);gp=oldprev+g@wr.T
    grad+=.0002*w/P
    return loss+.0001*np.mean(w*w),grad,c

def objective(x,y,w,arm,h=6,query=16,anchor=None):
    c,_=forward(x,w,arm,h,query,anchor);l,_=population.loss_counts(c,y,'CE');return l+.0001*np.mean(w*w)
