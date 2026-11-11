import numpy as np
import population
def attention(I,H,full,anchor=None):
    H=np.roll(H,-1,axis=-1);H[...,-1]=I;raw=I-H.mean(-1)
    S=np.maximum(raw,0) if anchor is None else np.maximum(anchor['raw'],0)+(anchor['raw']>0)*(raw-anchor['raw'])
    Q=2*S[...,None];K=H*1.;V=H*1.
    if full:
        score=Q*K;index=np.argmax(score,-1) if anchor is None else anchor['index'];maximum=np.max(score,-1,keepdims=True) if anchor is None else np.take_along_axis(score,index[...,None],-1)
        e=np.exp(1.*(score-maximum));denom=e.sum(-1,keepdims=True)+1e-9;alpha=e/denom;C=(alpha*V).sum(-1)
    else:score=None;index=None;denom=None;alpha=None;C=V.mean(-1)
    gate=np.tanh(S);A=gate*C
    return A,H,dict(H=H,raw=raw,S=S,Q=Q,K=K,V=V,score=score,index=index,denom=denom,alpha=alpha,C=C,gate=gate,A=A)
def vjp(a,gA,full):
    gC=gA*a['gate'];gS=gA*a['C']*(1-a['gate']**2);H=a['H']
    if full:
        alpha=a['alpha'];ga=gC[...,None]*H;dot=(ga*alpha).sum(-1,keepdims=True);gs=alpha*(ga-dot)-np.eye(5)[a['index']]*(1e-9/a['denom'])*dot
        gS+=(gs*2*H).sum(-1);gH=gC[...,None]*alpha+gs*2*a['S'][...,None]
    else:gH=np.broadcast_to(gC[...,None]/5,H.shape).copy()
    gr=gS*(a['raw']>0);gH-=gr[...,None]/5;gH[...,-1]+=gr;return gH
def forward(x,w,arm,q,anchor=None):
    wi,wr,db,vb,wo,ob=population.unpack(w,6);n,T,_=x.shape;d=np.zeros((n,12));v=np.zeros((n,6));o=np.zeros((n,2));p=v.copy();H=np.zeros((n,12,5)) if arm in ('FULL','NOATT') else None;counts=np.zeros((n,2));cache=[]
    for t in range(T):
        anc=None if anchor is None else anchor[t];I=x[:,t]@wi+p@wr+db
        if arm in ('FULL','NOATT'):A,H,a=attention(I,H,arm=='FULL',None if anc is None else anc['attention'])
        else:A=I if arm=='GAIN2' else np.zeros_like(I);a=None
        dc=.95*d+.25*(I+A);loc=np.maximum(dc,0) if anc is None else np.maximum(anc['dc'],0)+(anc['dc']>0)*(dc-anc['dc']);d=dc
        vc=.85*v+.5*loc.reshape(n,6,2).sum(2)+vb;hs=(vc>.5).astype(float) if anc is None else anc['hs']+population.slope(anc['vc'])*(vc-anc['vc']);v=vc*(1-hs)
        oc=.85*o+p@wo+ob;os=(oc>.5).astype(float) if anc is None else anc['os']+population.slope(anc['oc'])*(oc-anc['oc']);o=oc*(1-os)
        if t>=q:counts+=os
        cache.append(dict(I=I,A=A,attention=a,dc=dc,vc=vc,oc=oc,hs=hs,os=os,prev=p,states=np.concatenate((d,v,o),1)));p=hs
    return counts,cache
def loss_grad(x,y,w,arm,q,zero_surrogate=False):
    wi,wr,db,vb,wo,ob=population.unpack(w,6);c,cache=forward(x,w,arm,q);loss,gcount=population.loss_counts(c,y,'CE');grad=np.zeros_like(w);gwi,gwr,gdb,gvb,gwo,gob=population.unpack(grad,6);n,T,_=x.shape
    gd=np.zeros((n,12));gv=np.zeros((n,6));go=np.zeros((n,2));gp=np.zeros((n,6));pending=np.zeros((T,n,12));sg=(lambda c:np.zeros_like(c)) if zero_surrogate else population.slope
    for t in reversed(range(T)):
        a=cache[t];goc=go*(1-a['os'])+((gcount if t>=q else 0)-go*a['oc'])*sg(a['oc']);gwo+=a['prev'].T@goc;gob+=goc.sum(0);oldprev=goc@wo.T;go=.85*goc
        gvc=gv*(1-a['hs'])+(gp-gv*a['vc'])*sg(a['vc']);gvb+=gvc.sum(0);gv=.85*gvc;gdc=gd+.5*np.repeat(gvc,2,axis=1)*(a['dc']>0);direct=.25*gdc;gd=.95*gdc
        if arm in ('FULL','NOATT'):
            gh=vjp(a['attention'],direct,arm=='FULL')
            for lag in range(min(t+1,5)):pending[t-lag]+=gh[...,4-lag]
            gi=direct+pending[t]
        else:gi=direct*(2 if arm=='GAIN2' else 1)
        gwi+=x[:,t].T@gi;gwr+=a['prev'].T@gi;gdb+=gi.sum(0);gp=oldprev+gi@wr.T
    grad+=.0002*w/len(w);return loss+.0001*np.mean(w*w),grad,c
def objective(x,y,w,arm,q,anchor=None):return population.loss_counts(forward(x,w,arm,q,anchor)[0],y,'CE')[0]+.0001*np.mean(w*w)
def evaluate(x,w,arm,q):
    counts=[];resources=[];saved=None;wi,wr,db,vb,wo,ob=population.unpack(w,6)
    for start in range(0,len(x),32):
        xx=x[start:start+32];c,k=forward(xx,w,arm,q);counts.append(c);r=np.zeros((len(xx),6))
        for t,a in enumerate(k):
            r[:,0]+=a['hs'].sum(1);r[:,1]+=a['os'].sum(1);r[:,2]+=a['prev']@(wr!=0).sum(1);r[:,3]+=a['prev']@(wo!=0).sum(1)
            if 16<=t<q:r[:,4]+=a['hs'].sum(1);r[:,5]+=a['os'].sum(1)
        resources.append(r)
        if start==0:
            saved={key:np.stack([a[field][:8] for a in k],1) for key,field in [('states','states'),('currents','I'),('injections','A')]};saved['spikes']=np.stack([np.concatenate((a['hs'],a['os']),1)[:8] for a in k],1);saved['candidates']=np.stack([np.concatenate((a['dc'],a['vc'],a['oc']),1)[:8] for a in k],1)
            if arm in ('FULL','NOATT'):
                for name,field in [('histories','H'),('surprises','S'),('contexts','C')]:saved[name]=np.stack([a['attention'][field][:8] for a in k],1)
    saved['counts']=np.concatenate(counts);saved['resources']=np.concatenate(resources);return saved
