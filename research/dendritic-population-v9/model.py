"""V2 native dynamics, with explicit BPTT surrogate derivatives for training only."""
import numpy as np
P=176
SLICES=[(0,72,(6,12)),(72,144,(6,12)),(144,156,(12,)),(156,162,(6,)),(162,174,(6,2)),(174,176,(2,))]
def unpack(t):return [t[a:b].reshape(s) for a,b,s in SLICES]
def initialize(seed):
    r=np.random.default_rng(seed)
    return np.concatenate([r.normal(0,.45,72),r.normal(0,.25,72),np.full(12,.15),np.full(6,.05),r.normal(0,.5,12),np.full(2,.1)])
def slope(c):return 1/(1+5*np.abs(c-.5))**2

def forward(x,t,arm,anchor=None):
    wi,wr,db,vb,wo,ob=unpack(t);n,T,_=x.shape
    d=np.zeros((n,12));v=np.zeros((n,6));o=np.zeros((n,2));prev=np.zeros((n,6));counts=np.zeros((n,2));cache=[]
    for k in range(T):
        dc=.85*d+.25*(x[:,k]@wi+prev@wr+db)
        if arm=='DEND':
            bs=np.zeros_like(dc);loc=np.maximum(dc,0) if anchor is None else np.maximum(anchor[k]['dc'],0)+(anchor[k]['dc']>0)*(dc-anchor[k]['dc']);d=dc
        else:
            bs=(dc>.5).astype(float) if anchor is None else anchor[k]['bs']+slope(anchor[k]['dc'])*(dc-anchor[k]['dc'])
            loc=bs;d=dc*(1-bs)
        vc=.65*v+.5*loc.reshape(n,6,2).sum(2)+vb
        hs=(vc>.5).astype(float) if anchor is None else anchor[k]['hs']+slope(anchor[k]['vc'])*(vc-anchor[k]['vc'])
        v=vc*(1-hs)
        oc=.65*o+prev@wo+ob
        os=(oc>.5).astype(float) if anchor is None else anchor[k]['os']+slope(anchor[k]['oc'])*(oc-anchor[k]['oc'])
        o=oc*(1-os)
        if k>=16:counts+=os
        cache.append(dict(dc=dc,vc=vc,oc=oc,bs=bs,hs=hs,os=os,prev=prev,states=np.concatenate((d,v,o),1)))
        prev=hs
    return counts,cache

def loss_counts(c,y,kind):
    n=len(y)
    if kind=='CE':
        z=.5*c;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
        loss=float(-np.log(p[np.arange(n),y]).mean());g=p.copy();g[np.arange(n),y]-=1;g*=.5/n
    else:
        target=np.full_like(c,.125);target[np.arange(n),y]=.75;delta=c/8-target
        loss=float(np.mean(delta*delta));g=delta/(8*n)
    return loss,g

def objective(x,y,t,arm,kind,anchor=None):
    c,_=forward(x,t,arm,anchor);l,_=loss_counts(c,y,kind)
    return l+.0001*np.mean(t*t)

def loss_grad(x,y,t,arm,kind,zero_surrogate=False):
    wi,wr,db,vb,wo,ob=unpack(t);counts,cache=forward(x,t,arm);loss,gcount=loss_counts(counts,y,kind)
    grad=np.zeros(P);gwi,gwr,gdb,gvb,gwo,gob=unpack(grad);n=len(y)
    gd=np.zeros((n,12));gv=np.zeros((n,6));go=np.zeros((n,2));gprev=np.zeros((n,6))
    sg=(lambda c:np.zeros_like(c)) if zero_surrogate else slope
    for k in reversed(range(x.shape[1])):
        a=cache[k]
        goc=go*(1-a['os'])+((gcount if k>=16 else 0)-go*a['oc'])*sg(a['oc'])
        gwo+=a['prev'].T@goc;gob+=goc.sum(0);oldprev=goc@wo.T;go=.65*goc
        gvc=gv*(1-a['hs'])+(gprev-gv*a['vc'])*sg(a['vc'])
        gvb+=gvc.sum(0);gloc=.5*np.repeat(gvc,2,axis=1);gv=.65*gvc
        if arm=='DEND':gdc=gd+gloc*(a['dc']>0)
        else:gdc=gd*(1-a['bs'])+(gloc-gd*a['dc'])*sg(a['dc'])
        q=.25*gdc;gwi+=x[:,k].T@q;gwr+=a['prev'].T@q;gdb+=q.sum(0)
        gprev=oldprev+q@wr.T;gd=.85*gdc
    grad+=.0002*t/P
    return loss+.0001*np.mean(t*t),grad,counts

def metrics(x,y,t,arm):
    c,cache=forward(x,t,arm);pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1));spikes=np.stack([np.concatenate([a['bs'],a['hs'],a['os']],1) for a in cache],1)
    return dict(accuracy=float(np.mean(pred==y)),coverage=float(np.mean(pred>=0)),abstentions=int(np.sum(pred<0)),events_mean=spikes.sum(1).mean(0).tolist(),counts=c,prediction=pred,spikes=spikes,states=np.stack([a['states'] for a in cache],1),candidates=np.stack([np.concatenate([a['dc'],a['vc'],a['oc']],1) for a in cache],1))
