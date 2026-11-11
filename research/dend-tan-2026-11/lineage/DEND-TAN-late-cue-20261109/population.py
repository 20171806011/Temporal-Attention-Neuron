"""Dimension-generalized V2 native equations; surrogate derivatives train only.

The h=6 algebra and initialization draw order match the frozen V8 implementation.
No label is an argument to forward(). h counts hidden somata, not all cells.
"""
import numpy as np

def parameter_count(h):
    assert isinstance(h, (int, np.integer)) and h > 0
    return 2*h*h+17*h+2

def unpack(t,h=6):
    shapes=((6,2*h),(h,2*h),(2*h,),(h,),(h,2),(2,));out=[];start=0
    assert t.shape==(parameter_count(h),)
    for shape in shapes:
        size=int(np.prod(shape));out.append(t[start:start+size].reshape(shape));start+=size
    return out

def initialize(seed,h=6):
    r=np.random.default_rng(seed)
    scale=np.sqrt(6/h)
    return np.concatenate([r.normal(0,.45,12*h),r.normal(0,.25,2*h*h)*scale,np.full(2*h,.15),np.full(h,.05),r.normal(0,.5,2*h)*scale,np.full(2,.1)])

def slope(c):return 1/(1+5*np.abs(c-.5))**2

def forward(x,t,arm,h=6,anchor=None,ret=(.85,.65),query=16,intervention=None):
    assert arm in ('DEND','POINT')
    wi,wr,db,vb,wo,ob=unpack(t,h);n,T,_=x.shape
    d=np.zeros((n,2*h));v=np.zeros((n,h));o=np.zeros((n,2));prev=np.zeros((n,h));counts=np.zeros((n,2));cache=[]
    assert T==query+8 and query>=16
    intervention=intervention or {'kind':'none'}
    for k in range(T):
        if k==intervention.get('tick'):
            if intervention['kind']=='sham': d=d.copy();v=v.copy();o=o.copy();prev=prev.copy()
            elif intervention['kind']=='zero':
                if intervention['part'] in ('d','all'):d=np.zeros_like(d)
                if intervention['part'] in ('v','all'):v=np.zeros_like(v)
                if intervention['part']=='all':o=np.zeros_like(o);prev=np.zeros_like(prev)
        recurrent=prev@wr
        if intervention['kind']=='cut_recurrence' and 16<=k<query:recurrent=np.zeros_like(recurrent)
        dc=ret[0]*d+.25*(x[:,k]@wi+recurrent+db)
        if arm=='DEND':
            bs=np.zeros_like(dc);loc=np.maximum(dc,0) if anchor is None else np.maximum(anchor[k]['dc'],0)+(anchor[k]['dc']>0)*(dc-anchor[k]['dc']);d=dc
        else:
            bs=(dc>.5).astype(float) if anchor is None else anchor[k]['bs']+slope(anchor[k]['dc'])*(dc-anchor[k]['dc'])
            loc=bs;d=dc*(1-bs)
        vc=ret[1]*v+.5*loc.reshape(n,h,2).sum(2)+vb
        hs=(vc>.5).astype(float) if anchor is None else anchor[k]['hs']+slope(anchor[k]['vc'])*(vc-anchor[k]['vc'])
        v=vc*(1-hs)
        oc=ret[1]*o+prev@wo+ob
        os=(oc>.5).astype(float) if anchor is None else anchor[k]['os']+slope(anchor[k]['oc'])*(oc-anchor[k]['oc'])
        o=oc*(1-os)
        if k>=query:counts+=os
        cache.append(dict(dc=dc,vc=vc,oc=oc,bs=bs,hs=hs,os=os,prev=prev,states=np.concatenate((d,v,o),1)))
        prev=hs
    return counts,cache

def loss_counts(c,y,kind):
    n=len(y)
    if kind=='CE':
        z=.5*c;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
        loss=float(-np.log(p[np.arange(n),y]).mean());g=p.copy();g[np.arange(n),y]-=1;g*=.5/n
    else:
        assert kind=='RATE'
        target=np.full_like(c,.125);target[np.arange(n),y]=.75;delta=c/8-target
        loss=float(np.mean(delta*delta));g=delta/(8*n)
    return loss,g

def objective(x,y,t,arm,kind,h=6,anchor=None,ret=(.85,.65),query=16,intervention=None):
    c,_=forward(x,t,arm,h,anchor,ret,query,intervention);l,_=loss_counts(c,y,kind)
    return l+.0001*np.mean(t*t)

def loss_grad(x,y,t,arm,kind,h=6,zero_surrogate=False,ret=(.85,.65),query=16,intervention=None):
    assert intervention is None or intervention.get('kind')=='none', 'Transient interventions are inference-only; use fixed ret for constrained retraining.'
    wi,wr,db,vb,wo,ob=unpack(t,h);counts,cache=forward(x,t,arm,h,ret=ret,query=query,intervention=intervention);loss,gcount=loss_counts(counts,y,kind)
    P=parameter_count(h);grad=np.zeros(P);gwi,gwr,gdb,gvb,gwo,gob=unpack(grad,h);n=len(y)
    gd=np.zeros((n,2*h));gv=np.zeros((n,h));go=np.zeros((n,2));gprev=np.zeros((n,h))
    sg=(lambda c:np.zeros_like(c)) if zero_surrogate else slope
    for k in reversed(range(x.shape[1])):
        a=cache[k]
        goc=go*(1-a['os'])+((gcount if k>=query else 0)-go*a['oc'])*sg(a['oc'])
        gwo+=a['prev'].T@goc;gob+=goc.sum(0);oldprev=goc@wo.T;go=ret[1]*goc
        gvc=gv*(1-a['hs'])+(gprev-gv*a['vc'])*sg(a['vc'])
        gvb+=gvc.sum(0);gloc=.5*np.repeat(gvc,2,axis=1);gv=ret[1]*gvc
        if arm=='DEND':gdc=gd+gloc*(a['dc']>0)
        else:gdc=gd*(1-a['bs'])+(gloc-gd*a['dc'])*sg(a['dc'])
        q=.25*gdc;gwi+=x[:,k].T@q;gwr+=a['prev'].T@q;gdb+=q.sum(0)
        gprev=oldprev+q@wr.T;gd=ret[0]*gdc
    grad+=.0002*t/P
    return loss+.0001*np.mean(t*t),grad,counts

def metrics(x,y,t,arm,h=6,ret=(.85,.65),query=16,intervention=None):
    c,cache=forward(x,t,arm,h,ret=ret,query=query,intervention=intervention);pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1));spikes=np.stack([np.concatenate([a['bs'],a['hs'],a['os']],1) for a in cache],1)
    return dict(counts=c,prediction=pred,spikes=spikes,states=np.stack([a['states'] for a in cache],1),candidates=np.stack([np.concatenate([a['dc'],a['vc'],a['oc']],1) for a in cache],1))
