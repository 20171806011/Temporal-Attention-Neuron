"""New objectives; unchanged native forward and the same anchored spike proxy."""
import numpy as np
import model,population

LOSSES=('CE','COUNT','CECOM')
DIAGNOSTIC_STEPS=np.array(list(range(0,400,20))+[399])
def count_terms(c,y,kind):
    n=len(y);s=c.sum(1);sign=np.eye(2)[y]*2-1;d=(c*sign).sum(1)
    common=float(np.mean((s-8)**2/32));gc=np.repeat(((s-8)/(16*n))[:,None],2,axis=1)
    if kind=='COUNT':disc=float(np.mean((d-4)**2/32));gd=sign*((d-4)/(16*n))[:,None]
    else:disc,gd=population.loss_counts(c,y,'CE')
    if kind=='CE':common=0.;gc=np.zeros_like(c)
    if kind not in LOSSES:raise ValueError(kind)
    return disc,common,gd,gc

def backward(x,w,arm,q,cache,gcount):
    wi,wr,db,vb,wo,ob=population.unpack(w,6);grad=np.zeros_like(w);gwi,gwr,gdb,gvb,gwo,gob=population.unpack(grad,6);n,T,_=x.shape
    gd=np.zeros((n,12));gv=np.zeros((n,6));go=np.zeros((n,2));gp=np.zeros((n,6));pending=np.zeros((T,n,12));sg=population.slope
    for t in reversed(range(T)):
        a=cache[t];goc=go*(1-a['os'])+((gcount if t>=q else 0)-go*a['oc'])*sg(a['oc']);gwo+=a['prev'].T@goc;gob+=goc.sum(0);oldprev=goc@wo.T;go=.85*goc
        gvc=gv*(1-a['hs'])+(gp-gv*a['vc'])*sg(a['vc']);gvb+=gvc.sum(0);gv=.85*gvc;gdc=gd+.5*np.repeat(gvc,2,axis=1)*(a['dc']>0);direct=.25*gdc;gd=.95*gdc
        if arm in ('FULL','NOATT'):
            gh=model.vjp(a['attention'],direct,arm=='FULL')
            for lag in range(min(t+1,5)):pending[t-lag]+=gh[...,4-lag]
            gi=direct+pending[t]
        else:gi=direct*(2 if arm=='GAIN2' else 1)
        gwi+=x[:,t].T@gi;gwr+=a['prev'].T@gi;gdb+=gi.sum(0);gp=oldprev+gi@wr.T
    return grad

def loss_grad(x,y,w,arm,q,kind,diagnostic=False):
    c,cache=model.forward(x,w,arm,q);disc,common,gd,gc=count_terms(c,y,kind)
    task=backward(x,w,arm,q,cache,gd+gc);reg=.0002*w/len(w);l2=.0001*np.mean(w*w);total=disc+common+l2
    components=np.array([disc,common,l2,total,np.linalg.norm(gd),np.linalg.norm(gc),np.linalg.norm(gd+gc)])
    if diagnostic:
        cg=backward(x,w,arm,q,cache,gc) if kind!='CE' else np.zeros_like(w)
        gradients=np.stack((task-cg,cg,reg))
    else:gradients=None
    return total,task+reg,c,components,gradients

def objective(x,y,w,arm,q,kind,anchor=None):
    c=model.forward(x,w,arm,q,anchor)[0];disc,common,_,_=count_terms(c,y,kind)
    return disc+common+.0001*np.mean(w*w)
