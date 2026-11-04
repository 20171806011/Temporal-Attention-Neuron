"""Selected literal-function transcriptions from temporal_distractor.py at fixed commit.
NOT the full archived file. Imports/path side effects removed; FIT_RECORDS logging
added after minimize, without changing objective, gradient, options or predictions.
Source ranges read: constants/generator/readout/metrics lines70..239; main240..437.
"""
import numpy as np
from scipy.optimize import minimize
SEEDS=[20260904,20260905,20260906]
N_TRAIN=4000;N_TEST=2000;N_CATCH_FRAC=1.0/3.0
L_BLK=14;Q_IDX=8;TGT_IDX=4;D_IDX=slice(5,8);L2=1e-4
A_LO_0,A_HI_0=.55,1.05
A_LO_1,A_HI_1=.95,1.45
FIT_RECORDS=[]
def sample_amp(rng,n,cls):
    cls=np.asarray(cls,dtype=int).reshape(-1)
    lo=np.full(n,A_LO_0);hi=np.full(n,A_HI_0)
    m=cls==1;lo[m]=A_LO_1;hi[m]=A_HI_1
    return rng.uniform(lo,hi)
def make_trials(rng,n):
    n_catch=int(round(N_CATCH_FRAC*n));n_present=n-n_catch
    n_dist=n_present//2;n_clean=n_present-n_dist
    cT=rng.integers(0,2,size=n_present)
    cQ=rng.integers(0,2,size=n_present)
    X=np.zeros((n,L_BLK));y=np.full(n,-1,dtype=int)
    cT_all=np.full(n,-1,dtype=int);cQ_all=np.full(n,-1,dtype=int)
    idx=0
    for (cond_n,n_d) in ((n_dist,3),(n_clean,0)):
        for j in range(idx,idx+cond_n):
            jj=j-idx
            X[j,TGT_IDX]=sample_amp(rng,1,[cT[jj]])[0]
            X[j,Q_IDX]=sample_amp(rng,1,[cQ[jj]])[0]
            if n_d:
                X[j,D_IDX]=sample_amp(rng,n_d,rng.integers(0,2,size=n_d))
            y[j]=int(cT[jj]);cT_all[j]=int(cT[jj]);cQ_all[j]=int(cQ[jj])
        idx+=cond_n
    n_catch_dist=n_catch//2
    for k in range(n_catch):
        j=idx+k;cq=int(rng.integers(0,2))
        X[j,Q_IDX]=sample_amp(rng,1,[cq])[0]
        if k<n_catch_dist:
            X[j,D_IDX]=sample_amp(rng,3,rng.integers(0,2,size=3))
        y[j]=2;cQ_all[j]=cq
    perm=rng.permutation(n)
    present=(cT_all>=0)[perm]
    return (X[perm],y[perm],cT_all[perm],cQ_all[perm],present)
def resp_features(su):
    u=su['u'][:,Q_IDX:Q_IDX+5]
    y=(u>.5).astype(float)
    return np.column_stack([u,y])
def hist5_features(X):return X[:,TGT_IDX:Q_IDX+1]
def softmax_fit(Xtr,ytr):
    n,d=Xtr.shape;K=3;y=np.asarray(ytr,dtype=int)
    mu=Xtr.mean(axis=0);sd=Xtr.std(axis=0);sd[sd<1e-9]=1.0
    Xs=(Xtr-mu)/sd
    def loss(w):
        W=w.reshape(K,d+1);Z=Xs@W[:,1:].T+W[:,0]
        Zm=Z-Z.max(axis=1,keepdims=True);e=np.exp(Zm);p=e/e.sum(axis=1,keepdims=True)
        ll=-np.mean(np.log(p[np.arange(n),y]+1e-12))
        return ll+0.5*L2*float((W[:,1:]**2).sum())
    def grad(w):
        W=w.reshape(K,d+1);Z=Xs@W[:,1:].T+W[:,0]
        Zm=Z-Z.max(axis=1,keepdims=True);e=np.exp(Zm);p=e/e.sum(axis=1,keepdims=True)
        T=np.zeros_like(p);T[np.arange(n),y]=1.0;g=p-T
        gW=np.empty(K*(d+1)).reshape(K,d+1)
        gW[:,0]=g.mean(axis=0);gW[:,1:]=(g.T@Xs)/n+L2*W[:,1:]
        return gW.ravel()
    res=minimize(loss,np.zeros(K*(d+1)),jac=grad,method='L-BFGS-B',options=dict(maxiter=800,ftol=1e-12))
    FIT_RECORDS.append({'success':bool(res.success),'status':int(res.status),'message':str(res.message),'nit':int(res.nit),'nfev':int(res.nfev),'loss':float(res.fun),'jac_inf':float(np.max(np.abs(res.jac)))})
    return mu,sd,res.x

def predict(mu,sd,w,Xte):
    Xs=(Xte-mu)/sd;W=w.reshape(3,Xte.shape[1]+1)
    Z=Xs@W[:,1:].T+W[:,0];Zm=Z-Z.max(axis=1,keepdims=True)
    e=np.exp(Zm);return e/e.sum(axis=1,keepdims=True)

def bal3(pred,ytrue,mask=None):
    sel=np.arange(len(ytrue)) if mask is None else np.where(mask)[0];rec=[]
    for k in range(3):
        m=ytrue[sel]==k
        if m.sum():rec.append(float((pred[sel][m]==k).mean()))
    return float(np.mean(rec)) if rec else float('nan')

def confusion(pred,ytrue):
    C=np.zeros((3,3),dtype=int)
    for t,p in zip(ytrue,pred):C[t,p]+=1
    return C

def macro_f1(pred,ytrue,mask=None):
    sel=np.arange(len(ytrue)) if mask is None else np.where(mask)[0];fs=[]
    for k in range(3):
        tp=int(((pred[sel]==k)&(ytrue[sel]==k)).sum());fp=int(((pred[sel]==k)&(ytrue[sel]!=k)).sum());fn=int(((pred[sel]!=k)&(ytrue[sel]==k)).sum())
        if tp+fp==0 and fn==0:continue
        prec=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.
        fs.append(2*prec*rec/(prec+rec) if prec+rec else 0.)
    return float(np.mean(fs)) if fs else float('nan')
