"""Independent augmented-design ridge least squares; no normal-equation solve.

Objective: mean squared residual + alpha*sum(beta**2). Intercept is unpenalized.
Only fitting features determine standardization and the constant-column mask.
"""
import numpy as np
STD_FLOOR=1e-8
SCORE_ZERO=1e-10

def fit(X,y,alpha=.1):
    """y contains signed regression targets, normally -1/+1, not class labels."""
    X=np.asarray(X,float);y=np.asarray(y,float)
    assert X.ndim==2 and y.shape[0]==len(X) and y.ndim in (1,2) and alpha>0
    n,p=X.shape;mean=X.mean(0);std=np.sqrt(np.mean((X-mean)**2,axis=0));constant=std<STD_FLOOR;scale=np.where(constant,1.,std)
    Z=(X-mean)/scale;Z[:,constant]=0
    design=np.empty((n+p,p+1));design[:n,:p]=Z/np.sqrt(n);design[:n,p]=1/np.sqrt(n)
    design[n:,:p]=np.sqrt(alpha)*np.eye(p);design[n:,p]=0
    multi=y.ndim==2;yy=y if multi else y[:,None]
    target=np.zeros((n+p,yy.shape[1]));target[:n]=yy/np.sqrt(n)
    solution=np.linalg.lstsq(design,target,rcond=None)[0]
    coef=solution[:p];coef[constant]=0;intercept=solution[p]
    return dict(mean=mean,scale=scale,constant=constant,coef=coef if multi else coef[:,0],intercept=intercept if multi else intercept[0])

def scores(X,fitted):
    Z=(np.asarray(X,float)-fitted['mean'])/fitted['scale'];Z[:,fitted['constant']]=0
    return Z@fitted['coef']+fitted['intercept']

def classify(score):
    score=np.asarray(score,float).copy();score[np.abs(score)<=SCORE_ZERO]=0
    return (score>0).astype(int)

def metrics(score,y):
    pred=classify(score);y=np.asarray(y,int)
    assert pred.shape==y.shape
    recall=[float(np.mean(pred[y==j]==j)) for j in (0,1)]
    return dict(balanced_accuracy=sum(recall)/2,accuracy=float(np.mean(pred==y)),recalls=recall,mse=float(np.mean((np.asarray(score)-(2*y-1))**2)),score_rms=float(np.sqrt(np.mean(np.asarray(score)**2))))
