import numpy as np
ALPHA=.1
def permutations(base,seed):
    rng=np.random.default_rng(seed);y=base['y'];out=np.empty((len(y),20),dtype=np.int64)
    for j in range(20):
        for k in range(len(y)//4):
            ix=np.flatnonzero(base['quartet']==k);out[ix,j]=rng.permutation(y[ix])
    return out
def transform(x,mean,scale,active):
    z=(x-mean)/scale;z[:,~active]=0;return np.column_stack((z,np.ones(len(x))))
def fit(x,labels):
    mean=x.mean(0);sd=x.std(0);active=sd>=1e-8;scale=np.maximum(sd,1e-8);z=transform(x,mean,scale,active);target=2*labels-1;reg=np.eye(z.shape[1])*ALPHA;reg[-1,-1]=0
    coef=np.linalg.solve(z.T@z/len(x)+reg,z.T@target/len(x))
    return dict(mean=mean,scale=scale,active=active,coef=coef,fit_scores=z@coef)
def predict(fit,x):return transform(x,fit['mean'],fit['scale'],fit['active'])@fit['coef']
def metrics(scores,y):
    scores=np.asarray(scores);p=(scores>1e-10).astype(int);target=2*y-1
    return dict(balanced_accuracy=float((np.mean(p[y==0]==0)+np.mean(p[y==1]==1))/2),mse=float(np.mean((scores-target)**2)),score_rms=float(np.sqrt(np.mean(scores**2))))
