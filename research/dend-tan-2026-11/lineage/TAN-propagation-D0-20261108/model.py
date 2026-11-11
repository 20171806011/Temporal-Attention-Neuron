import numpy as np
import core,population
def effective(w,G):
    z=w.copy();z[:144]*=G;z[162:174]*=G;return z
def forward(x,w,arm,G,L,anchor=None):return core.forward(x,effective(w,G),arm,6,16,anchor,coupling=L)
def loss_grad(x,y,w,arm,G,L,reg):
    loss,g,c=core.loss_grad(x,y,effective(w,G),arm,6,16,coupling=L,reg=0)
    g[:144]*=G;g[162:174]*=G;g+=2*reg*w/len(w)
    return loss+reg*np.mean(w*w),g,c
def objective(x,y,w,arm,G,L,reg,anchor=None):
    c,_=forward(x,w,arm,G,L,anchor);return population.loss_counts(c,y,'CE')[0]+reg*np.mean(w*w)
def evaluate(x,w,arm,G,L):
    counts=[];resources=[];saved=None;ww=effective(w,G);wi,wr,db,vb,wo,ob=population.unpack(ww,6)
    for start in range(0,len(x),32):
        xx=x[start:start+32];c,cache=forward(xx,w,arm,G,L);counts.append(c);r=np.zeros((len(xx),8))
        for t,a in enumerate(cache):
            r[:,0]+=a['bs'].sum(1);r[:,1]+=a['hs'].sum(1);r[:,2]+=a['os'].sum(1);r[:,3]+=a['prev']@(wr!=0).sum(1);r[:,4]+=a['prev']@(wo!=0).sum(1)
            if t>=16:r[:,5]+=a['bs'].sum(1);r[:,6]+=a['hs'].sum(1);r[:,7]+=a['os'].sum(1)
        resources.append(r)
        if start==0:
            saved=dict(states=np.stack([a['states'][:8] for a in cache],1),spikes=np.stack([np.concatenate((a['bs'],a['hs'],a['os']),1)[:8] for a in cache],1))
            for name,field in [('histories','history'),('candidates','candidate'),('surprises','surprise'),('contexts','context')]:saved[name]=np.stack([np.concatenate([a[k][field] for k in ('b','h','o')],axis=1)[:8] for a in cache],1)
            saved['currents']=np.stack([np.concatenate((xx[:,t]@wi+a['prev']@wr+db,L*a['bs'].reshape(len(xx),6,2).sum(2)+vb,a['prev']@wo+ob),1)[:8] for t,a in enumerate(cache)],1)
    saved['counts']=np.concatenate(counts);saved['resources']=np.concatenate(resources);return saved
