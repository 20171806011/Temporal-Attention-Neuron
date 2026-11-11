import numpy as np
import native_tan,population
from common import structural_cost
def evaluate(x,w,arm,h,q):
    counts=[];resources=[];saved=None;_,wr,_,_,wo,_=population.unpack(w,h);tan=arm.startswith('TAN_')
    for start in range(0,len(x),64):
        xx=x[start:start+64];c,cache=native_tan.forward(xx,w,arm,h,q);counts.append(c);r=np.zeros((len(xx),8))
        for t,a in enumerate(cache):
            r[:,0]+=a['bs'].sum(1);r[:,1]+=a['hs'].sum(1);r[:,2]+=a['os'].sum(1);r[:,3]+=a['prev']@(wr!=0).sum(1);r[:,4]+=a['prev']@(wo!=0).sum(1)
            if 16<=t<q:r[:,5]+=a['bs'].sum(1);r[:,6]+=a['hs'].sum(1);r[:,7]+=a['os'].sum(1)
        resources.append(r)
        if start==0:
            saved=dict(states=np.stack([a['states'][:8] for a in cache],1),spikes=np.stack([np.concatenate((a['bs'],a['hs'],a['os']),1)[:8] for a in cache],1))
            if tan:
                for name,field in [('histories','history'),('candidates','candidate'),('surprises','surprise'),('contexts','context')]:saved[name]=np.stack([np.concatenate([a[k][field] for k in ('b','h','o')],axis=1)[:8] for a in cache],1)
            else:saved['candidates']=np.stack([np.concatenate((a['dc'],a['vc'],a['oc']),axis=1)[:8] for a in cache],1)
    saved['counts']=np.concatenate(counts);saved['resources']=np.concatenate(resources)
    names=('first_layer_events','hidden_events','output_events','recurrent_deliveries','output_deliveries','hold_first_layer_events','hold_hidden_events','hold_output_events');cost={k:float(saved['resources'][:,j].mean()) for j,k in enumerate(names)}
    cost.update(first_layer_local_deliveries=cost['first_layer_events'],internal_deliveries=cost['first_layer_events']+cost['recurrent_deliveries']+cost['output_deliveries'],**structural_cost(arm,h,x.shape[1]))
    return saved,cost
