"""DEND-POP-1 native-spiking feedforward population; no output pulse generator."""
import numpy as np
PARTITIONS=np.array([[[0,1],[2,3]],[[0,2],[1,3]],[[0,3],[1,2]]])
ARMS=('DEND','POOLED','POINT_NETWORK')

def native_step(old,current,leak=.5,threshold=.5):
    candidate=leak*old+current
    spike=candidate>threshold
    return np.where(spike,0.,candidate),spike,candidate

def simulate(inputs,arm):
    """Only supplied waveforms enter updates. No labels or evaluator state accepted."""
    if arm not in ARMS:raise ValueError(arm)
    n,steps,channels=inputs.shape
    assert channels==4 and np.isfinite(inputs).all() and np.min(inputs)>=0
    d=np.zeros((n,3,2));h=np.zeros((n,3));o=np.zeros((n,3));previous=np.zeros((n,3),bool)
    states=np.zeros((n,steps,12));candidates=np.zeros_like(states)
    spikes=np.zeros((n,steps,12),np.uint8)
    costs=np.zeros((n,6)) # sensory branch L1/L2; hidden L1/L2; output L1/L2
    for t in range(steps):
        drive=inputs[:,t,PARTITIONS].sum(axis=-1)
        dc=.8*d+.2*drive
        if arm=='POINT_NETWORK':
            branch_spike=dc>1.1;d=np.where(branch_spike,0.,dc)
            current=branch_spike.sum(axis=2).astype(float)
        else:
            d=dc;branch_spike=np.zeros_like(d,bool)
            current=np.maximum(d-1.1,0).sum(axis=2) if arm=='DEND' else np.maximum(d.sum(axis=2)-1.1,0)
        h,hs,hc=native_step(h,current)
        output_current=.6*previous.astype(float)
        o,os,oc=native_step(o,output_current)
        previous=hs.copy()
        states[:,t]=np.concatenate((d.reshape(n,6),h,o),axis=1)
        candidates[:,t]=np.concatenate((dc.reshape(n,6),hc,oc),axis=1)
        spikes[:,t]=np.concatenate((branch_spike.reshape(n,6),hs,os),axis=1)
        for j,arr in enumerate((drive.reshape(n,6),current,output_current)):
            costs[:,2*j]+=np.abs(arr).sum(axis=1)
            costs[:,2*j+1]+=(arr*arr).sum(axis=1)
    counts=spikes[:,:,9:12].sum(axis=1)
    best=counts.max(axis=1)
    valid=(best>0)&((counts==best[:,None]).sum(axis=1)==1)
    predictions=np.where(valid,counts.argmax(axis=1),-1)
    return dict(states=states,candidates=candidates,spikes=spikes,costs=costs,
                counts=counts,predictions=predictions)
