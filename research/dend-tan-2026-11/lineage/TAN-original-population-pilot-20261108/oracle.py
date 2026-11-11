"""Small fixture oracle: original scalar class per neuron, independent wiring."""
import numpy as np
from original_tan import make_variant

def split(w,h):
    parts=[];pos=0
    for shape in ((6,2*h),(h,2*h),(2*h,),(h,),(h,2),(2,)):
        length=int(np.prod(shape));parts.append(w[pos:pos+length].reshape(shape));pos+=length
    return parts

def replay(x,w,arm,h,q):
    wi,wr,db,vb,wo,ob=split(w,h);n,T,_=x.shape
    all_counts=[];all_states=[];all_histories=[];all_spikes=[]
    variant='TAN-full' if arm=='TAN_FULL' else 'TAN-no-attention'
    for episode in range(n):
        cells=[make_variant(variant) for _ in range(3*h+2)];previous=np.zeros(h);count=np.zeros(2);states=[];histories=[];spikes=[]
        for t in range(T):
            first=[]
            for j in range(2*h):
                current=sum(x[episode,t,k]*wi[k,j] for k in range(6))+sum(previous[k]*wr[k,j] for k in range(h))+db[j]
                first.append(cells[j].forward(current))
            middle=[]
            for j in range(h):middle.append(cells[2*h+j].forward(.5*(first[2*j]+first[2*j+1])+vb[j]))
            out=[]
            for j in range(2):out.append(cells[3*h+j].forward(sum(previous[k]*wo[k,j] for k in range(h))+ob[j]))
            if t>=q:count+=out
            states.append(np.array([float(c.h) for c in cells]));histories.append(np.stack([c.history.copy() for c in cells]));spikes.append(np.r_[first,middle,out]);previous=np.array(middle,float)
        all_counts.append(count);all_states.append(states);all_histories.append(histories);all_spikes.append(spikes)
    return dict(counts=np.array(all_counts),states=np.array(all_states),histories=np.array(all_histories),spikes=np.array(all_spikes))
