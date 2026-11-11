"""Inference only. Native equations with explicitly timed interventions."""
import numpy as np
import model
from common import RET,PARTS
def evaluate(x,w,arm,h,q,case):
    wi,wr,db,vb,wo,ob=model.unpack(w,h);T=x.shape[1];cs=[];rs=[];trace=None;events=None
    cut=case.get('cut',[-1,-1]);zero_tick=case.get('zero_tick',-1);sham=case.get('sham_tick',-1)
    for b in range(0,len(x),64):
        xx=x[b:b+64];n=len(xx);d=np.zeros((n,2*h));v=np.zeros((n,h));o=np.zeros((n,2));p=np.zeros((n,h));c=np.zeros((n,2));r=np.zeros((n,8));ss=[];sp=[]
        for t in range(T):
            if t==zero_tick:
                state=[d,v,o,p]
                for j in PARTS[case['zero_part']]:state[j]=np.zeros_like(state[j])
                d,v,o,p=state
            if t==sham:d=d.copy();v=v.copy();o=o.copy();p=p.copy()
            off=cut[0]<=t<cut[1]
            rec=np.zeros_like(d) if off else p@wr
            dc=RET[0]*d+.25*(xx[:,t]@wi+rec+db)
            if arm=='DEND':bs=np.zeros_like(dc);loc=np.maximum(dc,0);d=dc
            else:bs=(dc>.5).astype(float);loc=bs;d=dc*(1-bs)
            vc=RET[1]*v+.5*loc.reshape(n,h,2).sum(2)+vb;hs=(vc>.5).astype(float);v=vc*(1-hs)
            oc=RET[1]*o+p@wo+ob;os=(oc>.5).astype(float);o=oc*(1-os)
            if t>=q:c+=os
            r[:,0]+=bs.sum(1);r[:,1]+=hs.sum(1);r[:,2]+=os.sum(1)
            if not off:r[:,3]+=p@(wr!=0).sum(1)
            r[:,4]+=p@(wo!=0).sum(1)
            if 16<=t<q:r[:,5]+=bs.sum(1);r[:,6]+=hs.sum(1);r[:,7]+=os.sum(1)
            if b==0:ss.append(np.concatenate((d,v,o),1)[:8]);sp.append(np.concatenate((bs,hs,os),1)[:8])
            p=hs
        cs.append(c);rs.append(r)
        if b==0:trace=np.stack(ss,1);events=np.stack(sp,1)
    resources=np.concatenate(rs);saved=dict(counts=np.concatenate(cs),states=trace,spikes=events,resources=resources)
    names=('branch_events','hidden_events','output_events','recurrent_deliveries','output_deliveries','hold_branch_events','hold_hidden_events','hold_output_events');cost={k:float(resources[:,j].mean()) for j,k in enumerate(names)}
    cutsteps=sum(cut[0]<=t<cut[1] for t in range(T));reset=sum((2*h,h,2,h)[j] for j in PARTS[case['zero_part']]) if 'zero_part' in case else 0
    cost.update(branch_deliveries=cost['branch_events'],internal_deliveries=cost['branch_events']+cost['recurrent_deliveries']+cost['output_deliveries'],state_updates=T*(3*h+2),graded_local_transfers=T*2*h if arm=='DEND' else 0,forward_macs=T*(2*h*h+14*h),executed_forward_macs=T*(2*h*h+14*h)-cutsteps*2*h*h,cut_steps=cutsteps,reset_values=reset,parameters=model.parameter_count(h),continuous_slots=3*h+2,persistent_continuous=3*h+2,previous_spike_registers=h,output_counters=2,spiking_cells=h+2 if arm=='DEND' else 3*h+2,graded_branches=2*h if arm=='DEND' else 0)
    return saved,cost
