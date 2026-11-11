"""Native intervention forward. Labels are absent from this module's forward API."""
import numpy as np
from train_model import unpack

COMPONENTS = {'d': (0,), 'v': (1,), 'o': (2,), 'p': (3,), 'dv': (0,1), 'all': (0,1,2,3)}

def specifications():
    specs = [dict(name='baseline', kind='none')]
    for phase, tick, window in [('write',5,[4,5]),('hold',9,[8,9]),('query',16,[16,17])]:
        specs.append(dict(name=f'{phase}_sham',kind='identity',tick=tick))
        specs.append(dict(name=f'{phase}_path_sham',kind='path_sham',window=window))
        for component in ('d','v','o','p','all'):
            specs.append(dict(name=f'{phase}_zero_{component}',kind='zero',tick=tick,component=component))
        for path in ('local','self','output'):
            specs.append(dict(name=f'{phase}_cut_{path}',kind='path',window=window,path=path))
    for donor in ('flip','same'):
        for component in COMPONENTS:
            specs.append(dict(name=f'hold_{donor}_{component}',kind='transplant',tick=9,component=component,donor=donor))
    return specs

def forward(x, theta, arm, h=6, ret=(.85,.65), spec=None, donor=None):
    wi,wr,db,vb,wo,ob = unpack(theta,h)
    n,T,_ = x.shape
    st = [np.zeros((n,2*h)),np.zeros((n,h)),np.zeros((n,2)),np.zeros((n,h))]
    memory = np.empty((n,T+1,4*h+2));memory[:,0] = np.concatenate(st,1)
    spikes = np.zeros((n,T,3*h+2));candidates = np.empty_like(spikes)
    states = np.empty_like(spikes);counts=np.zeros((n,2))
    recurrent=np.zeros(n);outgoing=np.zeros(n);delta=np.zeros(n)
    spec = spec or dict(kind='none')
    slices = [slice(0,2*h),slice(2*h,3*h),slice(3*h,3*h+2),slice(3*h+2,4*h+2)]
    selfmask = np.arange(h)[:,None] == np.arange(2*h)[None,:]//2
    for k in range(T):
        if k == spec.get('tick'):
            if spec['kind']=='identity': st=[a.copy() for a in st]
            elif spec['kind'] in ('zero','transplant'):
                for group in COMPONENTS[spec['component']]:
                    replacement = np.zeros_like(st[group]) if spec['kind']=='zero' else donor[:,k,slices[group]].copy()
                    delta += np.abs(replacement-st[group]).sum(1)
                    st[group] = replacement
        d,v,o,p = st
        active = spec['kind']=='path' and k in spec['window']
        current_wr = np.where(selfmask,0,wr) if active and spec['path']=='self' else wr
        current_wo = np.zeros_like(wo) if active and spec['path']=='output' else wo
        rec=p@current_wr
        dc=ret[0]*d+.25*(x[:,k]@wi+rec+db)
        if arm=='DEND': bs=np.zeros_like(dc);local=np.maximum(dc,0);d=dc
        else: bs=(dc>.5).astype(float);local=bs;d=dc*(1-bs)
        if active and spec['path']=='local': local=np.zeros_like(local)
        if spec['kind']=='path_sham' and k in spec['window']:
            local=local*1;current_wo=current_wo*1
        vc=ret[1]*v+.5*local.reshape(n,h,2).sum(2)+vb
        hs=(vc>.5).astype(float);v=vc*(1-hs)
        oc=ret[1]*o+p@current_wo+ob
        os=(oc>.5).astype(float);o=oc*(1-os)
        if k>=16: counts+=os
        recurrent += p@np.count_nonzero(current_wr,axis=1)
        outgoing += p@np.count_nonzero(current_wo,axis=1)
        st=[d,v,o,hs]
        memory[:,k+1]=np.concatenate(st,1)
        states[:,k]=np.concatenate([d,v,o],1)
        spikes[:,k]=np.concatenate([bs,hs,os],1)
        candidates[:,k]=np.concatenate([dc,vc,oc],1)
    return dict(counts=counts,memory=memory,states=states,spikes=spikes,candidates=candidates,
                recurrent_deliveries=recurrent,output_deliveries=outgoing,intervention_l1=delta)

def measure(v,y,pattern=None,h=6,arm='DEND'):
    c=v['counts'];pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1))
    recalls=[float(np.mean(pred[y==j]==j)) for j in (0,1)]
    cov=float(np.mean(pred>=0));ba=float(np.mean(recalls));z=v['spikes'];T=z.shape[1]
    events=[float(z[:,:,:2*h].sum((1,2)).mean()),float(z[:,:,2*h:3*h].sum((1,2)).mean()),float(z[:,:,3*h:].sum((1,2)).mean())]
    cm=[[int(np.sum((y==j)&(pred==k))) for k in (0,1,-1)] for j in (0,1)]
    rec=float(v['recurrent_deliveries'].mean());out=float(v['output_deliveries'].mean())
    q=dict(accuracy=float(np.mean(pred==y)),balanced_accuracy=ba,coverage=cov,class_recall=recalls,confusion=cm,
           silent_fraction=float(np.mean(np.all(c==0,1))),nonsilent_tie_fraction=float(np.mean((pred<0)&np.any(c!=0,1))),
           flipped_label_fraction=float(np.mean(pred==1-y)),passed=bool(ba>=.9 and cov>=.9),
           branch_events=events[0],hidden_events=events[1],output_events=events[2],total_events=sum(events),
           continuous_state_updates=int((3*h+2)*T),continuous_states=3*h+2,spike_registers=h,decoder_counters=2,
           parameters=2*h*h+17*h+2,spiking_cells=h+2 if arm=='DEND' else 3*h+2,
           recurrent_event_deliveries=rec,hidden_output_deliveries=out,
           branch_event_deliveries=events[0],consumed_internal_event_deliveries=rec+out+events[0],
           scheduled_graded_local_transfers=2*h*T if arm=='DEND' else 0,
           dense_forward_macs=int((2*h*h+14*h)*T),input_macs=int(12*h*T),
           intervention_l1_mean=float(v['intervention_l1'].mean()))
    if pattern is not None:q['pattern_recalls']={str(p):float(np.mean(pred[pattern==p]==y[pattern==p])) for p in np.unique(pattern)}
    return q
