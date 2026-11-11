"""Balanced temporal selection quartets; targets never use neural equations."""
import numpy as np
CONDITIONS=('nominal','clean','strong','cue_gap3','tag_removed','query_only')
def make_base(n,seed):
    assert n%4==0;r=np.random.default_rng(seed);k=n//4
    amplitude=r.uniform(.8,1.2,k);tag_amplitude=r.uniform(.8,1.2,k);background=r.uniform(0,.03,(k,8,2));background[:,[2,6],:]=0;distractor=r.uniform(.2,1.2,(k,3));permutation=r.permutation(n)
    quartet=np.repeat(np.arange(k),4);selected=np.tile([0,0,1,1],k);first_bit=np.tile([0,1,0,1],k);y=first_bit^selected
    return dict(amplitude=amplitude,tag_amplitude=tag_amplitude,background=background,distractor=distractor,permutation=permutation,quartet=quartet[permutation],selected=selected[permutation],first_bit=first_bit[permutation],y=y[permutation],seed=np.array(seed))
def assemble(base,condition='nominal'):
    assert condition in CONDITIONS;n=len(base['y']);x=np.zeros((n,24,6));q=16
    for j in range(n):
        k=base['quartet'][j];b=int(base['first_bit'][j]);s=int(base['selected'][j]);x[j,8:16,:2]=base['background'][k];x[j,10,b]+=base['amplitude'][k];x[j,14,1-b]+=base['amplitude'][k]
        if condition!='tag_removed':x[j,(7 if condition=='cue_gap3' else 8)+4*s,2]=base['tag_amplitude'][k]
        if condition!='clean':x[j,[9,13,15],3]=base['distractor'][k]*(2 if condition=='strong' else 1)
    x[:,16:,4]=1
    if condition=='query_only':x[:,:16]=0
    return x,q
