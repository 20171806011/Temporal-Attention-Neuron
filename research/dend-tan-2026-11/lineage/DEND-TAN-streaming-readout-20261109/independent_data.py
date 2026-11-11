"""Independent stimulus reconstruction and temporal-selection invariants.

Targets depend only on which of two ordered pulses the cue selects. This module
does not import experimental data generation, neural equations or classifiers.
"""
import numpy as np
CONDITIONS=('nominal','clean','strong','cue_gap3','tag_removed','query_only')

def make_base(n,seed):
    assert n>0 and n%4==0
    rng=np.random.default_rng(seed);K=n//4
    amplitude=rng.uniform(.8,1.2,K);tag_amplitude=rng.uniform(.8,1.2,K)
    background=rng.uniform(0,.03,(K,8,2))
    for t in (2,6):background[:,t,:]=0
    distractor=rng.uniform(.2,1.2,(K,3));permutation=rng.permutation(n)
    records=np.array([(k,s,b,b if s==0 else 1-b) for k in range(K) for s in (0,1) for b in (0,1)],dtype=int)
    records=records[permutation]
    return dict(amplitude=amplitude,tag_amplitude=tag_amplitude,background=background,distractor=distractor,permutation=permutation,quartet=records[:,0],selected=records[:,1],first_bit=records[:,2],y=records[:,3],seed=np.array(seed))

def assemble(base,condition='nominal'):
    assert condition in CONDITIONS
    n=len(base['y']);x=np.zeros((n,24,6),dtype=float)
    for j,(k,s,b) in enumerate(zip(base['quartet'],base['selected'],base['first_bit'])):
        for relative in range(8):
            for ch in (0,1):x[j,8+relative,ch]=base['background'][k,relative,ch]
        x[j,10,int(b)]=base['amplitude'][k]
        x[j,14,1-int(b)]=base['amplitude'][k]
        if condition!='tag_removed':
            cue=(10 if s==0 else 14)-(3 if condition=='cue_gap3' else 2)
            x[j,cue,2]=base['tag_amplitude'][k]
        if condition!='clean':
            for pos,t in enumerate((9,13,15)):x[j,t,3]=(2 if condition=='strong' else 1)*base['distractor'][k,pos]
    for t in range(16,24):x[:,t,4]=1
    if condition=='query_only':x[:,:16,:]=0
    return x,16

def pairs(base):
    """Same ordered pair of candidates, opposite selection and label."""
    out=[]
    for k in range(len(base['amplitude'])):
        for b in (0,1):
            candidates=np.flatnonzero((base['quartet']==k)&(base['first_bit']==b))
            assert len(candidates)==2
            i=candidates[base['selected'][candidates]==0][0]
            j=candidates[base['selected'][candidates]==1][0]
            assert base['y'][i]==1-base['y'][j]
            out.append((int(i),int(j)))
    return out

def check(base,x,condition):
    """Raise on a broken data invariant; return a compact validation record."""
    assert condition in CONDITIONS and x.shape==(len(base['y']),24,6)
    n=len(base['y']);K=n//4
    assert n%4==0 and np.isfinite(x).all() and np.all(x>=0)
    assert np.array_equal(np.sort(base['permutation']),np.arange(n))
    assert np.sum(base['y']==0)==np.sum(base['y']==1)==n//2
    assert np.all(base['background'][:,[2,6],:]==0)
    assert np.all(x[:,:,5]==0) and np.all(x[:,16:,4]==1)
    assert np.all(x[:,:16,4]==0) and np.all(x[:,16:,:4]==0)
    for k in range(K):
        ix=np.flatnonzero(base['quartet']==k);assert len(ix)==4
        assert set(zip(base['selected'][ix].tolist(),base['first_bit'][ix].tolist()))=={(0,0),(0,1),(1,0),(1,1)}
        assert np.sum(base['y'][ix])==2
        for j in ix:
            b=int(base['first_bit'][j]);s=int(base['selected'][j]);assert int(base['y'][j])==(b if s==0 else 1-b)
            if condition=='query_only':continue
            assert x[j,10,b]==x[j,14,1-b]==base['amplitude'][k]
            assert x[j,10,1-b]==x[j,14,b]==0
            for relative in range(8):
                if relative not in (2,6):assert np.array_equal(x[j,8+relative,:2],base['background'][k,relative,:])
            if condition=='tag_removed':assert np.all(x[j,:,2]==0)
            else:
                ticks=np.flatnonzero(x[j,:,2]);assert len(ticks)==1
                assert ticks[0]==(10 if s==0 else 14)-(3 if condition=='cue_gap3' else 2)
                assert x[j,ticks[0],2]==base['tag_amplitude'][k]
            if condition=='clean':assert np.all(x[j,:,3]==0)
            else:
                assert np.array_equal(np.flatnonzero(x[j,:,3]),np.array([9,13,15]))
                assert np.array_equal(x[j,[9,13,15],3],base['distractor'][k]*(2 if condition=='strong' else 1))
        if condition!='query_only':
            totals=x[ix,:,:2].sum(1)
            assert np.allclose(totals,totals[0],rtol=0,atol=1e-12)
    duplicate_pairs=0
    if condition=='tag_removed':
        for i,j in pairs(base):assert np.array_equal(x[i],x[j]);duplicate_pairs+=1
    if condition=='query_only':
        assert np.all(x[:,:16]==0) and np.array_equal(x,np.broadcast_to(x[0],x.shape))
    expected,q=assemble(base,condition)
    assert q==16 and np.array_equal(x,expected)
    return dict(n=n,quartets=K,balanced_labels=True,equal_pulse_amplitudes=True,matched_nuisance_quartets=True,exact_opposite_label_duplicate_pairs=duplicate_pairs,all_inputs_identical=condition=='query_only')
