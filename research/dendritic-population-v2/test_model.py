import numpy as np
from model import forward,unpack,initialize,P
assert len(initialize(0))==P==176
z=np.zeros((2,24,6));theta=np.zeros(P)
for arm in ('DEND','POINT'):
    a=forward(z,theta,arm,trace=True)
    assert not a['spikes'].any() and np.all(a['prediction']==-1)
    assert a['states'].shape==(2,24,20)
    w=unpack(theta);w['output_bias'][:]=.5
    a=forward(z,theta,arm,trace=True)
    assert not a['spikes'][:,0,18:].any() and a['spikes'][:,1,18:].all()
    w['output_bias'][:]=0
# Cross-cell recurrent weights never affect a cut circuit, but self connections do.
t=initialize(41);u=t.copy();wu=unpack(u);mask=np.arange(6)[:,None]!=np.arange(12)[None,:]//2
wu['recurrent'][mask]+=1
for arm in ('DEND','POINT'):
    a=forward(np.ones((2,24,6)),t,arm,cut_cross=True,trace=True)
    b=forward(np.ones((2,24,6)),u,arm,cut_cross=True,trace=True)
    assert np.array_equal(a['states'],b['states'])
print('Native threshold/reset, quiet state, dimensions and cross-cut fixtures PASS')
