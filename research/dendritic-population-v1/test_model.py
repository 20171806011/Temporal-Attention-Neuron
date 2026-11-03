"""Mechanism/contract fixtures, not benchmark tuning."""
import numpy as np
from model import simulate,native_step,ARMS

def main():
    h,s,u=native_step(np.array([0.,0.]),np.array([.5,np.nextafter(.5,np.inf)]))
    assert list(s)==[False,True] and h[0]==.5 and h[1]==0
    for arm in ARMS:
        z=simulate(np.zeros((2,32,4)),arm)
        assert not z['spikes'].any() and not z['states'].any()
        assert np.all(z['predictions']==-1) # no forced or scheduled spikes
        x=np.zeros((1,32,4));x[:,4:14,0:2]=1.
        out=simulate(x,arm)
        assert out['states'].shape==(1,32,12)
        hs=out['spikes'][0,:,6:9];os=out['spikes'][0,:,9:12]
        assert not os[0].any()
        # Since .6 > output threshold .5, each native input spike produces
        # an output spike one step later, with no output residual or extra event.
        assert np.array_equal(os[1:],hs[:-1])
        assert not os[np.concatenate(([True],~hs[:-1].any(axis=1)))].any()
        assert np.all(out['states'][0,:,6:9][hs.astype(bool)]==0)
    # Changing evaluation labels is impossible through this API: only x and arm.
    print('PASS: strict-threshold, zero-input silence, abstention, reset, state shape and one-step propagation fixtures')
if __name__=='__main__':main()
