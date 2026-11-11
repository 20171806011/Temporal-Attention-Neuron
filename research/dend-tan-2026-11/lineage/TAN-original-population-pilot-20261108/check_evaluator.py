"""Independent evaluator and batching contract, before scientific data generation."""
import time,numpy as np
import native_tan,independent,evaluate,population,base_data
from common import *
def main():
    tic=time.perf_counter();checks=0;maxerr=0.
    for name,m in read(R/'SOURCE_LOCK.json')['source_files'].items():assert sha(R/name)==m['sha256'];checks+=1
    b=base_data.make_base(80,79967001);w=population.initialize(79968001,H)
    # Deliberate activation is fixture-only, not the scientific initialization.
    wi,wr,db,vb,wo,ob=population.unpack(w,H);wi*=2;vb[:]=.8;wo[:]=np.random.default_rng(79968002).normal(.6,.7,wo.shape);ob[:]=.4
    for D in (0,24,96):
        x,q=assemble(b,D)
        for arm in ARMS:
            saved,cost=evaluate.evaluate(x,w,arm,H,q);ref=independent.replay(x,w,arm,H,q)
            for k in ('counts','spikes'):assert np.array_equal(saved[k],ref[k]),k;checks+=1
            for k in ('states','candidates','resources')+(('histories','surprises','contexts') if arm.startswith('TAN_') else ()):
                err=float(np.max(abs(saved[k]-ref[k])));maxerr=max(maxerr,err);assert err<1e-8,(arm,D,k,err);checks+=1
            assert cost['parameters']==176 and cost['output_counters']==2;checks+=1
            if arm.startswith('TAN_'):assert cost['history_slots']==100 and cost['persistent_continuous_slots']==120;checks+=1
            if arm=='TAN_FULL':assert cost['attention_exp_evaluations']==(q+8)*100 and cost['attention_cached_argmax_comparisons']==(q+8)*80 and cost['persistent_array_bytes_per_episode']==1024;checks+=1
            if arm=='TAN_NOATT':assert cost['attention_exp_evaluations']==0 and cost['noatt_context_mean_divisions']==(q+8)*20;checks+=1
            assert np.all(x[:,16:q]==0);checks+=1
    assert not (R/'train').exists() and not (R/'evaluate').exists();checks+=1
    dump(R/'CONTRACT_CHECK.json',dict(status='PASS',checks=checks,max_numeric_error=maxerr,wall_seconds=time.perf_counter()-tic,scientific_seeds_used=False,fixture_seeds=[79967001,79968001,79968002],scope='Independent native equations versus batched experimental evaluator across four arms and three delays, with activated fixture-only cells.'))
    print(read(R/'CONTRACT_CHECK.json'))
if __name__=='__main__':main()
