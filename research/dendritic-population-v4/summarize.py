"""Read-only scientific aggregation; does not alter models or selection."""
import json
import numpy as np
from common import OUT,dump

def main():
    rows=json.loads((OUT/'development_log.json').read_text());reps=json.loads((OUT/'replication_log.json').read_text()) if (OUT/'replication_log.json').exists() else [];a=[]
    for arm in ('DEND','POINT'):
        for lr in (.01,.03):
            for method in ('SPSA','ADAM'):
                for n in ([100] if method=='SPSA' else [100,200]):
                    group=sorted([r for r in rows if r['stage']=='A' and r['arm']==arm and r['loss']=='CE' and r['lr']==lr and r['method']==method and r['steps']==n],key=lambda z:z['seed'])
                    a.append(dict(arm=arm,lr=lr,method=method,updates=n,accuracy=[r['validation']['accuracy'] for r in group],coverage=[r['validation']['coverage'] for r in group]))
    costs=[]
    for stage in ('A','B','C','R'):
        for arm in ('DEND','POINT'):
            for method in ('ADAM','SPSA'):
                unique={r['trace']:r for r in rows+reps if r['stage']==stage and r['arm']==arm and r['method']==method and r['trace']}
                if not unique:continue
                updates=sum(r['trajectory_steps'] for r in unique.values());nf=updates*(2 if method=='SPSA' else 1);nb=updates if method=='ADAM' else 0
                costs.append(dict(stage=stage,arm=arm,method=method,trajectories=len(unique),updates=updates,forward_passes=nf,backward_passes=nb,forward_matrix_MAC_terms=nf*128*24*156,backward_matrix_MAC_terms=nb*128*24*240,seconds=sum(r['seconds_trajectory'] for r in unique.values())))
    dump(OUT/'aggregation.json',dict(A_CE_paired_controls=a,costs=costs,notes='A100/A200 deduplicated by trajectory trace. MAC terms omit scalar nonlinearities, resets, biases, loss, clipping and Adam arithmetic; not full compute or energy equality.'))
if __name__=='__main__':main()
