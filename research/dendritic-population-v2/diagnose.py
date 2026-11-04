"""Post-test descriptive audit only; never changes parameters or primary criteria."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent/'results'
lock=json.loads((R/'SELECTION.json').read_text());out=[]
for r in lock['selected']:
    s,a=r['seed'],r['arm'];v=np.load(R/f'test_{s}_{a}_intact.npz');cut=np.load(R/f'test_{s}_{a}_cut_cross.npz')
    init=np.load(R/f'initial_{s}.npy');theta=np.load(R/r['file']);unique,freq=np.unique(v['counts'],axis=0,return_counts=True)
    change=theta-init
    out.append(dict(seed=s,arm=a,readout_count_histogram=[{'counts':u.tolist(),'episodes':int(f)} for u,f in zip(unique,freq)],cross_cut_changes_hidden_spikes_fraction=float(np.any(v['spikes'][:,:,12:18]!=cut['spikes'][:,:,12:18],axis=(1,2)).mean()),cross_cut_changes_output_counts_fraction=float(np.any(v['counts']!=cut['counts'],axis=1).mean()),parameter_change_norms={'input':float(np.linalg.norm(change[:72])),'recurrent':float(np.linalg.norm(change[72:144])),'biases':float(np.linalg.norm(np.concatenate([change[144:162],change[174:]]))),'output':float(np.linalg.norm(change[162:174]))},max_absolute_state=float(np.abs(v['states']).max())))
(R/'diagnostics.json').write_text(json.dumps({'status':'post-test descriptive; no tuning or new training','rows':out},indent=2)+'\n')
print(json.dumps(out,indent=2))
