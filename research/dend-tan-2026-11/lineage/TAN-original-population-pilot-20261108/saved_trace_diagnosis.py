"""Post-hoc description of stored records only; no forward calls or new fits."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    assert read(R/'AUDIT.json')['status']=='PASS'
    out=[];training=[]
    for tr in read(R/'train/ROWS.json'):
        z=np.load(R/'train'/f'{tr["id"]}.npz');w=z['initial'].copy();m=np.zeros_like(w);v=np.zeros_like(w);steps=[]
        for k,g in enumerate(z['gradients']):
            norm=z['history'][k,1];regularizer=.0002*w/len(w);task=g*max(1,norm)-regularizer;task_norm=float(np.linalg.norm(task));regularizer_norm=float(np.linalg.norm(regularizer))
            if k in (0,1,2,5,10,25,50,100,199):steps.append(dict(completed_updates=k,task_gradient_norm=task_norm,regularizer_gradient_norm=regularizer_norm,task_to_regularizer_norm_ratio=task_norm/regularizer_norm if regularizer_norm else None,biases=dict(first=w[144:156].tolist(),hidden=w[156:162].tolist(),output=w[174:176].tolist())))
            m=.9*m+.1*g;v=.999*v+.001*g*g;w=np.clip(w-tr['lr']*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
        assert np.array_equal(w,z['theta'])
        training.append(dict(id=tr['id'],arm=tr['arm'],lr=tr['lr'],seed=tr['seed'],parameter_change_norm=float(np.linalg.norm(w-z['initial'])),initial_parameter_norm=float(np.linalg.norm(z['initial'])),final_parameter_norm=float(np.linalg.norm(w)),all_200_query_counts_zero=bool(np.all(z['counts']==0)),final_hidden_bias=w[156:162].tolist(),final_output_bias=w[174:176].tolist(),steps=steps))
        for D in (0,24,48,96):
            z=np.load(R/'evaluate'/f'{tr["id"]}_d{D}.npz');c=z['counts'];patterns,frequency=np.unique(c,axis=0,return_counts=True);q=16+D;layer=[]
            for name,sl in (('first',slice(0,12)),('hidden',slice(12,18)),('output',slice(18,20))):
                segments={}
                for segment,ts in (('warmup',slice(0,8)),('write',slice(8,16)),('hold',slice(16,q)),('query',slice(q,q+8))):
                    a=z['candidates'][:,ts,sl]
                    if not a.size:continue
                    info=dict(candidate_min=float(a.min()),candidate_max=float(a.max()),candidate_mean=float(a.mean()),spikes_in_first8=float(z['spikes'][:,ts,sl].sum()))
                    if tr['arm'].startswith('TAN_'):
                        surprise=z['surprises'][:,ts,sl];info.update(zero_surprise_fraction=float(np.mean(surprise==0)),surprise_max=float(surprise.max()))
                    segments[segment]=info
                layer.append(dict(layer=name,segments=segments))
            out.append(dict(id=tr['id'],arm=tr['arm'],lr=tr['lr'],seed=tr['seed'],delay=D,n_all=256,n_traces=8,count_patterns=[dict(count=a.astype(int).tolist(),n=int(b)) for a,b in zip(patterns,frequency)],layers_first8=layer,all256_mean_events=dict(zip(('first','hidden','output','recurrent_deliveries','output_deliveries','hold_first','hold_hidden','hold_output'),z['resources'].mean(0).tolist()))))
    result=dict(analysis_status='POST_HOC_SAVED_RECORDS_ONLY',new_forward_runs=0,new_fits=0,scope='Every native output count and event record uses all 256 episodes. Candidate/surprise extrema describe only the eight saved full traces. Reconstructed task gradients subtract declared L2 after undoing norm clipping; they are not exact hard-threshold derivatives.',training=training,evaluations=out)
    (R/'SAVED_TRACE_DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for arm in ('TAN_FULL','TAN_NOATT'):
        chosen=[r for r in out if r['arm']==arm and r['delay']==24]
        print(arm,json.dumps([dict(id=r['id'],patterns=r['count_patterns'],events=r['all256_mean_events']) for r in chosen]))
if __name__=='__main__':main()
