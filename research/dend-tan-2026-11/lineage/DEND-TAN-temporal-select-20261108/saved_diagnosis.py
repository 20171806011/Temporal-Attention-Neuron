"""Post-hoc stored-count/gradient diagnosis; zero new neural evaluations."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text(encoding='utf-8'))
def main():
    assert read('AUDIT.json')['status']=='PASS';rows=[]
    for t in read('train/ROWS.json'):
        tr=np.load(R/'train'/(t['id']+'.npz'));z=np.load(R/'evaluate'/(t['id']+'_nominal.npz'));b=np.load(R/'evaluate'/f'test_s{t["seed"]}.npz');c=z['counts'];pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1));u,n=np.unique(c,axis=0,return_counts=True)
        category='variable_decision_or_tie'
        if np.all(c==0):category='all_silent_0_0'
        elif np.all(c==8):category='all_saturated_8_8'
        elif np.all(c[:,0]==c[:,1]):category='all_other_ties'
        cells=[]
        for s in (0,1):
            for f in (0,1):
                ix=(b['selected']==s)&(b['first_bit']==f);cells.append(dict(selected=s,first_bit=f,n=int(ix.sum()),accuracy=float(np.mean(pred[ix]==b['y'][ix])),coverage=float(np.mean(pred[ix]>=0))))
        rows.append(dict(id=t['id'],arm=t['arm'],lr=t['lr'],seed=t['seed'],category=category,patterns=[dict(count=k.tolist(),n=int(v)) for k,v in zip(u,n)],conditional_cells=cells,initial_training_query_spikes_mean=float(tr['counts'][0].sum(1).mean()),last_training_task_gradient_norm=float(tr['history'][-1,2]),last_training_regularizer_gradient_norm=float(tr['history'][-1,3])))
    totals={k:sum(r['category']==k for r in rows) for k in ('all_silent_0_0','all_saturated_8_8','all_other_ties','variable_decision_or_tie')}
    # Algebraic count-space demonstration, not a new model run or fitted comparison.
    grid=[]
    for k in range(9):
        counts=np.full((2,2),float(k));logits=.5*counts;ex=np.exp(logits-logits.max(1,keepdims=True));p=ex/ex.sum(1,keepdims=True);loss=float(-np.log(p[[0,1],[0,1]]).mean());grid.append(dict(common_count=k,CE_task_loss=loss))
    out=dict(status='POSTHOC_SAVED_RECORDS_AND_COUNT_ALGEBRA',new_model_runs=0,new_training_runs=0,totals=totals,rows=rows,equal_count_CE_grid=grid,interpretation='CE(.5*counts) is invariant to a common additive count shift. This count-space fact does not establish the cause of failed native training; proxy gradients may remain nonzero and input/task/initialization effects are not separated.')
    (R/'SAVED_DIAGNOSIS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(totals))
if __name__=='__main__':main()
