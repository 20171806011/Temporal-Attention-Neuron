"""Summarize audited saved frozen-network probes; never run a neural model."""
from pathlib import Path
import csv,json
import numpy as np
R=Path(__file__).resolve().parent
ARMS=('DEND','FULL','NOATT','GAIN2');LOSSES=('CE','COUNT','CECOM')
CONDITIONS=('nominal','tag_removed','query_only');TIMES=(9,10,14,15,19,23)
BASE_GROUPS=('branch','soma','previous_spikes','output','joint')
DIMENSIONS=dict(branch=12,soma=6,previous_spikes=6,output=2,joint=26,history=60,augmented=86,counts=2)
BOOT=np.random.default_rng(4001900001).integers(0,3,(20000,3))

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(name,value):(R/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ci(values):
    x=np.asarray(values,float);assert x.shape==(3,) and np.all(np.isfinite(x))
    return dict(values=x.tolist(),mean=float(x.mean()),min=float(x.min()),max=float(x.max()),std=float(x.std(ddof=1)),n=3,ci95=np.quantile(x[BOOT].mean(1),[.025,.975]).tolist())
def csvwrite(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def configurations(arm):
    groups=BASE_GROUPS+(('history','augmented') if arm in ('FULL','NOATT') else ())
    return [(group,time) for group in groups for time in TIMES]+[('counts',23)]

def main():
    audit=read(R/'AUDIT.json');assert audit['status']=='PASS'
    fits=read(R/'fits/ROWS.json');probes=read(R/'evaluate/ROWS.json');native=read(R/'evaluate/NATIVE_ROWS.json')
    fit_index={row['probe_id']:row for row in fits};assert len(fits)==len(fit_index)==1332
    for row in probes:
        fit=fit_index[row['probe_id']]
        for key in ('id','arm','loss','seed','group','time','dimension'):assert row[key]==fit[key]
        row.update({key:fit[key] for key in ('fit_ba','fit_mse')})
    assert len(probes)==3996 and len(native)==108
    index={};native_index={};negative=[];nulls=[]
    for row in probes:
        key=tuple(row[k] for k in ('arm','loss','seed','group','time','condition'));assert key not in index;index[key]=row
        assert row['dimension']==DIMENSIONS[row['group']] and (row['group'],row['time']) in configurations(row['arm'])
        if row['condition']!='nominal':
            assert abs(row['balanced_accuracy']-.5)<1e-12
            negative.append(dict(id=row['id'],group=row['group'],time=row['time'],condition=row['condition'],balanced_accuracy=row['balanced_accuracy']))
        else:
            nullrow={key:row[key] for key in ('id','arm','loss','seed','group','time','dimension','balanced_accuracy','mse','score_rms')}
            for key in ('fit_shuffle_ba','fit_shuffle_mse','fit_shuffle_score_rms','test_shuffle_ba','test_shuffle_mse'):
                values=np.asarray(row[key],float);assert values.shape==(20,) and np.all(np.isfinite(values))
                nullrow[key]=dict(values=values.tolist(),mean=float(values.mean()),min=float(values.min()),max=float(values.max()))
            nulls.append(nullrow)
    for row in native:
        key=tuple(row[k] for k in ('arm','loss','seed','condition'));assert key not in native_index;native_index[key]=row
        assert abs(row['coverage']+row['silence']+row['saturated_tie']+row['intermediate_tie']-1)<1e-12
        if row['condition']!='nominal':
            assert abs(row['forced_ba']-.5)<1e-12 and abs(row['strict_ba']-.5*row['coverage'])<1e-12
    assert len(negative)==2664 and len(nulls)==1332
    groups={};native_groups={};summary=[]
    for arm in ARMS:
        for loss in LOSSES:
            group={};groups[f'{arm}_{loss}']=group;native_groups[f'{arm}_{loss}']={}
            for condition in CONDITIONS:
                rr=[native_index[(arm,loss,s,condition)] for s in range(3)]
                native_groups[f'{arm}_{loss}'][condition]={key:ci([r[key] for r in rr]) for key in ('strict_ba','forced_ba','coverage','silence','saturated_tie','intermediate_tie')}
            for feature,time in configurations(arm):
                cell=dict(group=feature,time=time,dimension=DIMENSIONS[feature]);group[f'{feature}_t{time}']=cell
                for condition in CONDITIONS:
                    rr=[index[(arm,loss,s,feature,time,condition)] for s in range(3)]
                    cell[condition]={key:ci([r[key] for r in rr]) for key in ('balanced_accuracy','mse','score_rms','fit_ba','fit_mse')}
                    b=cell[condition]['balanced_accuracy'];summary.append(dict(arm=arm,loss=loss,group=feature,time=time,dimension=DIMENSIONS[feature],condition=condition,ba_mean=b['mean'],ba_values=b['values'],ba_min=b['min'],ba_max=b['max'],ba_std=b['std'],ba_ci95=b['ci95'],mse_mean=cell[condition]['mse']['mean'],score_rms_mean=cell[condition]['score_rms']['mean']))
    def probe_values(arm,loss,feature='joint',time=15,condition='nominal'):return np.asarray(groups[f'{arm}_{loss}'][f'{feature}_t{time}'][condition]['balanced_accuracy']['values'])
    effects=[]
    for arm in ARMS:
        for loss in LOSSES:
            joint=probe_values(arm,loss);forced=np.asarray(native_groups[f'{arm}_{loss}']['nominal']['forced_ba']['values']);counts=probe_values(arm,loss,'counts',23)
            for name,baseline in (('joint15_minus_native_forced',forced),('joint15_minus_counts_probe23',counts)):
                effects.append(dict(arm=arm,loss=loss,comparison=name,condition='nominal',primary=True,dimension=26,**ci(joint-baseline)))
    attention=[dict(loss=loss,comparison='FULL_minus_NOATT_joint15',condition='nominal',primary=True,dimension=26,**ci(probe_values('FULL',loss)-probe_values('NOATT',loss))) for loss in LOSSES]
    assert len(effects)==24 and len(attention)==3 and len(summary)==1332
    early=[r for r in probes if r['time']==9 and r['condition']=='nominal'];assert len(early)==216
    early_max=max(abs(r['balanced_accuracy']-.5) for r in early)
    result=dict(study_status='EXPLORATORY_FROZEN_STATE_LINEAR_PROBES',groups=groups,native_groups=native_groups,primary_effects=effects,attention_primary_effects=attention,shuffle_controls=nulls,negative_controls=dict(probe_rows=2664,native_rows=72,probe_identity='Forced binary probes must have BA=.5 on identical opposite-label inputs.',native_identity='Strict BA=.5*coverage; forced-tie0 BA=.5.',max_probe_residual=max(abs(r['balanced_accuracy']-.5) for r in negative),pre_content_t9_nominal_rows=len(early),pre_content_t9_max_residual=early_max),dimensions=DIMENSIONS,times=list(TIMES),conditions=list(CONDITIONS),counts=dict(frozen_neural_models=36,neural_updates=0,probe_configurations=1332,probe_test_rows=3996,native_test_rows=108,fit_label_shuffles_per_config=20,test_label_shuffles_per_config=20),inference='Three paired frozen-model/data blocks;20000 whole-block bootstrap resamples,seed4001900001;pointwise unadjusted95% intervals across27 prespecified primary contrasts. Exploratory, not confirmatory.',probe_definition='Nominal FIT256 only, mean squared error to +/-1 with ridge alpha .1; train-only standardization, std<1e-8 columns zeroed in all conditions; unpenalized intercept. Scores within1e-10 of zero are forced to label0, otherwise score>0 predicts1.',null_scope='FIT-label shuffles refit twenty nuisance readouts and evaluate true TEST labels. TEST-label shuffles keep true-fit scores and permute TEST labels. Values are descriptive controls, not adjusted p-values or proof of absence/presence of information.',claim_boundary='Linear accessibility in frozen post-step states does not establish native causal use, changed native performance, general retrieval, or delay-dependent storage. The fixed cue-position/opposite-candidate task is already logically answerable at t10. Extra history dimensions are reported explicitly.')
    dump('RESULTS.json',result);csvwrite('probe_metrics.csv',probes);csvwrite('native_metrics.csv',native);csvwrite('group_summary.csv',summary);csvwrite('primary_effects.csv',effects+attention)
    print(json.dumps(dict(primary_effects=effects,attention_primary_effects=attention,negative_controls=result['negative_controls']),indent=2))
if __name__=='__main__':main()
