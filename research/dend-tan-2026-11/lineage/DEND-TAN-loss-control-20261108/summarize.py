"""Saved-record objective comparisons and gradient diagnostics; no model runs."""
from pathlib import Path
import csv,json
import numpy as np
R=Path(__file__).resolve().parent
ARMS=('DEND','FULL','NOATT','GAIN2');LOSSES=('CE','COUNT','CECOM')
CONDITIONS=('nominal','clean','strong','cue_gap3','tag_removed','query_only')
RESOURCE_KEYS=('soma_events','output_events','recurrent_deliveries','output_deliveries','hold_soma_events','hold_output_events')
COMPONENT_KEYS=('discriminative_loss','common_loss','l2_loss','total_loss','discriminative_count_gradient_norm','common_count_gradient_norm','total_count_gradient_norm')
BOOT=np.random.default_rng(3991900001).integers(0,3,(20000,3))
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(name,x):(R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ci(values):
    x=np.asarray(values,dtype=float);assert x.shape==(3,)
    return dict(values=x.tolist(),mean=float(x.mean()),min=float(x.min()),max=float(x.max()),std=float(x.std(ddof=1)),n=3,ci95=np.quantile(x[BOOT].mean(1),[.025,.975]).tolist())
def csvwrite(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def count_metrics(c,y):
    eq=c[:,0]==c[:,1];silent=c.sum(1)==0;saturated=np.all(c==8,axis=1);pred=np.where(eq,-1,c.argmax(1));coverage=float(np.mean(~eq));correct=float(np.mean(pred==y))
    return dict(balanced_accuracy=float(sum(np.mean(pred[y==j]==j) for j in (0,1))/2),accuracy=correct,correct_coverage=correct,coverage=coverage,decision_coverage=coverage,silence=float(np.mean(silent)),tie_rate=float(np.mean(eq)),nonzero_tie_rate=float(np.mean(eq&~silent)),saturated_tie_rate=float(np.mean(saturated)),intermediate_tie_rate=float(np.mean(eq&~silent&~saturated)),prediction0_rate=float(np.mean(pred==0)),prediction1_rate=float(np.mean(pred==1)),conditional_accuracy=correct/coverage if coverage else None,query_output_spikes_mean=float(np.mean(c.sum(1))),mean_count_difference=float(np.mean(abs(c[:,0]-c[:,1]))))
def main():
    assert read(R/'AUDIT.json')['status']=='PASS'
    raw=read(R/'evaluate/ROWS.json');training=read(R/'train/ROWS.json');assert len(raw)==216 and len(training)==36
    rows=[];negative=[]
    for source in raw:
        row=dict(source);assert row['lr']==.0025 and row['loss'] in LOSSES
        base=np.load(R/'evaluate'/f'test_s{row["seed"]}.npz');z=np.load(R/'evaluate'/f'{row["id"]}_{row["condition"]}.npz');computed=count_metrics(z['counts'],base['y'])
        for key in ('balanced_accuracy','accuracy','coverage','silence'):assert abs(row[key]-computed[key])<1e-12
        row.update(computed);row.update(zip(RESOURCE_KEYS,row['mean_resources']));row['internal_pulse_deliveries']=row['recurrent_deliveries']+row['output_deliveries']
        assert abs(row['tie_rate']-(row['silence']+row['saturated_tie_rate']+row['intermediate_tie_rate']))<1e-12
        assert row['hold_soma_events']==row['hold_output_events']==0
        if row['condition'] in ('tag_removed','query_only'):
            residual=row['balanced_accuracy']-.5*row['coverage'];assert abs(residual)<1e-12
            row['negative_control_ba_minus_half_coverage']=residual;negative.append({k:row[k] for k in ('id','arm','loss','seed','condition','balanced_accuracy','coverage','silence','negative_control_ba_minus_half_coverage')})
        rows.append(row)
    groups={};metrics=('balanced_accuracy','accuracy','correct_coverage','coverage','decision_coverage','silence','tie_rate','nonzero_tie_rate','saturated_tie_rate','intermediate_tie_rate','prediction0_rate','prediction1_rate','query_output_spikes_mean','mean_count_difference')+RESOURCE_KEYS+('internal_pulse_deliveries',)
    for arm in ARMS:
        for loss in LOSSES:
            group={};groups[f'{arm}_{loss}']=group
            for condition in CONDITIONS:
                rr=sorted((r for r in rows if (r['arm'],r['loss'],r['condition'])==(arm,loss,condition)),key=lambda r:r['seed']);assert [r['seed'] for r in rr]==[0,1,2]
                group[condition]={key:ci([r[key] for r in rr]) for key in metrics};group[condition]['ba90_count']=sum(r['balanced_accuracy']>=.9 for r in rr)
                group[condition]['ba75_and_coverage90_count']=sum(r['balanced_accuracy']>=.75 and r['coverage']>=.9 for r in rr)
                group[condition]['ba90_and_coverage90_count']=sum(r['balanced_accuracy']>=.9 and r['coverage']>=.9 for r in rr)
                conditional=[r['conditional_accuracy'] for r in rr];group[condition]['conditional_accuracy']=dict(values=conditional,defined_count=sum(v is not None for v in conditional),definition='Correct among decided episodes; undefined for a seed with no decisions. No partial-seed mean or CI is reported.')
    def values(arm,loss,condition,key='balanced_accuracy'):return np.asarray(groups[f'{arm}_{loss}'][condition][key]['values'])
    effects=[]
    for arm in ARMS:
        for loss in ('COUNT','CECOM'):
            for condition in CONDITIONS:
                effects.append(dict(arm=arm,loss=loss,comparison=loss+'_minus_CE',condition=condition,primary=condition=='nominal',negative_control=condition in CONDITIONS[4:],**ci(values(arm,loss,condition)-values(arm,'CE',condition)),coverage_difference=ci(values(arm,loss,condition,'coverage')-values(arm,'CE',condition,'coverage'))))
    attention=[]
    for loss in LOSSES:
        for condition in CONDITIONS:attention.append(dict(loss=loss,condition=condition,comparison='FULL_minus_NOATT',primary=False,negative_control=condition in CONDITIONS[4:],**ci(values('FULL',loss,condition)-values('NOATT',loss,condition))))
    alternative_effects=[dict(arm=arm,condition='nominal',comparison='COUNT_minus_CECOM',primary=False,**ci(values(arm,'COUNT','nominal')-values(arm,'CECOM','nominal'))) for arm in ARMS]
    diagnostics=[];component_diagnostics=[]
    for tr in training:
        z=np.load(R/'train'/f'{tr["id"]}.npz');base=np.load(R/'train'/f'base_s{tr["seed"]}.npz');y=base['y']
        assert z['counts'].shape==(400,64,2) and z['history'].shape==(400,4) and z['components'].shape==(400,7)
        meta={k:tr[k] for k in ('id','seed','arm','loss','lr')}
        for k in (0,1,2,5,10,25,50,100,200,399):
            history=z['history'][k];row=dict(**meta,completed_updates=k,next_update=k+1,condition='nominal',**count_metrics(z['counts'][k],y),training_loss=float(history[0]),gradient_norm=float(history[1]),task_gradient_norm=float(history[2]),regularizer_gradient_norm=float(history[3]));row.update(zip(COMPONENT_KEYS,z['components'][k].tolist()));diagnostics.append(row)
        steps=z['diagnostic_steps'];components=z['component_gradients'];assert steps.shape==(21,) and components.shape==(21,3,176)
        assert np.array_equal(steps,np.array(list(range(0,400,20))+[399]))
        for j,k0 in enumerate(steps):
            k=int(k0);gd,gc,gr=components[j];nd=float(np.linalg.norm(gd));nc=float(np.linalg.norm(gc));nr=float(np.linalg.norm(gr));total=float(np.linalg.norm(gd+gc+gr));raw=float(z['history'][k,1]);assert np.isclose(total,raw,atol=1e-8,rtol=1e-10)
            component_diagnostics.append(dict(**meta,completed_updates=k,discriminative_parameter_gradient_norm=nd,common_parameter_gradient_norm=nc,l2_parameter_gradient_norm=nr,combined_parameter_gradient_norm=total,global_clip_factor=1/max(1,raw),common_to_discriminative_norm_ratio=nc/nd if nd else None,discriminative_common_cosine=float(gd@gc)/(nd*nc) if nd and nc else None,**dict(zip(COMPONENT_KEYS,z['components'][k].tolist()))))
    assert len(diagnostics)==360 and len(component_diagnostics)==756 and len(effects)==48 and len(attention)==18 and len(negative)==72
    result=dict(study_status='EXPLORATORY_OBJECTIVE_CONTROL',fixed_learning_rate=.0025,objectives=list(LOSSES),conditions=list(CONDITIONS),groups=groups,effects=effects,primary_effects=[e for e in effects if e['primary']],secondary_objective_effects=[e for e in effects if not e['primary']],attention_effects=attention,alternative_objective_effects=alternative_effects,negative_controls=dict(status='PASS',rows=negative,max_abs_residual=max(abs(r['negative_control_ba_minus_half_coverage']) for r in negative),identity='BA = 0.5 * decision coverage on identical opposite-label inputs; ties count wrong.'),inference='Three paired independently fitted seeds;20000 whole-block bootstrap resamples,seed3991900001;pointwise unadjusted95% intervals. Eight nominal alternative-objective-minus-CE primary comparisons. All settings retained. No broad superiority or unique-cause claim.',metric_definitions=dict(correct_coverage='Fraction of all episodes correctly decided; equals accuracy here.',decision_coverage='Fraction with unequal native query counts.',saturated_tie='Both query counts exactly8.',silence='Both query counts exactly0.',intermediate_tie='Equal positive counts below8.',conditional_accuracy='Correct / decided; undefined when no decisions.'),gradient_diagnostic_scope='Stored raw parameter-gradient components precede clipping and Adam; norms and cosines do not measure their individual contributions to nonlinear optimizer updates. Alternative objectives pay for21 additional diagnostic backwards; training updates remain400 for all.',diagnostic_timing='Counts at k precede update k+1; final test uses400 completed updates.',objective_algebra='COUNT=COMMON+mean((d-4)^2/32), where s=c0+c1,d=c_correct-c_incorrect,COMMON=mean((s-8)^2/32). CECOM=CE+COMMON; CE=mean(log(1+exp(-.5d))).')
    dump('RESULTS.json',result);dump('DIAGNOSTICS.json',diagnostics);dump('COMPONENT_DIAGNOSTICS.json',component_diagnostics);csvwrite('metrics.csv',rows);csvwrite('diagnostics.csv',diagnostics);csvwrite('component_diagnostics.csv',component_diagnostics)
    print(json.dumps(dict(primary=result['primary_effects'],nominal={key:{m:g['nominal'][m]['mean'] for m in ('balanced_accuracy','coverage','silence','saturated_tie_rate')} for key,g in groups.items()},negative_control_max_residual=result['negative_controls']['max_abs_residual']),indent=2))
if __name__=='__main__':main()
