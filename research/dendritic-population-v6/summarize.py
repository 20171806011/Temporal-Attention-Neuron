"""Tables/figures and planning arithmetic from saved V6 artifacts only."""
from pathlib import Path
import json,csv,math,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/tan-v6-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';F=R/'figures'

def main():
    F.mkdir(exist_ok=True);train=json.loads((O/'TRAIN_LOCK.json').read_text());test=json.loads((O/'TEST_SUMMARY.json').read_text());stats=test['statistics']
    cost=[]
    for arm in ('DEND','POINT'):
        for recipe in ('RATE_0.01','CE_0.03'):
            rows=[r for r in train['rows'] if (r['arm'],r['recipe'])==(arm,recipe)];updates=sum(r['steps'] for r in rows)
            cost.append(dict(arm=arm,recipe=recipe,trajectories=len(rows),parameter_updates=updates,
                forward_dense_MAC_terms=updates*128*24*156,backward_dense_MAC_terms=updates*128*24*240,
                checkpoint_training_forward_batches=len(rows)*9,checkpoint_forward_dense_MAC_terms=len(rows)*9*128*24*156,
                test_forward_dense_MAC_terms=len(rows)*512*24*156,recorded_training_seconds=sum(r['seconds'] for r in rows)))
    costs=dict(cells=cost,unique_training_trajectories=80,updates=16000,training_phase_wall_seconds=train['seconds'],
        test_artifact_phase_wall_seconds=test['seconds'],test_inference_seconds=sum(r['inference_seconds'] for r in test['rows']),
        test_model_episodes=40960,unique_test_histories=10240,checkpoint_training_forward_batches=720,
        note='Dense matrix MAC terms exclude scalar activation/reset, losses,Adam,I/O,data generation,audits and posthoc; not FLOP/energy/hardware equivalence. Checkpoint passes include final training metrics.')
    (O/'COST.json').write_text(json.dumps(costs,indent=2)+'\n')
    primary=stats['intervals'][stats['primary']];delta=np.asarray(primary['paired_values']);sd=float(np.std(delta,ddof=1))
    plans=[dict(target_half_width=h,normal_approx_pairs=math.ceil((1.959963984540054*sd/h)**2) if sd>0 else None,
        inflated_SD_1_5_pairs=math.ceil((1.959963984540054*1.5*sd/h)**2) if sd>0 else None) for h in (.05,.10)]
    planning=dict(primary=stats['primary'],pilot_pairs=20,paired_accuracy_SD_ddof1=sd,mean_precision_illustrations=plans,
        pilot_zero_failures_one_sided95_upper=1-.05**(1/20),
        future_single_failure_proportion_worstcase_normal_pairs_for_halfwidth_0_10=97,
        future_single_failure_proportion_worstcase_normal_pairs_for_halfwidth_0_05=385,
        fresh_zero_failure_runs_for_one_sided95_upper_below_0_05=59,
        upper_if_zero_of59=1-.05**(1/59),
        limitations='Illustrative normal precision estimates, not guaranteed power or required/authorized runs.20pairpilotSD is uncertain, especially with silent/success mixtures.1.5xSD is sensitivity,not a confidence bound.0SD would not justify n=0. Paired failure differences additionally require discordance assumptions.Confirmatory study needs new seeds and independent implementation/team review; none run here.')
    (O/'PRECISION_PLAN.json').write_text(json.dumps(planning,indent=2)+'\n')
    fields=['split','id','seed','arm','recipe','accuracy','coverage','abstentions','abstention_rate','silent_fraction','nonsilent_tie_fraction','both_saturated_fraction','failed','fully_silent','fully_abstaining','class_recall','confusion','mean_signed_count_margin','signed_count_margin_histogram','readout_hidden_event_rate','readout_output_event_rate','output_candidate_margin_quantiles']
    with (O/'per_seed_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for split,rows in [('train',train['rows']),('test',test['rows'])]:
            for r in rows:
                d={k:(json.dumps(r[k]) if isinstance(r.get(k),list) else r[k]) for k in fields if k in r};d['split']=split;writer.writerow(d)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'DEND':'#176b87','POINT':'#b7542c'}
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    for ax,arm in zip(axs[0],('DEND','POINT')):
        vals=[]
        for recipe in ('RATE_0.01','CE_0.03'):
            vals.append(np.array([next(r['accuracy'] for r in test['rows'] if (r['arm'],r['recipe'],r['seed'])==(arm,recipe,s)) for s in range(20)])*100)
        for s in range(20):ax.plot([0,1],[vals[0][s],vals[1][s]],color=colors[arm],alpha=.4,lw=.8,marker='o',ms=3)
        ax.set_xticks([0,1],['RATE / .01','CE / .03']);ax.set_xlim(-.25,1.25);ax.set_ylim(-4,104);ax.set_ylabel('Test accuracy (%)')
        iv=stats['intervals']['accuracy_'+arm+'_CE_minus_RATE'];lo,hi=np.array(iv['percentile95'])*100
        ax.set_title(arm+(' (primary)' if arm=='DEND' else ' (secondary)'))
        ax.set_xlabel(f"Paired change: {iv['mean']*100:.2f} pp; 95% [{lo:.2f}, {hi:.2f}]")
    labels=['DEND\nRATE','DEND\nCE','POINT\nRATE','POINT\nCE']
    for ax,event,title in zip(axs[1],('failed','fully_silent'),('Failed runs','Fully output-silent runs')):
        for i,c in enumerate(stats['cells']):
            d=c['run_events'][event];y=d['fraction']*100;lo,hi=np.array(d['wilson95'])*100
            ax.errorbar(i,y,yerr=np.array([[max(0,y-lo)],[max(0,hi-y)]]),fmt='o',color=colors[c['arm']],capsize=5)
            ax.annotate(str(d['count'])+'/20',(i,y),xytext=(8,0),textcoords='offset points',fontsize=9)
        ax.set_xticks(range(4),labels);ax.set_xlim(-.4,3.7);ax.set_ylim(-5,110);ax.set_ylabel('Run fraction (%)');ax.set_title(title+' (Wilson 95%)')
    fig.suptitle('V6: 20 paired training seeds, four fixed recipes/structures',fontsize=12)
    fig.tight_layout(rect=(0,.03,1,.97));fig.text(.5,.01,'200 updates; 512 test episodes per seed. Accuracy intervals use training-seed pairs, not episodes.',ha='center',fontsize=9)
    fig.savefig(F/'stability.png',dpi=170);fig.savefig(F/'stability.svg');plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for ax,(arm,recipe) in zip(axs.flat,[(a,r) for a in ('DEND','POINT') for r in ('RATE_0.01','CE_0.03')]):
        for row in train['rows']:
            if (row['arm'],row['recipe'])!=(arm,recipe):continue
            z=np.load(O/row['trace'],allow_pickle=False);y=np.r_[z['history'][:,3],row['accuracy']]*100
            ax.plot(np.arange(201),y,color=colors[arm],alpha=.4,lw=.8)
        ax.axhline(90,ls=':',lw=.7,color='#777');ax.set_title(arm+' '+recipe);ax.set_xlabel('Completed parameter updates');ax.set_ylabel('Training accuracy (%)');ax.set_ylim(-3,103)
    fig.suptitle('All 80 saved training trajectories (no runs excluded)',fontsize=12);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(F/'training_trajectories.png',dpi=170);fig.savefig(F/'training_trajectories.svg');plt.close(fig)
    for svg in F.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({'cost':costs,'precision_plan':planning},indent=2))

if __name__=='__main__':main()
