"""Render saved results and cost accounting only; never train or evaluate a model."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
O = R / 'results'
F = R / 'figures'
F.mkdir(exist_ok=True)

def read(name):
    return json.loads((O / name).read_text())

def main():
    pre, fin, test = [read(n) for n in ('PREPARE_LOCK.json', 'FINAL_LOCK.json', 'TEST_SUMMARY.json')]
    rows = pre['rows'] + fin['rows']
    unique = list({r['trace']: r for r in rows}.values())
    phases = [('A', 'fixed'), ('search', 'candidate'), ('B', 'fixed'), ('B', 'selected')]
    cost = []
    for stage, role in phases:
        for arm in ('DEND', 'POINT'):
            rr = [r for r in unique if (r['stage'], r['role'], r['arm']) == (stage, role, arm)]
            updates = sum(r['steps'] for r in rr)
            cost.append(dict(stage=stage, role=role, arm=arm, trajectories=len(rr),
                updates=updates, batch_forward_calls=updates, batch_backward_calls=updates,
                forward_dense_MAC_terms=updates*128*24*156,
                backward_dense_MAC_terms=updates*128*24*240,
                recorded_training_seconds=sum(r['seconds'] for r in rr)))
    inference = list({r['file']: r for r in test['rows']}.values())
    summary = dict(training=cost, unique_trajectories=len(unique),
        aliases=len(rows)-len(unique), total_updates=sum(r['steps'] for r in unique),
        training_phase_wall_seconds=pre['seconds']+fin['seconds'],
        test_artifact_phase_wall_seconds=test['seconds'], unique_test_model_evaluations=len(inference),
        test_model_episodes=len(inference)*512, unique_test_histories=test['unique_test_histories'],
        test_forward_dense_MAC_terms=len(inference)*512*24*156,
        measured_test_inference_seconds=sum(r['inference_seconds'] for r in inference),
        validation_forward_calls=60, validation_forward_dense_MAC_terms=60*128*24*156,
        limits='Dense MAC terms count matrix products, excluding scalar activation/reset, Adam, data generation, compression, audits, and hardware effects. Forward 72+72+12=156; backward 12+12+72+72+72=240 per episode/time step. Wall time is observed serial CPU timing, not matched compute or energy evidence.')
    (O / 'COST.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (O / 'per_seed_metrics.csv').open('w', newline='') as f:
        fields = ['phase','role','mode','arm','seed','accuracy','coverage','abstentions',
                  'abstention_rate','silent_fraction','both_saturated_fraction','failed',
                  'class0_recall','class1_recall','confusion']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in test['rows']:
            d = {k:r[k] for k in fields if k in r}
            d.update(class0_recall=r['class_recall'][0], class1_recall=r['class_recall'][1],
                     confusion=json.dumps(r['confusion']))
            writer.writerow(d)

    def values(phase, role, arm, mode='intact', metric='accuracy'):
        return np.array([next(r[metric] for r in test['rows'] if
            (r['phase'],r['role'],r['arm'],r['mode'],r['seed']) ==
            (phase,role,arm,mode,s)) for s in range(10)])*100

    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    colors = {'DEND':'#176b87', 'POINT':'#b7542c'}
    fig, axes = plt.subplots(1,3,figsize=(12,4.4),sharey=True)
    for ax,(phase,role,title) in zip(axes,[('A','fixed','A: fixed RATE / .01'),
                                         ('B','fixed','B: fixed RATE / .01'),
                                         ('B','selected','B: independently selected (primary)')]):
        d,p = values(phase,role,'DEND'),values(phase,role,'POINT')
        for s in range(10):
            ax.plot([0,1],[d[s],p[s]],color='#b7bec5',lw=.9,zorder=1)
        for x,arm,y in [(0,'DEND',d),(1,'POINT',p)]:
            ax.scatter(np.full(10,x),y,color=colors[arm],s=32,zorder=2,alpha=.85)
            ax.scatter([x],[y.mean()],color=colors[arm],marker='_',s=420,zorder=3)
        iv = test['intervals'][f'{phase}_{role}_DEND_minus_POINT']
        lo,hi=np.array(iv['percentile95'])*100
        ax.set_title(title,fontsize=10)
        ax.set_xticks([0,1],['DEND','POINT']);ax.set_xlim(-.35,1.35)
        ax.set_ylim(-4,104);ax.axhline(90,color='#999',lw=.7,ls=':')
        ax.set_xlabel(f"DEND - POINT: {iv['mean']*100:.2f} pp\n95% seed-pair interval [{lo:.2f}, {hi:.2f}]")
    axes[0].set_ylabel('Unconditional accuracy (%)')
    fig.suptitle('Delayed XOR: each line is one independently trained seed pair',fontsize=12)
    fig.text(.5,.01,'10 seed pairs per panel; 512 balanced test episodes per seed. B panels share data and initialization.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(F/'paired_accuracy.png',dpi=180);fig.savefig(F/'paired_accuracy.svg');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4.2))
    for ax,arm in zip(axes[:2],('DEND','POINT')):
        fixed,selected=values('B','fixed',arm),values('B','selected',arm)
        for s in range(10):
            ax.plot([0,1],[fixed[s],selected[s]],color=colors[arm],alpha=.45,marker='o',ms=4)
        ax.set_xticks([0,1],['Fixed','Selected']);ax.set_xlim(-.3,1.3);ax.set_ylim(-4,104)
        ax.set_title(arm+': same B seeds');ax.set_ylabel('Accuracy (%)')
    ax=axes[2]
    for phase,role,marker in [('A','fixed','o'),('B','fixed','s'),('B','selected','^')]:
        for arm in ('DEND','POINT'):
            ax.scatter(values(phase,role,arm,'no_history','coverage'),values(phase,role,arm,'no_history'),
                       color=colors[arm],marker=marker,s=24,alpha=.65,
                       label=f'{phase} {role} {arm}')
    ax.plot([0,100],[0,50],'k--',lw=1,label='Population expectation: coverage / 2')
    ax.set_xlim(-3,103);ax.set_ylim(-3,58);ax.set_xlabel('Coverage after prefix removal (%)')
    ax.set_ylabel('Accuracy after prefix removal (%)');ax.set_title('No-history diagnostic')
    ax.text(.04,.95,'Markers: A fixed o; B fixed square;\nB selected triangle. Blue DEND; orange POINT.',
            transform=ax.transAxes,va='top',fontsize=7)
    fig.suptitle('Configuration transfer and input-prefix dependence',fontsize=12)
    fig.text(.5,.01,'Prefix removal changes the input distribution; it does not establish a specific memory mechanism.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(F/'configuration_diagnostic.png',dpi=180);fig.savefig(F/'configuration_diagnostic.svg');plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='training'},indent=2))

if __name__ == '__main__':
    main()
