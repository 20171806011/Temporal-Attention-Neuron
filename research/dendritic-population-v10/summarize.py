"""Reporting-only transformations of saved V10 outputs; no data/model execution."""
from pathlib import Path
import csv,json,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/tan-v10-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/tan-v10-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent
O=R/'results';F=R/'figures'
SIZES=(6,4,2,1);ARMS=('DEND','POINT');COLORS={'DEND':'#196f91','POINT':'#bd562d'}

def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def pct(x):return f'{100*x:.2f}'
def csvout(name,fields,rows):
    with (O/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for row in rows:w.writerow({k:json.dumps(row[k]) if isinstance(row.get(k),(dict,list)) else row.get(k,'') for k in fields})
def savefig(fig,name):
    fig.savefig(F/(name+'.png'),dpi=170);fig.savefig(F/(name+'.svg'));plt.close(fig)
    svg=F/(name+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

def accounting(train,test):
    cells=[]
    for h in SIZES:
        fwd=2*h*h+14*h;back=4*h*h+16*h
        for arm in ARMS:
            rows=[r for r in train['rows'] if (r['h'],r['arm'])==(h,arm)]
            cells.append(dict(h=h,arm=arm,runs=len(rows),parameters=2*h*h+17*h+2,float_states=3*h+2,previous_hidden_spike_bits=h,readout_counters=2,
                represented_cells=h+2 if arm=='DEND' else 3*h+2,graded_compartments=2*h if arm=='DEND' else 0,spiking_branch_cells=0 if arm=='DEND' else 2*h,
                local_branch_nonlinearities=2*h,hard_thresholds=h+2 if arm=='DEND' else 3*h+2,reset_operations=h+2 if arm=='DEND' else 3*h+2,
                forward_macs_per_episode_step=fwd,backward_macs_per_episode_step=back,updates=len(rows)*200,
                training_forward_macs=len(rows)*200*128*24*fwd,training_backward_macs=len(rows)*200*128*24*back,
                diagnostic_forward_batches=len(rows)*9,diagnostic_forward_macs=len(rows)*9*128*24*fwd,
                test_forward_macs=len(rows)*512*24*fwd,recorded_training_seconds=sum(r['seconds'] for r in rows)))
    keys=('runs','updates','training_forward_macs','training_backward_macs','diagnostic_forward_batches','diagnostic_forward_macs','test_forward_macs')
    audits={}
    for name in ('verification','gradient_audit'):
        obj=read(O/(name+'.json'));assert obj['status']=='PASS',name
        audits[name]={'status':obj['status'],'seconds':obj.get('seconds'),'scope':'Internal workflow check, not external replication'}
    result=dict(cells=cells,totals={k:sum(c[k] for c in cells) for k in keys},training_wall_seconds=train['seconds'],test_artifact_wall_seconds=test['seconds'],
        test_model_call_seconds=sum(r['inference_seconds'] for r in test['rows']),audits=audits,
        unique_training_episodes=5*128,unique_test_episodes=5*512,model_test_episodes=40*512,independent_seed_data_blocks=5,
        limitations=['Matrix MAC counts exclude scalar nonlinearities, resets, loss, Adam, I/O and audit work.',
            'Same h matches parameters, continuous states, dense matrix terms, batch, optimizer and decoder between arms; represented cells, nonlinearities/reset derivatives and activation patterns differ.',
            'h counts hidden somata, not total neurons; POINT has 2h additional native branch cells and DEND has 2h graded compartments.',
            'Per-weight initialization distributions remain fixed; aggregate drive variance and per-coordinate mean-L2 strength change with size. This is a complete recipe/size comparison, not pure capacity isolation.',
            'Spike counts, dense event-delivery proxy, state variation and dimensionless latency are not energy, ATP, milliseconds or physical hardware efficiency.',
            'Wall times depend on CPU contention/compression; overlapping audit times are not added as elapsed workflow time.'])
    dump(R/'RESOURCE_COST.json',result)
    lines=['# V10 resource and compute accounting','','All numbers refer to the frozen 40-run RATE/.01/200-update study; each arm/size has 5 independent paired seed/data blocks.','','| h | Arm | Parameters | Float states | Cells | Graded branches | Spiking branches | Forward/backward MAC per episode-step |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for c in cells:lines.append(f"| {c['h']} | {c['arm']} | {c['parameters']} | {c['float_states']} | {c['represented_cells']} | {c['graded_compartments']} | {c['spiking_branch_cells']} | {c['forward_macs_per_episode_step']} / {c['backward_macs_per_episode_step']} |")
    lines+=['','Parameters: 12h input weights +2h² recurrent weights +2h branch biases +h soma biases +2h output weights +2 output biases =2h²+17h+2. Continuous states: 2h branches +h somata +2 outputs; separately store h previous binary spikes and 2 count accumulators.','','Forward dense MAC terms: 12h+2h²+2h. Backward terms: output-weight gradient 2h + previous-hidden gradient 2h + input-weight gradient 12h + recurrent-weight gradient 2h² + recurrent-hidden gradient 2h². No input gradient is needed.','','| h | Arm | Updates | Training forward MAC | Training backward MAC | Nine-diagnostic MAC | Test MAC | Train wall seconds |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for c in cells:lines.append(f"| {c['h']} | {c['arm']} | {c['updates']:,} | {c['training_forward_macs']:,} | {c['training_backward_macs']:,} | {c['diagnostic_forward_macs']:,} | {c['test_forward_macs']:,} | {c['recorded_training_seconds']:.3f} |")
    t=result['totals'];lines+=['',f"Totals: {t['updates']:,} parameter updates; {t['training_forward_macs']:,} training-forward and {t['training_backward_macs']:,} training-backward dense MAC terms. Nine diagnostics per fit contribute {t['diagnostic_forward_macs']:,}; testing contributes {t['test_forward_macs']:,}. Final training metrics are already included in the ninth diagnostic.",'',f"Unique training episodes 640; unique held-out episodes 2,560; held-out model-episodes 20,480. The independent inferential units remain five seed/data blocks, not 20,480 independently trained models.",'',f"Training with artifacts: {train['seconds']:.3f}s. Test with artifacts: {test['seconds']:.3f}s. Test model calls: {result['test_model_call_seconds']:.3f}s.",'','Audits: '+', '.join(f"{k}: {v['seconds']:.3f}s" for k,v in audits.items())+'.','','Limits:','']+[f'- {x}' for x in result['limitations']]+['']
    (R/'RESOURCE_COST.md').write_text('\n'.join(lines));return result

def tables(train,test):
    rows=[dict(split=split,**r) for split,source in [('train',train),('test',test)] for r in source['rows']]
    fields=['split','id','seed','arm','h','balanced_accuracy','accuracy','coverage','class_recall','pattern_recall','abstentions','abstention_rate','silent_fraction','nonsilent_tie_fraction','failed','fully_silent','fully_abstaining','confusion','total_spike_events_mean','spike_events_mean','state_l1_change_mean','dense_event_delivery_proxy_mean','correct_settled_fraction_by_step','unique_settled_fraction_by_step','capped_correct_settled_latency_mean','mean_signed_count_margin']
    csvout('per_seed_metrics.csv',fields,rows)
    patterns=[dict(split=r['split'],id=r['id'],seed=r['seed'],arm=r['arm'],h=r['h'],pattern=p,
        bits=''.join(str((p>>j)&1) for j in range(4)),target=int(((p&1)&((p>>1)&1))|(((p>>2)&1)&((p>>3)&1))),recall=r['pattern_recall'][p]) for r in rows for p in range(16)]
    csvout('per_seed_pattern_recall.csv',['split','id','seed','arm','h','pattern','bits','target','recall'],patterns)
    csvout('cell_metrics.csv',list(test['statistics']['cells'][0]),test['statistics']['cells'])
    csvout('paired_contrasts.csv',['contrast','mean','percentile95','paired_values','n_pairs','degenerate','unit'],[dict(contrast=k,**v) for k,v in test['statistics']['intervals'].items()])
    lines=['# V10 per-seed results','','All percentages include abstentions as errors. Pass means BA>=90% AND coverage>=90%, using the frozen RATE/.01/200-update recipe. Each split has all 40 runs.','','| Split | h | Arm | Seed | BA % | Accuracy % | Coverage % | Recall0 % | Recall1 % | Silent % | Non-silent tie % | Pass |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['split']} | {r['h']} | {r['arm']} | {r['seed']} | {pct(r['balanced_accuracy'])} | {pct(r['accuracy'])} | {pct(r['coverage'])} | {pct(r['class_recall'][0])} | {pct(r['class_recall'][1])} | {pct(r['silent_fraction'])} | {pct(r['nonsilent_tie_fraction'])} | {'no' if r['failed'] else 'yes'} |")
    lines+=['','All 16 pattern recalls for every split/seed are retained in [per_seed_pattern_recall.csv](results/per_seed_pattern_recall.csv); bits are printed in input-channel order b0,b1,b2,b3. No pattern or failed fit is omitted. Full confusion matrices, activity and retrospective settling curves are in [per_seed_metrics.csv](results/per_seed_metrics.csv).','']
    (R/'TABLES.md').write_text('\n'.join(lines))

def figures(train,test):
    F.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for arm in ARMS:
        a=np.array([[next(r['balanced_accuracy'] for r in test['rows'] if (r['h'],r['arm'],r['seed'])==(h,arm,s)) for h in SIZES] for s in range(5)])*100
        for row in a:axes[0].plot(range(4),row,color=COLORS[arm],alpha=.27,marker='o',lw=.8)
        axes[0].plot(range(4),a.mean(0),color=COLORS[arm],marker='D',lw=2,label=arm+' mean, 5 blocks')
    axes[0].axhline(90,color='#777777',ls=':',lw=.8);axes[0].set_xticks(range(4),SIZES);axes[0].set_xlabel('Hidden somata h (not total cells)');axes[0].set_ylabel('Held-out balanced accuracy (%)');axes[0].set_ylim(-3,103);axes[0].legend(fontsize=8)
    for i,h in enumerate(SIZES):
        q=test['statistics']['intervals'][f'h{h}_DEND_minus_POINT_balanced_accuracy'];lo,hi=np.array(q['percentile95'])*100
        axes[1].plot([lo,hi],[i,i],color='#333333',lw=2);axes[1].scatter(q['mean']*100,i,color='#196f91',zorder=3)
        axes[1].scatter(np.array(q['paired_values'])*100,np.full(5,i)+np.linspace(-.12,.12,5),color='#999999',s=13,alpha=.6)
    axes[1].axvline(0,color='#777777',lw=.8);axes[1].set_yticks(range(4),[f'h={h}' for h in SIZES]);axes[1].invert_yaxis();axes[1].set_xlabel('DEND − POINT BA (percentage points)');axes[1].set_title('Pointwise 95% paired-bootstrap intervals')
    fig.suptitle('V10A: fixed RATE/.01, 200 updates, 5 paired seed/data blocks');fig.tight_layout(rect=(0,.06,1,.94));fig.text(.5,.02,'All failures retained. Four exploratory intervals are not multiplicity-adjusted; no best-size selection.',ha='center',fontsize=9);savefig(fig,'size_and_paired_difference')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));markers={6:'o',4:'s',2:'^',1:'D'}
    for arm in ARMS:
        for h in SIZES:
            rows=[r for r in test['rows'] if (r['h'],r['arm'])==(h,arm)]
            for ax,key,label in zip(axes,['total_spike_events_mean','state_l1_change_mean'],['Mean total native spikes/episode','Mean L1 state change/episode']):
                ax.scatter([r[key] for r in rows],[100*r['balanced_accuracy'] for r in rows],color=COLORS[arm],marker=markers[h],alpha=.75,label=f'{arm}, h={h}');ax.set_xlabel(label);ax.set_ylabel('Held-out BA (%)');ax.set_ylim(-3,103)
    axes[1].legend(fontsize=7,ncol=2);fig.suptitle('Activity/cost proxies across all 40 fits');fig.tight_layout(rect=(0,.08,1,.93));fig.text(.5,.025,'These are computational proxies, not joules or ATP; low cost on a failed task is not efficient successful computation.',ha='center',fontsize=8.5);savefig(fig,'activity_tradeoffs')
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for ax,h in zip(axes.flat,SIZES):
        for arm in ARMS:
            cell=next(c for c in test['statistics']['cells'] if (c['h'],c['arm'])==(h,arm))
            ax.plot(range(1,9),100*np.array(cell['correct_settled_fraction_by_step']),marker='o',color=COLORS[arm],label=arm)
        ax.set_title(f'h={h}');ax.set_xlabel('Readout prefix step');ax.set_ylabel('Correctly settled episodes (%)');ax.set_ylim(-3,103);ax.legend(fontsize=8)
    fig.suptitle('Retrospective stable-leader curves; all errors/abstentions remain in denominator');fig.tight_layout(rect=(0,.055,1,.94));fig.text(.5,.02,'Future prefixes are used for scoring. This is not an online stopping rule or a changed decoder.',ha='center',fontsize=9);savefig(fig,'retrospective_settling')
    matrix=np.array([np.mean([r['pattern_recall'] for r in test['rows'] if (r['h'],r['arm'])==(h,arm)],axis=0) for h in SIZES for arm in ARMS]);fig,ax=plt.subplots(figsize=(13,4.8));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_xticks(range(16),[''.join(str((p>>j)&1) for j in range(4)) for p in range(16)],rotation=45,ha='right');ax.set_yticks(range(8),[f'{arm} h={h}' for h in SIZES for arm in ARMS]);ax.set_xlabel('Latent bits in channel order b0,b1,b2,b3');ax.set_title('Held-out pattern recall, mean of 5 blocks; abstentions are errors');fig.colorbar(im,ax=ax,label='Recall');fig.tight_layout();savefig(fig,'pattern_recall')

def report(train,test,cost):
    stats=test['statistics'];audit=read(O/'gradient_audit.json');verification=read(O/'verification.json');pre=read(R/'GRADIENT_CHECK.json')
    point_small_pass=[r['seed'] for r in test['rows'] if r['h']==1 and r['arm']=='POINT' and not r['failed']]
    positive_intervals=sum(q['percentile95'][0]>0 for q in stats['intervals'].values())
    lines=['# V10A report: learned feature binding across four fixed network sizes','',
        'This exploratory pilot tests DEND and the existing strong POINT branch-cell network on one noisy four-feature binding distribution, using RATE/.01, 200 updates and five independent paired seed/data blocks per size. It retains every fit, including silent, tied and failed outcomes. The input ends before count readout, so this is binding with brief retention, not a purely static classification test.','',
        '| h | DEND mean BA % | POINT mean BA % | DEND passes/5 | POINT passes/5 | Paired DEND−POINT BA pp | Pointwise 95% interval pp |','|---:|---:|---:|---:|---:|---:|---|']
    for h in SIZES:
        d=next(c for c in stats['cells'] if (c['h'],c['arm'])==(h,'DEND'));p=next(c for c in stats['cells'] if (c['h'],c['arm'])==(h,'POINT'));q=stats['intervals'][f'h{h}_DEND_minus_POINT_balanced_accuracy'];lo,hi=q['percentile95']
        lines.append(f"| {h} | {pct(d['mean_balanced_accuracy'])} | {pct(p['mean_balanced_accuracy'])} | {5-d['failures']} | {5-p['failures']} | {100*q['mean']:.2f} | [{100*lo:.2f},{100*hi:.2f}] |")
    lines+=['','BA is mean class recall, with abstentions wrong. Pass requires BA>=90% AND coverage>=90%. Intervals resample the five paired training/data blocks, not individual episodes; they are exploratory, pointwise and not multiplicity-adjusted. A zero-crossing interval does not establish equivalence, and a degenerate interval does not establish population equality or noninferiority.','',
        f"Under this single binding distribution, fixed RATE/.01 recipe, 200-update budget and five paired blocks, {positive_intervals} of the four pointwise intervals lie above zero. This is evidence about this complete training setup, not an architecture-only effect or a confirmed general ranking. At h=1, POINT passes the frozen BA/coverage gate in seed(s) {point_small_pass}; that model has two branch cells, one hidden soma and two outputs, so it is a five-cell network, not one point neuron. Its verified passing case prevents interpreting the minimum tested POINT size as inherently unable to meet this task gate.",'',
        '![Size comparison and paired intervals](figures/size_and_paired_difference.png)','',
        '## Methods','',
        'The frozen task samples all 16 latent Boolean patterns equally and labels (b0 AND b1) OR (b2 AND b3). High amplitudes are U[.8,1.2], lows U[0,.1]; features persist on steps 8–15, readout cue on 16–23, a label-independent nuisance on 7–9, and independent Gaussian noise SD .03 is clipped at zero. Each block has 128 training and 512 held-out episodes shared across arms/sizes. Pattern balance gives 7/16 positives; a majority classifier obtains 56.25% ordinary accuracy and 50% BA. Four-high and three-high inputs are positive; cross-pair-only patterns are negative. Held-out examples are new amplitudes/noise on the same 16 patterns, not unseen Boolean combinations.','',
        'The h=6 equations and initialization match V8 exactly. At each h, DEND and POINT share dense input/recurrent/output access, parameter count 2h²+17h+2, continuous state count 3h+2 and paired initial parameters. DEND has 2h graded ReLU compartments; POINT has 2h additional native branch cells. Both retain .85 branch and .65 soma/output retention. Hidden/output spikes, and POINT branch spikes, use strict>.5 threshold/reset; the decoder counts only native output spikes in steps 16–23 and requires a unique largest count. No external pulse generator or analog decoder is added.','',
        'All 40 fits use 200 full-batch Adam updates at RATE/.01, inherited initialization distributions, norm clipping 1, parameter range [-2,2], fixed final checkpoint and surrogate slope 1/(1+5|candidate−.5|)². This is a proxy learning rule, not the exact derivative of the hard-spike model. No validation selection, tuning, restart, early stopping or best-size choice is made.','',
        '## Complete outcomes and computational proxies','',
        'Every seed, class recall, coverage, silence and non-silent tie rate appears in [TABLES.md](TABLES.md). All 16 per-seed pattern recalls appear in [the CSV](results/per_seed_pattern_recall.csv). Raw weights, gradients, checkpoints, native trajectories and datasets remain in results/.','',
        '| h | Arm | Accuracy % | Coverage % | Failures/5 | Fully silent/5 | Mean spikes | Mean L1 state change | Capped correct-settled step |','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in stats['cells']:lines.append(f"| {c['h']} | {c['arm']} | {pct(c['mean_accuracy'])} | {pct(c['mean_coverage'])} | {c['failures']} | {c['fully_silent']} | {c['mean_total_spikes']:.3f} | {c['mean_state_l1_change']:.3f} | {c['mean_capped_correct_settled_latency']:.3f} |")
    lines+=['',
        '![Pattern-level outcomes](figures/pattern_recall.png)','',
        'Native branch/hidden/output events are retained separately. DEND has no branch-spike events by definition but performs graded branch updates, so total spike differences do not alone measure total cost. State changes use model coordinates. Correct settling is determined retrospectively from all remaining prefix-count decisions; wrong or abstaining episodes receive capped latency 9 rather than being excluded. These are dimensionless computational proxies, not ATP, joules, milliseconds or a realizable online stopping rule.','',
        '![Activity proxies](figures/activity_tradeoffs.png)','',
        '![Retrospective settling](figures/retrospective_settling.png)','',
        '## Verification and provenance','',
        f"Protocol/source commit before scientific data: `{train['protocol_commit']}`. Final training/checkpoint commit before held-out data: `{test['checkpoint_commit']}`. Both gates are checked against Git blobs by the execution scripts. The seed registry, source audit and freeze hashes are retained; test evaluation is guarded by TEST_STARTED.json.",'',
        f"Deterministic parity/native-boundary fixtures passed. Before training, {pre['finite_difference_parameter_checks']} anchored finite-difference parameter checks had maximum residual {max(r['max_surrogate_chain_rule_error'] for r in pre['rows']):.6g}. After training, {audit['parameter_derivative_checks']} final-checkpoint parameter checks had maximum residual {audit['max_chain_rule_residual']:.6g}; {audit['sampled_updates']} sampled updates had maximum clipped-gradient residual {audit['max_sampled_gradient_residual']:.6g}. These finite differences validate the locally anchored surrogate graph, not hard-spike derivatives.",'',
        'The separately authored native/data/optimizer verifier reports '+verification['status']+'. It checks saved initialization/data, all Adam updates/checkpoints, native final training and test trajectories, exact spikes/counts/predictions, metrics and seed-level statistics. Its JSON records state residuals and complete counts. These remain checks within the same AI-assisted workflow, not external independent replication or peer review.','',
        '## Resources and execution','',
        f"The fixed 40 fits used 8,000 updates: {cost['totals']['training_forward_macs']:,} training-forward and {cost['totals']['training_backward_macs']:,} training-backward dense matrix MAC terms; diagnostic and test terms are reported separately in [RESOURCE_COST.md](RESOURCE_COST.md). Training/artifact wall time was {train['seconds']:.3f}s and held-out/artifact wall time {test['seconds']:.3f}s. No paid compute was used. The h-axis is hidden-soma count; total represented cells differ between architectures.",'',
        '## What this does not establish','',
        'A single static affine threshold cannot implement the four-bit target, as shown by the positive/negative vector-sum contradiction in [DESIGN_REVIEW.md](DESIGN_REVIEW.md). Two ordinary threshold subunits can implement it. The current POINT baseline already has branch cells, recurrence and analog subthreshold state; the static theorem is not an impossibility result for this baseline. Observed finite-budget failures concern this recipe and data distribution. No new pooling intervention is included, so V10 cannot identify causal necessity of local nonlinear partition. It does not establish a universal architecture ranking, energy/hardware efficiency, biological realization, long-delay retention or compositional extrapolation.','',
        'The proposed two-timescale experiment was not run. The older 150 ms/10 ms result used a different modified model and analog inverse decoder. It cannot be imported into these native count-decoded population equations, and graded memory has not been shown to be zero energy. Biological motivation is limited to the primary compartmental-model and dendritic-integration studies cited in [DESIGN_REVIEW.md](DESIGN_REVIEW.md).','',
        '## Limitations','',
        'There are only five independent training/data blocks, four unadjusted exploratory primary intervals, one task distribution, one training recipe and 200 updates. Large episode counts do not remedy low seed-level precision. Dense shared input access avoids privileged DEND wiring but does not force a specific learned anatomical decomposition. Size changes alter state, parameters, matrix cost, aggregate initialization drive variance and the per-coordinate mean-L2 coefficient; the curves describe the full fixed setup rather than pure representational capacity. Nonlinearities, reset derivatives and activation patterns are not matched. Identical nominal optimizer settings do not establish equal optimization quality or effective updates between architectures. The brief retention interval can affect the size results.','',
        'Every size and failed fit is retained. Lower activity on an inaccurate fit is not efficient successful computation. An interval containing zero is inconclusive rather than evidence of equivalence; interval degeneration is not a noninferiority bound. Future work requires a distinct justified protocol and new data, not an automatic sweep or retuning of these results.','',
        'Research direction and native firing constraints were proposed by Li Zexu. Protocol formalization, implementation, execution, numerical verification and this report draft were AI-assisted. Earlier results remain preserved; this bounded study does not automatically authorize another phase or public submission.','']
    (R/'REPORT.md').write_text('\n'.join(lines))

def main():
    train=read(O/'TRAIN_LOCK.json');test=read(O/'TEST_SUMMARY.json')
    assert len(train['rows'])==len(test['rows'])==40
    assert {(r['h'],r['arm'],r['seed']) for r in test['rows']}=={(h,a,s) for h in SIZES for a in ARMS for s in range(5)}
    assert len(test['statistics']['cells'])==8 and len(test['statistics']['intervals'])==4
    cost=accounting(train,test);tables(train,test);figures(train,test);report(train,test,cost)
    print('Saved V10 report, tables, resources and four PNG/SVG figures; no model or data generator calls.')
if __name__=='__main__':main()
