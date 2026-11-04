"""V9 reporting only: read saved, audited results; never fit/evaluate a model."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tan-v9-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
O = R / 'results'
F = R / 'figures'
ARMS = ('DEND', 'POINT')
CONDITIONS = ('intact', 'retrained', 'cut')
COLORS = {'intact': '#23658a', 'retrained': '#b65c34', 'cut': '#777777'}
LABELS = {'intact': 'Intact', 'retrained': 'Retrained no-cross', 'cut': 'Inference cut'}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(value.rstrip() + '\n')


def dump(path, value):
    write(path, json.dumps(value, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(x):
    return f'{100*x:.2f}'


def interval(x):
    return f'[{pct(x[0])}, {pct(x[1])}]'


def write_csv(name, fields, rows):
    with (O / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), (list, dict))
                             else row.get(key, '') for key in fields})


def save_figure(fig, stem):
    fig.savefig(F / (stem + '.png'), dpi=170)
    path = F / (stem + '.svg')
    fig.savefig(path, metadata={'Date': None})
    write(path, '\n'.join(line.rstrip() for line in path.read_text().splitlines()))
    plt.close(fig)


def selected(rows, arm, condition):
    return sorted([r for r in rows if (r['arm'], r['condition']) == (arm, condition)],
                  key=lambda row: row['seed'])


def resources(train, test, audits):
    cells = []
    for arm in ARMS:
        for condition in CONDITIONS:
            rows = selected(train['rows'], arm, condition)
            tests = selected(test['rows'], arm, condition)
            updates = sum(row['steps'] for row in rows)
            cells.append(dict(arm=arm, condition=condition, trajectories=len(rows),
                parameter_updates=updates, stored_parameters=176,
                effective_trainable_parameters=116 if condition == 'retrained' else (176 if condition == 'intact' else 0),
                forward_dense_MAC_terms=updates*128*24*156,
                backward_dense_MAC_terms=updates*128*24*240,
                checkpoint_forward_batches=len(rows)*9,
                checkpoint_forward_dense_MAC_terms=len(rows)*9*128*24*156,
                test_forward_dense_MAC_terms=len(tests)*512*24*156,
                recorded_training_seconds=sum(row['seconds'] for row in rows),
                recorded_test_inference_seconds=sum(row['inference_seconds'] for row in tests)))
    additive = ('trajectories', 'parameter_updates', 'forward_dense_MAC_terms',
                'backward_dense_MAC_terms', 'checkpoint_forward_batches',
                'checkpoint_forward_dense_MAC_terms', 'test_forward_dense_MAC_terms')
    audit_cost = {name: dict(status=obj['status'], recorded_wall_seconds=obj['seconds'])
                  for name, obj in audits.items()}
    result = dict(cells=cells, totals={key: sum(c[key] for c in cells) for key in additive},
        shared_resources=dict(stored_float64_parameters=176, continuous_states=20,
            previous_hidden_spike_states=6, output_count_accumulators=2,
            hidden_somata=6, output_cells=2, branch_units=12,
            DEND_branch_type='graded', POINT_branch_type='native spiking with reset',
            training_batch=128, sequence_steps=24, readout_steps=8, updates_per_trajectory=200,
            training_seed_blocks=5, test_episodes_per_block=512,
            unique_training_histories=640, unique_test_histories=2560,
            model_test_episodes=15360, trained_models=20, derived_inference_cut_models=10,
            forward_dense_MAC_terms_per_episode_step=156,
            backward_dense_MAC_terms_per_episode_step=240,
            checkpoint_updates=[0, 1, 5, 10, 20, 50, 100, 150, 200]),
        intervention=dict(cross_cell_slots_fixed_zero=60, within_cell_recurrent_slots_retained=12,
            intact_effective_trainable_parameters=176, retrained_effective_trainable_parameters=116,
            masked_regularizer_denominator=176,
            no_sparse_compute_optimization=True),
        training_phase_wall_seconds=train['seconds'],
        test_artifact_phase_wall_seconds=test['seconds'],
        test_inference_seconds=sum(row['inference_seconds'] for row in test['rows']),
        audits=audit_cost, versions=train['versions'],
        limitations=[
            'Dense matrix MAC terms exclude scalar nonlinearities, reset, losses, optimizer, generation, I/O and numerical verification.',
            'Intact and retrained have unequal effective trainable parameter counts (176 versus 116), despite equal storage and dense matrix work.',
            'DEND and POINT have unequal scalar nonlinearities, reset derivatives, branch-event semantics and realized activation/gradient patterns.',
            'Recorded wall times depend on machine contention and artifact compression; audits may overlap and must not be added as total elapsed workflow time.',
            'Spike totals and state L1 variation are computational activity proxies, not physical energy, ATP, biological metabolism or hardware efficiency.',
            'The inference-cut condition has no additional fit; its training origin is the corresponding intact model.'])
    dump(R / 'RESOURCE_COST.json', result)
    totals = result['totals']
    lines = ['# V9 resource and computation accounting', '',
        'The fixed experiment has 20 trained models and 10 inference-only cuts. Five seed/data blocks are the statistical units. Matrix multiply-accumulate (MAC) terms describe dense arithmetic, not measured hardware operations or energy.', '',
        '| Resource | Intact DEND / POINT | Retrained no-cross DEND / POINT | Inference cut |',
        '|---|---:|---:|---:|',
        '| Stored float64 parameters | 176 | 176 | 176 |',
        '| Effective trainable parameters during fitting | 176 | 116 | No additional fitting |',
        '| Cross-cell / self recurrent slots active | 60 / 12 | 0 / 12 | 0 / 12 |',
        '| Continuous states / previous hidden spikes / output counters | 20 / 6 / 2 | 20 / 6 / 2 | 20 / 6 / 2 |',
        '| Branch units / hidden somata / outputs | 12 / 6 / 2 | 12 / 6 / 2 | 12 / 6 / 2 |',
        '| Training batch / sequence steps / updates | 128 / 24 / 200 | 128 / 24 / 200 | No additional fitting |',
        '| Forward / backward dense MAC terms per episode-step | 156 / 240 | 156 / 240 | 156 / no backward |', '',
        'DEND branches retain graded states and have no branch spike events; POINT branch cells spike and reset. Scalar nonlinearities, reset derivatives, active states and gradient patterns differ. The 60 removed weights still occupy dense matrix positions. The original L2 regularizer denominator remains 176.', '',
        '| Arm | Condition | Fits | Updates | Training forward MAC | Training backward MAC | Diagnostic forward MAC | Test forward MAC | Training seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        lines.append(f"| {c['arm']} | {c['condition']} | {c['trajectories']} | {c['parameter_updates']:,} | {c['forward_dense_MAC_terms']:,} | {c['backward_dense_MAC_terms']:,} | {c['checkpoint_forward_dense_MAC_terms']:,} | {c['test_forward_dense_MAC_terms']:,} | {c['recorded_training_seconds']:.3f} |")
    lines += ['',
        f"Totals: {totals['parameter_updates']:,} parameter updates, {totals['forward_dense_MAC_terms']:,} training-forward MAC terms, {totals['backward_dense_MAC_terms']:,} training-backward MAC terms, {totals['checkpoint_forward_dense_MAC_terms']:,} diagnostic-forward terms and {totals['test_forward_dense_MAC_terms']:,} test-forward terms. Nine saved diagnostics per fit include the final training diagnostic, which is not counted again.", '',
        'There are 640 unique training histories and 2,560 unique held-out histories. Inputs are shared across architectures/conditions within each block. The 15,360 test model-episodes are not independent training replicates.', '',
        f"Recorded training and artifact phase: {train['seconds']:.3f} s. Held-out evaluation and artifact phase: {test['seconds']:.3f} s. Native test calls alone: {result['test_inference_seconds']:.3f} s. CPU-only NumPy, one BLAS/OpenMP thread; 300-second scientific stage caps.", '',
        '| Additional verification | Recorded wall seconds | Status |', '|---|---:|---|']
    for name, obj in audit_cost.items():
        lines.append(f"| {name} | {obj['recorded_wall_seconds']:.3f} | {obj['status']} |")
    lines += ['',
        'Audit work is outside the training/test matrix counts: full Adam reconstruction, scalar-equation native replay, 60 sampled full-batch gradient replays, and 2,920 final-checkpoint local surrogate finite differences. Pre-data fixtures include 704 intact and 232 masked local derivative checks. Audit files retain exact counts, timing and residuals.', '',
        *['- ' + item for item in result['limitations']]]
    write(R / 'RESOURCE_COST.md', '\n'.join(lines))
    return result


def tables(train, test):
    stats = test['statistics']
    fields = ['split', 'id', 'seed', 'arm', 'condition', 'accuracy', 'balanced_accuracy',
        'coverage', 'abstentions', 'abstention_rate', 'silent_fraction',
        'nonsilent_tie_fraction', 'both_saturated_fraction', 'failed', 'fully_silent',
        'fully_abstaining', 'class_recall', 'confusion', 'mean_signed_count_margin',
        'median_signed_count_margin', 'signed_count_margin_histogram', 'spike_events_mean',
        'total_spike_events_mean', 'state_l1_change_mean', 'correct_settled_fraction_by_step',
        'unique_settled_fraction_by_step', 'branch_event_rate', 'hidden_event_rate',
        'output_event_rate', 'readout_hidden_event_rate', 'readout_output_event_rate',
        'output_candidate_margin_quantiles']
    rows = [dict(split=split, **row) for split, items in
            [('train', train['rows']), ('test', test['rows'])] for row in items]
    assert len(rows) == 50
    write_csv('per_seed_metrics.csv', fields, rows)
    write_csv('cell_metrics.csv', ['arm', 'condition', 'n', 'mean_accuracy',
        'mean_balanced_accuracy', 'mean_coverage', 'failures', 'failure_wilson95',
        'fully_silent', 'fully_abstaining', 'mean_total_spikes', 'mean_state_l1_change',
        'correct_settled_fraction_by_step'], stats['cells'])
    contrasts = [dict(contrast=name, primary_descriptive=name in stats['primary_family'], **obj)
                 for name, obj in stats['intervals'].items()]
    write_csv('paired_contrasts.csv', ['contrast', 'primary_descriptive', 'mean', 'percentile95',
        'n_pairs', 'degenerate', 'unit', 'paired_values'], contrasts)
    lines = ['# V9 complete result tables', '',
        'Abstentions are counts; accuracy, balanced accuracy, coverage, silence, nonzero ties and class recalls are percentages. Every failed fit is retained. A pass requires both accuracy and coverage at least 90%. Class recalls include abstentions as errors. Confusion rows are true class 0/1 and columns predict 0, predict 1, abstain. All datasets have balanced class counts.', '']
    for split, items in [('Training at update 200 (20 fits)', train['rows']),
                         ('Fresh held-out evaluation (30 models)', test['rows'])]:
        lines += ['## ' + split, '',
            '| ID | Accuracy | Balanced acc. | Coverage | Abstentions | Silence | Nonzero ties | Recall 0 / 1 | Pass |',
            '|---|---:|---:|---:|---:|---:|---:|---|---|']
        for r in sorted(items, key=lambda r: (r['arm'], CONDITIONS.index(r['condition']), r['seed'])):
            lines.append(f"| {r['id']} | {pct(r['accuracy'])} | {pct(r['balanced_accuracy'])} | {pct(r['coverage'])} | {r['abstentions']} | {pct(r['silent_fraction'])} | {pct(r['nonsilent_tie_fraction'])} | {pct(r['class_recall'][0])} / {pct(r['class_recall'][1])} | {'yes' if not r['failed'] else 'no'} |")
        lines.append('')
    lines += ['## Held-out confusion and signed count margin', '',
        'Margin = true-class count minus other-class count, over all episodes. The CSV retains all 17 histogram bins from −8 through 8.', '',
        '| ID | Confusion | Mean margin | Median margin | Both saturated (%) |',
        '|---|---|---:|---:|---:|']
    for r in test['rows']:
        lines.append(f"| {r['id']} | `{json.dumps(r['confusion'])}` | {r['mean_signed_count_margin']:.4f} | {r['median_signed_count_margin']:.4f} | {pct(r['both_saturated_fraction'])} |")
    lines += ['', '## Held-out activity and retrospective settling', '',
        'Events are means per episode over all 24 steps, including failed predictions. State variation includes the transition from zero initial state. Curves list correct-and-settled percentages at readout steps 1–8; they use future prefixes and are not online stopping times or energy measurements.', '',
        '| ID | Branch events | Soma events | Output events | Total events | State L1 change | Correct-settled curve (%) |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in test['rows']:
        e = r['spike_events_mean']
        curve = ', '.join(pct(v) for v in r['correct_settled_fraction_by_step'])
        lines.append(f"| {r['id']} | {e['branch']:.3f} | {e['hidden']:.3f} | {e['output']:.3f} | {r['total_spike_events_mean']:.3f} | {r['state_l1_change_mean']:.3f} | {curve} |")
    lines += ['', '## Paired seed-block contrasts', '',
        'All intervals are 95% percentile bootstrap intervals from 20,000 resamples of five complete paired blocks. Values below are percentage points. There is no family-wide multiplicity correction; all eight are exploratory. Negative failure differences mean fewer intact failures. A degenerate or zero-crossing interval does not establish equivalence.', '',
        '| Contrast | Primary descriptive | Mean | Pointwise 95% interval | Five paired values |',
        '|---|---|---:|---|---|']
    for r in contrasts:
        lines.append(f"| {r['contrast']} | {'yes' if r['primary_descriptive'] else 'no'} | {pct(r['mean'])} | {interval(r['percentile95'])} | {', '.join(pct(v) for v in r['paired_values'])} |")
    lines += ['', '## Frozen seed registry', '',
        '| Block | Training data | Test data | Initialization |', '|---|---:|---:|---:|']
    for row in read(R / 'SEEDS.json')['pairs']:
        lines.append(f"| {row['seed']} | {row['train_seed']} | {row['test_seed']} | {row['initial_seed']} |")
    lines += ['', 'The analysis RNG seed is 2090030001. Initialization is shared on retained coordinates. The ten cut models have no separate training fit; their provenance is the corresponding intact vector multiplied by the fixed mask. Full train activity and all native traces remain in JSON/NPZ and the CSV.']
    write(R / 'TABLES.md', '\n'.join(lines))


def figures(test):
    F.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'TAN-V9-fixed'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for ax, arm in zip(axes, ARMS):
        values = np.array([[next(r['accuracy'] for r in test['rows']
                               if (r['arm'], r['condition'], r['seed']) == (arm, c, s))
                            for c in CONDITIONS] for s in range(5)])*100
        for s, value in enumerate(values):
            ax.plot(range(3), value, marker='o', alpha=.65, lw=1, label=f'Block {s}')
        ax.plot(range(3), values.mean(0), color='black', marker='D', lw=2.3, label='Mean')
        ax.axhline(90, ls=':', color='#555555', lw=.8)
        ax.set_xticks(range(3), ['Intact', 'Retrained\nno-cross', 'Inference\ncut'])
        ax.set_ylim(-4, 104)
        ax.set_title(arm + ': five paired seed/data blocks')
    axes[0].set_ylabel('Unconditional held-out accuracy (%)')
    axes[1].legend(fontsize=8, ncol=2, loc='lower left')
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.text(.5, .02, 'All fits retained; 200 RATE/.01 updates, 512 test episodes per block. Dotted line: accuracy component of 90% gate.', ha='center', fontsize=8)
    save_figure(fig, 'paired_accuracy')

    fig, ax = plt.subplots(figsize=(10, 4.7))
    labels = []
    for y, (arm, condition) in enumerate((a, c) for a in ARMS for c in ('retrained', 'cut')):
        obj = test['statistics']['intervals'][f'{arm}_intact_minus_{condition}_accuracy']
        mean = obj['mean']*100
        lo, hi = np.array(obj['percentile95'])*100
        ax.plot([lo, hi], [y, y], color=COLORS[condition], lw=2)
        ax.plot(mean, y, 'o', color=COLORS[condition], ms=7)
        ax.text(hi+1, y+.08, f'{mean:.2f} [{lo:.2f}, {hi:.2f}]', fontsize=8)
        labels.append(arm + ': intact − ' + LABELS[condition].lower())
    ax.axvline(0, color='#555555', ls=':', lw=1)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Paired accuracy difference (percentage points)')
    ax.set_title('V9 pointwise exploratory intervals: five paired training/data blocks')
    ax.margins(x=.35, y=.2)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.text(.5, .02, '20,000 paired bootstrap resamples; no family-wide correction. Degenerate intervals do not establish equivalence.', ha='center', fontsize=8)
    save_figure(fig, 'paired_contrasts')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, arm in zip(axes, ARMS):
        for condition in CONDITIONS:
            cell = next(c for c in test['statistics']['cells'] if (c['arm'], c['condition']) == (arm, condition))
            ax.plot(range(1, 9), np.array(cell['correct_settled_fraction_by_step'])*100,
                    marker='o', label=LABELS[condition], color=COLORS[condition])
        ax.set_title(arm)
        ax.set_xticks(range(1, 9))
        ax.set_xlabel('Readout step (1–8)')
        ax.set_ylim(-4, 104)
    axes[0].set_ylabel('Correct and retrospectively settled (%)')
    axes[1].legend(fontsize=9, loc='lower right')
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.text(.5, .02, 'All episodes remain in denominator. Settling uses future prefixes; curves do not define an online stopping rule.', ha='center', fontsize=8)
    save_figure(fig, 'retrospective_settling')


def report(train, test, audits, cost, args):
    stats = test['statistics']
    cells = {(c['arm'], c['condition']): c for c in stats['cells']}
    initial_checks = read(R / 'GRADIENT_CHECK.json')
    masked_checks = read(R / 'MASKED_CHECK.json')
    max_initial_error = max(r['max_surrogate_chain_rule_error'] for r in initial_checks['rows'])
    max_masked_error = max(r['max_proxy_derivative_error'] for r in masked_checks['rows'])
    native_error = max(audits['verification']['training']['max_training_state_candidate_residual'],
                       audits['verification']['tests']['max_test_state_candidate_residual'])
    lines = ['# V9: Cross-cell recurrence controls on native-spike delayed XOR', '',
        'Date: 2026-11-04. Status: completed bounded experiment; internal checks passed. This report resumes the research under the new blueprint while preserving all V1–V8 artifacts and the historical V8 pause.', '',
        '## Finding', '',
        'The experiment tests learned solutions under one fixed RATE/.01 recipe, 200 updates, five paired seed/data blocks, 128 training and 512 fresh held-out episodes per block. It compares intact training, training with cross-cell recurrence fixed to zero, and cutting an intact model only at inference.']
    for arm in ARMS:
        intact, retrained, cut = (cells[(arm, c)] for c in CONDITIONS)
        delta = stats['intervals'][f'{arm}_intact_minus_retrained_accuracy']
        lines += ['',
            f"**{arm}:** intact passes {5-intact['failures']}/5; retrained no-cross passes {5-retrained['failures']}/5; inference cut passes {5-cut['failures']}/5. Mean held-out accuracies are {pct(intact['mean_accuracy'])}%, {pct(retrained['mean_accuracy'])}%, and {pct(cut['mean_accuracy'])}%, respectively. The intact-minus-retrained difference is {pct(delta['mean'])} percentage points, pointwise exploratory 95% paired-bootstrap interval {interval(delta['percentile95'])}, in these five blocks and this fixed budget."]
        if retrained['failures'] < 5:
            lines.append(f"The {5-retrained['failures']} passing no-cross {arm} fits demonstrate observed solutions that meet this finite task's accuracy/coverage gate without cross-cell recurrence. This result concerns a population with retained self-feedback and state; it does not show that one cell solves every possible delayed-XOR input.")
        else:
            lines.append('No retrained no-cross fit passed this gate. Five failures under this recipe and budget do not prove physical necessity, representational impossibility, or failure under other learning rules.')
    lines += ['',
        'Inference cutting tests dependence of a particular learned solution; it does not test whether alternative no-cross solutions can be trained. These two controls must be interpreted separately. No claim of a general DEND-versus-POINT ordering is made.', '',
        '## Frozen design and provenance', '',
        f"- Preserved V8 starting commit: `9cddbbad7f1722d8a116c0a8245fc40d3a5e97ca`.",
        f"- Protocol/source/seed commit, before scientific data: `{args.protocol_commit}`.",
        f"- Training data, all 20 fits and 10 derived cuts committed before test generation: `{args.checkpoint_commit}`.",
        f"- Freeze SHA-256: `{sha(R / 'FREEZE.json')}`.",
        f"- Training-lock SHA-256: `{sha(O / 'TRAIN_LOCK.json')}`.",
        '- [Protocol](PROTOCOL.md), [seed registry](SEEDS.json), [all per-seed tables](TABLES.md), [resource accounting](RESOURCE_COST.md), and [audit outputs](results/verification.json) retain the details.', '',
        'The forward equations, initialization distribution and task generator are preserved from V8/V2. Both architectures use native hard spikes with strict candidate > 0.5 and reset at hidden/output layers; POINT branches also spike/reset, while DEND branches remain graded. There is no external pulse generator, label-conditioned forward signal, analog decoder or altered tie rule. Each episode resets its state independently. The labels are used only in the loss and evaluation.', '',
        'Each 24-step history presents two independently balanced bit cues; the informative second cue ends by step 15. The common readout occupies steps 16–23. The target is their XOR. Native output spike counts over those eight steps predict only when the winner is unique; ties and silence abstain and count as errors. Frozen RATE targets are .75/.125 of the eight-count scale; Adam uses learning rate .01, standard .9/.999 moments, global gradient-norm clip 1 and parameter clipping to [−2,2]. The learning slope 1/(1+5|candidate−.5|)^2 is a surrogate, not the derivative of the hard-spike function.', '',
        'The retrained intervention zeros the 60 recurrent weights from a soma to branches of other cells before training. Twelve within-cell feedback weights survive. Gradients are masked before global clipping; parameters and moments remain zero on the removed slots. Intact and retrained initialization agree on retained coordinates. Retrained capacity is 116 effective parameters versus 176 intact, although storage and the unchanged regularizer denominator remain 176. Derived cut vectors equal the intact final weights multiplied by this same mask, and receive no optimization.', '',
        'No validation selection, checkpoint selection, hyperparameter search, seed replacement or test-driven refinement was performed. All 20 fixed fits and 30 final model evaluations are retained. Pre-data deterministic fixtures check state reset and batch independence; no additional held-out no-history ablation was performed. The recurrence controls therefore do not identify the detailed mechanism that stores either bit.', '',
        '## Results and uncertainty', '',
        '| Arm | Condition | Mean accuracy (%) | Mean coverage (%) | Failures / 5 | Failure risk Wilson 95% (%) | Fully silent / 5 |',
        '|---|---|---:|---:|---:|---|---:|']
    for c in stats['cells']:
        lines.append(f"| {c['arm']} | {c['condition']} | {pct(c['mean_accuracy'])} | {pct(c['mean_coverage'])} | {c['failures']} | {interval(c['failure_wilson95'])} | {c['fully_silent']} |")
    lines += ['',
        'A failure is accuracy < 90% or coverage < 90%; means alone do not determine a pass. Class recalls, balanced accuracy, confusion matrices, silence and nonzero ties appear for every block in [TABLES.md](TABLES.md) and [the 50-row training/test CSV](results/per_seed_metrics.csv). All labels are balanced, so balanced and ordinary accuracy agree; both are recorded. Label-independent binary guessing with coverage c has expected unconditional accuracy c/2, not a universal 50% baseline when abstention occurs.', '',
        '![Five paired seed outcomes](figures/paired_accuracy.png)', '',
        '| Paired accuracy contrast | Mean (pp) | Pointwise exploratory 95% interval (pp) |',
        '|---|---:|---|']
    for arm in ARMS:
        for condition in ('retrained', 'cut'):
            d = stats['intervals'][f'{arm}_intact_minus_{condition}_accuracy']
            lines.append(f"| {arm}: intact − {condition} | {pct(d['mean'])} | {interval(d['percentile95'])} |")
    lines += ['',
        'The two intact-minus-retrained accuracy contrasts were prespecified as primary descriptive outcomes. Four accuracy and four failure-risk contrasts are reported with the same five-block resampling and no multiplicity protection. They are exploratory, not confirmatory evidence from a protected hypothesis family. The independently trained seed/data block is the statistical unit: neither 512 episodes per model nor 20,000 bootstrap draws increases n beyond five.', '',
        'Intervals are percentile paired bootstrap intervals. Degenerate intervals such as [0,0] reflect the observed five blocks; they do not establish population equivalence or a noninferiority bound. Even zero observed failures in five blocks has a marginal 95% Wilson upper bound of about 43.45%. Crossing zero likewise does not establish equivalence.', '',
        '![Pointwise paired intervals](figures/paired_contrasts.png)', '',
        '## Activity, settling and cost', '',
        '| Arm | Condition | Mean total native events / episode | Mean state L1 variation / episode | Correct-settled by first readout step (%) | By eighth step (%) |',
        '|---|---|---:|---:|---:|---:|']
    for c in stats['cells']:
        lines.append(f"| {c['arm']} | {c['condition']} | {c['mean_total_spikes']:.3f} | {c['mean_state_l1_change']:.3f} | {pct(c['correct_settled_fraction_by_step'][0])} | {pct(c['correct_settled_fraction_by_step'][-1])} |")
    lines += ['',
        'The activity table includes all episodes, including failures and abstentions. DEND branch states produce no discrete branch events but still require state updates. Raw event totals are not a matched-performance energy comparison. State L1 variation depends on the chosen state representation and scale; neither measure estimates joules, ATP or biological metabolism.', '',
        'A prediction is retrospectively settled at a readout prefix only when its unique winner matches the final winner at all later prefixes. Correct-and-settled curves retain wrong/abstaining examples in their denominator, so their final point equals accuracy. This metric uses future observations and does not replace the fixed eight-step decoder with an online early stopping policy. Steps are not interpreted as biological milliseconds.', '',
        '![Retrospective settling](figures/retrospective_settling.png)', '',
        f"The 4,000 Adam updates use {cost['totals']['forward_dense_MAC_terms']:,} training-forward and {cost['totals']['backward_dense_MAC_terms']:,} backward dense MAC terms. Nine diagnostic forwards per fit add {cost['totals']['checkpoint_forward_dense_MAC_terms']:,}; held-out inference adds {cost['totals']['test_forward_dense_MAC_terms']:,}. Dense zero-weight multiplication is still counted. Recorded training/artifact time is {train['seconds']:.3f} s and held-out/artifact time {test['seconds']:.3f} s on this execution. These times and arithmetic counts are not energy or hardware efficiency measurements. The [complete resource table](RESOURCE_COST.md) identifies unequal effective parameter counts, branch nonlinearities, reset derivatives and activity patterns.", '',
        '## Numerical verification', '',
        f"Pre-data deterministic fixtures passed native forward identity with the preserved V2 implementation, hard thresholds/reset, binary spikes, batch independence and constrained masks. The 704 intact local surrogate derivative checks had maximum error {max_initial_error:.6g}; the 232 active-coordinate masked checks had maximum error {max_masked_error:.6g}. These fixtures generated no scientific training or test data.", '',
        f"Both saved audits report PASS: [native/data/Adam/statistics replay](results/verification.json) and [gradient audit](results/gradient_audit.json). The gradient audit replays {audits['gradient_audit']['sampled_updates']} full-batch gradients at updates 0/99/199; maximum residual is {audits['gradient_audit']['max_sampled_gradient_residual']:.6g}. Its {audits['gradient_audit']['parameter_derivative_checks']:,} final-checkpoint local surrogate finite differences have maximum absolute residual {audits['gradient_audit']['max_chain_rule_residual']:.6g}.", '',
        f"The scalar native/optimizer/data/statistics checker passed {audits['verification']['checks']:,} assertions. Maximum final-training/test state or candidate residual was {native_error:.6g}; spikes, counts and predictions matched exactly. It verified 3,200 distinct train/test histories without exact duplicates.", '',
        'The separately written scalar native verifier regenerates data and initialization, reconstructs all 4,000 Adam updates from saved clipped gradients, checks all 180 checkpoints, verifies exact masked slots/moments and cut provenance, and replays every final-training and held-out model episode. Spikes, counts and predictions must match exactly; states/candidates use the frozen 1e-10 tolerance. It independently recomputes metrics, activity, settling and paired statistics. Sampling the frozen analytic backward is not a fresh derivation of every gradient. Finite differences apply to a locally anchored surrogate graph, not to the discontinuous native-spike map.', '',
        'All checks belong to this same AI-assisted workflow. They are not external independent replication and not peer review. Source and artifact hashes allow others to inspect and rerun the study.', '',
        '## What this does not establish', '',
        '- Finite-budget retraining failure cannot prove that cross-cell recurrence is physically necessary or that a representation is impossible.',
        '- A passing no-cross population is not a one-neuron solution and does not prove exact correctness on all possible task inputs.',
        '- Cutting an intact solution does not establish the absence of alternative trainable solutions without that pathway.',
        '- These five blocks and one task do not rank architectures generally, establish equivalence, or justify a breakthrough claim.',
        '- Event counts and state changes do not establish energy, hardware, biological or metabolic benefits.',
        '- Retrospective settling does not establish an online stopping rule or a physical decision-time benefit.', '',
        '## Limitations', '',
        'The study has five independent paired training/data blocks, one delayed-XOR distribution and one fixed training recipe/budget. There is no independent replication or broad task transfer. Both initial weights and datasets vary together across blocks; their variance contributions cannot be separated. Holding the optimizer recipe fixed isolates the preregistered intervention within that recipe, not optimization quality across all recipes.', '',
        'Masking reduces the number of effective parameters and changes the gradient-normalization geometry. The no-cross model retains graded or spiking branch states, self recurrence and a multi-cell output network. This study does not distinguish those remaining memory/computation pathways. Pointwise bootstrap intervals at n=5 are limited and do not protect the full eight-contrast family. Test episodes provide held-out observations within the balanced block design, not independent fitted models.', '',
        'Dense MAC terms omit scalar operations, optimizer work and audit costs. Native branch-event semantics differ between DEND and POINT. Activity and retrospective settling are descriptive recorded outcomes, not new losses or selection criteria. Earlier studies remain frozen; V10 has a separate protocol and cannot retrospectively change these V9 endpoints.', '',
        '## Attribution and reproducibility', '',
        'Research direction and native-firing requirements were proposed by Li Zexu. Protocol refinement, implementation, local execution, numerical checking and this report draft were AI-assisted. No journal submission or peer-review status is implied.', '',
        'Read [README.md](README.md) for artifact locations and replay commands. `summarize.py` reads only saved audited outputs and never trains or evaluates a model. The final file manifest, maintained separately, covers scientific sources, datasets, weights, traces, audits, tables and figures. All V1–V8 files and main-branch history are preserved.']
    write(R / 'REPORT.md', '\n'.join(lines))
    write(R / 'README.md', '\n'.join(['# V9 native-spike cross-cell recurrence controls', '',
        'This independent research directory preserves the earlier studies and tests intact, retrained no-cross and inference-cut conditions on delayed XOR. It contains 20 training fits and 30 held-out evaluations across five paired blocks, with all failed fits retained.', '',
        '- [Report](REPORT.md) and [all per-seed tables](TABLES.md).',
        '- [Frozen protocol](PROTOCOL.md), [seeds](SEEDS.json), and [source freeze](FREEZE.json).',
        '- [Resource/accounting details](RESOURCE_COST.md) and [machine-readable costs](RESOURCE_COST.json).',
        '- `results/TRAIN_LOCK.json`: all 20 fitted models, 10 derived cuts and pre-test artifact hashes.',
        '- `results/*_trace.npz`: every clipped gradient, pre-update count and training diagnostic.',
        '- `results/*_checkpoints.npz`: nine parameter checkpoints per fit; matching JSON retains diagnostics.',
        '- `results/train_s*.npz`, `test_s*.npz`: complete fixed datasets and generation seeds.',
        '- `results/*_train_native.npz`, `*_test_native.npz`: native states, candidates, hard spikes, counts and predictions.',
        '- [Held-out summary](results/TEST_SUMMARY.json), [native/optimizer verification](results/verification.json), [gradient audit](results/gradient_audit.json).',
        '- [50 train/test metric rows](results/per_seed_metrics.csv), [six cell summaries](results/cell_metrics.csv), [eight paired intervals](results/paired_contrasts.csv).',
        '- `figures/`: PNG and SVG paired-outcome, interval and retrospective-settling figures.', '',
        'In a separate copy of this directory, verify the existing scientific artifacts (NumPy required). These scripts rewrite their own audit JSON; leave the archived copy unchanged:', '',
        '```bash', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_gradients.py', '```', '',
        'Reporting only (NumPy and Matplotlib; no training/model calls):', '',
        '```bash', f'python summarize.py --protocol-commit {args.protocol_commit} --checkpoint-commit {args.checkpoint_commit}', '```', '',
        'The scientific runner refuses an existing result directory and the evaluator refuses an existing test marker. Do not delete these guards or rerun the same test as a new observation. Reproduction that generates new artifacts belongs in a separate copy/new output location, preserving this frozen evidence.', '',
        'No-cross still contains multiple cells, internal state, self-feedback and output integration. Passing this finite gate does not prove exact correctness over all inputs; failure does not prove physical necessity. All eight intervals are exploratory pointwise estimates from five training/data blocks. Activity is not energy, and retrospective settling is not online stopping. Verification is internal, not outside replication or peer review.']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-commit', required=True)
    parser.add_argument('--checkpoint-commit', required=True)
    args = parser.parse_args()
    for commit in (args.protocol_commit, args.checkpoint_commit):
        assert len(commit) == 40 and all(ch in '0123456789abcdef' for ch in commit)
    audits = {name: read(O / (name + '.json')) for name in ('verification', 'gradient_audit')}
    assert all(obj['status'] == 'PASS' for obj in audits.values()), 'Require both audits before reporting'
    train, test = read(O / 'TRAIN_LOCK.json'), read(O / 'TEST_SUMMARY.json')
    assert len(train['rows']) == 20 and len(train['cut_rows']) == 10 and len(test['rows']) == 30
    assert all(r['steps'] == 200 for r in train['rows'])
    assert len(test['statistics']['intervals']) == 8
    tables(train, test)
    cost = resources(train, test, audits)
    figures(test)
    report(train, test, audits, cost, args)
    print(json.dumps(dict(status='REPORTS_WRITTEN', train_rows=20, test_rows=30,
                          statistical_units=5, paired_contrasts=8), indent=2))


if __name__ == '__main__':
    main()
