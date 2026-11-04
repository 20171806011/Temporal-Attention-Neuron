"""V8 reporting only: transform saved data; never train or evaluate a model."""
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tan-v8-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
O = R / 'results'
F = R / 'figures'
V7 = R.parent / 'dendritic-population-v7/results/TEST_SUMMARY.json'
RECIPES = ('RATE_0.01', 'RATE_0.03', 'CE_0.01', 'CE_0.03')
PRIMARY = 'accuracy_POINT_loss_lr_interaction'
N = 104
RUNS = 416
COLOR = '#b7542c'


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


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
    fig.savefig(path)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def resources(train, test):
    cells = []
    for recipe in RECIPES:
        rows = [row for row in train['rows'] if row['recipe'] == recipe]
        updates = sum(row['steps'] for row in rows)
        cells.append(dict(arm='POINT', recipe=recipe, trajectories=len(rows), parameter_updates=updates,
            forward_dense_MAC_terms=updates * 128 * 24 * 156,
            backward_dense_MAC_terms=updates * 128 * 24 * 240,
            checkpoint_training_forward_batches=len(rows) * 9,
            checkpoint_forward_dense_MAC_terms=len(rows) * 9 * 128 * 24 * 156,
            test_forward_dense_MAC_terms=len(rows) * 512 * 24 * 156,
            recorded_training_seconds=sum(row['seconds'] for row in rows)))
    additive = ['trajectories', 'parameter_updates', 'forward_dense_MAC_terms',
                'backward_dense_MAC_terms', 'checkpoint_training_forward_batches',
                'checkpoint_forward_dense_MAC_terms', 'test_forward_dense_MAC_terms']
    audits = {}
    for name in ('verification', 'gradient_audit'):
        obj = read(O / (name + '.json'))
        assert obj['status'] == 'PASS', name
        audits[name] = dict(status=obj['status'], recorded_wall_seconds=obj['seconds'])
        if name == 'verification':
            audits[name]['training_counts'] = obj['training']
            audits[name]['test_counts'] = obj['tests']
        else:
            audits[name].update({key: obj[key] for key in
                ('sampled_updates', 'final_checkpoints', 'parameter_derivative_checks')})
    result = dict(cells=cells, totals={key: sum(cell[key] for cell in cells) for key in additive},
        shared_resources=dict(arm='POINT', trainable_float64_parameters=176, continuous_states=20,
            previous_hidden_spike_states=6, output_count_accumulators=2,
            native_spiking_branch_cells=12, hidden_somata=6, output_cells=2,
            training_batch=128, sequence_steps=24, updates_per_trajectory=200,
            test_episodes_per_block=512, training_seed_blocks=N, cells_per_block=4,
            unique_training_histories=N*128, unique_test_histories=N*512,
            model_test_episodes=RUNS*512, forward_dense_MAC_terms_per_episode_step=156,
            backward_dense_MAC_terms_per_episode_step=240,
            checkpoint_updates=[0, 1, 5, 10, 20, 50, 100, 150, 200]),
        training_phase_wall_seconds=train['seconds'], test_artifact_phase_wall_seconds=test['seconds'],
        test_inference_seconds=sum(row['inference_seconds'] for row in test['rows']), audits=audits,
        caps_seconds=dict(training_with_diagnostics=900, test_with_artifacts=600, each_audit=900),
        visual_reference=dict(v7_summary_sha256=hashlib.sha256(V7.read_bytes()).hexdigest(),
            purpose='Historical pilot interval shown for visual context only; no pooling or cross-phase hypothesis test.'),
        limitations=[
            'Dense matrix MAC counts exclude scalar nonlinearities, resets, loss computation, Adam, data generation, I/O and numerical audits.',
            'Recorded wall times depend on CPU contention and file compression; audit times may overlap and must not be summed as elapsed workflow time.',
            'All four cells are POINT; this experiment cannot compare architecture resources or structural superiority.',
            'Matching matrix terms and nominal learning rates does not establish equal effective updates, energy or hardware cost.',
            'Checkpoint forward counts include the final training diagnostic; no validation or hyperparameter selection runs were performed.'])
    dump(R / 'RESOURCE_COST.json', result)
    totals = result['totals']
    lines = ['# V8 resource and compute accounting', '',
        'Counts cover the fixed 416-run POINT factorial replication. Dense matrix multiply-accumulate (MAC) terms are an algorithmic count, not measured FLOPs, energy, or hardware efficiency.', '',
        '| Quantity | Each POINT configuration |', '|---|---:|',
        '| Trainable float64 parameters | 176 |', '| Continuous state values | 20 |',
        '| Previous hidden spikes / output counters | 6 / 2 |',
        '| Native branch cells / hidden somata / output cells | 12 / 6 / 2 |',
        '| Batch / sequence steps / updates | 128 / 24 / 200 |',
        '| Paired seed/data blocks | 104 |',
        '| Forward / backward dense MAC terms per episode-step | 156 / 240 |', '',
        '| Recipe | Runs | Updates | Forward MAC | Backward MAC | Diagnostic MAC | Test MAC | Recorded train seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for cell in cells:
        lines.append(f"| {cell['recipe']} | {cell['trajectories']} | {cell['parameter_updates']:,} | {cell['forward_dense_MAC_terms']:,} | {cell['backward_dense_MAC_terms']:,} | {cell['checkpoint_forward_dense_MAC_terms']:,} | {cell['test_forward_dense_MAC_terms']:,} | {cell['recorded_training_seconds']:.3f} |")
    lines += ['',
        f"Total: {totals['parameter_updates']:,} parameter updates; {totals['forward_dense_MAC_terms']:,} training-forward MAC terms and {totals['backward_dense_MAC_terms']:,} training-backward MAC terms. The {totals['checkpoint_training_forward_batches']:,} additional training-diagnostic forward batches contribute {totals['checkpoint_forward_dense_MAC_terms']:,} terms; held-out inference contributes {totals['test_forward_dense_MAC_terms']:,}. Nine diagnostics include update 200, so final training metrics are not counted twice.", '',
        'There are 13,312 unique training histories and 53,248 unique held-out histories, reused across four models within each paired block. The 212,992 test model-episodes are not 212,992 independent training units; the inferential unit is 104 seed/data blocks.', '',
        f"Training and artifact phase: {train['seconds']:.3f} s. Held-out evaluation and artifact phase: {test['seconds']:.3f} s. Native test model calls total {result['test_inference_seconds']:.3f} s; the remaining time includes metrics, compression and I/O.", '',
        '| Separate audit | Recorded wall seconds | Status |', '|---|---:|---|']
    for name, audit in audits.items():
        lines.append(f"| {name} | {audit['recorded_wall_seconds']:.3f} | {audit['status']} |")
    lines += ['', *result['limitations'], '',
        'Numerical checking is additional work outside the training/test matrix budget: 1,248 sampled backward replays, 73,216 final-checkpoint local proxy-derivative checks, complete Adam reconstruction, native trajectory replay, and statistical regeneration. Audit JSON files retain complete counts and residuals.', '',
        'CPU-only NumPy execution uses one BLAS/OpenMP thread. Fixed caps are 900 seconds for training plus diagnostics, 600 seconds for held-out evaluation plus artifacts, and 900 seconds per audit. No paid compute, architecture search, external output pulse generator, or analog decoder was introduced.', '',
        'V7 appears in one interval figure solely as historical pilot context. Its 20 blocks are not pooled with the 104 new V8 blocks, and that plot is not a formal between-phase test.', '']
    (R / 'RESOURCE_COST.md').write_text('\n'.join(lines))


def tables(train, test):
    stats = test['statistics']
    fields = ['split', 'id', 'seed', 'arm', 'recipe', 'accuracy', 'coverage', 'abstentions',
        'abstention_rate', 'silent_fraction', 'nonsilent_tie_fraction', 'both_saturated_fraction',
        'failed', 'fully_silent', 'fully_abstaining', 'class_recall', 'confusion',
        'mean_signed_count_margin', 'median_signed_count_margin', 'signed_count_margin_histogram',
        'branch_event_rate', 'hidden_event_rate', 'output_event_rate', 'readout_hidden_event_rate',
        'readout_output_event_rate', 'output_candidate_margin_quantiles']
    rows = [dict(split=split, **row) for split, data in
            [('train', train['rows']), ('test', test['rows'])] for row in data]
    write_csv('per_seed_metrics.csv', fields, rows)
    fields = ['arm', 'recipe', 'n', 'mean_accuracy', 'mean_coverage', 'mean_silent_fraction',
        'mean_abstention_rate', 'mean_nonsilent_tie_fraction', 'mean_train_accuracy', 'run_events', 'pass_transitions']
    write_csv('cell_metrics.csv', fields, stats['cells'])
    rows = [dict(contrast=name, **value) for name, value in stats['intervals'].items()]
    write_csv('paired_contrasts.csv', ['contrast', 'mean', 'student_t95', 'percentile95',
        'sample_sd', 'standard_error', 'degrees_of_freedom', 't_critical', 'primary_accuracy',
        'n_pairs', 'unit', 'degenerate', 'paired_values'], rows)


def figures(train, test):
    F.mkdir(exist_ok=True)
    stats = test['statistics']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    values = np.array([[next(row['accuracy'] for row in test['rows']
                       if (row['recipe'], row['seed']) == (recipe, seed))
                       for recipe in RECIPES] for seed in range(N)]) * 100
    for value in values:
        ax.plot(range(4), value, color=COLOR, alpha=.11, lw=.65, marker='o', ms=2)
    ax.plot(range(4), values.mean(0), color='#111111', lw=2.5, marker='D', ms=6, label='Mean of 104 blocks')
    ax.axhline(90, ls=':', color='#777777', lw=.8, label='90% accuracy gate')
    ax.set_xticks(range(4), ['RATE / .01', 'RATE / .03', 'CE / .01', 'CE / .03'])
    ax.set_ylabel('Held-out unconditional accuracy (%)')
    ax.set_ylim(-4, 104)
    ax.set_title('V8: 104 paired POINT seed/data blocks across four fixed recipes')
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.text(.5, .02, 'Each thin line retains one block across four cells. All failed, silent and tied runs are included.', ha='center', fontsize=9)
    save_figure(fig, 'paired_cells')

    current = stats['intervals'][PRIMARY]
    pilot = read(V7)['statistics']['intervals'][PRIMARY]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 1.25]})
    delta = np.array(current['paired_values']) * 100
    edges = np.arange(-200, 201, 10)
    axes[0].hist(delta, bins=edges, color=COLOR, alpha=.65, edgecolor='white')
    axes[0].axvline(0, color='#555555', lw=.9)
    axes[0].axvline(current['mean']*100, color='#111111', lw=1.6, label='V8 mean')
    axes[0].set_xlim(max(-200, 10*np.floor(delta.min()/10)-10), min(200, 10*np.ceil(delta.max()/10)+10))
    axes[0].set_xlabel('Within-block interaction (percentage points)')
    axes[0].set_ylabel('Training/data blocks')
    axes[0].set_title('V8 interaction distribution: n=104')
    axes[0].legend(fontsize=8)
    specs = [(current, 'student_t95', 'V8 primary: paired t 95%', COLOR, 'o'),
             (current, 'percentile95', 'V8 sensitivity: bootstrap 95%', '#176b87', 's'),
             (pilot, 'percentile95', 'V7 pilot: bootstrap 95%', '#777777', 'D'),
             (pilot, 'bonferroni_family95', 'V7 pilot: adjusted 99.1667%', '#aaaaaa', 'D')]
    for i, (record, key, label, color, marker) in enumerate(specs):
        lo, hi = np.array(record[key])*100
        axes[1].plot([lo, hi], [i, i], color=color, lw=2.2)
        axes[1].plot(record['mean']*100, i, marker, color=color, ms=6)
    axes[1].axvline(0, color='#555555', lw=.9)
    axes[1].axvline(-5, color='#999999', ls=':', lw=1)
    axes[1].set_yticks(range(len(specs)), [spec[2] for spec in specs], fontsize=8.7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Mean interaction (percentage points)')
    axes[1].set_title('Prespecified V8 primary and sensitivity intervals')
    fig.tight_layout(rect=(0, .13, 1, .98))
    fig.text(.5, .067, 'Interaction = (CE/.03 − CE/.01) − (RATE/.03 − RATE/.01). Primary interpretation uses the V8 t interval only.', ha='center', fontsize=9)
    fig.text(.5, .022, 'V7 is historical context only: different cohort and interval methods; no pooling or between-phase hypothesis test. Dotted line: −5 pp.', ha='center', fontsize=8.5)
    save_figure(fig, 'primary_interaction')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, recipe in zip(axes.flat, RECIPES):
        for row in train['rows']:
            if row['recipe'] != recipe:
                continue
            with np.load(O / row['trace'], allow_pickle=False) as trace:
                accuracy = np.r_[trace['history'][:, 3], row['accuracy']] * 100
            ax.plot(np.arange(201), accuracy, color=COLOR, alpha=.16, lw=.65)
        ax.axhline(90, ls=':', lw=.7, color='#777777')
        ax.set_title('POINT ' + recipe.replace('_', ' / '))
        ax.set_xlabel('Completed updates')
        ax.set_ylabel('Training accuracy (%)')
        ax.set_ylim(-3, 103)
    fig.suptitle('V8: all 416 saved training trajectories; final checkpoint fixed before test')
    fig.tight_layout(rect=(0, 0, 1, .96))
    save_figure(fig, 'training_trajectories')


def main():
    train = read(O / 'TRAIN_LOCK.json')
    test = read(O / 'TEST_SUMMARY.json')
    assert len(train['rows']) == len(test['rows']) == RUNS
    assert all(row['arm'] == 'POINT' for row in train['rows'] + test['rows'])
    assert len(test['statistics']['cells']) == 4
    assert test['statistics']['primary_family'] == [PRIMARY]
    assert len(test['statistics']['intervals']) == 18
    resources(train, test)
    tables(train, test)
    figures(train, test)
    print('Saved V8 resources, three CSV tables and three PNG/SVG figures; no model calls.')


if __name__ == '__main__':
    main()
