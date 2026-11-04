"""V7 reporting only: read saved results; never train or evaluate models."""
from pathlib import Path
import csv
import json
import os
import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tan-v7-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
O = R / 'results'
F = R / 'figures'
ARMS = ('DEND', 'POINT')
RECIPES = ('RATE_0.01', 'RATE_0.03', 'CE_0.01', 'CE_0.03')
COLORS = {'DEND': '#176b87', 'POINT': '#b7542c'}


def read(path):
    return json.loads(path.read_text())


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def write_csv(name, fields, rows):
    with (O / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k]) if isinstance(row.get(k), (list, dict))
                             else row.get(k, '') for k in fields})


def save_figure(fig, stem):
    fig.savefig(F / (stem + '.png'), dpi=170)
    svg = F / (stem + '.svg')
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def resources(train, test):
    cells = []
    for arm in ARMS:
        for recipe in RECIPES:
            rows = [r for r in train['rows'] if (r['arm'], r['recipe']) == (arm, recipe)]
            updates = sum(r['steps'] for r in rows)
            cells.append(dict(arm=arm, recipe=recipe, trajectories=len(rows), parameter_updates=updates,
                forward_dense_MAC_terms=updates * 128 * 24 * 156,
                backward_dense_MAC_terms=updates * 128 * 24 * 240,
                checkpoint_training_forward_batches=len(rows) * 9,
                checkpoint_forward_dense_MAC_terms=len(rows) * 9 * 128 * 24 * 156,
                test_forward_dense_MAC_terms=len(rows) * 512 * 24 * 156,
                recorded_training_seconds=sum(r['seconds'] for r in rows)))
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
            audits[name]['sampled_updates'] = obj['sampled_updates']
            audits[name]['final_checkpoints'] = obj['final_checkpoints']
            audits[name]['parameter_derivative_checks'] = obj['parameter_derivative_checks']
    result = dict(cells=cells, totals={k: sum(c[k] for c in cells) for k in additive},
        shared_resources=dict(trainable_float64_parameters=176, continuous_states=20,
            previous_hidden_spike_states=6, output_count_accumulators=2, training_batch=128,
            sequence_steps=24, updates_per_trajectory=200, test_episodes_per_block=512,
            training_seed_blocks=20, cells_per_block=8, unique_training_histories=20*128,
            unique_test_histories=20*512, model_test_episodes=160*512,
            forward_dense_MAC_terms_per_episode_step=156,
            backward_dense_MAC_terms_per_episode_step=240,
            checkpoint_updates=[0, 1, 5, 10, 20, 50, 100, 150, 200]),
        training_phase_wall_seconds=train['seconds'], test_artifact_phase_wall_seconds=test['seconds'],
        test_inference_seconds=sum(r['inference_seconds'] for r in test['rows']), audits=audits,
        limitations=['Dense matrix MAC counts exclude scalar nonlinearities, resets, loss computation, Adam, data generation, I/O and numerical audits.',
            'Recorded wall times depend on CPU contention and file compression; separate audit times may overlap and must not be summed as elapsed workflow time.',
            'DEND has 12 graded compartments while POINT has 12 additional spiking branch cells; scalar work, activation patterns and reset derivatives differ.',
            'Matching matrix terms and nominal learning rates does not establish equal effective updates, energy or hardware cost.',
            'Checkpoint forward counts include the final training diagnostic; no validation or hyperparameter selection runs were performed.'])
    dump(R / 'RESOURCE_COST.json', result)
    t = result['totals']
    lines = ['# V7 resource and compute accounting', '',
        'Counts below cover the fixed 160-run factorial. Dense matrix multiply-accumulate (MAC) terms are an algorithmic count, not measured FLOPs, energy, or hardware efficiency.', '',
        '| Quantity | DEND | POINT |', '|---|---:|---:|',
        '| Trainable float64 parameters | 176 | 176 |', '| Continuous state variables | 20 | 20 |',
        '| Previous hidden spikes / output counters | 6 / 2 | 6 / 2 |',
        '| Branch computation | 12 graded compartments | 12 native-spiking cells |',
        '| Hidden somata / output cells | 6 / 2 | 6 / 2 |',
        '| Batch / sequence steps / updates | 128 / 24 / 200 | 128 / 24 / 200 |',
        '| Fixed loss–learning-rate cells / seeds per cell | 4 / 20 | 4 / 20 |',
        '| Forward / backward dense MAC terms per episode-step | 156 / 240 | 156 / 240 |', '',
        '| Cell | Runs | Updates | Forward MAC | Backward MAC | Diagnostic MAC | Test MAC | Recorded train seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        lines.append(f"| {c['arm']} {c['recipe']} | {c['trajectories']} | {c['parameter_updates']:,} | {c['forward_dense_MAC_terms']:,} | {c['backward_dense_MAC_terms']:,} | {c['checkpoint_forward_dense_MAC_terms']:,} | {c['test_forward_dense_MAC_terms']:,} | {c['recorded_training_seconds']:.3f} |")
    lines += ['', f"Total: {t['parameter_updates']:,} parameter updates; {t['forward_dense_MAC_terms']:,} training-forward MAC terms and {t['backward_dense_MAC_terms']:,} training-backward MAC terms. The {t['checkpoint_training_forward_batches']:,} additional training-diagnostic forwards contribute {t['checkpoint_forward_dense_MAC_terms']:,} terms; testing contributes {t['test_forward_dense_MAC_terms']:,}. Nine diagnostics include update 200, so final training metrics are not counted twice.", '',
        'There are 2,560 unique training histories and 10,240 unique held-out histories, reused across eight models within each paired block. The 81,920 test model-episodes are not 81,920 independent training units; the inferential unit remains 20 seed/data blocks.', '',
        f"Training and artifact phase: {train['seconds']:.3f} s. Held-out evaluation and artifact phase: {test['seconds']:.3f} s. The native test model calls themselves total {result['test_inference_seconds']:.3f} s; the rest includes compression, metrics and I/O.", '',
        '| Separate audit | Recorded wall seconds | Status |', '|---|---:|---|']
    for name, a in audits.items():
        lines.append(f"| {name} | {a['recorded_wall_seconds']:.3f} | {a['status']} |")
    lines += ['', *[p for p in result['limitations']], '',
        'Numerical checking is extra work outside the training/test matrix budget: 480 sampled backward replays, 28,160 final-checkpoint local proxy-derivative checks, full Adam reconstruction, native trajectory replay, and statistical regeneration. See the audit JSON for the complete counts and residuals.', '',
        'CPU-only NumPy execution used one BLAS/OpenMP thread. Each training, test/artifact and audit stage had a 300-second cap. No paid compute, architecture search, external output pulse generator, or analog decoder was introduced.', '']
    (R / 'RESOURCE_COST.md').write_text('\n'.join(lines))


def tables(train, test):
    stats = test['statistics']
    fields = ['split', 'id', 'seed', 'arm', 'recipe', 'accuracy', 'coverage', 'abstentions',
        'abstention_rate', 'silent_fraction', 'nonsilent_tie_fraction', 'both_saturated_fraction',
        'failed', 'fully_silent', 'fully_abstaining', 'class_recall', 'confusion',
        'mean_signed_count_margin', 'median_signed_count_margin', 'signed_count_margin_histogram',
        'branch_event_rate', 'hidden_event_rate', 'output_event_rate', 'readout_hidden_event_rate',
        'readout_output_event_rate', 'output_candidate_margin_quantiles']
    rows = [dict(split=split, **r) for split, rr in [('train', train['rows']), ('test', test['rows'])] for r in rr]
    write_csv('per_seed_metrics.csv', fields, rows)
    fields = ['arm', 'recipe', 'n', 'mean_accuracy', 'mean_coverage', 'mean_silent_fraction',
        'mean_abstention_rate', 'mean_nonsilent_tie_fraction', 'mean_train_accuracy', 'run_events', 'pass_transitions']
    write_csv('cell_metrics.csv', fields, stats['cells'])
    rows = [dict(contrast=k, **v) for k, v in stats['intervals'].items()]
    write_csv('paired_contrasts.csv', ['contrast', 'mean', 'percentile95', 'bonferroni_family95',
        'primary_accuracy', 'n_pairs', 'unit', 'degenerate', 'paired_values'], rows)


def figures(train, test):
    F.mkdir(exist_ok=True)
    stats = test['statistics']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, arm in zip(axes, ARMS):
        values = np.array([[next(r['accuracy'] for r in test['rows']
                          if (r['arm'], r['recipe'], r['seed']) == (arm, recipe, seed))
                          for recipe in RECIPES] for seed in range(20)]) * 100
        for v in values:
            ax.plot(range(4), v, color=COLORS[arm], alpha=.22, lw=.7, marker='o', ms=2.6)
        ax.plot(range(4), values.mean(0), color='#111111', lw=2.2, marker='D', ms=5, label='Mean of 20 seeds')
        ax.axhline(90, ls=':', color='#777777', lw=.8)
        ax.set_xticks(range(4), ['RATE\n.01', 'RATE\n.03', 'CE\n.01', 'CE\n.03'])
        ax.set_title(arm)
        ax.set_ylabel('Held-out unconditional accuracy (%)')
        ax.set_ylim(-4, 104)
        ax.legend(loc='lower left', fontsize=8)
    fig.suptitle('V7: all 20 paired seed blocks across the fixed 2 × 2 grid')
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.text(.5, .015, 'Each thin line is one paired seed/data block. All failed, silent and tied models are included.', ha='center', fontsize=9)
    save_figure(fig, 'paired_cells')

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    labels = []
    for i, name in enumerate(stats['primary_family']):
        v = stats['intervals'][name]
        arm = 'DEND' if '_DEND_' in name else 'POINT'
        lo, hi = np.array(v['bonferroni_family95']) * 100
        a, b = np.array(v['percentile95']) * 100
        ax.plot([lo, hi], [i, i], color=COLORS[arm], lw=1.1)
        ax.plot([a, b], [i, i], color=COLORS[arm], lw=4)
        ax.plot(v['mean']*100, i, 'o', color=COLORS[arm], ms=6)
        label = name.replace('accuracy_', '').replace('_loss_lr_interaction', ': loss × LR')
        label = label.replace('_loss_main', ': loss main').replace('_lr_main', ': LR main')
        labels.append(label)
    ax.axvline(0, color='#444444', lw=1)
    ax.axvline(-5, color='#aaaaaa', ls=':', lw=.7)
    ax.axvline(5, color='#aaaaaa', ls=':', lw=.7)
    ax.set_yticks(range(6), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Paired accuracy effect (percentage points)')
    ax.set_title('Six prespecified primary contrasts: exploratory seed-level intervals')
    fig.tight_layout(rect=(0, .11, 1, .96))
    fig.text(.5, .05, 'Thick: pointwise 95%; thin: nominal 99.1667% marginal intervals for six-contrast Bonferroni adjustment.', ha='center', fontsize=8.6)
    fig.text(.5, .017, '20 paired blocks; approximate bootstrap coverage. Dotted ±5 pp is a planning threshold, not an equivalence test.', ha='center', fontsize=8.6)
    save_figure(fig, 'primary_contrasts')

    fig, axes = plt.subplots(2, 4, figsize=(13, 7), sharex=True, sharey=True)
    for ax, (arm, recipe) in zip(axes.flat, [(a, r) for a in ARMS for r in RECIPES]):
        for row in train['rows']:
            if (row['arm'], row['recipe']) != (arm, recipe):
                continue
            with np.load(O / row['trace'], allow_pickle=False) as trace:
                accuracy = np.r_[trace['history'][:, 3], row['accuracy']] * 100
            ax.plot(np.arange(201), accuracy, color=COLORS[arm], alpha=.35, lw=.7)
        ax.axhline(90, ls=':', lw=.7, color='#777777')
        ax.set_title(arm + ' ' + recipe.replace('_', ' / '))
        ax.set_xlabel('Completed updates')
        ax.set_ylabel('Training accuracy (%)')
        ax.set_ylim(-3, 103)
    fig.suptitle('All 160 saved training trajectories; final checkpoint fixed before test')
    fig.tight_layout(rect=(0, 0, 1, .96))
    save_figure(fig, 'training_trajectories')


def main():
    train = read(O / 'TRAIN_LOCK.json')
    test = read(O / 'TEST_SUMMARY.json')
    assert len(train['rows']) == len(test['rows']) == 160
    assert len(test['statistics']['cells']) == 8
    assert len(test['statistics']['primary_family']) == 6
    assert len(test['statistics']['intervals']) == 50
    resources(train, test)
    tables(train, test)
    figures(train, test)
    print('Saved V7 resource accounting, three CSV tables and three PNG/SVG figures; no model calls.')


if __name__ == '__main__':
    main()
