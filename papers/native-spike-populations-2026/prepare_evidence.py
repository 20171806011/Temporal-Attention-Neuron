"""Publication-only extraction and plotting of frozen V6–V10 outputs.

Never imports an experimental model, generates task data, trains, or evaluates
a network. Running this script rewrites only this paper's evidence and figures.
"""
from pathlib import Path
import hashlib
import json
import math
import os

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tan-population-paper-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/tan-population-paper-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE_COMMIT = 'c75814eac5202e6a265ca42e78805f906d302249'
REPO_URL = 'https://github.com/20171806011/Temporal-Attention-Neuron'
GATES = {
    6: ('26d85e9d6176f7b0d335d55527cd0c8049f55a75', '0cb7d16aad5fe897406110fc5c79fd483347aab3'),
    7: ('8ad0b69e65074563929dff905acc0067837e3778', '041693b7297b888ed31ac78f8d5677408669f1fe'),
    8: ('cd5cd4ef1cc65fd098b3a73d36d68e7a49307d17', '3ec9ce117ad3ce1a20d6e5ef4f2395916f120821'),
    9: ('ad1d807b1d48be559fd67d4f6d196b8693b4da90', 'd4fde77eb49e21c00950cb93050819ab72032e90'),
    10: ('4dd342161cee0d2d0f0a8383cec6aba5758d514d', 'e77dd9555ce15f94b9f5bcba12650c48aa9ce19d'),
}
BLOCKS = {6: 20, 7: 20, 8: 104, 9: 5, 10: 5}
FIT_COUNTS = {6: 80, 7: 160, 8: 416, 9: 20, 10: 40}
COLORS = {'DEND': '#176d8d', 'POINT': '#c15b28'}
RECIPES = ('RATE_0.01', 'RATE_0.03', 'CE_0.01', 'CE_0.03')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def extract():
    sources = {}
    studies = {}
    for version in range(6, 11):
        directory = ROOT / 'research' / f'dendritic-population-v{version}'
        test = read(directory / 'results/TEST_SUMMARY.json')
        train = read(directory / 'results/TRAIN_LOCK.json')
        audit = read(directory / 'results/verification.json')
        gradient = read(directory / 'results/gradient_audit.json')
        cost_name = 'results/COST.json' if version == 6 else 'RESOURCE_COST.json'
        cost = read(directory / cost_name)
        assert audit['status'] == gradient['status'] == 'PASS'
        assert len(train['rows']) == FIT_COUNTS[version]
        cells = []
        for original in test['statistics']['cells']:
            cell = dict(original)
            if 'run_events' in original:
                cell['failures'] = original['run_events']['failed']['count']
                cell['fully_silent'] = original['run_events']['fully_silent']['count']
                cell['fully_abstaining'] = original['run_events']['fully_abstaining']['count']
                cell['failure_wilson95'] = original['run_events']['failed']['wilson95']
            cell['passes'] = cell['n'] - cell['failures']
            cells.append(cell)
        keep = ('id', 'seed', 'arm', 'recipe', 'condition', 'h', 'accuracy',
                'balanced_accuracy', 'coverage', 'abstentions', 'abstention_rate',
                'silent_fraction', 'nonsilent_tie_fraction', 'both_saturated_fraction',
                'class_recall', 'failed', 'fully_silent', 'fully_abstaining',
                'spike_events_mean', 'total_spike_events_mean', 'state_l1_change_mean',
                'capped_correct_settled_latency_mean', 'pattern_recall')
        rows = [{k: row[k] for k in keep if k in row} for row in test['rows']]
        studies[str(version)] = {
            'phase': f'V{version}' if version != 10 else 'V10A',
            'independent_paired_blocks': BLOCKS[version],
            'new_training_fits': len(train['rows']),
            'derived_inference_cuts': len(train.get('cut_rows', [])),
            'test_model_conditions': len(rows),
            'unique_training_episodes': BLOCKS[version]*128,
            'unique_test_episodes': test['unique_test_histories'],
            'test_model_episode_evaluations': test['model_test_episodes'],
            'protocol_commit': GATES[version][0],
            'checkpoint_commit': GATES[version][1],
            'training_wall_seconds': train['seconds'],
            'test_artifact_wall_seconds': test['seconds'],
            'primary_family': test['statistics'].get('primary_family', [test['statistics'].get('primary')]),
            'cells': cells, 'intervals': test['statistics']['intervals'],
            'test_rows': rows, 'verification': audit, 'gradient_audit': gradient,
            'resource_accounting': cost,
            'primary_uncertainty': ('Approximate Student-t95, df103, sole primary interaction; bootstrap secondary' if version == 8 else
                                    'Six primary marginal percentile intervals with approximate Bonferroni family95; other intervals pointwise exploratory' if version == 7 else
                                    'Pointwise exploratory paired-block percentile95; no multiplicity protection'),
        }
        names = ['PROTOCOL.md', 'REPORT.md', 'TABLES.md', 'SEEDS.json', 'FREEZE.json',
                 'MANIFEST.json', 'model.py', 'data.py', 'common.py', 'runner.py', 'evaluate.py',
                 'verify.py', 'audit_gradients.py', 'check_contracts.py', 'check_gradients.py',
                 'results/TEST_SUMMARY.json', 'results/TRAIN_LOCK.json',
                 'results/verification.json', 'results/gradient_audit.json', cost_name]
        for optional in ('PROVENANCE.json', 'PROVENANCE.md', 'CONTRASTS.json', 'PLANNING.json',
                         'POSTHOC_V5.md', 'posthoc_v5.json', 'DESIGN_REVIEW.md', 'SOURCE_AUDIT.json'):
            if (directory / optional).is_file():
                names.append(optional)
        for name in names:
            path = directory / name
            relative = path.relative_to(ROOT).as_posix()
            sources[relative] = {'sha256': sha(path), 'size_bytes': path.stat().st_size,
                                 'pinned_url': f'{REPO_URL}/blob/{SOURCE_COMMIT}/{relative}'}
    # Earlier independently selected configuration is context, not a new pooled cohort.
    for name in ('research/dendritic-population-v5/REPORT.md', 'research/dendritic-population-v5/PROTOCOL.md'):
        path = ROOT / name
        sources[name] = {'sha256': sha(path), 'size_bytes': path.stat().st_size,
                         'pinned_url': f'{REPO_URL}/blob/{SOURCE_COMMIT}/{name}'}
    totals = {
        'new_training_fits': sum(s['new_training_fits'] for s in studies.values()),
        'derived_inference_cuts': sum(s['derived_inference_cuts'] for s in studies.values()),
        'test_model_conditions': sum(s['test_model_conditions'] for s in studies.values()),
        'test_model_episode_evaluations': sum(s['test_model_episode_evaluations'] for s in studies.values()),
        'unique_test_episodes_across_distinct_studies': sum(s['unique_test_episodes'] for s in studies.values()),
        'adam_updates_reconstructed': sum(s['verification']['training']['reconstructed_updates'] for s in studies.values()),
        'checkpoints_replayed': sum(s['verification']['training']['native_training_checkpoints'] for s in studies.values()),
        'saved_checker_assertions': sum(s['verification']['checks'] for s in studies.values()),
        'sampled_gradient_updates': sum(s['gradient_audit']['sampled_updates'] for s in studies.values()),
        'final_local_surrogate_derivative_checks': sum(s['gradient_audit']['parameter_derivative_checks'] for s in studies.values()),
        'maximum_saved_native_state_candidate_residual': max(max(s['verification']['training']['max_training_state_candidate_residual'], s['verification']['tests']['max_test_state_candidate_residual']) for s in studies.values()),
        'maximum_final_surrogate_chain_rule_residual': max(s['gradient_audit']['max_chain_rule_residual'] for s in studies.values()),
        'not_a_pooled_inferential_sample': True,
    }
    return {'schema_version': 1, 'source_commit': SOURCE_COMMIT,
            'scope': 'Reporting-only extraction of completed studies. No new training, task data or network inference. Cohorts remain separate; saved audits are internal checks, not external replication.',
            'sources': sources, 'studies': studies, 'totals': totals}


def save_figure(fig, stem):
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    fig.savefig(out / (stem+'.png'), dpi=220, bbox_inches='tight')
    fig.savefig(out / (stem+'.pdf'), bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(out / (stem+'.svg'), bbox_inches='tight', metadata={'Date': None})
    p = out / (stem+'.svg')
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)


def figures(evidence):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.hashsalt': 'tan-v6-v10-paper'})
    s = evidence['studies']
    fig = plt.figure(figsize=(7.6, 7.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, .8])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    ax = axes[0]
    for arm in ('DEND', 'POINT'):
        vals = np.array([[next(r['accuracy'] for r in s['6']['test_rows'] if r['seed'] == seed and r['arm'] == arm and r['recipe'] == recipe) for recipe in ('RATE_0.01', 'CE_0.03')] for seed in range(20)])*100
        for row in vals: ax.plot([0, 1], row, color=COLORS[arm], alpha=.13, lw=.7)
        ax.plot([0, 1], vals.mean(0), 'o-', color=COLORS[arm], lw=2.2, label=arm)
    ax.set(xticks=[0, 1], xticklabels=['RATE / .01', 'CE / .03'], ylim=(-2, 103), ylabel='Held-out accuracy (%)', title='A  V6: two bundled recipes\n20 paired blocks')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for version, style, marker in (('7', '--', 's'), ('8', '-', 'o')):
        for loss, color in (('RATE', '#176d8d'), ('CE', '#c15b28')):
            means = [100*next(c['mean_accuracy'] for c in s[version]['cells'] if c['arm'] == 'POINT' and c['recipe'] == f'{loss}_{rate}') for rate in ('0.01', '0.03')]
            ax.plot([0, 1], means, color=color, linestyle=style, marker=marker, lw=2, label=f'V{version} {loss}')
    ax.set(xticks=[0, 1], xticklabels=['.01', '.03'], xlabel='Learning rate', ylim=(45, 100), ylabel='Mean POINT accuracy (%)', title='B  POINT factorial cell means\nV7: 20 blocks; V8: 104 blocks')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = axes[2]
    for row, version, key, label in ((1, '7', 'bonferroni_family95', 'V7: family-adjusted bootstrap'), (0, '8', 'student_t95', 'V8: primary Student-t95')):
        record = s[version]['intervals']['accuracy_POINT_loss_lr_interaction']
        lo, hi = np.array(record[key])*100
        mean = record['mean']*100
        ax.plot([lo, hi], [row, row], color='#343b42', lw=2.2)
        ax.plot(mean, row, 'o', color='#176d8d', ms=6)
        ax.text(mean, row+.17, f'{mean:.2f} [{lo:.2f}, {hi:.2f}]', ha='center', fontsize=8)
    ax.axvline(0, color='.65', lw=1)
    ax.axvline(-5, color='.65', lw=.8, ls=':')
    ax.set(yticks=[0, 1], yticklabels=['V8: primary t95\n(n=104)', 'V7: adjusted bootstrap\n(n=20)'], ylim=(-.45, 1.55), xlabel='(CE .03 − CE .01) − (RATE .03 − RATE .01), pp', title='C  Preserved POINT interaction intervals')
    ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.text(.5, .01, 'Separate cohorts; all failed fits retained. V6 changes loss and learning rate together.\nV7 six-primary family bootstrap coverage is approximate. V8 is a same-workflow, fresh-cohort test.\nDotted vertical line: prespecified −5 pp practical threshold.', ha='center', fontsize=7.6)
    save_figure(fig, 'optimization_interactions')

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.5), sharey=True)
    conditions = ('intact', 'retrained', 'cut')
    for ax, arm in zip(axes, ('DEND', 'POINT')):
        vals = np.array([[next(r['accuracy'] for r in s['9']['test_rows'] if (r['seed'], r['arm'], r['condition']) == (seed, arm, condition)) for condition in conditions] for seed in range(5)])*100
        for seed, row in enumerate(vals): ax.plot(range(3), row, 'o-', color=plt.cm.tab10(seed), alpha=.65, lw=1, ms=4)
        ax.plot(range(3), vals.mean(0), 'D-', color='#1b252c', lw=2.3, ms=6, label='Mean of 5 blocks')
        ax.axhline(90, color='.5', lw=.8, ls=':')
        for i, condition in enumerate(conditions):
            c = next(c for c in s['9']['cells'] if c['arm'] == arm and c['condition'] == condition)
            ax.text(i, 105, f"Pass: {c['passes']}/5", ha='center', fontsize=8)
        ax.set(xticks=range(3), xticklabels=['Intact', 'No-cross\nretrained', 'Inference\ncut'], ylim=(-2, 114), title=arm, ylabel='Held-out accuracy (%)' if arm=='DEND' else '')
    axes[1].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('V9: pathway dependence and alternative trained solutions', fontsize=11)
    fig.tight_layout(rect=(0, .12, 1, .95))
    fig.text(.5, .01, 'RATE/.01, 200 updates, 5 paired blocks. Pass requires accuracy AND coverage ≥90%.\nNo-cross retains a population, self-feedback and internal states.', ha='center', fontsize=7.6)
    save_figure(fig, 'recurrence_controls')

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.8), gridspec_kw={'width_ratios': [1.2, 1]})
    sizes = (6, 4, 2, 1)
    ax = axes[0]
    for arm in ('DEND', 'POINT'):
        vals = np.array([[next(r['balanced_accuracy'] for r in s['10']['test_rows'] if (r['seed'], r['arm'], r['h']) == (seed, arm, h)) for h in sizes] for seed in range(5)])*100
        for row in vals: ax.plot(range(4), row, 'o-', color=COLORS[arm], alpha=.2, lw=.8, ms=4)
        ax.plot(range(4), vals.mean(0), 'D-', color=COLORS[arm], lw=2.2, ms=5, label=arm)
    ax.axhline(90, color='.55', ls=':', lw=.8)
    ax.set(xticks=range(4), xticklabels=sizes, ylim=(-2, 104), ylabel='Held-out balanced accuracy (%)', xlabel='Hidden somata h (not total cells)', title='A  Fixed-recipe size curves')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for i, h in enumerate(sizes):
        row = 3-i
        record = s['10']['intervals'][f'h{h}_DEND_minus_POINT_balanced_accuracy']
        lo, hi = np.array(record['percentile95'])*100
        ax.plot([lo, hi], [row, row], color='#343b42', lw=2)
        ax.plot(record['mean']*100, row, 'o', color='#176d8d', ms=6)
        ax.scatter(np.array(record['paired_values'])*100, row+np.linspace(-.12,.12,5), color='.55', alpha=.6, s=14)
    ax.axvline(0, color='.6', lw=.8)
    ax.set(yticks=range(4), yticklabels=['h=1','h=2','h=4','h=6'], xlabel='DEND − POINT\nbalanced accuracy, pp', title='B  Pointwise paired\nbootstrap95 intervals')
    fig.suptitle('V10A: four-feature binding with brief retention', fontsize=12)
    fig.tight_layout(rect=(0, .13, 1, .95))
    fig.text(.5, .01, 'RATE/.01; 200 updates; 5 paired blocks; all failures retained. Four intervals are unadjusted.\nA POINT h=1 fit passes; finite failure is not representational impossibility.', ha='center', fontsize=7.6)
    save_figure(fig, 'binding_scaling')


def write_markdown(e):
    lines = ['# Evidence ledger for the V6–V10 population manuscript', '',
             f"Canonical source snapshot: `{SOURCE_COMMIT}`. All file paths and SHA-256 hashes are in [evidence.json](evidence.json).", '',
             'This publication reorganizes saved scientific results. It introduces no new training runs, task data, held-out evaluation or numerical-model replay. Original phase reports retain their execution-time status statements. Study cohorts and estimands are not pooled.', '',
             '## Study inventory and commit gates', '',
             '| Phase | Paired blocks | New fits | Derived cuts | Held-out model-episode evaluations | Protocol commit | Checkpoint commit |',
             '|---|---:|---:|---:|---:|---|---|']
    for s in e['studies'].values():
        lines.append(f"| {s['phase']} | {s['independent_paired_blocks']} | {s['new_training_fits']} | {s['derived_inference_cuts']} | {s['test_model_episode_evaluations']:,} | `{s['protocol_commit']}` | `{s['checkpoint_commit']}` |")
    lines += ['', 'The 716 fits yield 726 held-out model conditions because V9 derives ten inference cuts without training. There are 371,712 model-episode evaluations, not 371,712 independent training replications. Each study uses 128 training and 512 held-out episodes per paired block, with 200 updates per trained model.', '',
              '## Claim-to-source map', '',
              '| Claim | Saved source and exact field | Scope |', '|---|---|---|']
    claims = [
        (6, 'V6 DEND bundled-recipe contrast', '`statistics.intervals.accuracy_DEND_CE_minus_RATE`', '−13.564453 pp; pointwise bootstrap95 [−28.750000,+0.136719]; changes loss and LR together.'),
        (7, 'V7 POINT factorial interaction', '`statistics.intervals.accuracy_POINT_loss_lr_interaction`', '−33.837891 pp; six-primary-family adjusted percentile interval [−64.482666,−5.843058]; approximate exploratory coverage.'),
        (8, 'V8 primary POINT interaction', '`statistics.intervals.accuracy_POINT_loss_lr_interaction`', '−14.105694 pp; primary approximate Student-t95 [−22.603812,−5.607576],104 fresh blocks; negative direction repeats but the earlier crossover of simple-effect point estimates does not.'),
        (9, 'V9 dependence versus retrainable alternatives', '`statistics.cells`; `rows`', 'DEND intact/retrained/cut passes4/5,5/5,0/5; no-cross success concerns finite task/gate and retains multiple cells and state.'),
        (10, 'V10A fixed-recipe size comparison', '`statistics.cells`; `statistics.intervals`', 'DEND-minus-POINT BA contrasts at h6/4/2/1 are +9.82/+12.81/+19.58/+16.74pp; four unadjusted intervals, five blocks; not a pure representation bound.'),
        (10, 'V10A minimum-size POINT passing case', '`rows[id=s0_POINT_h1]`', 'BA91.195437%,coverage97.0703125%; two branch cells+one hidden soma+two output cells; five cells, not one.'),
    ]
    for version, claim, field, scope in claims:
        path = f'../../research/dendritic-population-v{version}/results/TEST_SUMMARY.json'
        lines.append(f'| {claim} | [TEST_SUMMARY.json]({path}), {field} | {scope} |')
    lines += ['', 'V7/V8 within-architecture loss-by-learning-rate contrasts use `(CE/.03 − CE/.01) − (RATE/.03 − RATE/.01)`. V8 uses a sole prespecified primary Student-t interval; its bootstrap interval is supplementary. V7 uses approximate Bonferroni-adjusted bootstrap intervals for six primary contrasts. V6, V9 and V10 intervals are pointwise exploratory. These methods must not be silently exchanged.', '',
              '## Complete cell means and failures', '',
              '| Phase | Architecture | Condition | n | Accuracy % | Balanced accuracy % | Coverage % | Failed | Fully silent | Fully abstaining |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in e['studies'].values():
        for c in s['cells']:
            condition = c.get('recipe', c.get('condition', f"h={c.get('h')}"))
            ba = c.get('mean_balanced_accuracy', c['mean_accuracy'])
            lines.append(f"| {s['phase']} | {c['arm']} | {condition} | {c['n']} | {100*c['mean_accuracy']:.6f} | {100*ba:.6f} | {100*c['mean_coverage']:.6f} | {c['failures']} | {c['fully_silent']} | {c['fully_abstaining']} |")
    lines += ['', 'Delayed XOR has balanced class counts, so ordinary and balanced accuracy coincide in V6–V9. V10 uniformly samples16 patterns with7 positive and9 negative patterns; balanced accuracy is the mean of class recalls, with abstentions wrong. Failure means either accuracy<90% or coverage<90% in V6–V9; balanced accuracy replaces accuracy in V10. Means do not determine the number of passing fits.', '',
              '## Saved numerical audits and execution costs', '',
              '| Phase | Checker assertions | Adam updates | Checkpoints | Final proxy derivatives | Sampled gradients | Training/artifact seconds | Test/artifact seconds |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in e['studies'].values():
        a=s['verification'];g=s['gradient_audit']
        lines.append(f"| {s['phase']} | {a['checks']:,} | {a['training']['reconstructed_updates']:,} | {a['training']['native_training_checkpoints']:,} | {g['parameter_derivative_checks']:,} | {g['sampled_updates']:,} | {s['training_wall_seconds']:.3f} | {s['test_artifact_wall_seconds']:.3f} |")
    lines += ['', 'Across the saved audits:978,440 assertions,143,200 reconstructed updates,6,444 checkpoints,121,806 final anchored-surrogate derivative checks and2,228 sampled gradients. Maximum native state/candidate residual is3.552713678800501e−15; maximum final local surrogate-chain-rule residual is7.041099786553318e−10. Spikes, counts and predictions match exactly in the saved full-trajectory replays. These are internal same-workflow checks, not external independent replication or peer review. Derivative checks validate a local surrogate graph, not the exact derivative of a hard-spike function.', '',
              'All per-phase dense matrix MAC accounting, separate diagnostic passes, test work and audit timing are copied into evidence.json from the original COST/RESOURCE_COST records. Wall time is CPU/environment dependent; overlapping audit wall times cannot be added as elapsed workflow time. Neither MACs nor model events measure joules, ATP or hardware energy. V6 scientific sources were frozen before training data; its separately written native verifier was added with the trained-checkpoint commit before held-out data, not before training. The phase-specific freeze records remain authoritative.', '',
              '## Publication figures', '',
              '- [Optimization and interaction figure](figures/optimization_interactions.png): V6 paired bundled recipes, V7/V8 separate POINT factorial means and their original primary interval conventions.',
              '- [Recurrence controls](figures/recurrence_controls.png): all five V9 seed outcomes for both architectures, including failed intact, retrained and cut conditions.',
              '- [Binding scale curves](figures/binding_scaling.png): all V10 outcomes, mean balanced accuracy and five-block paired contrasts at every size.',
              '', 'Each figure is also supplied as vector PDF and SVG. All are generated from saved JSON by prepare_evidence.py; figures do not imply pooled cohorts or equal biological/energy resources.', '',
              '## Evidence boundaries and prior context', '',
              'V5 independently selected POINT CE/.003; V6–V10 fixed grids did not retest that selected configuration. The uncertain V5 independently tuned comparison is not replaced by favorable fixed-recipe comparisons. Its original [report](../../research/dendritic-population-v5/REPORT.md) and [protocol](../../research/dendritic-population-v5/PROTOCOL.md) are separately hashed as context, not counted among the new V6–V10 fits.', '',
              'V9 removes60 cross-cell weights (176→116 effective trainable parameters), changes gradient normalization and retains self recurrence, branch states and output integration. Inference cuts concern an already trained solution; finite retraining failure cannot establish physical necessity. V10 matches parameters/state at each size but not cell counts, scalar nonlinearities, reset derivatives or effective optimization. Size changes initialization drive variance and per-coordinate mean-L2 strength. Its h=1 POINT success rules out interpreting this pilot as impossibility at the tested finite gate.', '',
              '## Validate the publication extraction', '',
              'Run `python validate_evidence.py` from this directory or any other working directory. It checks source SHA-256 identities against the pinned Git tree, headline arithmetic, paired contrast means, the V8 primary Student-t formula, complete fit retention and figure presence. It does not import scientific models or rerun experiments. Original experiment audit scripts may rewrite their logs; run those only in a separate copy.', '']
    (HERE/'EVIDENCE.md').write_text('\n'.join(lines))


def main():
    evidence = extract()
    (HERE/'evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    write_markdown(evidence)
    figures(evidence)
    print(json.dumps({'status':'PASS', 'sources':len(evidence['sources']), 'totals':evidence['totals'], 'scientific_execution':False}, indent=2))


if __name__ == '__main__':
    main()
