"""Independent verification of frozen native states, transplants and offline scores."""

from datetime import datetime
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_capture as capture
import independent_data as data
import independent_probe as probe
from reference import scalar_oracle


ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'DEND-TAN-frozen-probes-20261108'
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
LOSSES = ('CE', 'COUNT', 'CECOM')
META = ('id', 'seed', 'arm', 'loss')
NONSELF = ('cue_flip', 'content_flip', 'both_flip')
FEATURES = ('counts', 'raster')
SOURCES = {'model.py', 'population.py', 'reference.py', 'original_tan.py', 'task_data.py',
           'independent_data.py', 'probe.py', 'independent_probe.py', 'models.npz', 'MODELS.json'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, content):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Audit:
    def __init__(self):
        self.start = time.perf_counter()
        self.cpu = time.process_time()
        self.stage = 'source and phase locks'
        self.checks = self.baselines = self.fits = self.patches = self.steps = 0
        self.self_controls = self.early_controls = self.donor_controls = self.late_prefixes = 0
        self.scalar_models = 0
        self.max_numeric_error = self.max_coefficient_error = self.max_normal_residual = 0.

    def check(self, condition, context):
        self.checks += 1
        if time.perf_counter() - self.start > 300:
            raise TimeoutError(f'300 s audit cap during {self.stage}')
        if not condition:
            raise AssertionError(context)

    def exact(self, actual, expected, context):
        self.check(np.array_equal(actual, expected), context)

    def near(self, actual, expected, context, *, atol=1e-9, rtol=1e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.check(actual.shape == expected.shape, context + ' shape')
        self.check(np.isfinite(actual).all() and np.isfinite(expected).all(), context + ' finite')
        error = float(np.max(np.abs(actual - expected), initial=0.))
        self.max_numeric_error = max(error, self.max_numeric_error)
        self.check(np.allclose(actual, expected, atol=atol, rtol=rtol), f'{context}: max error {error:.6g}')
        return error


def patch_design(arm):
    early = [(9, 'full', 'content_flip')]
    mid_groups = ('full', 'hidden', 'branch', 'soma_p', 'output')
    if arm in capture.ATTENTION_ARMS:
        mid_groups += ('history',)
    return (early + [(15, 'full', 'self')] + list(product((15,), mid_groups, NONSELF))
            + [(19, 'full', 'self')] + list(product((19,), ('full', 'hidden', 'output'), NONSELF)))


def patch_id(ident, tap, group, donor):
    return f'{ident}_t{tap}_{group}_{donor}'


def index(audit, rows, field, expected):
    indexed = {row[field]: row for row in rows}
    audit.check(len(rows) == len(indexed) and set(indexed) == expected, f'complete unique {field} records')
    return indexed


def source_locks(audit):
    freeze, source, fit = (read(ROOT / name) for name in ('FREEZE.json', 'SOURCE_LOCK.json', 'FIT_LOCK.json'))
    audit.check(freeze['scientific_data_generated'] is False and freeze['neural_weights_fixed'] is True,
                'pre-science freeze and fixed weights')
    audit.check(fit['test_generated'] is False, 'no test before fit lock assertion')
    old_manifest_file = PRIOR / 'MANIFEST.json'
    manifest = read(old_manifest_file)['files']
    audit.check(source['previous_directory'] == f'../{PRIOR.name}' and
                source['previous_manifest_sha256'] == sha(old_manifest_file), 'parent manifest identity')
    audit.check(source['state_probe_source_prefix'] == 'fits/', 'old probe directory mapping')
    audit.check(read(PRIOR / 'AUDIT.json')['status'] == 'PASS' and
                sha(PRIOR / 'AUDIT.json') == manifest['AUDIT.json']['sha256'], 'audited parent batch')
    old_fit_lock_path = PRIOR / 'FIT_LOCK.json'
    audit.check(sha(old_fit_lock_path) == manifest['FIT_LOCK.json']['sha256'], 'parent fit lock identity')
    old_fit_lock = read(old_fit_lock_path)['files']

    rows = read(ROOT / 'MODELS.json')['rows']
    expected_models = set(product(range(3), ARMS, LOSSES))
    audit.check(len(rows) == 36 and {(r['seed'], r['arm'], r['loss']) for r in rows} == expected_models,
                '36 complete archived models')
    for i, row in enumerate(rows):
        audit.check(row['index'] == i and row['id'] == f"s{row['seed']}_{row['arm']}_{row['loss']}",
                    f'archive model row {i}')
    theta = load(ROOT / 'models.npz')
    audit.check(set(theta) == {'theta'} and theta['theta'].shape == (36, 176) and
                np.isfinite(theta['theta']).all(), 'finite archived weights')

    probe_files = {f"state_probes/{r['id']}_joint_t{t}.npz" for r in rows for t in capture.TAPS}
    audit.check(set(source['files']) == SOURCES | probe_files, 'all frozen source files and 108 state probes')
    for name, digest in source['files'].items():
        previous_name = name
        if name in probe_files:
            previous_name = 'fits/' + Path(name).name
            audit.check(digest == old_fit_lock[previous_name], f'{name} old fit-lock membership')
        entry = manifest[previous_name]
        audit.check(digest == entry['sha256'] == sha(ROOT / name) == sha(PRIOR / previous_name), f'{name} bytes')
        audit.check((ROOT / name).stat().st_size == entry['bytes'], f'{name} archive size')

    required = set(source['files']) | {'audit.py', 'independent_capture.py', 'run.py', 'PROTOCOL.md',
                                      'SOURCE_LOCK.json', 'PRECHECK.json', 'TIMING.json'}
    audit.check(required <= set(freeze['files']), 'freeze covers source, settings and verification')
    for name, digest in freeze['files'].items():
        relative = Path(name)
        audit.check(not relative.is_absolute() and '..' not in relative.parts, f'local freeze path {name}')
        audit.check(sha(ROOT / relative) == digest, f'frozen file {name}')
    for name in ('PRECHECK.json', 'TIMING.json'):
        audit.check(read(ROOT / name)['status'] == 'PASS', f'accepted frozen {name}')

    expected_fit = {'fit/ROWS.json'} | {f'data/fit_s{s}.npz' for s in range(3)}
    expected_fit |= {f"fit/{r['id']}_{kind}.npz" for r in rows for kind in ('baseline', *FEATURES)}
    audit.check(set(fit['files']) == expected_fit, 'complete calibration lock coverage')
    freeze_time = (ROOT / 'FREEZE.json').stat().st_mtime_ns
    fit_time = (ROOT / 'FIT_LOCK.json').stat().st_mtime_ns
    audit.check(datetime.fromisoformat(freeze['utc']) < datetime.fromisoformat(fit['utc']), 'lock UTC chronology')
    for name, digest in fit['files'].items():
        path = ROOT / name
        audit.check(freeze_time <= path.stat().st_mtime_ns <= fit_time and sha(path) == digest,
                    f'calibration artifact before test {name}')
    for block in range(3):
        audit.check((ROOT / f'data/test_s{block}.npz').stat().st_mtime_ns >= fit_time,
                    f'test block {block} generated after fit lock')
    return rows, theta['theta']


def read_data(audit, split, block, nuisance_keys):
    n, seed = (128, 4011010000) if split == 'fit' else (256, 4011020000)
    actual = load(ROOT / f'data/{split}_s{block}.npz')
    expected = data.make_base(n, seed + 100 * block)
    audit.check(set(actual) == set(expected), f'{split} s{block} data fields')
    for key in expected:
        audit.exact(actual[key], expected[key], f'{split} s{block} independently generated {key}')
    x, q = data.assemble(actual, 'nominal')
    audit.check(q == 16, 'query begins at t16')
    data.check(actual, x, 'nominal')
    keys = set(zip(actual['amplitude'], actual['tag_amplitude']))
    audit.check(len(keys) == n // 4 and not keys.intersection(nuisance_keys), 'distinct matched quartets across splits')
    nuisance_keys.update(keys)
    return actual, x


def donor_maps(audit, base, x):
    # Lexicographic sorting independently recovers the four entries of each quartet.
    quartet, selected, bit, labels = (base[k] for k in ('quartet', 'selected', 'first_bit', 'y'))
    lookup = np.lexsort((bit, selected, quartet)).reshape(-1, 2, 2)
    donors = {
        'self': np.arange(len(x)),
        'cue_flip': lookup[quartet, 1 - selected, bit],
        'content_flip': lookup[quartet, selected, 1 - bit],
        'both_flip': lookup[quartet, 1 - selected, 1 - bit],
    }
    for name, ids in donors.items():
        audit.exact(quartet[ids], quartet, f'{name} donor stays within quartet')
        audit.exact(ids[ids], donors['self'], f'{name} reciprocal pairing')
        other_label = 1 - labels if name in NONSELF[:2] else labels
        audit.exact(labels[ids], other_label, f'{name} label relation')
        audit.exact(x[ids, 16:], x[:, 16:], f'{name} identical query input suffix')
    audit.exact(x[donors['content_flip'], :10], x[:, :10], 't9 identical input prefixes')
    return donors


def verify_baseline(audit, path, x, w, arm, taps):
    saved = load(path)
    independent = capture.capture(x, w, arm, taps=taps)
    audit.check(set(saved) == set(independent), f'{path.name} capture fields')
    for key, expected in independent.items():
        if key in ('raster', 'counts', 'resources', 'p'):
            audit.exact(saved[key], expected, f'{path.name} native {key}')
        else:
            audit.near(saved[key], expected, f'{path.name} native {key}')
    audit.exact(saved['counts'], saved['raster'].sum(axis=1), f'{path.name} counts equal query raster sums')
    audit.baselines += 1
    audit.steps += len(x) * 24
    return saved, independent


def verify_scalar(audit, x, w, arm, saved, ident):
    if arm not in capture.ATTENTION_ARMS:
        return
    scalar = scalar_oracle(x[:2], w, arm, 16)
    state = np.concatenate([saved[key][:2] for key in ('d', 'v', 'o')], axis=2)
    audit.near(state, scalar['states'][:, capture.TAPS], f'{ident} original scalar TAN states')
    audit.near(saved['H'][:2], scalar['histories'][:, capture.TAPS], f'{ident} original scalar TAN history')
    audit.exact(saved['p'][:2], scalar['spikes'][:, capture.TAPS, :6], f'{ident} original scalar TAN hidden spikes')
    audit.exact(saved['raster'][:2], scalar['spikes'][:, 16:, 6:], f'{ident} original scalar TAN query raster')
    audit.exact(saved['counts'][:2], scalar['counts'], f'{ident} original scalar TAN counts')
    audit.scalar_models += 1


def packed_fit(archive, column=0):
    return dict(mean=archive['mean'], scale=archive['scale'], constant=~archive['active'],
                coef=archive['coef'][:-1, column], intercept=archive['coef'][-1, column])


def features(snapshot, kind):
    if kind == 'counts':
        return snapshot['counts']
    return snapshot['raster'].reshape(len(snapshot['raster']), 16)


def verify_fit(audit, ident, kind, x, labels):
    archive = load(ROOT / f'fit/{ident}_{kind}.npz')
    audit.check(set(archive) == {'mean', 'scale', 'active', 'coef', 'fit_scores'} and
                archive['coef'].shape == (x.shape[1] + 1, 1) and archive['fit_scores'].shape == (len(x), 1),
                f'{ident} {kind} ridge artifact shape')
    target = 2 * labels - 1
    independent = probe.fit(x, target, alpha=.1)
    fitted = packed_fit(archive)
    audit.near(fitted['mean'], independent['mean'], f'{ident} {kind} fit-only mean')
    audit.exact(fitted['constant'], independent['constant'], f'{ident} {kind} fit-only constant mask')
    audit.near(fitted['scale'], np.where(independent['constant'], 1e-8, independent['scale']),
               f'{ident} {kind} fit-only scale', atol=1e-14)
    error = audit.near(archive['coef'][:, 0], np.r_[independent['coef'], independent['intercept']],
                       f'{ident} {kind} independent augmented ridge coefficients')
    audit.max_coefficient_error = max(audit.max_coefficient_error, error)
    score = archive['fit_scores'][:, 0]
    audit.near(score, probe.scores(x, independent), f'{ident} {kind} independent calibration scores')
    audit.near(score, probe.scores(x, fitted), f'{ident} {kind} archived calibration scores')
    z = (x - fitted['mean']) / fitted['scale']
    z[:, fitted['constant']] = 0
    residual = score - target
    normal = z.T @ residual / len(x) + .1 * fitted['coef']
    error = max(float(np.max(np.abs(normal))), abs(float(residual.mean())))
    audit.max_normal_residual = max(audit.max_normal_residual, error)
    audit.check(error < 1e-8, f'{ident} {kind} ridge normal residual')
    audit.fits += 1
    return fitted, score


def metric_values(score, labels, stem):
    values = probe.metrics(score, labels)
    return {stem + '_ba': values['balanced_accuracy'], stem + '_mse': values['mse']}


def native_metrics(counts, labels):
    margin = counts[:, 1].astype(np.int64) - counts[:, 0].astype(np.int64)
    tie = margin == 0
    forced = margin > 0
    strict = np.where(tie, -1, forced)
    silence = (counts == 0).all(axis=1)
    saturated = (counts == 8).all(axis=1)
    return dict(strict_ba=np.mean(strict == labels), forced_ba=np.mean(forced == labels),
                coverage=np.mean(~tie), silence=np.mean(silence), saturated_tie=np.mean(saturated),
                intermediate_tie=np.mean(tie & ~silence & ~saturated))


def verify_metrics(audit, record, expected, context):
    for key, value in expected.items():
        # With balanced labels and power-of-two N, binary rates are exact.
        if 'mse' in key or 'score_shift' in key:
            audit.near(record[key], value, f'{context} {key}')
        else:
            audit.exact(record[key], value, f'{context} {key}')


def joint_scores(state, fitted):
    return probe.scores(np.concatenate([state[key] for key in capture.STATE_FIELDS], axis=1), fitted)


def readout_scores(snapshot, fits):
    return {kind: probe.scores(features(snapshot, kind), fits[kind]) for kind in FEATURES}


def patch_metrics(baseline, raster, suffix, donor_index, labels, state_before, state_after,
                  readouts_before, readouts_after, suffix_resources):
    counts = raster.sum(axis=1)
    donor_y = labels[donor_index]
    before = baseline['counts'].astype(np.int64)
    after = counts.astype(np.int64)
    margin_before = before[:, 1] - before[:, 0]
    margin_after = after[:, 1] - after[:, 0]
    direction = 2 * donor_y - 1
    old_forced, new_forced = margin_before > 0, margin_after > 0
    raster_change = (raster != baseline['raster']).any(axis=(1, 2))
    count_change = (after != before).any(axis=1)
    forced_change = old_forced != new_forced
    transfer = ((labels != donor_y) & (probe.classify(state_before) == labels) &
                (probe.classify(state_after) == donor_y))

    def gain(before, after):
        return np.mean(after == donor_y) - np.mean(before == donor_y)

    donor_metrics = native_metrics(counts, donor_y)
    expected = native_metrics(counts, labels)
    for kind in FEATURES:
        expected.update(metric_values(readouts_after[kind], labels, f'{kind}_probe'))
    expected.update(
        donor_strict_ba=donor_metrics['strict_ba'], donor_forced_ba=donor_metrics['forced_ba'],
        donor_margin_shift=np.mean(direction * (margin_after - margin_before)),
        donor_forced_gain=gain(old_forced, new_forced),
        donor_count_probe_gain=gain(probe.classify(readouts_before['counts']), probe.classify(readouts_after['counts'])),
        donor_raster_probe_gain=gain(probe.classify(readouts_before['raster']), probe.classify(readouts_after['raster'])),
        donor_state_score_shift=np.mean(direction * (state_after - state_before)),
        raster_changed=np.mean(raster_change), count_changed=np.mean(count_change),
        margin_changed=np.mean(margin_before != margin_after), forced_changed=np.mean(forced_change),
        raster_changed_count_same=np.mean(raster_change & ~count_change),
        count_changed_forced_same=np.mean(count_change & ~forced_change),
        suffix_eq_donor=np.mean((suffix == baseline['raster'][donor_index, -suffix.shape[1]:]).all(axis=(1, 2))),
        state_to_donor=np.mean(transfer), state_to_donor_raster_same=np.mean(transfer & ~raster_change),
        state_to_donor_count_same=np.mean(transfer & ~count_change),
        state_to_donor_forced_same=np.mean(transfer & ~forced_change),
        mean_suffix_resources=suffix_resources.mean(axis=0),
    )
    return expected


def verify_patch(audit, model, record, w, x, labels, baseline, independent, donors, state_fit, scores_before, fits):
    ident, arm = model['id'], model['arm']
    tap, group, name = record['time'], record['group'], record['donor']
    tag = patch_id(ident, tap, group, name)
    audit.check(record['patch_id'] == tag and all(record[k] == model[k] for k in META), f'{tag} metadata')
    saved = load(ROOT / f'test/{tag}.npz')
    audit.check(set(saved) == {'donor_index', 'suffix_raster', 'suffix_resources', 'state_score'}, f'{tag} fields')
    donor = donors[name]
    audit.exact(saved['donor_index'], donor, f'{tag} matched donor index')

    state = capture.transplant(independent, tap, group, donor)
    suffix, suffix_resources = capture.continue_capture(x, w, arm, tap, state)
    audit.exact(saved['suffix_raster'], suffix, f'{tag} independently resumed query pulses')
    audit.exact(saved['suffix_resources'], suffix_resources, f'{tag} independently resumed resource ledger')

    # Score byte-verified archived snapshots to avoid amplifying replay roundoff by scaling.
    stored_state = capture.transplant(baseline, tap, group, donor)
    scored = joint_scores(stored_state, state_fit[tap])
    audit.near(saved['state_score'], scored, f'{tag} archived-state score', atol=1e-8)
    audit.exact(probe.classify(saved['state_score']), probe.classify(scored), f'{tag} archived-state decisions')

    prefix = max(0, tap - 15)
    raster = np.concatenate((baseline['raster'][:, :prefix], suffix), axis=1)
    audit.check(raster.shape == baseline['raster'].shape == (256, 8, 2), f'{tag} eight query steps')
    if tap == 19:
        audit.exact(raster[:, :4], baseline['raster'][:, :4], f'{tag} retained recipient prefix')
        audit.late_prefixes += 1
    if name == 'self' or tap == 9:
        audit.exact(raster, baseline['raster'], f'{tag} exact recipient raster control')
        audit.near(saved['state_score'], scores_before[tap], f'{tag} unchanged state scores', atol=1e-8)
        if tap == 9:
            for key in stored_state:
                actual = baseline[key][:, 0]
                if key == 'p':
                    audit.exact(actual, actual[donor], f'{tag} identical early spikes')
                else:
                    audit.near(actual, actual[donor], f'{tag} identical early {key}', atol=1e-12, rtol=0.)
            audit.early_controls += 1
        else:
            audit.self_controls += 1
    elif group == 'full':
        audit.exact(suffix, baseline['raster'][donor, prefix:], f'{tag} exact donor raster suffix')
        audit.near(saved['state_score'], scores_before[tap][donor], f'{tag} donor state scores', atol=1e-8)
        audit.donor_controls += 1

    readouts_before = readout_scores(baseline, fits)
    readouts_after = readout_scores(dict(counts=raster.sum(axis=1), raster=raster), fits)
    values = patch_metrics(baseline, raster, suffix, donor, labels, scores_before[tap], saved['state_score'],
                           readouts_before, readouts_after, suffix_resources)
    verify_metrics(audit, record, values, tag)
    audit.patches += 1
    audit.steps += len(x) * (23 - tap)


def verify_results(audit, models, theta):
    names = {model['id'] for model in models}
    patch_ids = {patch_id(model['id'], *case) for model in models for case in patch_design(model['arm'])}
    fit_rows = index(audit, read(ROOT / 'fit/ROWS.json'), 'id', names)
    base_rows = index(audit, read(ROOT / 'test/BASE_ROWS.json'), 'id', names)
    rows = index(audit, read(ROOT / 'test/ROWS.json'), 'patch_id', patch_ids)
    audit.check({p.stem for p in (ROOT / 'fit').glob('*.npz')} ==
                {f'{name}_{kind}' for name in names for kind in ('baseline', *FEATURES)}, 'all calibration files')
    audit.check({p.stem for p in (ROOT / 'test').glob('*.npz')} == patch_ids | {f'{name}_base' for name in names},
                'all baseline and transplant test files')

    nuisance_keys = set()
    for block in range(3):
        audit.stage = f'data block {block}'
        fit_data, fit_x = read_data(audit, 'fit', block, nuisance_keys)
        test_data, test_x = read_data(audit, 'test', block, nuisance_keys)
        fit_y, test_y = fit_data['y'], test_data['y']
        donors = donor_maps(audit, test_data, test_x)
        for model in (row for row in models if row['seed'] == block):
            ident, arm = model['id'], model['arm']
            audit.stage = ident
            w = theta[model['index']]
            audit.check(all(model[k] == fit_rows[ident][k] == base_rows[ident][k] for k in META),
                        f'{ident} calibration and baseline metadata')
            fit, _ = verify_baseline(audit, ROOT / f'fit/{ident}_baseline.npz', fit_x, w, arm, ())
            base, independent = verify_baseline(audit, ROOT / f'test/{ident}_base.npz', test_x, w, arm, capture.TAPS)
            verify_scalar(audit, test_x, w, arm, base, ident)
            fits = {}
            for kind in FEATURES:
                fits[kind], fit_scores = verify_fit(audit, ident, kind, features(fit, kind), fit_y)
                verify_metrics(audit, fit_rows[ident], metric_values(fit_scores, fit_y, f'{kind}_probe'), ident)
            base_scores = readout_scores(base, fits)
            state_fit = {t: packed_fit(load(ROOT / f'state_probes/{ident}_joint_t{t}.npz')) for t in capture.TAPS}
            state_score = {t: joint_scores(capture.transplant(base, t, 'full', donors['self']), state_fit[t])
                           for t in capture.TAPS}
            expected = native_metrics(base['counts'], test_y)
            expected.update(mean_resources=base['resources'].mean(axis=0), mean_query_counts=base['counts'].mean(axis=0))
            for kind in FEATURES:
                expected.update(metric_values(base_scores[kind], test_y, f'{kind}_probe'))
            for t, score in state_score.items():
                expected[f'joint{t}_ba'] = probe.metrics(score, test_y)['balanced_accuracy']
            verify_metrics(audit, base_rows[ident], expected, ident)
            audit.exact(expected['joint9_ba'], .5, f'{ident} identical-prefix state probe')
            for case in patch_design(arm):
                record = rows[patch_id(ident, *case)]
                verify_patch(audit, model, record, w, test_x, test_y, base, independent, donors, state_fit, state_score, fits)
        print(f'Independent readout-path audit block {block + 1}/3 complete', flush=True)


def verify_summary(audit):
    summary = read(ROOT / 'SCIENCE_SUMMARY.json')
    environment = read(ROOT / 'ENVIRONMENT.json')
    audit.check(read(ROOT / 'RUN_STATE.json')['status'] == 'complete' and summary['status'] == 'COMPLETE' and
                not (ROOT / 'SCIENCE_FAILURE.json').exists(), 'single completed scientific batch')
    audit.check((audit.baselines, audit.fits, audit.patches, audit.steps) == (72, 72, 1026, 2119680), 'audited planned scope')
    audit.check((audit.self_controls, audit.early_controls, audit.donor_controls, audit.late_prefixes, audit.scalar_models)
                == (72, 36, 216, 360, 18), 'audited control and scalar-source scope')
    expected = dict(models=36, neural_training_updates=0, probe_fits=72, baseline_captures=72, patch_conditions=1026,
                    forward_sample_steps=2119680, dense_forward_macs=330670080, cap_seconds=840)
    for key, value in expected.items():
        audit.check(summary[key] == value, f'science summary {key}')
    audit.check(0 < summary['wall_seconds'] <= 840 and summary['peak_working_set_bytes'] <= 1024**3,
                'science wall-clock and memory budgets')
    audit.check(0 < summary['artifact_bytes_at_science_end'] <= int(1.5 * 1024**3), 'science disk budget')
    audit.check(all(environment['threads'][key] == os.environ.get(key) == '1'
                    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')), 'single-thread limits')


def main():
    audit = Audit()
    try:
        audit.check(not (ROOT / 'AUDIT.json').exists(), 'audit already recorded')
        models, theta = source_locks(audit)
        verify_results(audit, models, theta)
        verify_summary(audit)
        result = dict(
            status='PASS', checks=audit.checks, models=36, neural_training_updates=0, probe_fits=audit.fits,
            baseline_captures=audit.baselines, patch_conditions=audit.patches, forward_sample_steps=audit.steps,
            self_controls=audit.self_controls, early_prefix_controls=audit.early_controls,
            donor_suffix_controls=audit.donor_controls, retained_t19_prefix_controls=audit.late_prefixes,
            source_scalar_attention_models=audit.scalar_models, source_scalar_episodes=2 * audit.scalar_models,
            max_numeric_error=audit.max_numeric_error, max_coefficient_error=audit.max_coefficient_error,
            max_normal_equation_residual=audit.max_normal_residual, wall_seconds=time.perf_counter() - audit.start,
            cpu_seconds=time.process_time() - audit.cpu, cap_seconds=300,
            scope=('Independent data, full and continued native forward, donor maps, raw pulses, '
                   'resource ledger, ridge fits and all metrics, fit-before-test locks, '
                   'recipient prefix and exact donor suffix controls, original scalar TAN checks.'),
        )
        save(ROOT / 'AUDIT.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        path = ROOT / 'AUDIT_FAILURE.json'
        if path.exists():
            path = ROOT / f'AUDIT_FAILURE_{time.time_ns()}.json'
        save(path, dict(status='FAIL', stage=audit.stage, checks=audit.checks, exception=type(error).__name__,
                        message=str(error), wall_seconds=time.perf_counter() - audit.start))
        raise


if __name__ == '__main__':
    main()
