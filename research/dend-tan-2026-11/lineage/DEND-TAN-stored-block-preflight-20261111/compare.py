"""Compare stored-block preflight outputs with sealed budget and optimization fixtures."""

from pathlib import Path
import time
import zipfile

import numpy as np

import block_audit as independent


TOLERANCE = 5e-12
EXPECTED_ISOLATED_PAIRS = 32
TRACE_EXAMPLES = 8
CANDIDATE_PACKET_SIZE = 256


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def compare_values(audit, current, saved, context):
    """Compare all logical JSON values without relaxing discrete metadata."""
    audit.check(type(current) is type(saved), f'{context}: metadata type')
    if isinstance(saved, dict):
        audit.check(set(current) == set(saved), f'{context}: exact keys')
        for key in saved:
            compare_values(audit, current[key], saved[key], f'{context}/{key}')
    elif isinstance(saved, list):
        audit.check(len(current) == len(saved), f'{context}: exact list length')
        for index, (left, right) in enumerate(zip(current, saved)):
            compare_values(audit, left, right, f'{context}/{index}')
    elif isinstance(saved, float):
        audit.close(current, saved, context, TOLERANCE)
    else:
        audit.check(current == saved, f'{context}: exact metadata value')


def untimed_rows(rows):
    return [{key: value for key, value in row.items() if not key.endswith('_seconds')} for row in rows]


def compare_trial(directory, baseline, guard, examples, reference_packet_size=None):
    """Check a complete trial against its same-size sealed budget or stored fixture."""
    started = time.perf_counter()
    directory, baseline = Path(directory), Path(baseline)
    audit = independent.Audit(guard)
    audit.check(examples in (512, 256), 'one of the two frozen preflight sizes')
    paths = {f'{group}_{rule}.npz' for group in independent.GROUPS for rule in independent.RULES}
    arrays = paths | {'base.npz'}
    for folder in (directory, baseline):
        audit.check({path.name for path in folder.glob('*.npz')} == arrays, f'{folder.name}: complete archives')

    before, after = load_arrays(baseline/'base.npz'), load_arrays(directory/'base.npz')
    audit.check(set(before) == set(after), 'fixed base archive schema')
    for key, value in before.items():
        audit.check(after[key].dtype == value.dtype and after[key].shape == value.shape, f'base/{key}: exact array type and shape')
        audit.exact(after[key], value, f'base/{key}: exact common data')
    before_w = np.load(baseline/'weights.npy', allow_pickle=False)
    after_w = np.load(directory/'weights.npy', allow_pickle=False)
    audit.check(after_w.dtype == before_w.dtype and after_w.shape == before_w.shape, 'exact common weight type and shape')
    audit.exact(after_w, before_w, 'exact common native weights and matcher')
    array_count = len(after)+1
    logical_bytes = after_w.nbytes+sum(array.nbytes for array in after.values())
    for name in sorted(paths):
        audit.guard()
        old, new = load_arrays(baseline/name), load_arrays(directory/name)
        audit.check(old['counts'].shape == (examples, 2) and old['resources'].shape == (examples, 6),
                    f'{name}: full reference native count and event scope')
        for key in independent.native.TRACES:
            audit.check(old[key].shape[:2] == (TRACE_EXAMPLES, independent.T), f'{name}/{key}: global first-eight reference traces')
        independent.compare_raw(audit, new, old, name)
        logical_bytes += sum(array.nbytes for array in new.values())
        array_count += len(new)

    old_rows, new_rows = (independent.read(folder/'ROWS.json') for folder in (baseline, directory))
    audit.check(len(old_rows) == len(new_rows) == len(paths), 'all logical scoring rows')
    compare_values(audit, untimed_rows(new_rows), untimed_rows(old_rows), 'all scores and logical row work')
    old_singles, new_singles = (independent.read(folder/'MICROBENCH.json') for folder in (baseline, directory))
    audit.check(len(old_singles) == len(new_singles) == EXPECTED_ISOLATED_PAIRS, 'all 32 isolated pairs')
    compare_values(audit, untimed_rows(new_singles), untimed_rows(old_singles), 'exact isolated selection, order and work')

    old_block, new_block = (independent.read(folder/'BLOCK.json') for folder in (baseline, directory))
    old_audit, new_audit = (independent.read(folder/'AUDIT.json') for folder in (baseline, directory))
    for block, result in ((old_block, old_audit), (new_block, new_audit)):
        audit.check(block['examples'] == result['examples'] == examples
                    and block['paths'] == result['paths'] == len(paths) and result['status'] == 'PASS',
                    f'complete audited {examples}-example block')
    old_work, new_work = old_block['work'].copy(), new_block['work'].copy()
    old_calls, new_calls = old_work.pop('shared_calls'), new_work.pop('shared_calls')
    saved_packet_size = old_audit.get('packet_size', 32)
    reference_packet_size = saved_packet_size if reference_packet_size is None else reference_packet_size
    audit.check(reference_packet_size == saved_packet_size and reference_packet_size in (32, CANDIDATE_PACKET_SIZE),
                'same packet size as the sealed reference ledger')
    if reference_packet_size == CANDIDATE_PACKET_SIZE:
        audit.check(examples == 256 and old_audit['compression'] == zipfile.ZIP_STORED, 'same-size stored optimization reference')
    audit.check(new_audit['packet_size'] == CANDIDATE_PACKET_SIZE and new_audit['compression'] == zipfile.ZIP_STORED,
                'candidate uses the fixed stored 256-example packet')
    groups = len(independent.GROUPS)
    audit.check(old_calls == groups*(examples//reference_packet_size)
                and new_calls == groups*(examples//CANDIDATE_PACKET_SIZE), 'exact reference and candidate packet-call counts')
    compare_values(audit, new_work, old_work, 'complete block logical work excluding the intended packet-count change')
    old_metrics, new_metrics = old_block['metrics'].copy(), new_block['metrics'].copy()
    old_bytes, new_bytes = old_metrics.pop('archived_bytes_before_block'), new_metrics.pop('archived_bytes_before_block')
    audit.check(old_bytes > 0 and new_bytes > 0, 'both physical archive sizes recorded')
    compare_values(audit, new_metrics, old_metrics, 'complete block non-timing metrics excluding physical file bytes')
    for key in ('work', 'retained_raw_array_bytes', 'prior_training_audit_forecast_seconds'):
        compare_values(audit, new_audit[key], old_audit[key], f'independent audit accounting/{key}')
    audit.guard(disk=True)
    return dict(status='PASS', examples=examples, paths=len(paths), checks=audit.checks,
                max_numeric_error=audit.max_error, arrays=array_count, logical_array_bytes=logical_bytes,
                isolated_pairs=len(new_singles), baseline_shared_calls=old_calls, shared_calls=new_calls,
                baseline_directory=baseline.as_posix(), baseline_packet_size=reference_packet_size,
                trace_examples_per_path=TRACE_EXAMPLES, trace_time_steps=independent.T,
                baseline_archived_bytes=old_bytes, archived_bytes=new_bytes, wall_seconds=time.perf_counter()-started)


def edge_reference(x, w, audit):
    packets = {rule: [] for rule in independent.RULES}
    for start in range(0, len(x), 8):
        audit.guard()
        replay = independent.independent_paths(x[start:start+8], w)
        for rule, arrays in replay.items():
            packets[rule].append(arrays)
    return {rule: {key: np.concatenate([packet[key] for packet in rows], axis=0) for key in rows[0]}
            for rule, rows in packets.items()}


def compare_edge(directory, baseline_fixture, edge_input, weights, guard, compression=None):
    """Check every edge trace against the sealed separate paths and independent replay."""
    started = time.perf_counter()
    directory, baseline_fixture = Path(directory), Path(baseline_fixture)
    audit = independent.Audit(guard)
    x, w = np.asarray(edge_input), np.asarray(weights)
    audit.check(x.shape == (12, independent.T, 6) and x.dtype == np.float64 and np.isfinite(x).all(), 'exact frozen edge input shape')
    audit.check(w.shape == (180,) and w.dtype == np.float64 and np.isfinite(w).all(), 'exact frozen edge weight shape')
    audit.exact(x, load_arrays(baseline_fixture/'fixture'/'edge.npz')['x'], 'same sealed edge inputs')
    audit.exact(w, load_arrays(baseline_fixture/'fixture'/'weights.npz')['edge'], 'same sealed edge weights')
    expected_names = {f'edge_{rule}.npz' for rule in independent.RULES}
    audit.check({path.name for path in directory.glob('*.npz')} == expected_names, 'all three candidate edge paths')
    if compression is not None:
        audit.check(compression in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED), 'frozen edge archive mode')

    expected = edge_reference(x, w, audit)
    array_count = logical_bytes = 0
    for rule in independent.RULES:
        path = directory/f'edge_{rule}.npz'
        raw = independent.load(audit, path, compression) if compression is not None else load_arrays(path)
        sealed = load_arrays(baseline_fixture/'evaluate'/f'edge_{rule}_separate.npz')
        independent.compare_raw(audit, raw, sealed, f'edge/{rule}/sealed')
        independent.compare_raw(audit, raw, expected[rule], f'edge/{rule}/independent')
        audit.check(all(len(array) == len(x) for array in raw.values()), f'edge/{rule}: complete all twelve trajectories')
        array_count += len(raw)
        logical_bytes += sum(array.nbytes for array in raw.values())
    audit.guard(disk=True)
    return dict(status='PASS', examples=len(x), paths=len(independent.RULES), checks=audit.checks,
                max_numeric_error=audit.max_error, arrays=array_count, logical_array_bytes=logical_bytes,
                wall_seconds=time.perf_counter()-started)
