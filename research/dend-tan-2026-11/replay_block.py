"""Verify one released block by rerunning its fixed checkpoint and saved task design."""

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))

import block
import block_audit
import capacity_data
import independent_capacity_data
import shared_model


def checked_array(value, expected, name):
    np.testing.assert_array_equal(value, expected, err_msg=name)


def run(seed):
    with np.load(ROOT/'evidence'/'confirmation_weights.npz', allow_pickle=False) as archive:
        w = archive['theta'][seed]
    with np.load(ROOT/'evidence'/'confirmation_counts.npz', allow_pickle=False) as archive:
        published = {key: archive[key][seed] for key in archive.files}
    stimulus_seed = 4087020000+100*seed
    base = capacity_data.make_base(256, stimulus_seed)
    other_base = independent_capacity_data.make_base(256, stimulus_seed)
    for key in base:
        checked_array(base[key], other_base[key], f'base/{key}')
    checked_array(base['selected'], published['selected'], 'target positions')
    checked_array(base['y'], published['y'], 'target labels')
    checked_array(base['group'] % 4, published['schedule'], 'schedules')
    inputs, independent_inputs = block.build_inputs(base), block_audit.build_inputs(other_base)
    audit = block_audit.Audit(lambda disk=False: None)
    for gi, geometry in enumerate(block.GEOMETRIES):
        for hi, history in enumerate(block.HISTORIES):
            group = f'{geometry}_{history}'
            x = inputs[group]
            checked_array(x, independent_inputs[group], f'{group} inputs')
            observed, _ = shared_model.evaluate(x, w, 28)
            expected = block_audit.independent_paths(independent_inputs[group], w)
            block._verify_paths(history, x, base, observed)
            match = block._first_read(x, observed['replay'], w)
            for ri, rule in enumerate(block.RULES):
                raw = observed[rule]
                audit_label = f'{group}/{rule}'
                block_audit.compare_raw(audit, raw, expected[rule], audit_label)
                index = (gi, hi, ri)
                checked_array(raw['counts'], published['counts'][index], f'{audit_label} counts')
                checked_array(raw['resources'], published['resources'][index], f'{audit_label} events')
                checked_array(raw['reset_steps'].any(axis=1), published['triggered'][index], f'{audit_label} triggers')
                for key, shape in (('replay_steps', (256,)), ('replay_resources', (256, 4))):
                    checked_array(raw.get(key, np.zeros(shape)), published[key][index], f'{audit_label} {key}')
                compared = raw.get('evicted_comparisons', np.zeros((256, 36))).sum(axis=1)
                checked_array(compared, published['evicted_comparisons'][index], f'{audit_label} comparisons')
                for key in ('first_read_scores', 'first_read_weights'):
                    np.testing.assert_allclose(match[key], published[key][index], atol=5e-12, rtol=0.,
                                               err_msg=f'{audit_label} {key}')
    print(f'PASS block {seed}: 24 paths, 256 episodes per path, {audit.checks} independent raw checks, '
          f'maximum continuous residual {audit.max_error:.3g}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block', type=int, choices=range(12), default=0)
    run(parser.parse_args().block)
