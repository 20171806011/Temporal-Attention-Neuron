"""Read-only publication audit; does not import or run any research model."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
checks = 0


def check(condition, description):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(description)


def close(a, b, description, tolerance=1e-12):
    check(abs(float(a)-float(b)) <= tolerance, description)


def main():
    e = json.loads((HERE/'evidence.json').read_text())
    baseline = e['source_commit']
    paths = list(e['sources'])
    raw = subprocess.check_output(['git', 'ls-tree', '-r', '--full-tree', baseline, '--', *paths], cwd=ROOT, text=True)
    tree = {}
    for line in raw.splitlines():
        meta, path = line.split('\t', 1)
        mode, kind, oid = meta.split()
        check(kind == 'blob', 'source is a regular Git blob '+path)
        tree[path] = oid
    for path, record in e['sources'].items():
        data = (ROOT/path).read_bytes()
        check(len(data) == record['size_bytes'], 'source length '+path)
        check(hashlib.sha256(data).hexdigest() == record['sha256'], 'source SHA256 '+path)
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        check(tree.get(path) == blob, 'source bytes pinned in original Git snapshot '+path)

    expected_fits = {'6':80, '7':160, '8':416, '9':20, '10':40}
    expected_blocks = {'6':20, '7':20, '8':104, '9':5, '10':5}
    originals = {}
    for version, study in e['studies'].items():
        r = ROOT/'research'/f'dendritic-population-v{version}'
        source = json.loads((r/'results/TEST_SUMMARY.json').read_text())
        train = json.loads((r/'results/TRAIN_LOCK.json').read_text())
        originals[version] = source
        check(study['new_training_fits'] == len(train['rows']) == expected_fits[version], 'all trained fits retained V'+version)
        check(study['independent_paired_blocks'] == expected_blocks[version], 'independent block count V'+version)
        check(study['derived_inference_cuts'] == (10 if version == '9' else 0), 'cuts not counted as fits V'+version)
        check(study['test_model_conditions'] == len(study['test_rows']) == len(source['rows']), 'every held-out model retained V'+version)
        check(study['test_model_episode_evaluations'] == len(source['rows'])*512, 'model-episode denominator V'+version)
        check(study['unique_test_episodes'] == expected_blocks[version]*512, 'unique held-out denominator V'+version)
        check(len({x['id'] for x in study['test_rows']}) == len(source['rows']), 'no duplicate model identifier V'+version)
        for row in study['test_rows']:
            original = next(x for x in source['rows'] if x['id'] == row['id'])
            check(all(original[k] == v for k,v in row.items()), 'exact source row '+row['id'])
            score = row.get('balanced_accuracy', row['accuracy'])
            close(score, sum(row['class_recall'])/2, 'class recall mean '+row['id'])
            check(row['failed'] == (score < .9 or row['coverage'] < .9), 'frozen failure definition '+row['id'])
            close(row['coverage'], 1-row['abstention_rate'], 'coverage and abstention '+row['id'])
            close(row['abstentions']/512, row['abstention_rate'], 'abstention count '+row['id'])
            check(0 <= row['accuracy'] <= row['coverage'] <= 1, 'unconditional accuracy '+row['id'])
        for cell in study['cells']:
            keys = [k for k in ('arm','recipe','condition','h') if k in cell]
            rows = [x for x in study['test_rows'] if all(x[k] == cell[k] for k in keys)]
            check(len(rows) == cell['n'] == expected_blocks[version], 'cell retains every block '+str(keys))
            for metric in ('accuracy','balanced_accuracy','coverage'):
                if 'mean_'+metric in cell:
                    close(statistics.mean(row.get(metric, row['accuracy']) for row in rows), cell['mean_'+metric], 'cell mean '+metric)
            check(sum(r['failed'] for r in rows) == cell['failures'], 'all cell failures')
            check(cell['passes'] == len(rows)-cell['failures'], 'cell pass count')
            check(sum(r['fully_silent'] for r in rows) == cell['fully_silent'], 'all silent fits')
            check(sum(r['fully_abstaining'] for r in rows) == cell['fully_abstaining'], 'all abstaining fits')
        check(study['intervals'] == source['statistics']['intervals'], 'intervals exactly retain original procedures V'+version)
        for name, interval in study['intervals'].items():
            values = interval['paired_values']
            check(len(values) == expected_blocks[version], 'interval paired unit '+name)
            close(statistics.mean(values), interval['mean'], 'paired contrast mean '+name)
        for kind, filename in (('verification','verification.json'),('gradient_audit','gradient_audit.json')):
            actual = json.loads((r/'results'/filename).read_text())
            check(study[kind] == actual and actual['status'] == 'PASS', 'saved audit identity V'+version+' '+kind)
        check(study['verification']['training']['reconstructed_updates'] == 200*expected_fits[version], 'all Adam updates V'+version)
        check(study['verification']['training']['native_training_checkpoints'] == 9*expected_fits[version], 'all checkpoints V'+version)
        ancestry = subprocess.run(['git','merge-base','--is-ancestor',study['protocol_commit'],study['checkpoint_commit']],cwd=ROOT,capture_output=True)
        check(ancestry.returncode == 0, 'protocol precedes checkpoint V'+version)

    def value(version, seed, arm, recipe=None, condition=None, h=None, metric='accuracy'):
        rows = e['studies'][version]['test_rows']
        return next(r[metric] for r in rows if r['seed']==seed and r['arm']==arm and (recipe is None or r.get('recipe')==recipe) and (condition is None or r.get('condition')==condition) and (h is None or r.get('h')==h))

    for version in ('7','8'):
        for arm in (('DEND','POINT') if version=='7' else ('POINT',)):
            record = e['studies'][version]['intervals'][f'accuracy_{arm}_loss_lr_interaction']
            for seed, stored in enumerate(record['paired_values']):
                delta = value(version,seed,arm,'CE_0.03')-value(version,seed,arm,'CE_0.01')-value(version,seed,arm,'RATE_0.03')+value(version,seed,arm,'RATE_0.01')
                close(delta,stored,'same-seed factorial interaction')
    record=e['studies']['8']['intervals']['accuracy_POINT_loss_lr_interaction']
    sd=statistics.stdev(record['paired_values']);se=sd/math.sqrt(104)
    close(sd,record['sample_sd'],'V8 primary SD')
    close(se,record['standard_error'],'V8 primary SE')
    close(record['mean']-record['t_critical']*se,record['student_t95'][0],'V8 primary lower')
    close(record['mean']+record['t_critical']*se,record['student_t95'][1],'V8 primary upper')
    check(record['student_t95'][1] < -.05,'V8 prespecified practical-threshold interpretation')
    # Preserve the failure of the V7 simple-effect point-estimate crossover to repeat.
    check(e['studies']['7']['intervals']['accuracy_POINT_loss_at_0.03']['mean'] < 0,'V7 high-LR simple effect negative')
    check(e['studies']['8']['intervals']['accuracy_POINT_loss_at_0.03']['mean'] > 0,'V8 high-LR simple effect positive')
    for arm in ('DEND','POINT'):
        for other in ('retrained','cut'):
            record=e['studies']['9']['intervals'][f'{arm}_intact_minus_{other}_accuracy']
            for seed, stored in enumerate(record['paired_values']):
                close(value('9',seed,arm,condition='intact')-value('9',seed,arm,condition=other), stored, 'V9 paired control contrast')
    for h in (6,4,2,1):
        record=e['studies']['10']['intervals'][f'h{h}_DEND_minus_POINT_balanced_accuracy']
        for seed, stored in enumerate(record['paired_values']):
            close(value('10',seed,'DEND',h=h,metric='balanced_accuracy')-value('10',seed,'POINT',h=h,metric='balanced_accuracy'), stored, 'V10 paired BA contrast')
    low_point=next(x for x in e['studies']['10']['test_rows'] if x['id']=='s0_POINT_h1')
    check(not low_point['failed'],'minimum-size POINT passing counterexample retained')
    totals=e['totals']
    expected={'new_training_fits':716,'derived_inference_cuts':10,'test_model_conditions':726,
              'test_model_episode_evaluations':371712,'unique_test_episodes_across_distinct_studies':78848,
              'adam_updates_reconstructed':143200,'checkpoints_replayed':6444,'saved_checker_assertions':978440,
              'sampled_gradient_updates':2228,'final_local_surrogate_derivative_checks':121806}
    for name,value_ in expected.items():check(totals[name]==value_,'publication total '+name)
    for stem in ('optimization_interactions','recurrence_controls','binding_scaling'):
        for suffix in ('png','pdf','svg'):
            p=HERE/'figures'/f'{stem}.{suffix}'
            check(p.is_file() and p.stat().st_size>1000,'publication figure '+p.name)
    print(json.dumps({'status':'PASS','checks':checks,'pinned_source_files':len(paths),'source_commit':baseline,
                      'new_training_or_inference':False,'scope':'Publication source identity and saved-result arithmetic only. No experiment rerun or outside replication. Figures also require visual inspection.'},indent=2))


if __name__ == '__main__':
    main()
