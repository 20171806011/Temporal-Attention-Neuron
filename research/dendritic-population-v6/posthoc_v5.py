"""Post hoc V5 checkpoint reconstruction; no training or new data generation.

Uses only saved V5 B DEND trajectories and saved final test outputs. Replays
training inputs at a sparse, explicitly documented set of optimizer checkpoints.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
import time
import numpy as np

R = Path(__file__).resolve().parent
V5 = R.parent / 'dendritic-population-v5'
O = V5 / 'results'
CHECKPOINTS = (0, 1, 5, 10, 20, 50, 100, 150, 200)
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('v5_posthoc_model', V5 / 'model.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    a = np.asarray(values)
    return dict(min=float(a.min()), q05=float(np.quantile(a, .05)),
                median=float(np.median(a)), mean=float(a.mean()),
                q95=float(np.quantile(a, .95)), max=float(a.max()))


def summarize(v, y):
    c = v['counts']; pred = v['prediction']; z = v['spikes']; cand = v['candidates']
    margin = c[np.arange(len(y)), y] - c[np.arange(len(y)), 1-y]
    output_margin = cand[:, 16:, 18:] - .5
    cm = np.zeros((2, 3), dtype=int)
    for truth, p in zip(y, pred):
        cm[int(truth), int(p) if p >= 0 else 2] += 1
    accuracy = float(np.mean(pred == y)); coverage = float(np.mean(pred >= 0))
    return dict(episodes=len(y), accuracy=accuracy, coverage=coverage,
        failed=bool(accuracy < .9 or coverage < .9),
        abstention_rate=float(np.mean(pred < 0)),
        silence_rate=float(np.mean(np.all(c == 0, axis=1))),
        tied_nonzero_rate=float(np.mean((c[:, 0] == c[:, 1]) & (c[:, 0] > 0))),
        both_saturated_rate=float(np.mean(np.all(c == 8, axis=1))),
        recall=[float(np.mean(pred[y == k] == k)) for k in (0, 1)],
        confusion=cm.tolist(), signed_count_margin=distribution(margin),
        signed_count_margin_by_class=[distribution(margin[y == k]) for k in (0, 1)],
        count_margin_positive=float(np.mean(margin > 0)),
        count_margin_zero=float(np.mean(margin == 0)),
        count_margin_negative=float(np.mean(margin < 0)),
        hidden_spikes_per_episode_all24=float(z[:, :, 12:18].sum((1, 2)).mean()),
        hidden_spikes_per_episode_readout8=float(z[:, 16:, 12:18].sum((1, 2)).mean()),
        hidden_per_cell_spikes_all24=z[:, :, 12:18].sum(1).mean(0).tolist(),
        output_spikes_per_episode_all24=float(z[:, :, 18:].sum((1, 2)).mean()),
        output_spikes_per_episode_readout8=float(c.sum(1).mean()),
        output_per_cell_counts=c.mean(0).tolist(),
        output_candidate_minus_threshold_readout=distribution(output_margin),
        output_candidate_minus_threshold_by_output=[distribution(output_margin[:, :, k]) for k in (0, 1)],
        output_candidate_near_threshold_fraction=float(np.mean(np.abs(output_margin) <= .01)),
        hidden_candidate_minus_threshold_readout=distribution(cand[:, 16:, 12:18]-.5))


def main():
    start = time.perf_counter()
    pre = json.loads((O / 'PREPARE_LOCK.json').read_text())
    fin = json.loads((O / 'FINAL_LOCK.json').read_text())
    test = json.loads((O / 'TEST_SUMMARY.json').read_text())
    source_files = {str(p.relative_to(V5)): sha(p) for p in
                    (V5/'model.py', O/'PREPARE_LOCK.json', O/'FINAL_LOCK.json', O/'TEST_SUMMARY.json')}
    rows = []
    for r in fin['rows']:
        if r['arm'] != 'DEND':
            continue
        assert (r['loss'], r['lr']) in [('RATE', .01), ('CE', .03)]
        data = np.load(O/r['data_file'], allow_pickle=False)
        tr = np.load(O/r['trace'], allow_pickle=False)
        theta = np.load(O/r['initial_file'], allow_pickle=False).copy()
        m = np.zeros(176); vv = np.zeros(176)
        snapshots = []; data_grad_norms = []
        for k in range(201):
            if k in CHECKPOINTS:
                value = model.metrics(data['x'], data['y'], theta, 'DEND')
                summary = summarize(value, data['y'])
                summary.update(updates=k, objective=float(model.loss_counts(value['counts'],data['y'],r['loss'])[0]+.0001*np.mean(theta**2)),
                               theta_sha256=hashlib.sha256(theta.tobytes()).hexdigest(),
                               parameters_at_bounds=int(np.sum(np.abs(theta) == 2)))
                snapshots.append(summary)
            if k == 200:
                break
            clipped = tr['clipped_gradients'][k]
            norm = tr['history'][k, 1]
            raw = clipped * max(1, norm)
            data_grad_norms.append(float(np.linalg.norm(raw-.0002*theta/176)))
            m = .9*m+.1*clipped; vv=.999*vv+.001*clipped*clipped
            theta = np.clip(theta-r['lr']*(m/(1-.9**(k+1)))/(np.sqrt(vv/(1-.999**(k+1)))+1e-8), -2, 2)
        final = np.load(O/r['file'], allow_pickle=False)
        assert np.array_equal(theta, final), r['id']
        ts = next(a for a in test['rows'] if (a['phase'],a['seed'],a['arm'],a['role'],a['mode']) == ('B',r['seed'],'DEND',r['role'],'intact'))
        testfile = O/ts['file']; saved = np.load(testfile, allow_pickle=False)
        testdata = O/f"test_B_s{r['seed']}.npz"; td = np.load(testdata, allow_pickle=False)
        final_test = summarize(saved, td['y'])
        for metric in ['accuracy','coverage']:
            assert final_test[metric] == ts[metric]
        assert final_test['silence_rate'] == ts['silent_fraction']
        first_full_silent = next((s['updates'] for s in snapshots if s['silence_rate'] == 1), None)
        row = dict(id=r['id'], seed=r['seed'], role=r['role'], loss=r['loss'],lr=r['lr'],
                   initial_seed=r['initial_seed'], training_seed=int(data['train_seed']),
                   reconstructed_final_exact=True, checkpoint_updates=list(CHECKPOINTS),
                   training_snapshots=snapshots, final_test=final_test,
                   first_sampled_all_silent_training_checkpoint=first_full_silent,
                   training_loss_history=tr['history'][:,0].tolist(),
                   raw_surrogate_gradient_norm_history=tr['history'][:,1].tolist(),
                   update_norm_history=tr['history'][:,2].tolist(),
                   inferred_data_surrogate_gradient_norm_history=data_grad_norms,
                   last20_raw_surrogate_gradient_norm=distribution(tr['history'][-20:,1]),
                   last20_data_surrogate_gradient_norm=distribution(data_grad_norms[-20:]),
                   clipped_update_fraction=float(np.mean(tr['history'][:,1]>1)),
                   final_train_minus_test_accuracy=snapshots[-1]['accuracy']-final_test['accuracy'])
        rows.append(row)
        for name in (r['data_file'],r['trace'],r['initial_file'],r['file'],ts['file'],testdata.name):
            source_files['results/'+name] = sha(O/name)
    assert len(rows) == 20
    selected_dev = [r for r in pre['rows'] if r['stage']=='search' and r['arm']=='DEND' and (r['loss'],r['lr'])==('CE',.03)]
    assert len(selected_dev)==5
    result = dict(status='PASS', analysis_type='POST HOC, descriptive, no causal attribution',
        source_completed_commit='73c1bff6eebf0cb98502f19f8b08d15e2a261182',
        source_files_sha256=source_files, no_new_training_or_data=True,
        final_test_reused_saved_outputs=True,
        scope='All ten V5 B DEND seed pairs, fixed RATE/.01 and selected CE/.03; no POINT post hoc analysis here.',
        definitions={'checkpoint':'after named number of Adam updates; zero is initialization',
                     'margin':'native correct-output count minus wrong-output count; zero abstains',
                     'candidate_margin':'prethreshold candidate minus .5; >0 emits a spike',
                     'silence':'both output counts zero over steps16–23; not absence of all hidden activity',
                     'gradient':'saved surrogate gradient, not derivative of the hard spike objective',
                     'data_gradient_norm':'saved raw gradient reconstructed from clipped gradient and saved norm, minus known L2 regularizer gradient'},
        selected_development_validation_accuracy=[r['validation']['accuracy'] for r in selected_dev],
        rows=rows, reconstructed_trajectories=len(rows), training_forward_replays=len(rows)*len(CHECKPOINTS),
        training_episode_replays=len(rows)*len(CHECKPOINTS)*128,
        unavailable=['No parameter intervention or alternative optimization replay was performed.',
                     'Intermediate training activity was recomputed only at nine checkpoints, not every update; full loss/norm history exists.',
                     'No intermediate held-out predictions or gradients were saved; final held-out arrays are reused.',
                     'Observational trajectories cannot determine whether a particular state, bias, recurrent path, loss, learning rate, or surrogate rule caused collapse.'],
        seconds=time.perf_counter()-start)
    (R/'posthoc_v5.json').write_text(json.dumps(result,indent=2)+'\n')
    write_report(result)
    print(json.dumps({k:result[k] for k in ['status','reconstructed_trajectories','training_forward_replays','seconds']},indent=2))


def write_report(result):
    rows=result['rows'];by={(r['seed'],r['role']):r for r in rows}
    lines=['# V5 failures inspected before V6: post hoc descriptive analysis','',
        '**This is a post hoc audit of already observed V5 results.** It generated no new data, trained no new models, and makes no causal claim. It uses all ten final B DEND seeds for CE/.03 and their matched RATE/.01 controls, including successful seeds; it is not an independently tuned new comparison.','',
        'The 20 complete Adam trajectories were reconstructed from saved clipped gradients and exactly reproduce every final weight array. Native forward replay uses the frozen V5 implementation on the saved 128-episode training sets at updates 0, 1, 5, 10, 20, 50, 100, 150 and 200. Final 512-episode test outputs are read from the saved V5 arrays, not regenerated or reevaluated. Thus these observations do not create additional independent experiments.','',
        '## Final training versus held-out behavior','',
        'All accuracies count ties and silence as errors. A failure is accuracy <90% or coverage <90%. The table reports each training seed, not episode-pooled pseudo-replicates.','',
        '| Seed | RATE train acc | RATE test acc | CE train acc | CE test acc | CE train coverage | CE train silence | CE test silence |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in range(10):
        f=by[s,'fixed'];c=by[s,'selected'];ft=f['training_snapshots'][-1];ct=c['training_snapshots'][-1]
        vals=[ft['accuracy'],f['final_test']['accuracy'],ct['accuracy'],c['final_test']['accuracy'],ct['coverage'],ct['silence_rate'],c['final_test']['silence_rate']]
        lines.append('| '+str(s)+' | '+' | '.join(f'{100*v:.4f}%' for v in vals)+' |')
    lines += ['', '## Failed CE/.03 seeds and observed checkpoints','',
        'Counts below concern the native output readout at steps16–23. Candidate margins are prethreshold output values minus .5, so a nonpositive margin cannot emit an output spike at that step. Hidden activity is separately reported; output silence must not be called whole-network silence.','',
        '| Seed | Updates | Train acc | Coverage | Silence | Mean signed count margin | Hidden spikes / episode, all24 | Output spikes / episode, all24 | Max output candidate margin, readout |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in (5,7,9):
        for q in by[s,'selected']['training_snapshots']:
            lines.append(f"| {s} | {q['updates']} | {q['accuracy']*100:.2f}% | {q['coverage']*100:.2f}% | {q['silence_rate']*100:.2f}% | {q['signed_count_margin']['mean']:.4f} | {q['hidden_spikes_per_episode_all24']:.4f} | {q['output_spikes_per_episode_all24']:.4f} | {q['output_candidate_minus_threshold_readout']['max']:.6g} |")
    lines += ['', '## Final native margins and activity for all CE/.03 seeds','',
        '| Seed | Train count margin mean [min,max] | Test count margin mean [min,max] | Train hidden spikes / episode, readout8 | Test hidden spikes / episode, readout8 | Train output candidate margin max | Test output candidate margin max | Last20 data surrogate gradient norm median |',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for s in range(10):
        r=by[s,'selected'];a=r['training_snapshots'][-1];b=r['final_test'];am=a['signed_count_margin'];bm=b['signed_count_margin']
        lines.append(f"| {s} | {am['mean']:.4f} [{am['min']:.0f},{am['max']:.0f}] | {bm['mean']:.4f} [{bm['min']:.0f},{bm['max']:.0f}] | {a['hidden_spikes_per_episode_readout8']:.4f} | {b['hidden_spikes_per_episode_readout8']:.4f} | {a['output_candidate_minus_threshold_readout']['max']:.6g} | {b['output_candidate_minus_threshold_readout']['max']:.6g} | {r['last20_data_surrogate_gradient_norm']['median']:.6g} |")
    lines += ['', '## Interpretation and unavailable evidence','',
        'For CE/.03 seeds 5, 7 and 9, the final training accuracies are respectively 50%, 0% and 0%, exactly matching their held-out accuracies. Thus the three observed failures already exist on the training examples; they are not a collapse occurring only on held-out examples. The other seven CE seeds fit all training examples and score 99.4141–100% on their saved tests. These descriptions concern this observed cohort, not a universal generalization guarantee.','',
        'Seed 7 is fully output-silent by the sampled checkpoint after 5 updates and at every later sampled checkpoint. Seed 9 is fully readout-silent by checkpoint 10 and at all later sampled checkpoints; at checkpoint 10 it still has a small amount of output activity outside the readout window. Seed 5 is also fully readout-silent at checkpoints 10, 20, 50 and 100, then partially recovers to 25% training accuracy at 150 and 50% at 200. This temporary collapse would be missed by looking only at its final 50% score. Sparse activity checkpoints do not locate the exact transition update or exclude unsampled excursions.','',
        'The silent final seeds 7/9 retain hidden activity: respectively 16.0/30.10 hidden spikes per training episode during the eight readout steps. Their maximum output candidate-minus-threshold margins are −0.6067/−2.0798 on training data, with similar held-out margins. Their last 20 median data-surrogate gradient norms are approximately 9.13e-6/5.08e-5, small but nonzero; this does not prove a zero-gradient absorbing state. In contrast, RATE/.01 reaches 98.4375% training and 99.4141% test accuracy for seed 5, and 100% training / 99.6094% test for seed 9. Its matched seed 7 also fails at 75% training and test accuracy, so RATE does not rescue every problematic initialization in this cohort.','',
        'The five development seeds used to select DEND CE/.03 all had 100% validation accuracy. The B seed cohort was retrained from new data and initializations. Failure to transfer that development-selected recipe across training seeds is distinct from a train/test generalization gap within one fitted seed. No final test was used for retraining or selection in this audit.','',
        'Both loss and learning rate differ between CE/.03 and RATE/.01. Their paired observational contrast cannot assign a failure to CE alone, learning rate alone, a specific recurrent connection, the surrogate gradient, a particular threshold margin or representational capacity. Even a temporally preceding change is not a causal intervention. The gradient summaries describe the prescribed surrogate rule. They are not exact hard-spike derivatives.','',
        'Loss and gradient/update norms are available at all 200 preupdate steps. Activity and margins here are available at only nine reconstructed training checkpoints; statements about an onset must say “by the sampled checkpoint” and must not assert an exact collapse update. Intermediate held-out activity was not recorded and is not recreated. The JSON retains all sampled metrics, per-class margins and recalls, full saved loss/norm sequences, source hashes and exact-reconstruction checks.','',
        'This audit is internal to the same AI-assisted workflow. It is not external independent replication or peer review, and it does not add to V5’s ten independently trained seed pairs.','',
        f"Runtime for reconstruction and replay: {result['seconds']:.3f} seconds. {result['training_forward_replays']} training-batch forward replays, {result['training_episode_replays']} training episode replays; 20 saved final test arrays inspected. Full numerical details: [posthoc_v5.json](posthoc_v5.json)."]
    (R/'POSTHOC_V5.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
