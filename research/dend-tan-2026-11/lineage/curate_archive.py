"""Build a compact, sanitized historical archive. Does not import experiment code.

Usage: python curate_archive.py SOURCE_OUTPUTS PUBLIC_RESEARCH_DIRECTORY
The destination must already contain this script under lineage/.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

MD_NAMES = set('PROTOCOL.md COMPLETION_PROTOCOL.md REPORT.md INTERPRETATION.md MODEL_SCOPE.md AMENDMENTS.md CORRECTIONS.md AUDIT_NOTE.md PRECHECK_NOTE.md TIMING_RULE.md RESOURCE_BLOCKER.md IMPLEMENTATION.md PLAN.md HYPOTHESES.md'.split())
JSON_NAMES = set('AUDIT.json AUDIT_explore.json AUDIT_COMPLETE.json AUDIT_new_v2.json AUDIT_old_v2.json AUDIT_old_FAILURE.json AUDIT_REVISION.json AUDIT_NUMERIC_DIAGNOSTIC.json VERIFICATION.json VERIFY.json TIMING_AUDIT.json PREFLIGHT_VERIFICATION.json SUMMARY.json SCIENCE_SUMMARY.json TRAIN_SUMMARY.json TRAIN_AUDIT.json RUN_SUMMARY.json WORK_STATUS.json RUN_STATE.json STOPPED.json STOP_STATUS.json COMPLETION_STATE.json EVALUATION_ACCOUNTING.json PRECHECK_FAILURE_1.json PRECHECK_INITIAL_INFEASIBLE.json TIMING_INITIAL_INFEASIBLE.json TIMING.json RESOURCE_CORRECTION.json MODEL_SCOPE.json PASSIVE_CHECK.json SAVED_ANALYSIS.json SAVED_DIAGNOSIS.json SAVED_TRACE_DIAGNOSIS.json tune_FAILURE.json evaluate_FAILURE.json RESOURCE_CONTRACT.json'.split())
PROVENANCE_NAMES = set('MANIFEST.json PREFLIGHT_MANIFEST.json FREEZE.json EXPLORE_FREEZE.json COMPLETION_FREEZE.json SOURCE_LOCK.json SOURCE_AUDIT.json PROVENANCE.json SEED_REGISTRY.json HYPOTHESIS_LOCK.json DESIGN.json TASK_LOCK.json'.split())
PY_NAMES = set('amplitude_model.py analyze.py audit_completed.py audit_revised.py audit_timing.py audit.py base_data.py binding_data.py block_audit.py block.py budget_io.py capacity_data.py capacity_metrics.py capacity_model.py capture.py check_evaluator.py common.py compare.py complete_evaluation.py confirm_data.py continuation.py core_model.py core.py correct_resource_accounting.py data_check.py data.py design.py donor_map.py dynamics.py engine.py evaluate.py event_model.py finalize.py fixed_count_diagnosis.py fixture_data.py formal.py gradient_check.py guard_model.py independent_amplitude.py independent_binding_data.py independent_capacity_data.py independent_capacity.py independent_capture.py independent_confirm.py independent_data.py independent_donor.py independent_event.py independent_guard_confirm.py independent_guard.py independent_interference.py independent_key.py independent_learned_data.py independent_learned.py independent_loss.py independent_opt.py independent_probe.py independent_replay.py independent_reset.py independent_scale.py independent_stats.py independent_training.py independent_warm.py independent.py interference_model.py intervene.py key_model.py learned_data.py learned_model.py lock_hypotheses.py metrics.py model.py native_tan.py opt_model.py oracle.py passive_check.py pilot.py population_independent.py population.py precheck.py prefix.py preflight_attempt_1.py preflight.py probe.py readout.py reference.py reset_model.py run.py saved_analysis.py saved_diagnosis.py saved_trace_diagnosis.py scale_model.py seal.py shared_model.py study.py summarize.py task_data.py timing_stats.py timing.py train_model.py trainer.py training_audit.py training.py verify_study.py verify.py warm_model.py'.split())
RESULT_KEYS = set('study_status updates_per_fit fixed_learning_rate objectives conditions inference inference_limits decision primary primary_effects attention_primary_effects primary_paired secondary_paired primary_width_effects secondary_DEND_minus_POINT attention_effects secondary_effects alternative_objective_effects secondary_objective_effects effects restoration_contrasts full_minus_noatt_d0 latency_effects equivalence_supported negative_controls control_checks overall contrast_signs coverage resources dimensions times counts probe_definition null_scope claim_boundary metric_definitions gradient_diagnostic_scope diagnostic_timing objective_algebra diagnostic_timing resource_scope max_abs_l2_residual max_abs_vector_residual max_norm_residual max_vector_residual selection'.split())
COMPLETE_PACKAGES = {'DEND-TAN-evicted-key-shared-confirm-20261111', 'DEND-TAN-replay-confirm-20261111'}

# English curatorial descriptions are distinct from historical source records.
STUDIES = [
('TAN-state-study', 'State mechanisms', 'Completed', 'DEND/POINT circuit study, not original TAN attention. Branch-versus-soma adjusted transfer was 96.45 percentage points [93.52,99.26]; matched-budget comparisons remained task-specific. A delivered-event accounting correction is retained.', []),
('TAN-large-explore', 'State mechanisms', 'Completed exploration', 'Larger h=24 circuit exploration completed 32 tuning fits, 12 new fits and 144 evaluations. Candidate-entry requirements were not all met; no confirmation claim.', ['TAN-state-study']),
('TAN-memory-transfer-20261108', 'State mechanisms', 'Completed with audit amendment', 'Restricted retraining demonstrated alternative state-dependent solutions; branch retention was not universally necessary. The initial numerical audit failure and its explicit error-propagation amendment are preserved.', ['TAN-large-explore']),
('TAN-width-study-20261108', 'State mechanisms', 'Stopped at frozen budget', 'Tuning stopped after 17 complete fits and one 199-update partial fit. No final training or test generation occurred in this stage.', ['TAN-large-explore']),
('TAN-width-validation-20261108', 'State mechanisms', 'Completed through declared continuation', 'Width effects were nonmonotonic and recipe-confounded. The original evaluation timeout, 81 subsequent completion evaluations and separate completion protocol remain visible.', ['TAN-width-study-20261108']),
('TAN-width-step-validation-20261108', 'State mechanisms', 'Completed', 'Fixed-CE width-by-learning-rate comparison: h96 DEND D24 BA changed from 0% to 100% when learning rate decreased from .01 to .0025. This is a bounded optimizer intervention, not a scaling law.', ['TAN-width-validation-20261108']),
('TAN-recurrence-recovery-20261108', 'State mechanisms', 'Completed', 'All 24 frozen models and 864 conditions were retained. Recovery-window recurrence had a causal effect in the selected h96 low-rate cohort; this did not establish complete forgetting followed by recovery, attractor counts, or original TAN attention.', ['TAN-width-step-validation-20261108']),
('TAN-original-population-pilot-20261108', 'Original TAN and branch attention', 'Completed negative pilot', 'The original scalar TAN cell was integrated into a population. All 12 TAN/NOATT models produced [0,0] at every tested delay, including D0. Strict BA and coverage were zero; this is recipe failure, not universal ineffectiveness.', ['TAN-recurrence-recovery-20261108']),
('TAN-propagation-D0-20261108', 'Original TAN and branch attention', 'Stopped before science', 'User scope correction stopped this proposal before scientific fits/calibration. Only 64 fixture timing updates had occurred; it is not a scientific negative result.', ['TAN-original-population-pilot-20261108']),
('DEND-TAN-branch-pilot-20261108', 'Original TAN and branch attention', 'Completed negative pilot', 'DEND continuous branches received TAN-inspired history injection, not whole TAN cells. Both fixed-rate D24 FULL-minus-NOATT primary means were zero.', ['TAN-original-population-pilot-20261108']),
('DEND-TAN-temporal-select-20261108', 'Original TAN and branch attention', 'Completed exploration', 'FULL-minus-NOATT primary BA differences were -25.00 and -4.82 percentage points at the two fixed rates. Opposite candidate contents allowed a temporal-XOR solution; this was not general retrieval.', ['DEND-TAN-branch-pilot-20261108']),
('DEND-TAN-loss-control-20261108', 'Readout diagnostics', 'Completed negative control study', '36 fits and 14,400 updates. All eight alternative-objective primary mean differences were negative. Descriptive CE/COUNT/CECOM BA was 61.33/22.75/17.68%; count regulation increased nonzero ties.', ['DEND-TAN-temporal-select-20261108']),
('DEND-TAN-frozen-probes-20261108', 'Readout diagnostics', 'Completed frozen-model diagnosis', 'All 36 models were evaluated without neural updates. Mean t15 joint-probe BA was 83.52%, count-probe BA 63.29%, native strict BA 33.75%, and native tie-to-zero BA 57.32%. Decodability is not causal use or native performance.', ['DEND-TAN-loss-control-20261108']),
('DEND-TAN-readout-path-20261109', 'Readout diagnostics', 'Completed frozen-model intervention', 'Complete t15 counterlabel state transfer changed query rasters, counts and forced native classes in 67.66%, 50.84% and 26.91% of recipients. These are different stages and denominators, not interchangeable measures.', ['DEND-TAN-frozen-probes-20261108']),
('DEND-TAN-timing-increment-20261109', 'Readout diagnostics', 'Completed frozen-model diagnosis', 'At fixed two-channel counts, true temporal ordering outperformed shuffled ordering by 10.50 points [7.75,12.05] in the descriptive three-block comparison; 35/36 models had eligible pairs.', ['DEND-TAN-readout-path-20261109']),
('DEND-TAN-streaming-readout-20261109', 'Readout diagnostics', 'Completed frozen-model diagnosis', 'Equal-dimensional MOMENT2 versus COUNT2 mean BA difference was +0.63 points with inconsistent original-block directions. These are calibrated terminal readouts, not changed neural dynamics.', ['DEND-TAN-timing-increment-20261109']),
('DEND-TAN-late-cue-20261109', 'Attention and retrieval', 'Completed exploration', 'Independent candidate contents and a later selector: D16 FULL and NOATT BA were 65.62% and 81.77%, a -16.15-point mean difference across three training blocks.', ['DEND-TAN-temporal-select-20261108']),
('DEND-TAN-hold-paths-20261109', 'Attention and retrieval', 'Stopped before freeze', 'Priority changed before preflight, freeze, scientific data generation or scientific runs. Preserve as unexecuted setup, not negative evidence.', ['DEND-TAN-late-cue-20261109']),
('DEND-TAN-scale-attention-20261109', 'Attention and retrieval', 'Completed exploration', 'Three-block D16 SCALE-minus-SCALE_MEAN mean was +3.39 points with interval spanning zero; the comparison with FULL changed several mechanisms together.', ['DEND-TAN-late-cue-20261109']),
('DEND-TAN-scale-confirm-20261109', 'Attention and retrieval', 'Completed confirmation without joint success', '12 fresh blocks: SCALE-minus-MEAN +0.91 points [-8.79,9.11]; SCALE-minus-GAIN2 +18.49 [0.13,38.02]. The requirement that both primary differences be positive was not met.', ['DEND-TAN-scale-attention-20261109']),
('DEND-TAN-scale-retrieval-20261109', 'Attention and retrieval', 'Completed reused-weight intervention', 'Using the confirmation weights, trace transplantation transferred answers by +5.24 points [2.28,8.56], while dynamic versus mean cue/query reading had zero BA effect in every block.', ['DEND-TAN-scale-confirm-20261109']),
('DEND-TAN-scale-prefix-20261109', 'Attention and retrieval', 'Completed reused-weight negative intervention', 'Both prefix total-effect and trace-source primary BA contrasts were exactly zero in all 12 blocks. The two boundary trace sources were numerically identical; this does not negate different-content transplantation.', ['DEND-TAN-scale-confirm-20261109']),
('DEND-TAN-scale-training-20261109', 'Attention and retrieval', 'Completed training-path comparison', '12 fresh blocks: common-mean evaluation gave DYN-minus-MEAN -1.11 points [-6.84,2.15] and DYN-minus-DETACH +1.43 [-5.92,8.66]. Neither direction was established.', ['DEND-TAN-scale-prefix-20261109']),
('DEND-TAN-key-binding-20261110', 'Addressed memory', 'Completed', 'Random write order and visible discrete addresses: KEY-minus-KEY_MEAN opposite-content BA +23.18 points [10.42,38.15]; swapped-slot counterlabel pair gain was 100 points in each of 12 blocks.', ['DEND-TAN-scale-training-20261109']),
('DEND-TAN-key-amplitude-20261110', 'Addressed memory', 'Completed reused-weight intervention', 'Matched global injection norm: swapping direction increased counterlabel paired answers by 99.87 points [99.48,100]; correct direction versus matched mean gained 93.88 [90.62,96.22]. Per-branch amplitude/sign still changed.', ['DEND-TAN-key-binding-20261110']),
('DEND-TAN-learned-key-20261110', 'Addressed memory', 'Completed', '12 blocks, 36 fits: learned bilinear matching versus uniform reading gained 52.28 points [42.71,62.37]; swapped reading gained 96.48 [91.54,99.74] counterlabel paired answers. FIFO, slots and matching form remained supplied priors.', ['DEND-TAN-key-binding-20261110']),
('DEND-TAN-learned-capacity-20261111', 'Interference and repair', 'Completed reused-weight intervention', 'Under identical four writes, four-slot versus two-slot early-record BA gained 21.97 points [18.85,25.00]; equal-norm top-score reading versus soft reading gained 2.18 [0.55,4.23].', ['DEND-TAN-learned-key-20261110']),
('DEND-TAN-learned-interference-20261111', 'Interference and repair', 'Completed reused-weight intervention', 'Crossing full and early-write-deleted states localized recent-record gains to native-state replacement (+10.30 points [8.61,12.00]) and a smaller value-state effect (+0.67 [0.13,1.22]).', ['DEND-TAN-learned-capacity-20261111']),
('DEND-TAN-selective-reset-20261111', 'Interference and repair', 'Completed candidate not adopted', 'Selective reset recent-record gain +4.36 points [-5.11,10.71] failed the positive-gain requirement. Early-record noninferiority passed, but both conditions were required.', ['DEND-TAN-learned-interference-20261111']),
('DEND-TAN-warm-reference-20261111', 'Interference and repair', 'Completed candidate not adopted', 'Warm-state replacement recent-record gain +7.00 points [-4.10,13.15] failed the positive-gain requirement; a large negative tail block remained.', ['DEND-TAN-selective-reset-20261111']),
('DEND-TAN-donor-content-20261111', 'Interference and repair', 'Completed reused-weight intervention', 'With recipient input and read injection fixed, same-example versus target-bit-flipped natural donor state changed aligned count margin by 5.769 spikes [5.089,6.435]. This is a content-specific intervention, not an implementable label oracle.', ['DEND-TAN-warm-reference-20261111']),
('DEND-TAN-event-replay-20261111', 'Interference and repair', 'Completed candidate exploration', 'Visible-event full-prefix replay gained 10.71 recent-record BA points [7.88,13.80]; evicted-record change -0.33 [-1.30,0] met the -2-point bound. Reused weights required fresh-block confirmation.', ['DEND-TAN-donor-content-20261111']),
('DEND-TAN-replay-opt-20261111', 'Interference and repair', 'Completed optimization with latency negatives', 'Caching plus zero-row multiplication skipping reduced replay MACs by 44.15% but increased triggered-batch time by 4.738 ms and single-trigger time by .961 ms. Full-prefix replay was retained.', ['DEND-TAN-event-replay-20261111']),
('DEND-TAN-replay-confirm-20261111', 'Independent confirmations', 'Completed fresh-block confirmation', '12 new fits, 7,200 updates, 512 examples/block and 96 paths: recent-record gain +5.27 points [3.42,7.16], evicted-record change -.13 [-.52,0]. Both frozen criteria passed.', ['DEND-TAN-replay-opt-20261111']),
('DEND-TAN-evicted-key-20261111', 'Independent confirmations', 'Completed reused-weight candidate exploration', 'Near-key competition veto: evicted-target BA +1.17 points [.52,2.02], recent BA -.03 [-.13,0]. Weights came from the replay confirmation; this was a new candidate needing independent confirmation.', ['DEND-TAN-replay-confirm-20261111']),
('DEND-TAN-evicted-key-confirm-20261111', 'Execution engineering', 'Stopped before formal budget', 'Both original test-size budget tiers were infeasible. Zero formal fits, updates or evaluations; this directory was never silently resumed.', ['DEND-TAN-evicted-key-20261111']),
('DEND-TAN-shared-replay-fixture-20261111', 'Execution engineering', 'Completed synthetic equivalence fixture', 'Shared execution matched separate trajectories on 2,060 input/history combinations and 27 paths, including 12 hand-designed edges. No formal confirmation or accuracy endpoint.', ['DEND-TAN-evicted-key-confirm-20261111']),
('DEND-TAN-shared-block-budget-20261111', 'Execution engineering', 'Completed infeasible budget preflight', 'Both complete-block tiers exceeded frozen science/audit forecasts despite passing independent audits. No formal training or test data.', ['DEND-TAN-shared-replay-fixture-20261111']),
('DEND-TAN-block-optimization-20261111', 'Execution engineering', 'Completed synthetic optimization', 'Packet256 and uncompressed NPZ preserved logical outputs. Uncompressed archives reduced write time at about four times compressed storage. Two rounds do not estimate host-load uncertainty.', ['DEND-TAN-shared-block-budget-20261111']),
('DEND-TAN-stored-block-preflight-20261111', 'Execution engineering', 'Completed feasible budget preflight', '512-example forecasts failed; 256-example forecasts passed (science159.10 s, audit251.30 s). Both repeated timings were retained without selecting favorable load.', ['DEND-TAN-block-optimization-20261111']),
('DEND-TAN-evicted-key-shared-confirm-20261111', 'Independent confirmations', 'Completed fresh-block confirmation', '12 new fits, 7,200 updates and 288 paths: near-key evicted BA +.78125 points [.13021,1.5625], four positive/eight zero blocks; recent change zero in all blocks. This restored legacy-replay loss to original-model BA, not lost memory content.', ['DEND-TAN-evicted-key-20261111', 'DEND-TAN-stored-block-preflight-20261111']),
]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def redact_text(text):
    text = re.sub(r'(?i)(?:\\\\\?\\)?[A-Z]:[\\/][^\n\r\"\'`<>]*', 'REDACTED_LOCAL_PATH', text)
    text = re.sub(r'\b(?:TABLET|DESKTOP|LAPTOP)-[A-Z0-9-]+\b', 'REDACTED_HOST', text, flags=re.I)
    text = text.replace('\u674e\u5219\u5f90', 'REDACTED_PERSON')
    text = re.sub(r'(?:\.\./)+(?:TAN-latest|task|vendor)(?:/[^\s\"\'`<>]*)?', 'EXTERNAL_SOURCE_NOT_INCLUDED', text)
    text = re.sub(r'https://(?:github\.com/[^\s/)]+/)?[^\s)]*(?:access_token|token|api_key)=[^\s)]+', 'REDACTED_SENSITIVE_URL', text, flags=re.I)
    return text

def sanitize(value):
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {redact_text(str(key)): ('REDACTED_HOST' if str(key).lower() in ('host','hostname','machine','computername') else sanitize(item)) for key,item in value.items()}
    return value

def main(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    root = destination/'lineage'
    actual = {path.name for path in source.iterdir() if path.is_dir()}
    names = {item[0] for item in STUDIES}
    assert len(names) == len(STUDIES) and actual == names, (len(actual), len(names), actual ^ names)
    plans = {}
    for study, *_ in STUDIES:
        plans[study] = {}
        for path in (source/study).iterdir():
            if not path.is_file():
                continue
            name = path.name
            if name in MD_NAMES or name in JSON_NAMES or name in PY_NAMES:
                plans[study][name] = name
            elif name in PROVENANCE_NAMES:
                plans[study][name] = name.replace('.json', '.redacted.json')
            elif name == 'RESULTS.json' and study not in COMPLETE_PACKAGES:
                plans[study][name] = 'RESULTS.summary.json'
        if study in COMPLETE_PACKAGES:
            plans[study] = {name: target for name,target in plans[study].items()
                            if name in MD_NAMES or name in ('AUDIT.json','TRAIN_AUDIT.json','WORK_STATUS.json','SCIENCE_SUMMARY.json','RUN_SUMMARY.json','TRAIN_SUMMARY.json')}

    def markdown(text, study):
        def link(match):
            image, label, raw = match.groups()
            target = unquote(raw.strip().strip('<>')).replace('\\', '/')
            if target.startswith(('https://','http://','mailto:')):
                return f'[{label}]({target})' if not image else f'{label} (external image omitted)'
            if target.startswith('#'):
                return label
            target_study = study
            parts = target.split('/')
            for part in parts:
                if part in names:
                    target_study = part
            filename = parts[-1].split('#')[0]
            if filename in plans[target_study] and not image:
                mapped = plans[target_study][filename]
                relative = mapped if study == target_study else f'../{target_study}/{mapped}'
                return f'[{label}]({relative})'
            safe_name = Path(filename).name or 'supporting asset'
            return f'{label} (`{safe_name}`; omitted from this compact archive)'
        text = re.sub(r'(!?)\[([^\]]*)\]\((<[^>]+>|[^)]+)\)', link, text)
        text = re.sub(r'<(?:img|video|audio)\b[^>]*>', '[media omitted from compact archive]', text, flags=re.I)
        return redact_text(text)

    index = dict(scope='Compact sanitized historical evidence; no experiments rerun', studies=[], files=[])
    for study, stage, status, outcome, parents in STUDIES:
        folder = root/study
        folder.mkdir(parents=True, exist_ok=True)
        records, source_states = [], {}
        for name, public in sorted(plans[study].items()):
            original = (source/study/name).read_bytes()
            content = original.decode('utf-8-sig')
            transformations = []
            if name.endswith('.json'):
                value = json.loads(content)
                if isinstance(value, dict) and 'status' in value:
                    source_states[name] = value['status']
                if name == 'RESULTS.json':
                    kept = {key:val for key,val in value.items() if key in RESULT_KEYS}
                    for key in ('block_rows','groups','group_rows','native_groups'):
                        if key in value and len(json.dumps(value[key])) <= 120000:
                            kept[key] = value[key]
                    value = dict(archive_kind='Explicit top-level summary projection, not the complete original result',
                                 original_top_level_keys=list(value), retained_top_level_keys=list(kept),
                                 omitted_top_level_keys=[key for key in value if key not in kept], data=kept)
                    transformations.append('Large result projected to explicit summary keys; primary and inference fields retained')
                changed = sanitize(value)
                if changed != value:
                    transformations.append('Personal machine/path metadata redacted')
                content = json.dumps(changed, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
            elif name.endswith('.md'):
                content = markdown(content, study)
                transformations.append('Links to included evidence rewritten; absent assets rendered as plain filenames')
                content = '> Historical archival document. Original language preserved. Personal paths/host metadata and unavailable media links were sanitized; consult README.md and ARCHIVE_INDEX.json.\n\n'+content
            else:
                content = redact_text(content)
                if content != original.decode('utf-8-sig'):
                    transformations.append('Local path or host strings redacted; archival inspection source, not standalone reproduction')
                ast.parse(content, filename=name)
            raw = content.encode('utf-8')
            (folder/public).write_bytes(raw)
            record = dict(source=f'outputs/{study}/{name}', public=f'lineage/{study}/{public}',
                          original_sha256=digest(original), public_sha256=digest(raw), original_bytes=len(original),
                          public_bytes=len(raw), transformations=transformations)
            records.append(record)
            index['files'].append(record)
        links = '\n'.join(f'- [{record["public"].split("/")[-1]}]({record["public"].split("/")[-1]})' for record in records)
        parent_links = ', '.join(f'[{parent}](../{parent}/README.md)' for parent in parents) or 'Initial indexed local study; earlier repository work is an external dependency.'
        package = '\nThe principal confirmation evidence is provided separately in the [complete-evidence section](../../evidence/). Large results and implementation files are not duplicated here.\n' if study in COMPLETE_PACKAGES else ''
        readme = f'''# {study}\n\nStage: **{stage}**. Curatorial status: **{status}**.\n\n{outcome}\n\nThis is an English curatorial description, not a new analysis or a translation of the frozen protocol. Numerical intervals are those reported by this stage; exploratory pointwise intervals and later two-endpoint 97.5% intervals have different inferential scopes.\n\nSource archive identifier: `outputs/{study}/` (workspace-relative identifier, not a private machine path).\n\nStudy/design ancestry: {parent_links}\n\nAn ancestry link can denote design development, source reuse, or weight reuse. It does not claim that all linked studies contain independent training blocks. The frozen protocol and source locks distinguish these relationships.\n{package}\n## Included historical evidence\n\n{links}\n\n## Archive limits\n\nOnly allowlisted root-level files are included. Raw model arrays, weights, trial subdirectories, figures, external checkouts, copied third-party scalar code and execution environments are omitted. Historical reports/protocols retain their original language. `RESULTS.summary.json`, when present, explicitly lists retained and omitted keys.\n\nSanitized files are archival inspection material. They are not byte-identical originals and are not a standalone runnable reproduction. Imports and source references can depend on omitted files. Original manifest digests describe the private source archive; public-file digests are separately recorded in [the archive index](../ARCHIVE_INDEX.json). No historical digest is claimed to authenticate a redacted file.\n'''
        (folder/'README.md').write_text(readme, encoding='utf-8')
        index['studies'].append(dict(name=study, stage=stage, curatorial_status=status, outcome=outcome, parents=parents,
                                    original_status_fields=source_states, included_files=len(records)))
    (root/'ARCHIVE_INDEX.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    rows = '\n'.join(f'| [{name}](lineage/{name}/README.md) | {stage} | {status} | {outcome} |' for name,stage,status,outcome,_ in STUDIES)
    lineage = f'''# Research lineage and negative outcomes\n\nThis index covers every related local output directory present at export: **{len(STUDIES)} stages**, including completed studies, budget stops and unexecuted proposals. It is a historical evidence archive, not 41 independent replications and not a single prospectively registered research program. Each stage has its own frozen scope.\n\n## Model identities and evidence progression\n\n1. **DEND/POINT state mechanisms:** the early `TAN-` directory prefix is historical. These circuits did not implement original TAN temporal attention. State interventions, restricted retraining, width and learning-rate studies have distinct causal scopes.\n2. **Original TAN and DEND branch attention:** the original-cell population pilot failed through output silence. The subsequent DEND-plus-history-injection model retained direct continuous integration and did not replace its cells with whole TAN cells. Its null/negative attention comparisons are retained.\n3. **Readout diagnosis and attention controls:** frozen probes, state transfer and timing diagnostics used the same original three training blocks. Probe decoding, forced tie-breaking, strict native accuracy and coverage are distinct outcomes. Later two-scale attention confirmation did not meet its joint hypothesis.\n4. **Addressed memory and interference:** discrete address selection, norm-controlled interventions and learned continuous-key matching established bounded routing effects. Capacity and cross-state interventions reused trained weights. Reset and fixed warm-state candidates failed their adoption criterion before visible-event replay was developed.\n5. **Independent confirmations:** replay was confirmed on new training blocks and four held-out schedules. The new evicted-key veto was first explored on those weights, then confirmed using another fresh set of 12 training blocks. These estimates are not pooled.\n6. **Execution engineering:** equivalence fixtures, failed budgets, batching and archive-format measurements enabled the final bounded run. They are not additional scientific accuracy replications or physical-energy experiments.\n\n## All indexed stages\n\n| Source stage and evidence | Family | Recorded interpretation | Outcome and boundary |\n|---|---|---|---|\n{rows}\n\n## Interpretation rules\n\nThe independent sampling unit is the paired training/data block, not an evaluation path, intervention, test example or audit check. Reused weights do not create fresh training replication. Later two-endpoint percentile-bootstrap intervals are nominal small-sample intervals; their multiplicity scope does not cover the whole adaptive exploration sequence. A locally frozen protocol is not a claim of external preregistration.\n\nNative count ties are errors under strict accuracy, while forced binary probes break ties; coverage must remain separate. A zero empirical interval or 100% observed score is not a population certainty guarantee. State transplantation does not establish exclusive storage, universal necessity or attractor counts. The final veto restores a small legacy-replay loss and preserves recent-record gains; it does not recover missing values or increase memory capacity.\n\nCounts of MACs, emitted spikes, delivered events, arrays and wall time are distinct. No joules, ATP or biological-time claims are supported. Shared execution time is counted once per group, not once per logical output. Isolated selected-call latency is descriptive. Independent equation implementations were within the same research workflow, not external peer-reviewed replication.\n\n## Public archive policy\n\nSee [curation policy](lineage/README.md) and [file index](lineage/ARCHIVE_INDEX.json). Historical documents preserve their original language; all new curatorial descriptions are in English. Original-file and sanitized-file hashes are distinguished. Missing supporting assets are written as plain filenames rather than broken links. The [complete-evidence section](evidence/) carries the selected principal confirmation evidence without duplication here. No experiments, model evaluations, Git actions or external publication were performed by this export.\n'''
    lineage = lineage.replace('not 41 independent', f'not {len(STUDIES)} independent')
    (destination/'LINEAGE.md').write_text(lineage, encoding='utf-8')
    policy = '''# Compact historical archive\n\nThis directory contains sanitized, allowlisted root-level records from all indexed local experiments. New explanatory text is English; historical protocols/reports preserve their original language and are clearly labeled.\n\n## Inclusion and omission\n\nThe exporter uses explicit filename allowlists for protocols, reports, result summaries, independent audits, stopped/failure records, source manifests and relevant first-party research Python modules. It never traverses experiment subdirectories. Array archives, trained weights, private environments, images, PDFs, logs, copied external repositories, vendor files and `original_tan.py` are excluded. The two principal confirmation packages are linked to the separate evidence directory rather than duplicated.\n\nLarge RESULTS records are explicitly projected to primary contrasts, inference definitions and selected compact summaries; each projection lists every omitted top-level field. The export makes no claim to preserve all raw results. Original manifest/index records are redacted provenance evidence, not manifests of this public tree.\n\n## Redaction and provenance\n\nPersonal absolute local paths, personal machine hostnames and private external checkout locations are replaced by neutral markers. Path/hash changes are recorded per file. Links to copied files are rewritten; links and embeds for omitted assets become plain filenames. This can remove local reproduction commands. Historical source hashes remain hashes of the original source bytes; the public hashes describe exported bytes. Sanitization does not alter stored numerical result values except string metadata, and no scientific statistics are recomputed.\n\nThe included Python files are for inspection and provenance. They have been syntax-parsed only. Omitted modules, source roots and raw arrays mean the compact archive is not a standalone runnable reproduction. Recreating experiments requires the documented dependencies, inputs and an explicitly prepared execution environment. No code in this archive was imported or run during export.\n\n## Verification\n\nThe exporter checks coverage of all input study directories, JSON readability, Python syntax, local Markdown link targets, known private-path/host markers, obvious credential-value patterns and the total size limit. This is a bounded export review, not a mathematical guarantee that arbitrary text can contain no sensitive material. Consult [ARCHIVE_INDEX.json](ARCHIVE_INDEX.json) for source/public digests and transformations, [EXPORT_CHECKS.json](EXPORT_CHECKS.json) for the final inspection, and [the English study map](../LINEAGE.md).\n'''
    (root/'README.md').write_text(policy, encoding='utf-8')
    policy_path = root/'README.md'
    policy_text = policy_path.read_text(encoding='utf-8').replace(
        'No code in this archive was imported or run during export.',
        'No historical experiment code was imported or run during export. Only this archive curation utility performed copying, sanitization and static checks.')
    policy_path.write_text(policy_text, encoding='utf-8')
    validate(root, destination)

def validate(root, destination):
    files = [path for path in root.rglob('*') if path.is_file()] + [destination/'LINEAGE.md']
    invalid, leaks = [], []
    private = re.compile(r'(?i)[A-Z]:[\\/]|\b(?:TABLET|DESKTOP|LAPTOP)-[A-Z0-9-]+|\u674e\u5219\u5f90')
    secret = re.compile(r'(?i)(?:api[_-]?key|password|authorization|access[_-]?token)\s*[=:]\s*[\"\'][A-Za-z0-9_./+\-=]{20,}')
    indexed = {item['public']: item for item in json.loads((root/'ARCHIVE_INDEX.json').read_text(encoding='utf-8'))['files']}
    matched_hashes = 0
    for path in files:
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        relative = path.relative_to(destination).as_posix()
        if relative in indexed:
            assert digest(raw) == indexed[relative]['public_sha256'], relative
            matched_hashes += 1
        if path.name != 'curate_archive.py' and (private.search(text) or secret.search(text)):
            leaks.append(path.relative_to(destination).as_posix())
        if path.suffix == '.json':
            json.loads(text)
        if path.suffix == '.py':
            ast.parse(text)
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', text):
                target = unquote(target.strip('<>')).split('#')[0]
                if not target or target.startswith(('https://','http://','mailto:')):
                    continue
                linked = path.parent/target
                expected_final_check = linked == root/'EXPORT_CHECKS.json'
                if not linked.exists() and not expected_final_check:
                    invalid.append(dict(file=path.relative_to(destination).as_posix(), target=target))
    total = sum(path.stat().st_size for path in files)
    assert not leaks and not invalid, dict(private_matches=leaks, broken_links=invalid)
    assert matched_hashes == len(indexed), 'every indexed public file exists and matches its export digest'
    assert total < 15*1024**2, total
    checks = dict(status='PASS', indexed_studies=len(STUDIES), files_examined=len(files), bytes_examined=total,
                  indexed_public_file_hashes_verified=matched_hashes,
                  size_limit_bytes=15*1024**2, broken_local_markdown_links=invalid, private_or_credential_matches=leaks,
                  json_parse=True, python_syntax_parse=True, experiment_runs=0, github_actions=0,
                  note='Export-only inspection; original and public hashes differ where sanitization occurred. Examined size is before this check record is updated.')
    (root/'EXPORT_CHECKS.json').write_text(json.dumps(checks, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(checks), flush=True)

if __name__ == '__main__':
    if sys.argv[1] == '--validate':
        public = Path(sys.argv[2]).resolve()
        validate(public/'lineage', public)
    else:
        main(*sys.argv[1:])
