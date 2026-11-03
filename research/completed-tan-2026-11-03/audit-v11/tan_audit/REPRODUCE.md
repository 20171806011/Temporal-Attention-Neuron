# Reproduce this audit

This public scientific copy contains tan_audit/ and a selected source snapshot under Temporal-Attention-Neuron/. It is not a verification of the unavailable earlier eight-stage archive. See ../PUBLICATION_NOTES.md for redactions, current checksums, historical hashes, and replay limitations.

Use Python3.12.14, NumPy2.3.5, SciPy1.17.0 and psutil7.2.2, the versions already installed for this run and recorded in `runtime_versions.json`. No dependency installation is performed by these scripts. In a fresh workspace, extract the package so its two directories are `tan_audit` and `Temporal-Attention-Neuron`. Do not overwrite an unrelated checkout. The main test and exact-rational reviewer locate siblings relative to themselves; the two original-task audit scripts retain their documented `Temporal-Attention-Neuron` source location.

Verify package bytes from the extraction root:

```sh
sha256sum -c SHA256SUMS.txt
```

To preserve the supplied results, copy `tan_audit` to a sibling `tan_audit_replay` before executing. Run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit_replay/variable_key_test.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit_replay/independent_variable_key_check.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit_replay/repository_boundary_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit_replay/floors/audit_floors.py
```

The scripts write only beside themselves. They never run the repository's full confirmatory workflow or smoke fixtures that overwrite historical artifacts. Expect2,560 attempted pairs,1,882 accepted (988 narrow,894 full), exact rational witnesses with gaps1/5 and6/5,1,536 native query checks with no counts above1, and exact floors25/54,1/10,3139/32000,13/9,1243/1600. Counts are adversarial-construction outcomes, not population-risk estimates.

Start with AUDIT_REPORT.md, PREREGISTRATION.md, and variable_key_review.md. Every failed primary pair remains in variable_key_results/attempts.jsonl. The initial repository-boundary audit seed typo was corrected before any trial generation and is disclosed in repo_review.md; the primary registered experiment completed on its first execution.

## Continuation included in archive version2

Start with FOLLOWUP_REPORT.md for the latest source audit and positive-control outcomes. The earlier report and outputs remain preserved. Additional completed commands, using the same installed dependencies, are:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/prior_sources/timescale-heldout/run_timescales.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/prior_sources/nonlinear-heldout/run_heldout.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/prior_sources/nonlinear-heldout/summarize_uncertainty.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/binding_control/run_control.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/binding_control/independent_review_check.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/timescale_review/independent_check.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/timescale_review/query_endpoint_supplement.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/nonlinear_review/independent_check.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/floors/native_support_recheck/check_native_support.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/floors/native_support_recheck/math_independent/rectangle_integral.py
```

Use a fresh extraction workspace for a full replay because some independent audit scripts use these absolute input paths; they must read the freshly replayed results. The scripts write only audit/source-replay output directories, not historical repository artifacts. Keep the delivered results elsewhere before replay if byte comparisons are intended. Runtime/memory observations naturally vary; scientific metrics, seeds, input ledgers and source hashes are the reproducibility targets.

The public copy retains executable `.py` sources whose original transport HTML entities were decoded; transport-only `.transmitted.txt` duplicates are omitted. They are direct-text reproductions, not historical archive-byte verification. The copied frozen models_p1b_v2.py in prior_sources has the same known SHA256 as the repository. SOURCE_PROVENANCE_RESOLVED.json describes the completed delivery scope; other unsupplied archive stages are not audited.

## Constructive interpolation control included in archive version3

Read LATEST_RESEARCH_REPORT.md and interpolation_control/REPORT.md. These add a different idealized online encoder, not a modification to original TAN. Protocol SHA256 is d1a76b593ef6887a4750847f144a8006cb671e18cbb35a0bada94af0e999b9f6. New input hash is7b65ce7549e6cd17d375cbf9c786a04eb32d3158bd2a489989567e5b97dc3f0e.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interpolation_control/theory_check.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interpolation_control/run_interpolation.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interpolation_control/review_check.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interpolation_control/range_precision_check.py
```

Expect512IID histories,18adversarial histories,1,024 dependent clean variants and14,166 paired-prediction rows. The script imports binding_control/run_control.py for the unchanged previous modified-carrier diagnostic, so retain the directory structure. Repository search evidence reflects the full inspected checkout; the package's minimal source subset cannot recreate that inventory without the same full commit, but includes the saved search ledger and relevant model/generator paths. Preserve supplied outputs before rerunning. This study does not reopen any prior fitting, parameter selection or original P1B label.

## Online precision comparison included in archive version4

Start with CURRENT_RESEARCH_REPORT.md and precision_binding/REPORT.md. The older LATEST_RESEARCH_REPORT.md remains the preserved version3 report. Primary-source comparison is in precision_binding/LITERATURE_REVIEW.md. This is one fixed 32-bit persistent-payload comparison, plus a 2-bit controller and explicitly excluded transient arithmetic/input/output resources; it is not a TAN dynamics implementation.

Preserve delivered results before replay, using a sibling audit-directory copy or a fresh extraction. Keep the frozen repository source subset beside it. Then run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/precision_binding/theory_check.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/precision_binding/run_precision.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/precision_binding/independent_review.py
```

Expect 512 fresh histories, 18 reused adversarial histories, three dependent variants per history, 1,590 input records, 6,360 arm rows, 12,720 operand outputs and 22,260 quantization events. All clean cases are valid and within the declared original ±0.5 operand tolerance. The separate local δ=0.05 metric and continuous errors distinguish the frozen coding schemes. No bit-width, range or quantizer sweep occurs.

Protocol SHA256: ae2c6180dd17eb085277153a58574970a03c413210a9db78c49ef8f8ce207db3. Input SHA256: 93fb40c3de844c59c83e93fd8d40f032fe2dc00d3795c6b0422cecace881ca16. The main script locates the generator relative to its audit directory; the independent review records its own source locations. Runtime observations can vary; inputs, integer states, metrics and scientific hashes are the comparison targets.

## Finite-time residual writer included in archive version5

Read finite_write/REPORT.md, finite_write/RESOURCE_CLARIFICATION.md and the independent REVIEW.md. This compares two fixed response/update budgets for one enriched residual mechanism. It neither modifies native TAN nor treats the prior exact-flow obstruction as a finite-error theorem. The continuous reference covers only second-stage slope evolution from the Euler-captured anchor.

Preserve delivered outputs in a separate workspace before replay. With the same installed dependencies and sibling frozen repository source subset, run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/finite_write/theory_certificates.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/finite_write/run_finite_write.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/finite_write/independent_review.py
```

Expect 512 fresh histories, 25 fixed fixtures, three dependent variants each, 1,611 inputs, 3,222 prediction rows, 6,444 operand outputs and 12 metric cells. A100 uses 100 literal second-write updates; FAST10 uses the analytically fixed 6,153 updates within the same nominal 10 ms. Both use 100 actual first-capture updates, three payload and three scratch arrays. No slope-division or closed-form endpoint replaces the writer. Full active-phase trajectories for 25 original fixtures per arm are saved as NPZ; their specified holding intervals complete the 100 ms timeline. The native-port collision and uncalibrated-capture fixture have separate artifacts.

The independent review regenerates random draws without importing the repository generator, checks all rows, snapshots, trajectory arrays, resource formulae and metrics, and verifies the exact 2,406-update minimax threshold. Runtime/memory vary; inputs, constants, model hashes and numerical outputs are the reproducibility targets. Preserve the dependency directory structure and read any reviewer source path assumptions when using a sibling replay directory.

Protocol SHA256: 01a94f178132b6b6210a8609dfcd4393c22f8c73f847cfddbc23fd410513a7a1. Input SHA256: b9c9b03937463bd519e3ada94bcec5c2c8551b48791d31641b2737dff9887100. There is no new gain, timing, noise or architecture sweep. Original P1B labels and historical outputs remain unchanged.

## Registered-interface reassessment included in archive version6

Read interface_reassessment/REPORT.md, SCOPE_CORRECTION.md, PROJECT_MAP.md, MANUSCRIPT_CONTRACT_ERRATA.md and REVIEW.md. The archive additionally includes the unchanged source documents needed to inspect the manuscript/protocol discrepancies; their hashes are in PROJECT_MAP_SOURCES.json and the package manifest.

Preserve delivered results before replay. With the same installed dependencies and sibling repository subset, run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interface_reassessment/theory_certificates.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interface_reassessment/boundary_check.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interface_reassessment/run_stratified_m1.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interface_reassessment/independent_review.py
```

Expected deterministic boundary coverage: six histories, three pairs, five physical carrier variants/eight runner models, three delays and independent role probes. There are144runner episodes and52,500 recorded physical Euler calls, with157,500 further runner-call comparisons. Positive-only equality of content is rejected as equality of the full input; the low and primed-low pairs retain exact full-path equality. M0 is explicitly outside the raw-drive collision bound.

Expected M1 reanalysis:4,000 original-generator histories,12,000 paired episode rows,24,000 role outputs and66whole/pattern/partition cells. Calibration is read unchanged from summary_p1b.json. All six unstratified M1 accuracies and mean Ecomp values match that file. The2,534-history sufficient-information oracle is evaluator-only and never changes model access. Do not interpret constructed pairs as prevalence or conditional strata as causal effects/new confirmatory labels.

Analysis-plan SHA256:8e0f71c73850097c6373ff3c1d312e5a2b74e4448f669cd0f0b185ca24213c68. Pre-execution amendment SHA256:39334648f9260fd86796c05d8e6ba10e44a589ae7a43e545fe02e6e5178e3d80. Stratified input SHA256:41d2a8c5613b6739942d60ade226b129a4133e6dc37485bde79e081322d78e23. Replay is conditional on the frozen generator and NumPy2.3.5, not a claim that unavailable historical trial bytes were recovered.

## Original-carrier localization and proposed corrections included in archive version7

Read stage_localization/REPORT.md, PREREGISTRATION.md, CORRECTION_DRAFT.md and both independent review files. The manuscript/readme patches are unapplied; the source subset includes the unchanged original files against which they were checked. Patch preview files are separate audit outputs.

Preserve delivered outputs before replay. With the same installed dependencies and the sibling frozen repository subset, run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/stage_localization/run_probe.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/stage_localization/independent_review.py
```

Expect 3,584 unique histories: 2,048 train, 512 validation, 1,024 test. All two queries/three delays remain in their history's split. There are 21,504 native query observations, 10,752 paired prediction rows with eight arms each, five fixed polynomial learners and 576 metric cells. Independent A interventions cover 96 query checkpoints/192 intervention windows. The experiment is an external supervised diagnostic of original M1, not a new neuronal implementation. All-positive subgroup metrics are descriptive; no subgroup fitting or model selection occurs.

Protocol SHA256: 8ffcda523dfc042f9d64473c361ecc39c6f69b1f9d2b2af7994bbd35dea6c1ff. Inputs SHA256: 3233ad20c7cc95c7581a9756fedb1a5669409a1fe9be5968dd341d7e8ea693ed. Fitted models SHA256: 6c51d7552cfa8554fe9db4fbdfbdedd0f47c6d3bfb4a93bcf18d82acc36cb660. These are the original-run hashes; numerical library/platform differences and ZIP timestamp metadata can change some serialized artifacts. Compare scientific inputs, allowed features, fitted numerical values, predictions and checker tolerances rather than runtime/memory or compressed NPZ metadata alone.

The independent checker resolves its audit directory from its own file path and reads the sibling repository directory. It reconstructs data and checks saved outputs rather than overwriting the primary experiment. Source hashes are mandatory. When the source snapshot omits .git, git-status provenance is explicitly unavailable; it is not represented as a verified clean checkout. The delivered check was completed against the full clean checkout. Frozen test outcomes are not for subsequent tuning.

For documentation-only regeneration, prepare_manuscript_patch.py and validate_manuscript_patch.py operate in the audit directory and check the frozen repository. They expect a real git checkout for git apply/status checks; the minimal archive source subset has no git internals. Existing PATCH_VALIDATION.json records completed checks in the full original checkout. The isolated replacement paragraphs compile with installed pdflatex; full manuscript release layout was not rebuilt.

## Original-project reassessment and Probe1 collision included in archive version8

Read project_reassessment_v8/REPORT.md, RESEARCH_AGENDA.md, P1B_CAUSAL_ACCOUNT.md, PREREGISTRATION.md and the independent theorem/numerical reviews. This version retains previous outputs and unapplied manuscript patches, adds the historical P1A ledgers needed for independent arithmetic replay, and runs one fixed original-TAN-I intervention family without fitting a decoder.

Preserve delivered primary outputs before replay. With sibling tan_audit/ and Temporal-Attention-Neuron/ directories, run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/project_reassessment_v8/theory_certificates.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/tan-mplconfig python3 tan_audit/project_reassessment_v8/run_probe1_collision.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/project_reassessment_v8/independent_review.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 tan_audit/project_reassessment_v8/p1a_ledger_check.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/project_reassessment_v8/p1a_input_field_check.py
```

Installed versions are in project_reassessment_v8/RUNTIME_VERSIONS.json: Python3.12.14, NumPy2.3.5, SciPy1.17.0, Matplotlib3.10.8. The original native module imports Matplotlib even though this study makes no figures. MPLCONFIGDIR points its optional cache to writable /tmp; without it this environment used Matplotlib's automatic /tmp fallback from the read-only home directory. No installation or protected-path change is needed.

Expected new experiment: one anchor,32corners,512seeded box pairs =545pairs/1090histories perarm; four native arms plus one t8-reset-suppressed B4 arm =5450primary trajectories/76300discrete updates. There are2725pair/model ledger rows,35saved NPZarrays and zero excludedcases. All545nativeB4 complete responses are bit-identical in the recorded implementation; the ideal whole-box equality is separately proved with exact rational exponential bounds. Analog and spike separation are reported separately for allcontrols. Metadata fields named *_euler_steps mean discrete original-TAN recurrence updates, not continuous-time Euler integration.

Protocol SHA256:d8f0b97300f7fd1d3e3b0c07cd5ca17c3cfb4b40b20efa25792e6ab3dd78df23. Input SHA256:1773c6bd6ee595fe1049b71ff6b8e34a3e7c9007f5de7f509d51d8536117fcf4. Primary script SHA256:697a6640b9cfc3cfc8959f9841ead9e88701416642614d752a3b7860bc40e725. The protocol's denominator-prefactor notation is clarified separately; no input/parameter/protocol bytes changed after execution. Runtime, compressed-container metadata and tiny scalar/vector arithmetic differences are not scientific outcome changes.

The independent checker locates the sibling repository relative to its own file path, does not import the primary simulator or production model, and keeps source-hash checks mandatory. If .git is absent in the delivered source snapshot, Git provenance is explicitly unavailable/null, not certified clean. ARCHIVE_REPLAY_REVIEW.json records a successful such replay. Preserve supplied primary results before running regeneration; independent checks only write their own audit outputs.

Expected historical P1A replay:60000condition/trial records,420000output rows,120000metric/vector query choices identical; zero hard discrepancies and soft discrepancies below1e-12. It reads the packaged compressed ledgers, not new RNG trials. Supplemental checks verifytarget fields,6705middle-key queries,and10000sharedC3/C4histories withscalednoise. It does not rerun G1 or claim60000independenthistories. Frozen repository source/documents and ledger hashes are in P1A_LEDGER_CHECKS.json and the package SHA256 manifest.


## Original noisy-step signal preservation included in archive version9

Read noise_preservation_v9/REPORT.md, PREREGISTRATION.md, REPORTING_CLARIFICATIONS.md, REVIEW.md and RESEARCH_AGENDA.md. This study retains the original signed-surprise Task B model; the rectified ablation and separate information-analysis model are not substitutes. Source nodes are safely extracted to omit unrelated import-time filesystem/plot actions, with exact segment/AST/hash verification.

The completed primary and independent commands were:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/noise_preservation_v9/run_noise_preservation.py
PYTHONDONTWRITEBYTECODE=1 python tan_audit/noise_preservation_v9/independent_review.py
```

Primary outputs are protected against accidental replacement. For regeneration, preserve the delivered data and supply a fresh output directory, for example `--output-dir /tmp/tan-noise-reproduction`. Both scripts resolve the sibling Temporal-Attention-Neuron/ source directory; TAN_REPO_ROOT can override it. The checker validates delivered primary artifacts and writes only its own review records; run it in a copied audit directory to preserve the supplied review unchanged. The source builder supports the same override, but is not needed for a replay because the exact source extraction and its manifest are already delivered.

Expected phase order:100exact original model/seed records, phase-A completion artifact, saved/hashed512fresh noise histories and uniforms, then clean/fresh native simulation and supplementary scores. Counts are2152native trajectories(100original+2048fresh+4clean),129120native updates,4096fresh observed-output rows,100original and4clean ledger rows, six2000-replicate paired fresh contrasts and one2000-replicate historical contrast. Native instrumentation also checks exact frozen-source classes at each update. No exclusions or fitted settings. Zero empirical errors and degenerate bootstrap intervals do not prove perfect population behavior.

Protocol SHA256:70a9fcebb77a223e321e34fa11eb18f1228431c3d3ef8ac40cfc1a4706eb87e1. Fresh-input NPZ SHA256:4d94d62d97f672478d62baaf859437ee2f3a1fa8e73fcc7593f41541564d45d9. Primary script SHA256:bd08b8aeef0ebdef4f69bba1e4efb8d9d4cb916800c0660f60c2bfd59c0e017f. Independent script SHA256:69a1b964f2ed04bf8cd5a05fc42de41d87dcb7cde90d59f64f850f880179c220. PRIMARY_MANIFEST.json and FRESH_INPUTS_FROZEN.json additionally record complete array schemas and raw numeric hashes. Compare scientific array values rather than runtime, provenance timestamps or compressed-container metadata.

The source snapshot excludes.git; replay must record repository provenance unavailable/null rather than claim a clean Git checkout. Source hashes remain mandatory. Original full-checkout execution/review confirmed the frozen commit and clean status. ARCHIVE_REPLAY_REVIEW.json records the separate no.git portability verification. No external package installation is needed: this continuation uses Python3.12.14 and NumPy2.3.5. Historical artifacts and all prior studies remain preserved.


## Finite-output information transfer included in archive version10

Read noise_transfer_v10/REPORT.md, PREREGISTRATION.md, POPULATION_CERTIFICATE_PROOF.md, CONTRACT_SCOPE_REVIEW.md and REVIEW.md. This uses the exact original Task B neurons and v9 balanced step/catch control, not the distinct W5 sinusoidal information-analysis model. Observations are the fixed five-step count C and already-fixed decision D. No continuous-noise density is estimated, and the count-to-decision operation is external.

Completed primary and independent commands:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/noise_transfer_v10/run_noise_transfer.py
PYTHONDONTWRITEBYTECODE=1 python tan_audit/noise_transfer_v10/independent_review.py
```

The primary refuses to replace completed output artifacts; use a fresh `--output-dir /tmp/tan-transfer-reproduction` for regeneration. Preserve the supplied v9 directory unchanged: v10 verifies its original replay, source extraction, audited instrumentation, clean traces and review hashes. Both scripts resolve sibling source/v9 directories; TAN_REPO_ROOT overrides the repository location. Run the checker in a copied audit tree to preserve the delivered review; by default it verifies primary outputs beside its own script, rather than automatically selecting a primary --output-dir elsewhere.

Expected coverage: 2,048 independent Gaussian paths, both step/catch conditions, 8,192 fresh native trajectories plus 4 clean controls, 491,760 recorded native updates, 16,384 fresh observed-law ledger rows. LIF thinning is analytically marginalized with p=529/1013, so its per-history pmfs and expected counts are not newly sampled spike trains or extra features granted to a decoder. All 52 information quantities/differences have 2,000 paired bootstrap replicates (104,000 values, 4,096,000 resampled history indices). The original late rate and resource proxies remain separate.

The joint native confidence certificate allocates .025 to four six-category histogram TV bounds and .025 to four one-sided exact miss/FPR binomial limits. It does not use the bootstrap for its population coverage, and does not apply binomial limits to fractional thinning probabilities. It binds target information and conditional path detail under the fixed balanced prior; no intrinsic denoising, energy or timing-capacity claim follows.

Protocol SHA256: 14ef73c1418b4dcea2a5a55de20391e3a8c82925bef70741ee3033ebcbfca179. Fresh-input SHA256: 9ea84952c119b8083e93d0ca7265e3cbc9d0927cb022a11cb9ade0b2c7bbfc15. Primary script SHA256: 039442feb8590ffc2e9684cbe319687060fea6ff0468b0ab0430f0cbaf964e9c. Independent script SHA256: f55d4554a29719608d0bae8cd150a8cfff60d7de8a4a681f224ccecb410b2356. Full schemas, raw numeric hashes and file hashes are in PRIMARY_MANIFEST.json and INPUTS_FROZEN.json. Installed Python3.12.14, NumPy2.3.5 and SciPy1.17.0 suffice.

ARCHIVE_REPLAY_REVIEW.json records successful isolated primary/reviewer execution without .git: mandatory source and v9 dependency checks passed, Git provenance was unavailable rather than fabricated, and all 16,159,416 array elements, 994 JSON numeric values, bootstrap/certificate values and scientific CSV bytes reproduced exactly. Container timestamps/runtime/provenance can differ; scientific arrays are the reproduction target. No copied replay outputs replaced the delivered primary or prior v9 results.


## Final temporal-correlation comparison in archive version11

Read gate_mechanism_v11/REPORT.md, PREREGISTRATION.md, REVIEW.md and ARCHIVE_REPLAY_REVIEW.json for the completed temporal-correlation comparison. The commands below document completed scientific work.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/gate_mechanism_v11/run_gate_mechanism.py
PYTHONDONTWRITEBYTECODE=1 python tan_audit/gate_mechanism_v11/independent_review.py
```

The primary preserves prior outputs: use a fresh `--output-dir /tmp/tan-gate-reproduction` for regeneration. The checker reads outputs beside its own script; use a copied audit tree to preserve delivered review records. Both resolve sibling Temporal-Attention-Neuron/ sources; TAN_REPO_ROOT can override that location. Source-only snapshots explicitly report Git unavailable, while mandatory source/v9 dependency hashes remain enforced. Preserve noise_preservation_v9/ unchanged, including its prior manifest and source extraction.

Exactly512 independent calibration paths,1024 paired iid/AR innovation histories and4 clean model/class controls were executed. Calibration uses only the pooled absolute current on512 paired step/catch paths. Gain.840938900477691 is frozen before test input generation. Ordinary ADAPT uses `.8*m+.2*x`, updated-mean contrast `(9/8)*(x-m_new)`, full-wave magnitude and common leak.8/strictthreshold1/reset0. It is not repository AdaptiveLIF. The two models create10244 total trajectories:2048 calibration,8192 test,4 clean. NativeTAN accounts for5122 trajectories and307320 updates, all checked against frozen-source behavior. The8192 test-ledger rows remain grouped into1024 independent innovations.

Seeds are2026110250 calibration,2026110251 test,2026110252 paired bootstrap. There are2000 whole-history bootstrap resamples,2048000 draw indices and424000 values for212 fixed metrics. Two primary contrasts use97.5% percentile intervals; other intervals are pointwise95%. Both primary intervals are negative. No exclusions, test calibration or threshold/model selection occurred.

Protocol SHA256:1d4a98c9c4b26583b3ab7d388659c4144c585756deebb29214137fdab600d362. Test-input SHA256:0a8824c5eb2f232722349e021c86aa239e0312b03caced61b3bd84450887d65f. Primary-script SHA256:e5435b9f48875dda21d20712d89b2b48fa8620e5b12c4073e592ab5d0544eb87. Manifests include all schemas/hashes; compare numeric arrays rather than container timestamps. No new dependencies were installed.

See ../PUBLICATION_NOTES.md for the public-copy integrity and reproducibility contract.
