# TAN independent audit and variable-key test — 2026-11-02

## Outcome

The supported TAN executor actually ran the frozen repository and a new preregistered experiment. The original tree remains unchanged at `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. Two substantive results follow.

1. The original clean-task spike channel is more constrained than its nominal codebook: at most one eligible spike, so Count outputs only −2 or silence fallback0. Its relaxed population composition-error floor is **13/9**, and the corresponding latency superset floor is **1243/1600**. These strengthen the codebook accounting without changing any registered failure labels.
2. The reconstructed content-to-gate candidate with tau_m10/tau_A150 admits **same-query, ideal same-state histories with different requested values even though the all-keys local inverse works**. Thus the reported conditional-retrieval result does not establish original variable-key addressing.

## Input scope and blocker

The earlier eight-stage archive bytes were unavailable to this initial audit. Selected source text was supplied later; see FOLLOWUP_REPORT.md. This scope limitation is retained, without operational delivery diagnostics.

Consequently **the archived scripts/protocols/results had not been read or directly rerun in this initial audit**. This audit checks the frozen repository and reconstructs only the supplied candidate definition. In particular, the prior .008093 sensor-only error, noise experiments, and full resource ledgers remained unverified at this stage. Later selected-source results are reported separately in FOLLOWUP_REPORT.md.

## Original task: independent audit

The actual per-history score is `(abs(eA+eB)+abs(eA-eB))/2=max(abs(eA),abs(eB))`. Independent exact rational integration reproduces the permissive alphabet floors 25/54 for Count and 3139/32000 for20 latency levels plus zero. PreregV2.6 already anticipated quantization and first-event erasure; neither is presented as a new discovery.

Original encoding ends with100 silent steps. Its cue-onset voltage is below `.99^100=.3660323413`, refractory state is zero, and clean query drive is at most2.5. The earliest threshold crossing is step36. After reset,20 refractory steps plus at least51 integration steps put the earliest second spike at107, beyond the99 eligible cue steps. Consequently Count only delivers −2, with0 used on abstention. Latency uses a subset of bins7–19 plus fallback0. Exact oracle integration gives:

| Original-task relaxation | Population E_comp lower bound |
|---|---:|
| Count {-2,0} | 13/9 =1.444444444 |
| Latency bins7–19 plus0 | 1243/1600 =.776875 |

Because abstention is not successful delivery, native Count joint delivery is at most1/36 under independent Uniform[-3,3] targets and strict±.50 tolerance. These are population statements, not deterministic finite-sample thresholds. They require frozen membrane dynamics, final silence, clean bounded query and zero content during cue; they must not be transferred to modified probes/time constants.

Fresh validation:256 histories,1,536 role queries across all delays,842 silence and694 one-spike; none above one. The conservative envelope continued120 steps spikes at36 and107. Selected original fixtures G1.0, G1.1, G1.5, G1.6a, G1.6b, G1.7 passed across the audit scripts. The old-A current rule, pre-reset/refractory readout and step100 exclusion are preserved. Exact metric identity was checked on9,409 rational pairs, with262,144-pair Monte Carlo validation per alphabet. Five frozen source hashes match the scientific closure table.

README's population-circuit necessity language exceeds the closure report, which explicitly declines a universal necessity claim. No repository text was edited. Details and source locations are in `repo_review.md` and `floors/REPORT.md`.

## Preregistered connection to variable-key retrieval

Protocol SHA256: `b7e8b21cb042512d0180c992c11520a391d7c574a4059f7d175225096a1d4cac`, saved before anchor generation. No training, optimizer, parameter search, or target-based root selection occurred.

Candidate encoding uses the repository Euler step with tau_m10,tau_A150 and active gate drive `max(0,abs(k)-1.4)+.1*(v+3)`. It retains the original five-slot timeline and state count. Test N=2, all10 possible chronological position pairs,64 anchors per pair in each of two value domains. For each anchor, apply both fixed queried-value shifts ±.2 on[-.5,.5] or ±1.2 on[-3,3]. Solve analytically for the second event's key/value to preserve ideal final u,A while holding the queried first key fixed. Every attempted pair, including rejection reasons, is saved.

| Domain | Attempts | Accepted | Max state residual | Smallest pre-reset threshold margin |
|---|---:|---:|---:|---:|
| Narrow[-.5,.5], target gap .2 |1,280|988|2.50e−16|.78639|
| Full[-3,3], target gap1.2 |1,280|894|3.33e−16|.57360|

Every position pair had accepted witnesses. Acceptance required legal value/key domains and spacing, positive-novelty branch, no spikes/clipping during encoding, refractory0, finite denominators, state residual≤1e−12 and diagnostic observation residual≤1e−11. The reconstruction wrapper exactly matches native encoding on64 fresh original-generator controls at eta0,tau_A15.

Accounting is exhaustive:2 domains ×10 chronological slot pairs ×64 anchors ×2 fixed signed shifts =2,560 attempts. The two shifts share an anchor; they are not independent trials. Narrow:988 accepted,277 rejected first for value-domain violation and15 for key-domain/spacing violation. Full:894 accepted,268 value-domain rejections and118 key-domain/spacing rejections. No attempt reached a denominator, branch, state-residual, or observation-residual rejection. Reasons are recorded in first-failure order (value-domain is checked before key-domain), so categories do not imply no other constraint would also fail. No redraw follows a failed transformation; only anchor key-spacing rejection sampling redraws as preregistered. All2,560 entries are in `variable_key_results/attempts.jsonl`, with cell totals in `summary.json`.

Example from the first full-domain accepted pair, same active slots0/1 and queried key `1.9249058855408148`:

| Quantity | History1 | History2 |
|---|---:|---:|
| Queried value |−2.8047912145471425|−1.6047912145471426|
| Other key |2.342028154804246|2.2460265618385553|
| Other value |1.97404267441867|1.883897316851479|
| Encoded u |.00020519717693189004|.00020519717693189004|
| Encoded A |.07749656384770424|.07749656384770427|
| Encoded r |0|0|

The float histories are **not literally identical states**. Independent120-digit reconstruction of the first witness in each domain gives state residual5e−121. A separate reviewer then derived a100-step block recurrence using exact Fraction arithmetic, without importing the primary test, and reconstructed witnesses with **exact rational u,A equality**, target separations1/5 and6/5, and rigorous whole-history membrane upper bounds .148511 and .533470. These rational witnesses are distinct from the rounded JSON histories. No impossibility claim is made about an arbitrary decoder exploiting floating-point bit differences.

For positive-novelty, no-reset branches, let `s_i=k_i+v_i`, `D_i=k_i-1.4+.1(v_i+3)`. The exact recurrence has `A=a1*D1+a2*D2` and `u=b11*D1*s1+(b12*D1+b22*D2)*s2`. After the fixed target-value shift, solve `D2'=(A-a1*D1')/a2`, then `s2'=(u-b11*D1'*s1')/(b12*D1'+b22*D2')`, finally `k2'=(D2'-.1*s2'+1.1)/.9`, `v2'=s2'-k2'`. These identities construct ideal equal states. The simulation verifies that the branch and task-domain restrictions hold with substantial margins.

On equal-weight members of such an ideal pair, any deterministic decoder receiving only current(state,q) has queried-value mean absolute error at least half the gap (.1 narrow,.6 full), and cannot deliver both members at strict tolerance .05 or .5 respectively. Deterministic future probes driven by that same state and q cannot separate them. This is an adversarial existence obstruction within the original task support, **not an IID population-error estimate**. It is specific to this reconstructed candidate, N=2, known timing and the no-reset branch; no general neuron impossibility is inferred.

## Resource and leakage accounting

The diagnostic all-keys inverse is given **the full analog final state u,A plus both event keys and known timing**. It is not a native readout and not an inverse of only the two probe samples. It selected a root only when exactly one solution lay in the specified value bounds; all3,764 accepted-history evaluations had unique in-domain roots, no abstentions, and max-operand error<.05. This confirms conditional invertibility on this selected witness set, not heldout generalization. Truth was used for evaluation and adversarial construction only, never for selecting an inverse root.

Native cue diagnostics use the original A-blind content0 recurrence. The separate .1-content100-step probe is an explicit extra resource and also receives q; some probes spike, so the samples alone are not claimed sufficient for inversion. The matching native/probe observations are indistinguishability cross-checks only. Changing a distractor key changes the privileged inverse's supplied information; this is why that inverse can recover different target values from ideal equal states.

## Reproduction and next decision

Completed primary command, exit0:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/variable_key_test.py > tan_audit/variable_key_run.log 2>&1
```

Other completed checks are `repository_boundary_audit.py` and `floors/audit_floors.py`. Python3.12.14, NumPy2.3.5, SciPy1.17.0 were installed already; no paid compute, dependency installation or credentials were used. Source inputs and all results are separate from the repository. The package includes a minimal frozen source subset plus file hashes and replay instructions.

The independent reviewer checked all2,560 ledger entries and all1,882 accepted constraints, the script/protocol hashes, algebra, branches, and inverse information contract. No material primary defect was found; `variable_key_review.md` records the readout qualifications and `independent_variable_key_check.py/.json` preserves the exact-rational cross-check.

The located information gap is concrete: on this two-event family, q already supplies K1 and timing/order is fixed, yet the decoder lacks K2. Supplying that one additional key, together with the privileged full state and known order, distinguishes these constructed histories and makes their unique in-domain inversion possible. This is a sufficient side-information control on this selected family, not a theorem that one extra scalar is the universal minimum. The original random-position task additionally needs the association of keys with event positions; N=3 is outside this test's sufficiency claim.

The next decisive positive control should test binding information explicitly on fresh **original-generator N=2 full-domain histories with random positions**, before adding noise or tuning timescales: retain the fixed candidate and compare its declared full-state inverse with (a) correct ordered key/slot metadata captured during encoding, (b) counterfactual swapped key/slot metadata, and (c) the exact M0 key–value buffer as an end-to-end oracle. Query and value swaps must be handled without revealing target values to the inverse. Freeze256 fresh histories, both role queries, unique-root/branch-validation abstention and strict±.5 delivery plus unconditional max-operand error before running. Report every root ambiguity or abstention; do not force100% by clipping or true-value root selection. Positive-control success would show that retained binding metadata repairs this interface on the tested support; failure with correct metadata would identify an additional global-inversion/observability bottleneck. The analog/full-state resource must remain explicit; original Count cannot be expected to cross its impossible codebook threshold.

This proposed control is not yet executed or preregistered as a new study. First review the missing final-stage source to reconcile its exact inverse/probe/noise contract with this independent counterexample; if the intended claim is two-probe retrieval, replay that exact observation interface rather than granting direct A. Further tuning of a known-all-keys inverse alone cannot resolve single-query addressing. No architecture change, publication, remote mutation, or additional experiment was silently launched. The running managed environment and local files are left available for continued work.
