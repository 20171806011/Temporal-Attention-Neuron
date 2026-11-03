# Independent repository audit (2026-11-02)

Repository: `Temporal-Attention-Neuron`, clean initial working tree at
`85213f0d417d54e4f48b301ed1c0e4200ef3c421`. The source audit did not modify repository files or historical results.

## Main finding: the clean-task Count alphabet is only {-2, 0}

The generic refractory capacity of five spikes is a permissive upper bound, not
the accessible capacity of the original clean P1B task. At encoding end the
carrier has always undergone the final 100 zero-current ISI steps (model source
lines 177–189). Since a non-reset membrane is strictly below threshold 1, its
cue-onset upper bound is `u0 < (1-.1/10)^100 = .3660323412732292`; any refractory
period only lowers it. The 10 ms gap also guarantees `r=0`. Nonnegative silence
delays do not increase this upper bound. During cue, content current is zero,
so A does not affect membrane evolution (lines 249–254).

For clean queries `q<=2.5`, a threshold from initial u is reached no faster than
the upper-drive recurrence `u_n = 2.5 + (u-2.5)*.99^n`. From the conservative
u0 bound, the first threshold is at step 36. From reset zero, a new threshold
needs 51 integration steps and 20 refractory steps first. Thus the second spike
cannot precede step `36+20+51=107`, outside the 99 eligible cue steps. Original
Count output can only be -2 (one spike) or 0 (silence fallback). The bounded
boundary test obtains spikes [36,107] when continued for 120 steps.

Fresh check: 128 histories each for N=2 and N=3, all three registered delays,
two roles each, continuous RandomState per N (seeds 2026110202, 2026110203).
All 1,536 role queries obeyed this bound: 842 silence, 694 one-spike. This is
a numerical cross-check of a model/task-specific proof, not a historical replay.
The generic fixture for counts 2–5 uses q=3.5,7,15,40 outside the clean task's
[.3,2.5] domain (`test_p1b_g1_fixtures.py:640–648`); those fixtures correctly
exercise implementation behavior but do not establish in-task code utilization.

For independent Uniform[-3,3] targets, allowing an oracle to choose freely
between {-2,0} gives scalar nearest-symbol error CDF

```
F(t) = 2t/3,            0 <= t <= 1
     = (t+3)/6,         1 <= t <= 3
```

The actual paired composition metric is the maximum operand error, so its
relaxed population floor is integral[1-F(t)^2] dt = 13/9. This is a stronger
task-reachability floor than 25/54 for the permissive full five-level Count
alphabet. It is neither a finite-sample deterministic threshold nor a change
to the registered .05 failure rule. It cannot be applied unchanged to modified
time constants, new content probes, noisy out-of-domain queries, or other models.
Under the original non-abstention rule the only deliverable symbol is -2;
accordingly any individual successful target must lie in (-2.5,-1.5), and joint
delivery population probability is at most 1/36 under independent target values.

Evidence: `repository_boundary_audit.py`, `native_boundary_check.json`,
`native_boundary_trials.csv`, `native_selected_fixtures.log` beside this report.
The four selected original fixtures (G1.1, G1.5, G1.6b, G1.7) all pass. The script
deliberately avoids original smoke fixtures that overwrite repository outputs.
An initial audit script invocation stopped before generating trials because a
12-digit seed exceeded RandomState's uint32 range; fixed seeds above were used
for the completed run. NumPy 2.3.5 and SciPy 1.17.0 were already installed.

## Source and inference checks

- `p1_experiments/task_generator.py:58–115`: keys are rejection sampled for
  spacing .05, randomly assigned among candidates; values are independent uniform
  [-3,3]; event positions are random; targets are distinct; clean query is only
  the selected key. `seed` in a trial is metadata; RNG state is supplied by the
  caller. `runner_p1b.py:37–54` uses a continuous RandomState for each N.
- `runner_p1b.py:153–159`: Ecomp averages absolute add and subtract errors. For
  operand errors eA,eB this equals max(|eA|,|eB|), not average operand MAE.
- `models_p1b_v2.py:82–107`: old A drives current membrane; returned spike and
  pre-reset candidate have distinct meaning. Lines 249–267 implement readout
  at step100 using the pre-reset candidate unless refractory; lines273 onward
  exclude spike100. This intentional difference follows preregV2.6.
- PreregV2.6 sections5.3,6,8.1 already anticipate quantization, first-event
  erasure, and a subthreshold state-collision fixture. Treating these as newly
  discovered would misstate prior work. Its collision fixture uses wK=wV=.1,
  not the native unit weights; preserve this qualification.
- README line37 claims necessity of distributed multichannel circuits. The
  final scientific closure lines16,106–109 explicitly does not establish such
  necessity or universal impossibility. The closure's narrower wording is
  supported; README wording exceeds the demonstrated result.
- Raw SHA256 hashes of task_generator, models_p1b_v2, runner_p1b, fixture, and
  preregV2.6 match the historical closure table. This verifies frozen source
  identity, not complete experimental provenance or a historical target ledger.

## Most decisive bridge to variable-key addressing

The known-all-keys inverse used for conditional retrieval has privileged side
information beyond the original per-role decoder contract. Original role A and
role B are independent probes from the same state; neither role receives the
other role's query or any candidate-key array. Supplying both queries together
must be explicitly marked as an additional resource, even for N=2.

For the modified gate, keep target key K1 and q=K1 fixed and retain fixed timing.
Vary distractor key K2, and solve for compensating values V1,V2 that give the
same encoded (u,A,r). On an unclipped nonspiking branch, with η=.1,
`D_i = max(0,K_i-1.4)+.1(V_i+3)` and `s_i=K_i+V_i`, final gate is linear
in D while membrane is quadratic through D*s. In the K_i>1.4 branch, write
`A=a1*D1+a2*D2` and `u=b11*D1*s1+b12*D1*s2+b22*D2*s2`. For a preregistered
perturbed V1', solve D2' from the first equation and s2' from the second, then
`K2'=(D2'-.1*s2'+1.1)/.9`, `V2'=s2'-K2'`.

Validate original domain, key spacing, all nonspiking/clipping branch conditions,
denominators, target separation, and full recurrence. Report analytic ideal-state
identity separately from tiny floating-point residuals. If a valid collision is
found with target separation above twice tolerance, every deterministic
state-and-current-query-only decoder fails at least one member of that pair;
additional repeated state-only probes cannot restore erased information. This
is an existence obstruction for this architecture and resource contract, not
a population failure rate or universal neuron theorem. Even granting known
timing strengthens the negative test: unknown positions are not needed to
produce the obstruction. If no valid pair is found within a fixed candidate
budget, report inconclusive rather than claiming injectivity.
