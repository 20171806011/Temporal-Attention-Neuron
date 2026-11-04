# Deterministic registered-input boundary audit

Completed2026-11-02 under ANALYSIS_PLAN.md and the pre-execution PLAN_AMENDMENT_1.md. The original frozen model, runner, generator, protocol and calibration summary were read and remain unchanged. The actual imported Euler function was wrapped only in process memory to record its calls, then restored; no repository file or historical output was written.

Executed command:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interface_reassessment/boundary_check.py > tan_audit/interface_reassessment/run_boundary.log 2>&1
```

It completed on its first run with exit0, wall time2.94s. The audit exercised six admissible histories, five physical carrier variants, all three registered delays and both independent role probes. It also called the original runner on all eight registered models:144 model-episodes and288 role outputs. A total210,000 actual Euler calls were checked:52,500 recorded physical-path calls plus157,500 runner calls whose complete arrays exactly matched the independently recorded encoding/delay/query concatenations. The same checkpoint was verified unchanged after delay/query calls.

## Where the boundary actually is

The Python `simulate_encoding` API receives raw key/value arrays and masks. The runner retains raw trial metadata and target values for scoring; M0 directly reads candidate keys/values. Those upstream arrays differ between collision histories. They are not being claimed identical or inaccessible to the whole software application.

The registered dynamical transition instead receives `I_content=k+v`, `novelty=max(0,abs(k)-1.4)`, `I_query`, current state and fixed parameters. Presentation masks, slot timing and active-event offsets additionally drive scheduling and the specified interventions. The audit included all of them. Thus the information-loss statement concerns the prescribed transduction/state-update contract, not the original window-based TAN, every possible wrapper implementation, or M0's raw cache.

M1/M2a/M2c start at `(0,0,0)`; M2b/M2d start at `(0,A0,0)` and clamp the gate. All use the same fixed calibration values from the original summary: A0=.06188672587424413, g_dec=.1326308169898006, b_dec=-.02681648355488859. M4 Count/Latency use M1 encoding. All active-event post-offset resets were checked separately: M2a/M2d reset u,r; M2c resets u,A,r. No initial state was manually precharged and no parameter was fitted to these examples.

## Three exact pairs

| Pair | Histories as ordered (key,value) events | Full input result |
|---|---|---|
| LowN2, slots0,4 | `[(.8,1),(1,-1)]` versus `[(1,.8),(.8,-.8)]` | Content `(1.8,0)`, novelty `(0,0)`, masks/timing and queries `(.8,1)` all coincide |
| PrimedN3, slots0,2,4 | `[(2,0),(.8,1.5),(1,-.5)]` versus `[(2,0),(1,1.3),(.8,-.3)]` | Content `(2,2.3,.5)`, novelty `(.6,0,0)`, masks/timing and queries `(.8,1)` all coincide |
| PositiveN2, slots0,4 | `[(1.6,1),(2,-1)]` versus `[(2,.6),(1.6,-.6)]` | Content `(2.6,1)` coincides, but novelty `(.2,.6)` versus `(.6,.2)` differs: **not a complete-input collision** |

Every exact rational key is in the registered positive key support, values are in the half-open value support, key spacings exceed.05, slots/masks agree, and each query is one of the stored keys. The corresponding binary64 TrialData objects were checked too. Query reuse is permitted and required by the registered task; roles probe independently from the same state rather than writing query values into the encoding stream.

For the lowN2 and primedN3 pairs, complete actual call arrays coincide exactly, including parameters, pre/post states, candidates, spike flags, delay and query calls. This holds for all five physical carriers. All seven native model/readout variants return identical paired answers at each delay. M0 returns the distinct correct values because its contract retains raw records.

The primed case is not merely the known first-event zero-gate erasure. In M1, the common positive primer makes the gate at the later event onsets equal to.0393397334640 and.00270916312552. Gate and content remain strictly positive throughout both ambiguous events. Their minimum gated currents are.0466621962054 and.000698571858548. The exact ideal gates are `(3/5)*(1-(149/150)^100)*(149/150)^300` and the same expression with exponent700. Floating trajectories agree to roundoff; exact rational formulas are not described as byte-identical floating states.

## Correct lower bound and positive-branch limitation

The lowN2 target vectors are `(1,-1)` and `(-.8,.8)`; primedN3 targets are `(1.5,-.5)` and `(-.3,1.3)`. Both differences have maximum norm9/5. Identical available input/query paths force a common prediction vector for a fixed native decoder. The triangle inequality therefore gives minimax Ecomp at least9/10 across either pair; the equally weighted two-history mean also has lower bound9/10. At most one history can meet the strict.5 joint operand tolerance. This is a constructed-pair bound, not a risk or prevalence estimate under the generator's continuous distribution.

No comparable ideal collision exists when all active events have positive novelty under the registered positive-key support. Full ports then invert event by event: `k=novelty+1.4`, `v=content-k`. Equal full waveforms, masks, timing and exact valid queries imply the same requested values. The positiveN2 pair above is deliberately a rejected projected-scalar collision.

In binary64, key-minus-background subtraction is exact on this positive branch under the usual Sterbenz condition, so the representable key is identified by novelty. Content-sum rounding can merge only ULP-scale value differences, bounded here by at most2^-50≈8.88e-16; literal floating injectivity of the complete pair is not asserted. Such rounding cannot explain a.5-scale target ambiguity.

The query sharpens the statement further: if at most one active event has zero novelty, the full waveform plus an exact valid query identifies the requested value. Positive events reveal their keys and values; a query not matching those keys must belong to the single remaining low event, whose value is `content-q`. This is an unrestricted information observer, not a buffer or oracle granted to M1. Multiple low events are necessary for the demonstrated ambiguity, but they do not imply every realized query/history is ambiguous.

## Saved evidence and limitations

- `exact_witnesses.json`: exact rational histories, masks/timing/port segments, target gaps, lower bounds and positive-branch inverse.
- `actual_calls.npz`:306 arrays containing complete actual calls and encoding schedules, with NaN only for semantic None values such as absent candidates/clamp settings.
- `boundary_trace_metadata.json`: full column definitions, initial/checkpoint states, query results, active-event interventions and hashes of each original-runner call stream.
- `boundary_outputs.jsonl`: all144 original-runner episode outputs and288 role values.
- `BOUNDARY_RESULTS.json`: counts, parameters, source hashes, equality checks and gate/current checks.

Main script SHA256: `ac81d140e6dbc89b9f9d76f4dc6ede5332bf707ddf42bbe8fe77adcca288d3aa`. The repository remained clean after execution. This deterministic audit neither changes original labels nor claims the separate novelty-stratified replay has completed. 
