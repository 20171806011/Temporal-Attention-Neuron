# Resources and differences from original P1-B

| Resource | Original M1 comparator | Bounded WHQ | Simple EMA |
|---|---|---|---|
| Active persistent variables | u,A,r (3); one inert padded coordinate for envelope/noise accounting | u,A0,A1,r (4) | u,A0,A1,r (4) |
| Finite-mode encoded storage | four int32 slots, 128 bits including inert pad | four int32 slots, 128 bits | same |
| State fractional grid | 20 bits; original recurrence is thereby explicitly wrapped/modified | same | same |
| Intermediate arithmetic | float64 | float64 | float64 |
| Memory state range | A in [0,5], refractory in [0,2]; u physical source behavior | A0/A1 in [0,5], initial 2.5, mapped m; same u/r threshold | same |
| Fixed address prior | current K/q supplied; no built-in two-mode address read in M1 | two hardwired centers + bandwidth | identical |
| Content holding | source A leak15ms/u10ms | content leak2000ms | identical |
| Readout observation | two analog candidate-voltage samples; native decoder uses last; auxiliary affine uses both/q | two samples + analytic filter | identical |
| Spike sampling | recorded for diagnosis; source native count/latency NOT the primary outputs here | no spikes in reported local test | same |
| Query scheduling | two sequential 10ms cues | same, no history/state clone between queries | same |

Total time per history: 100ms encoding + delay +20ms for both queries, i.e.
120/170/320ms for delays0/50/200. First query ends after10ms; the second after20ms.
Two observed scalar samples per query require two readout registers (here float64,
128 bits transient), fixed analytic coefficients, the current scalar query held
for its cue, and an external phase clock. These resources are not counted as
persistent content modes, but are NOT free. Both new models share them. The source
native affine has two saved parameters; its auxiliary local affine has four
calibrated coefficients, is explicitly diagnostic, and is not the P1-B decoder.

Readout is physically different from observing A0/A1 directly: samples are only
from u. Full internal states/targets saved by the evaluator for retention auditing
are never sent to the decoder. Vectorized histories and pre-generated noise arrays
are experiment-runner memory, not part of the proposed neuron. The learner sees
one event/value/query at a time.

The peak memory write rate, event durations, gain, readout hardware and input noise
are matched between WHQ and EMA. Sum |dm| and integral |I_query| are reported as
control proxies, not joules or biological energy. The different nonlinear write
function has a different computational implementation cost. No energetic or
capacity equivalence to original M1, or biological implementation, is claimed.

Quantization rounds after each step. A process perturbation is applied after the
threshold/reset operation, and analog samples use pre-reset candidate voltage
(or zero when refractory), following the source deadline convention. These exact
noise-injection points are part of the frozen test, not the original P1-B protocol.
