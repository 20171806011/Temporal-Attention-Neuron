# A local write/hold/query repair, not a general binding result

## 1. What was actually changed
The pinned P1-B M1 has state `(u,A,r)`. Its gate is driven by key novelty
`max(0,abs(K)-1.4)` and is independent of value; the current during encoding is
`A*(K+V)`. In cue steps `I_content=0` and `I_query=q`: native readout cannot use
A except through its earlier influence on u. The original Euler recurrence was
read via the GitHub connector. A scalar copy and equivalent vector implementation
are in this package; this is NOT an imported full repository model or a P1-B replay.

The new local model has `(u,A0,A1,r)`. Define `m_i=2*A_i/5-1`. A0 is repurposed
as a content carrier and A1 is an additional content carrier. It is thus +1 state,
not a cost-free three-state fix. The models are evaluated only on two FIXED,
pre-wired keys k0=1.8, k1=2.2. They do not store or identify arbitrary new keys.

`g_i(k)=exp(-(k-k_i)^2/(2*sigma^2))/sum_j exp(-(k-k_j)^2/(2*sigma^2))`, sigma=.08.
The two centers and bandwidth are explicit prior/ROM, identical in the simple
baseline. They are not a per-history external address/value array. Current key,
value and the event/cue phase are input/control signals; full histories are not
inputs to the state update function.

Let e(t)=1 only during the ten-ms active event. With tauH=2000ms:

 dm_i/dt = e(t)*g_i(k)*psi(v-m_i) - (1-e(t)*g_i(k))*m_i/tauH.

Repair: psi(z)=.25*tanh(z/.25) in value/ms.
Baseline: psi(z)=clip(z/1ms,-.25,.25), an ordinary rate-limited EMA/sample-hold.
Both have identical state count, addressing, timing, write-rate ceiling, holding
leak, observation apparatus, finite precision and disturbance sequence. Smooth
bounded residual writing is NOT the original key novelty rule. It is not claimed
as a novel algorithm, nor as an RLS solve. It is a small gated two-mode sample-hold.

During cue, m leaks at tauH; u/r use the original threshold/reset rule, with current
 I_query=.5+.6*sum_i g_i(q)*m_i.
Outside cue that readout current is zero. This explicitly changes the source's
query contract and makes both content coordinates observable through u. The
refractory state is retained, but no repaired/baseline spikes occurred in this
restricted experiment; this is a subthreshold analog recall result, not spike-only
binding or internal composition. Original M1 is also given the same two voltage
samples; its historical affine decoder uses only the deadline sample, and a
separate calibration-only affine diagnostic can use both samples and q.

## 2. Bounds and the observer
For v in [-.5,.5] the deterministic memory vector remains in that interval:
psi has the sign of the error and |psi(error)|<=|error|/1ms. The .1ms Euler update
is a non-overshooting convex relaxation plus a nonnegative small leak. Query
current is therefore in [.2,.8], and u remains below its threshold 1 from the
specified initialization. In finite-noise runs hardware clipping is possible and
recorded; no such event occurred for the two new models. Absence in this finite
sample is not a universal bounded-disturbance theorem.

Let h=.1ms, lambda=1-h/10=.99, mu=1-h/2000=.99995 and S0=sum g_i(q)m_i at cue onset.
In the clean subthreshold cue recurrence:

 u_n=lambda^n*u_0+.5*(1-lambda^n)
     +.6*(h/10)*S0*(mu^n-lambda^n)/(mu-lambda).

Two voltage samples at n1=10 and n2=100 yield
 a=lambda^(n2-n1),
 D=.6*(h/10)*(mu^n2-a*mu^n1)/(mu-lambda),
 S_hat=(y2-a*y1-.5*(1-a))/D.

This cancels nuisance starting membrane u0, not a stored value, because in the
new architecture contents reside in A0/A1. D is approximately .35607, not near
zero. Independent unit-SD sample noise is amplified by sqrt(1+a^2)/D~3.02974;
bounded sample noise |e_j|<=b yields <=(1+|a|)*b/D~3.94510*b. Process noise and
quantization are additional errors. No fitting or privileged endpoint memory
access is used by this decoder. It estimates the query-selected content at the
cue START, and does not undo the previous history's retention loss.

Earlier two-sample rank/conditioning results used a different carrier and resource
contract. These gains must not be interpreted as a matched reproduction of those
earlier experiments.

## 3. What finite holding buys, and where it fails
Without writing, each content mode decays by mu per step. Earlier values must
survive more of the fixed 100ms encoding schedule. A nearly fully written value
loses approximately |v|*(1-mu^age_steps); hence the 200ms stress condition can fail
without any spike or explosion. Slow holding is a resource/parameter change, not
an unexplained capability. Initial A0=A1=2.5 is a nonzero physical bias relative to
M1's A=0; no energetic claim is made.

A wrong query deliberately addresses the other pre-wired mode. This changes the
answer but not the target used by the evaluator. It exposes the reliance on a
correct address rather than a hidden free address correction operation.

## 4. Scientific scope
This study jointly changes content storage dimension, write law, hold lifetime and
query coupling. Its gains versus original M1 cannot be assigned to one of these
changes. The equal-resource EMA isolates only smooth tanh versus clipped linear
residual writing. Their near-identical outcomes provide no distinctive TAN
advantage here. Neither circuit handles unknown continuous keys, more than two
items, variable key dictionaries, arbitrary write timing or adversarial bounded
noise. The samples of bounded noise are iid draws, not an exhaustive worst-case
certificate. Model comparisons use a local value task and local thresholds; all
original P1-B outcome labels remain unchanged.
