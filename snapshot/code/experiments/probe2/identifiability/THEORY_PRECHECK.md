# Probe-2 Theory Precheck — why query-conditioned routing is structurally absent

Date: 2026-09-04 · Status: archived (AUDIT_FAILED, terminated negative result)

## 1. Frozen attention kernel

The Temporal Attention Neuron (TAN) attention energy at time `t` is

```
E_i = beta * Wq * Wk * S_t * x_i
```

with

```
S_t = [x_t - mu_t - epsilon]_+        (scalar surprise of the current input)
```

and weights

```
alpha_i = exp(E_i) / sum_j exp(E_j).
```

## 2. Key structural fact

At a fixed time step, `c = beta*Wq*Wk*S_t` is a **common scalar factor** for
every tap `i`:

```
E_i = c * x_i.
```

Therefore, whenever `S_t > 0`:

```
x_i > x_j   =>   E_i > E_j   =>   alpha_i > alpha_j.
```

Consequences:

- `argmax_i alpha_i = argmax_i x_i` (history taps) whenever `S_t > 0`;
- the query identity `q` enters the energy **only through the scalar
  `S_t`**, i.e. through a common multiplicative temperature/sharpness term;
- query identity can change attention *sharpness*, but cannot change the
  **relative ordering** of historical taps.

## 3. Predicted behaviours (all confirmed by the audit)

1. Same history, only query changed (`q=A -> q=B`):

   ```
   argmax_i alpha_i(q=A) = argmax_i alpha_i(q=B)      (history taps)
   ```

   => attention winner is query-independent; the counterfactual flip rate
   is expected to be ~0.

2. Attention ranking is amplitude-driven: with key amplitudes
   `K_A = 1.0 < K_B = 2.0` and value code `+/-0.25`, the B-coded event
   (1.75/2.25) dominates the A-coded event (0.75/1.25), so attention
   mechanically points at B regardless of `q`.

3. Self-attraction: with query amplitudes `x_q in {3, 4}` larger than all
   historical codes (max 2.25), the current tap is the window maximum, so
   the 5-tap softmax concentrates on the query tap itself.

4. Linear shortcuts for B2/B3: because "B is almost always the largest
   tap", a linear readout can approximate "decode the value of the maximum
   tap" which is correct on the 50% of trials with `q=B`.

## 4. Conclusion of the precheck

The frozen TAN provides *surprise-conditioned amplitude reweighting of
history*, not query-conditioned Q-K matching.  The audit was expected to
fail on routing criteria; the empirical audit confirmed this expectation
quantitatively.  This is a mechanistic boundary of the current model, not
an encoding accident and not a training failure.
