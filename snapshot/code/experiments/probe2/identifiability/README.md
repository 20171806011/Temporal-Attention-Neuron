# Probe-2 Identifiability Audit — Archive README

**This audit is a terminated negative-control / identifiability result.**
It must not be followed by parameter tuning within the current Probe-2
formulation.

## Files

| file | content |
| --- | --- |
| `probe2_identifiability_audit.py` | frozen audit script (single file, deterministic seed 20260904, N=1000, W=5, runtime < 1 s; SHA-256 `AABF8EBBB81E911E69117591F84E4F836CF59CF8C1B505DE53A8B7C1A88143C7`) |
| `THEORY_PRECHECK.md` | theoretical analysis of E_i = beta*Wq*Wk*S_t*x_i with scalar S_t |
| `AUDIT_REPORT.md` | full results (generator, Audit A/B/C1-C3/D), mechanism analysis, status |
| `AUDIT_OUTPUT.txt` | raw console output of the frozen script |

## Status

```
STATUS: TERMINATED / AUDIT_FAILED
```

## How to reproduce

```bash
# from the repository root (script imports the frozen model package):
export PYTHONPATH=$PWD/code        # Windows: $env:PYTHONPATH = code dir
python code/experiments/probe2/identifiability/probe2_identifiability_audit.py
```

The script does not write result files and does not modify any Phase
1/2/Probe-1 artifact.

## Prohibited follow-ups (no new instruction required to enforce)

- no adjustment of K_A/K_B, Delta, query amplitude, noise, beta, softmax
  temperature, Wq/Wk, classifier, feature set, or thresholds;
- no formal Probe-2 training / three-seed run;
- B4's 0.511 hit rate must not be written as routing evidence;
- hit(q=B) = 1.0 must not be written as query selection;
- this failure must not be reduced to "encoding failure" alone (see
  THEORY_PRECHECK.md: scalar S_t cannot reorder historical logits by
  query identity);
- no re-design of Probe-2 variants to make B4 pass (would turn the
  identifiability audit into post-hoc optimisation).

## Mechanistic boundary established

> TAN attention currently provides surprise-conditioned amplitude
> reweighting, not genuine Q-K routing.

Any future routing research must be a separate mechanism-extension
question ("what is the minimal mechanistic extension enabling genuine
query-conditioned selection?"), gated by the same-history / different-
query counterfactual:

```
alpha(X, q=A) != alpha(X, q=B)   with identical history X
```

with a direction of Delta C consistent with the binding target, and with
fixed-position, amplitude, query-leakage, uniform-integration, classifier,
self-attraction and random-position shortcuts re-excluded.
