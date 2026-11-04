# Independent review of final v11 round

Status: **PASS**.

Reconstructed original TAN using the reviewer's own earlier independent recurrence and implemented the frozen ADAPT baseline separately. No primary simulation or metric helper was imported. Used full-wave magnitude, pole.8, contrast scale9/8 and common leak.8/threshold1; no earlier alternative baseline was substituted.

Protocol SHA256: `1d4a98c9c4b26583b3ab7d388659c4144c585756deebb29214137fdab600d362`.
Reviewer SHA256: `fe35c0da97a61b52d3cd991d096d9542ccc27958e369b257d2a2cb2b675d3db3`.

Verified512 paired training histories,1024 held-out innovation histories in both noise regimes/classes/models, all four clean controls, all full trajectories, all per-trial/paired ledgers, all212 metric definitions, and all2,000 bootstrap rows. The gain was independently reconstructed from training currents only with math.fsum before test innovations were generated. AR initial noise is stationary, not zero. Baseline adaptation state is retained through spikes.

Independent gain: `0.84093890047769104`. Training current-L1 residual: `0`.
Largest native/adaptive trajectory residual: `0`; largest numerical comparison residual overall: `6.6613381477509392e-16`.

| Frozen primary contrast | Estimate | 97.5% paired percentile interval |
|---|---:|---|
| Delta_AR | -0.010253906 | [-0.020019531, -0.000488281] |
| Interaction | -0.076660156 | [-0.090820312, -0.062988281] |

Both required lower endpoints exceed zero: `False`.

The two primary intervals use quantiles.0125/.9875; all other intervals use.025/.975. These remain paired-bootstrap reporting criteria, not an exact finite-sample simultaneous theorem. The frozen calibration is not refit in any bootstrap. The reviewer retained misses with capped latency5 and checked conditional-on-hit latency with its hit coverage separately. Pre-onset alarms are counted once per innovation/model/regime; catch-window false alarms remain separate.

A disfavored correlation-advantage direction must be reported with the TPR/FPR, latency and current-budget tradeoffs. This comparison does not isolate TAN's gate causally, show universal inferiority/superiority, or match biological energy. Training current-L1 equality does not assert test-current equality. Exact input marginals/nominal step are preserved across regimes; realized finite input energies are allowed to differ.

Primary artifacts unchanged: `True`. Repository unchanged: `True`.
Repository provenance: `{'available': True, 'commit': '85213f0d417d54e4f48b301ed1c0e4200ef3c421', 'status': ''}`.

Completed command:

`PYTHONDONTWRITEBYTECODE=1 python tan_audit/gate_mechanism_v11/independent_review.py`

Numbered attempts preserve reviewer source snapshots, logs and every completed check. No original source, prior result, protocol or primary output was modified. No remaining blocker.

This completes the v11 separate computational review. It is AI-assisted verification, not external independent human certification.
