# Independent v10 runtime review

Status: **PASS**.

The reviewer reused its own independently written v9 neuron recurrence and independently implemented the finite-output laws, information calculations, whole-history bootstrap, and population certificate. No primary simulator or metric helper was imported. Primary source was consulted only for output schema after the independent core was written.

Protocol SHA256: `14ef73c1418b4dcea2a5a55de20391e3a8c82925bef70741ee3033ebcbfca179`.
Independent reviewer SHA256: `f55d4554a29719608d0bae8cd150a8cfff60d7de8a4a681f224ccecb410b2356`.

Verified all8,192 fresh and4clean native trajectories, including every spike, candidate, reset membrane, current, history, surprise, attention and query; all16,384 fresh and8clean output-law/resource ledger rows; every finite pmf; all104,000 bootstrap metric values and4,096,000 bootstrap indices. Clean native traces also match the independently audited v9 clean traces. Spikes, inputs and bootstrap indices agree exactly.

Maximum absolute numerical comparison residual: `5.3304582969815328e-14`.
Maximum full native float trajectory residual: `0`.

The axis check is explicit: condition0 is step/S=1, condition1 is catch/S=0. CP miss counts use zero step counts; FPR counts use positive catch counts. Symmetric entropy agreement alone was not used to verify this mapping.

| Native arm | Fixed-detector signal lower bound (bits) | Extra signal detail upper bound (bits) | Conditional path detail lower bound (bits) |
|---|---:|---:|---:|
| TAN | 0.970645382 | 0.029354618 | 0.503421211 |
| LIF | 0.882846313 | 0.117153687 | 0.124389312 |

These three statements hold jointly for both native arms with confidence at least95% under the frozen iid history design: .025 failure budget for four six-category histogram TV bounds and .025 for four one-sided exact binomial error limits. The reviewer calculated interior binomial upper limits by monotone tail inversion rather than the primary beta-quantile call, and checked tail residuals separately. The entropy-modulus proof and zero-error treatment are in POPULATION_CERTIFICATE_PROOF.md.

The positive path-detail lower bound, if present, concerns input-path information in the onset count discarded by the external fixed binary decision. It does not identify individual noise-caused spikes or prove a native filtering mechanism. The extra signal-information upper bound describes the same count-to-decision compression. No timing entropy, energy, novel information theory, or population-perfect-performance claim follows.

All52 reported information quantities have their complete2,000-replicate descriptive paired bootstrap distributions and pointwise intervals verified. The native population certificate is separate from those intervals; it remains nondegenerate for zero observed errors. CP is not applied to fractionally integrated thinning outputs. Analytic thinning agrees with Binomial(K,529/1013); no new thinning masks or fake thinned spike trains were introduced.

Every arm satisfies the finite-output chain rules and its count-to-decision information inequalities within1e-12. Valid thinning comparisons use native LIF count→thinned count→thinned decision. No generally invalid native-binary-decision→thinned-binary-decision information ordering is assumed.

Primary artifacts unchanged: `True`. Repository unchanged: `True`.
Repository provenance: `{'available': True, 'commit': '85213f0d417d54e4f48b301ed1c0e4200ef3c421', 'status': ''}`.

Completed command:

`PYTHONDONTWRITEBYTECODE=1 python tan_audit/noise_transfer_v10/independent_review.py`

All validation attempts retain their exact script snapshot, check JSON and log. No original source, prior result, primary artifact or protocol was changed by this review. No remaining blocker.
