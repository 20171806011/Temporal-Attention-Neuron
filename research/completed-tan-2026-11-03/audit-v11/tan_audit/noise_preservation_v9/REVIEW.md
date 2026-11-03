# Independent runtime review: original Task B signal preservation

Status: **PASS**. This review regenerated the frozen inputs and recurrences without importing the primary simulator, its metric helpers, or the extracted source classes. Source ASTs were parsed for provenance only.

Frozen protocol SHA256: `70a9fcebb77a223e321e34fa11eb18f1228431c3d3ef8ac40cfc1a4706eb87e1`.
Independent reviewer SHA256: `69a1b964f2ed04bf8cd5a05fc42de41d87dcb7cde90d59f64f850f880179c220`.

The independent run checked all 100 historical records, 2,152 full native trajectories (129,120 native update steps), all 4,096 fresh native/surrogate observed trajectories, all 4,096 fresh ledger rows plus100original and4clean rows, and all six fresh plus the original paired bootstrap contrasts. All spike decisions match exactly. Floating state, attention, current, resource, and metric comparisons use the preregistered absolute tolerance 1e-12.

Maximum absolute residual across all numerical comparisons: `4.7184478546569153e-16`.

| Arm | Step hit rate | Catch false-alarm rate | Balanced accuracy | Step late rate |
|---|---:|---:|---:|---:|
| TAN | 1.000000000 | 0.000000000 | 1.000000000 | 0.272900391 |
| LIF | 1.000000000 | 0.009765625 | 0.995117188 | 0.503759766 |
| LIF_THINNED | 0.828125000 | 0.005859375 | 0.911132812 | 0.264453125 |
| ZERO_OUTPUT | 0.000000000 | 0.000000000 | 0.500000000 | 0.000000000 |

The exact clean LIF certificate is spikes at indices 16,18,...,58, total22 and20/40=.5 in the historical late window. Thus clean stimulus-driven spikes contribute to that endpoint. Native clean TAN has zero surprise, gate, and subsequent spikes from index24 after its window fills.

A zero observed miss or false-alarm count can produce a degenerate empirical percentile-bootstrap interval. Such an interval is not evidence of perfect population sensitivity or specificity; it cannot represent unobserved error patterns. All confidence intervals here are descriptive empirical whole-history resampling intervals, with no multiplicity-adjusted claim.

The fixed detector sees only whether a spike occurs in the known onset window15:20. This is a narrow diagnostic, not general denoising or an optimal decoder. Thinning uses a label-blind fixed probability529/1013 and changes observed communication, while the LIF internal trajectory remains unchanged; it does not equalize physical energy. Existing P1A findings and historical Task B labels are untouched.

Both bootstrap streams were regenerated from their frozen seeds; all2,000fresh draws and all2,000original draws match exactly. Each draw resamples independent history groups and retains every model, condition, and thinning counterpart.

Primary artifacts unchanged during review: `True`.
Repository status unchanged: `True`. Repository provenance: `{'available': True, 'commit': '85213f0d417d54e4f48b301ed1c0e4200ef3c421', 'status': ''}`.

Completed command:

`PYTHONDONTWRITEBYTECODE=1 python tan_audit/noise_preservation_v9/independent_review.py`

Validation attempts and any harness fixes are preserved in numbered reviewer attempt JSON/log/script records; see INDEPENDENT_REVIEW_CHECKS.json. No protocol or primary outcome file was modified by this review.

No remaining blocker.
