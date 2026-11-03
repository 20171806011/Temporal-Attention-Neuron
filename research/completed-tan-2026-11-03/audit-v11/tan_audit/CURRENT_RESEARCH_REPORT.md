# Current continuation: finite precision during online binding

The new fixed-budget experiment shows that precision while holding the first event is a real cost of the ideal two-register interpolation writer. On 512 fresh original-generator N=2 histories, fixed raw ranges give mean Ecomp 0.006997; explicit phase-dependent coding gives 0.003198. A four-field cache with the same 32 payload bits gives 0.007872. The cache has a tighter observed tail than the raw-range writer. All arms pass the original ±0.5 operand tolerance; original P1B labels remain unchanged.

The primary contrast against a diagnostic retaining the first event at binary64 precision is +0.003850 Ecomp for the raw-range writer, paired 95% percentile interval [0.003228,0.004560]. That diagnostic uses 128 persistent bits before the second event and is not budget-matched. Phase-dependent coding controls the prefix error but requires explicit scale selection and a controller.

The formula is classical two-point interpolation. The contribution is online precision and resource accounting, not a new associative-memory identity. A scoped proof also shows that the literal noninjective coefficient overwrite cannot simply be assumed to arise from the existing invertible subthreshold TAN flow. Hybrid resets or other mechanisms remain possible and need their own implementation.

Read [the full report](precision_binding/REPORT.md), [primary-source comparison](precision_binding/LITERATURE_REVIEW.md), [frozen protocol](precision_binding/PREREGISTRATION.md), [exact theory](precision_binding/THEORY.md), and [independent audit](precision_binding/REVIEW.md). All 6,360 rows, 12,720 outputs, integer states and quantization traces are saved. There was one bit budget and no tuning sweep.

The 32-bit budget covers persistent payload only, plus a 2-bit controller. Raw input ports, external onset/reset/clock, binary64 temporary arithmetic, division, switched coding, perfect retention and high-precision output remain explicit additional assumptions. This is not a biological neuron, equal-energy comparison, optimal-bit lower bound or N=3 result.

Earlier reports remain unchanged: LATEST_RESEARCH_REPORT.md is the historical version3 interpolation report; FOLLOWUP_REPORT.md covers prior source and binding-control audits; AUDIT_REPORT.md covers the initial review.  
