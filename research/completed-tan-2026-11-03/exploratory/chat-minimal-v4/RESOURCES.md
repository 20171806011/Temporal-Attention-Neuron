# v4 resource ledger

## Persistent state: data words, not Python heap size

Capacity L=3. Persistent values are signed-int64 fixed-point codes. p=24 or p=40 is the number of fractional bits, not total bits per word. Both conditions use 64-bit words, with different dynamic ranges. All recorded states fit. `snapshot()` includes every mutable learner memory field.

| Resource | Redundant/common | Minimal/common | Explicit table |
|---|---:|---:|---:|
| key payload | m0..m6: 7 words | m1..m3: 3 words | 3 anchors |
| value payload | b0..b3: 4 words | b0..b2: 3 words | 3 values |
| count | 1 | 1 | 1 |
| mu,beta,sigma | 3 | 3 | sigma only: 1 |
| total mutable words | **15** | **10** | **8** |
| total mutable bits | **960** | **640** | **512** |

General formulas: redundant 3L+6 words; minimal 2L+4; table 2L+2. Core minimal payload equals table's 2L, but total minimal state costs two additional budget words. No claim of an information-theoretic minimum for arbitrary encodings.

The table's value quantization error is the fixed known h/2, so it does not need a dynamic beta word for this direct-overwrite model. Exact-reference computations have growing rational numerators/denominators and are not equal-bit competitors.

## Transaction/control accounting

A non-destructive write retains the original state and stages a complete candidate: additional 15/10/8 data words. The running Python implementation also creates a byte-for-byte pre-write snapshot to assert refusal invariance: another 15/10/8 word-equivalent bytes. Thus original + candidate + rollback-check bytes amount to 45/30/24 word equivalents before arithmetic workspace. This is an implementation accounting; an alternative transactional hardware implementation could organize logs differently, but none is claimed here.

The read-only program/configuration also contains L,p,h,Delta,epsilon,value/key ranges, representation descriptors and root-search limits. These are explicitly outside the above mutable-state counts, not free memory. Each request provides observed x, value v and error bound zeta. Here these are exact Fraction values, shared between methods; this is not an end-to-end fixed-bit analogue-input implementation.

The evaluator keeps true canonical keys, anchors, requested/latest labels and logs. They are never passed into write/query. They are experimental ground truth, not learner capacity. A full retained root/value certificate in external logs is not reused as hidden learner memory.

## Transient decoded roots and interval workspace

No persistent root list is stored for a moment learner. A view reconstructs roots every time. The following are distinct mathematical workspace objects in the implementation, with N<=L:

* midpoint Newton polynomial: N+1 rational coefficients;
* interval Newton polynomial: N+1 intervals (2N+2 rational endpoints);
* floating root proposal: polynomial/root arrays and library-owned numerical workspace;
* certified root centers: N rationals; root bounds: N intervals (2N endpoints);
* endpoint sign tests retained for audit: 2N intervals (4N endpoints);
* interval Lagrange coefficients: N^2 intervals (2N^2 endpoints);
* center Lagrange coefficients: N^2 rationals;
* reconstructed point values and bounds: N predictions, N errors, N value intervals (4N rationals/endpoints).

The Lagrange matrices alone therefore require **3N^2 rational slots**: 27 slots at N=3. A completed audit view retains 13N+2 rational slots across centers, roots, coefficients, endpoint tests, values/errors/value intervals: 41 slots at N=3. During candidate certification the pre-write view remains live while the candidate view and its matrices are built; these objects are additional to the candidate data words. The initial polynomial/proposal intermediates, Fraction arithmetic temporaries, copied decode vectors, list references and interval constructor temporaries also occupy memory. The exact Python peak heap was not measured; these counts are named mathematical workspace, not a false exact heap total.

The basic reconstruction is O(N^2) workspace in this implementation. Minimal payload deletion does not remove this workspace. The explicit table scans N anchors using O(N) address intervals (which could be streamed in a different implementation); it needs no polynomial/root/Lagrange reconstruction.

Instrumented exact-interval operand maxima in this run were **315 numerator bits and 326 denominator bits**. Those are maxima of observed operands, not the total working memory, a proven worst-case bit bound, or a fixed-width physical circuit. Instrumentation and returned proof records themselves cost memory. NumPy library scratch space and Python interpreter/object overhead are not reported as zero; no hardware-area or energy conclusion is drawn.

## Time measured on this host, first execution

These are 12 writes for each condition, not a throughput estimate for biological hardware. All proposals, interval checks and candidate copies are included in write time; scorer queries are excluded from these particular totals. Methods were executed in fixed order without benchmarking randomization. Values vary by run.

| Condition | Redundant/common write total | Minimal/common write total | Table write total |
|---|---:|---:|---:|
| separated q24 | 45.596 ms | 40.904 ms | 0.908 ms |
| clustered q24 | 70.807 ms | 65.492 ms | 0.751 ms |
| clustered q40 | 91.254 ms | 95.564 ms | 0.812 ms |

The minimal method spent respectively 26.658, 52.094 and 80.887 ms on root reconstruction/verification within writes (21, 23 and 21 root calls, including empty-state calls); value-certificate work was separately timed in `timing_v4.json`. Table root time is not applicable because it has no root reconstruction.

The moment methods perform identical root/value certification. The small timing difference between them is not reliable evidence of algorithmic speedup. Second-execution timing is also retained; correctness JSON, excluding times, was byte-identical. No energy measurement, hardware match, or complete fair-total-cost advantage is asserted.

## Arithmetic complexity note

Newton coefficients take O(N^2) scalar operations. This implementation builds all Lagrange polynomials naively with O(N^3) scalar arithmetic; root proposal has its own library work. Bit-operation costs grow with Fraction sizes. Neither the O(N^2) workspace label nor the 2L persistent payload implies O(L) reconstruction time.
