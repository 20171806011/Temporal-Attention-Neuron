# Reproducibility record

The specification was written before v4 implementation/execution; the learner and spec were hashed before the first run. `SOURCE_PRE_RUN.sha256` still matches.

One frozen 12-request x 3-condition x 3-method comparison was run twice only to check deterministic replay. No parameter changes, additional data grid, training, or tuning. Correctness JSON was byte-identical. Runtime measurements are separate and naturally differ.

The independent-path checker was written without importing the learner; it re-evaluates exact-rational interval proofs, state transitions, address outcomes and real-label scoring from the saved records. Its reporting variable-shadowing failure and fix are preserved in SCIENTIFIC_CORRECTIONS.md; learner and numerical data were not changed. It is a same-agent audit, not a third-party independent reproduction.

The historical preservation report records 20 pre-existing files, including v1/v2/v3 archives and their members, with identical SHA256 before and after. Hash equality does not constitute rerunning those experiments.

Transient root lists and variable-precision proof computation are included in the resource ledger. No state source is claimed to be original TAN production or v11.
