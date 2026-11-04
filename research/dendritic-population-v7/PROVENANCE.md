# V7 provenance and execution gates

- Baseline completed V6: `75a595f3c0c5aa3f50d7700b8c142906aca442b0`.
- Initial protocol/source commit: `ec1ca90`.
- Final pre-data freeze: `8ad0b69e65074563929dff905acc0067837e3778`. The only follow-on change before any scientific data was removing a trailing blank line from the independent verifier and updating its freeze hash/time; no executable behavior or design changed.
- Checkpoint gate: `041693b7297b888ed31ac78f8d5677408669f1fe`, containing all160 final training weights and traces before any held-out generation.
- Final result commit is the commit containing this document and MANIFEST.json; use git log for the full hash (a file cannot contain its own enclosing commit hash without recursion).

The scientific sources, seed schedule and contrast vectors are locked by FREEZE.json. The runner creates its output directory exclusively. The evaluator checks the training lock and refuses a second TEST_STARTED marker. All final weights, including failures, are evaluated. Guard files supplement, rather than replace, the documented commit order.

Single CPU process per scientific stage, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1, existing Python/NumPy environment. Protocol and checkpoint commits precede their respective scientific data generations. Generated reports and plots are descriptive transformations of saved results and do not retrain or retest models.

The original equations and generator are byte-identical to V6. SOURCE_AUDIT.json records earlier-manifest checks and the fresh seed namespace scan. PRESERVATION.json records the final check. The local branch is research/dendritic-population-v7. No push, merge, external message, paid compute, package installation or subsequent experimental phase is part of V7.

Research direction and mandatory native firing were proposed by Li Zexu. Protocol, code adaptation, execution, numerical verification, figures and report draft were AI-assisted. A second agent reviewed design and a separately written checker recomputed native equations, datasets, Adam reconstruction and statistics within the same workflow. This is not external independent replication or peer review.
