# Source-contract errata: Paper2 P1-B insertion

2026-11-02. Direct comparison of frozen source text with Protocol V2.6 and executable code. This is a proposed correction ledger, not an edit to the repository or a new experimental result.

The paper's P1-B paragraph mixes a different dynamical model into otherwise matching historical result numbers. The registered protocol and actual executable govern the input-boundary audit; do not use the manuscript's inconsistent equations as its contract.

| Paper2 source location | Current text/meaning | Registered executable contract | Correction needed |
|---|---|---|---|
| `snapshot/paper2/sections/s8_composition.tex:165–173` | u driven by I alone; A driven by sigma_A(u); r driven by A*u; r called a surprise-weighted carrier; readout g*r at query arrival. | Novelty drives A; old A multiplies content in u; r is a 2 ms refractory countdown/reset; query is a 10 ms membrane-current drive. Analog output uses deadline candidate u or reset u during refractoriness. Protocol:24–56,112–124; `models_p1b_v2.py:78–107,249–267`. | Replace the model/readout description with the registered hybrid Euler equations and exact deadline semantics. Do not change code or historical outputs. |
| Same file:174–176 | G1.5 explained as u below theta_A turning off sigma_A and causing all subthreshold differences to decay/erase. | G1.5 uses fixed positive-novelty keys, scaled content weights .1, and a compensating value relation to match final state. It is not a universal collapse of every subthreshold difference. Protocol:229–247. | Describe the actual affine collision construction and its branch/weight restrictions; separate it from first-event zero-novelty erasure. |
| Same file:182 | Prerequisite operand criterion rendered as E_bind<=.05. | Operand delivery is strict absolute error<.50 with no abstention. .05 is the separate composition-risk threshold for channel decisions. Protocol:171–178,216–223. | Restore distinct tolerances and strictness. |
| Same file:183 | TOST margin±.05; M2a called constant-gate. | TOST equivalence margin±.005; M2a resets u,r at active-event offsets and leaves dynamic A intact. M2b/M2d use A0. Protocol:93–106,199–214; `models_p1b_v2.py:121–133,167–175`. | Correct margin and arm definition; keep floor-limited label and no redundancy inference. |
| `README.md:37` | P1-B establishes distributed multi-channel necessity. | Closure:16,104–111 explicitly declines a population-coding necessity theorem; Paper2 s9:20–25 calls its ladder sufficient, not necessary. | Narrow summary to model-specific insufficiency and motivation for separately tested alternatives. |

The paper later correctly states the limited consensus in `s8_composition.tex:191–193`; the conflicting earlier description is still material because it changes what was simulated and which signal was observed. The historical numerical findings are not invalidated solely by a manuscript-description error. The source/protocol mismatch should be corrected in a separately reviewed manuscript change, preserving provenance and original experimental labels.

No repository mutation, publication or historical data regeneration was performed. Source SHA256 values are included in PROJECT_MAP_SOURCES.json.
