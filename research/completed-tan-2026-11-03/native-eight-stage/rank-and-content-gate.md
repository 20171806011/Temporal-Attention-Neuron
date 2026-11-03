# Conditional value rank: theorem and smallest same-state intervention

2026-11-02. Pinned production source85213f0d417d54e4f48b301ed1c0e4200ef3c421. Mathematical derivations below are new analysis of inspected equations, not claims of a new simulation or universal neuron-capacity theorem.

## 1. Exact affine-fiber theorem for registered carrier

Fix N candidate keys, event positions, event durations, initial state, and all parameters. Allow values v∈U⊂R^N to vary over an open set where the discrete Euler encoding remains subthreshold, with no reset/refractory branch. A's novelty drive depends only on keys. Hence A_n=A_n(K,timing) is independent of v, and r_n=0. Write a=1−h/τm=.99 and κ=(h/τm)g_A=.01. At endpoint Tsteps:

`u_T = b + cᵀv`,
`c_i = κ wV Σ_{n∈event_i} A_n a^(T−1−n)`.

Thus the full endpoint encoding E(v)=(u_T,A_T,0) is affine with rank≤1. If c≠0, its fibers have dimension N−1, and any sufficiently small δ with cᵀδ=0 yields E(v+δ)=E(v). If c=0, all values are locally erased. This exact affine identity establishes noninjectivity; merely finding a deficient derivative at a point would not (e.g. scalar x³ has derivative0 at0 but is injective).

For identical state and identical subsequent queries, any deterministic state-based future dynamics/readout, including nonlinear readout or repeated/adaptive querying, has identical outputs. It therefore cannot reconstruct every differing operand across a collision pair. If target i differs by d, at least one member incurs absolute error≥|d|/2. Query/adaptive policy must not secretly receive the original history, external encoder trace, or hidden target label. An external observer retaining encoding spikes changes the information available and is outside this endpoint theorem.

### Discrete spiking branches

At fixed finite Euler horizon, within an open region of values that produces exactly the same spike/reset/refractory schedule, A remains value-independent, refractory endpoint is fixed by that schedule, and u is again affine in the unreset inputs with rank≤1. Resets may make c vanish. A branch boundary can be nonsmooth; do not take derivatives across it or call a discontinuity a second continuous value coordinate. This is a statement about each open fixed branch, not a blanket numerical rank calculation over mixed branches.

Do not transfer this automatically to continuous-time event timing, stochastic encoders, retained spike trains, arbitrary adaptive intrinsic states, or all three-variable neurons. For stochastic *readout* from identical states under the same Markov transition law, output distributions remain identical. Stochastic encoding requires distributional/information analysis, not deterministic rank alone.

## 2. Robustness and general rank conditions

For smooth conditional encoder E:U⊂R^N→R^m, full column rankN at a point is sufficient for local reconstructability from someN endpoint coordinates by the inverse function theorem. If a differentiable local decoder D satisfies D(E(v))=v on a neighborhood, chain rule requires rank(DE)=N. Rank deficiency at a single point is not by itself an impossibility theorem; a constant-rank neighborhood r<N supplies nontrivial fibers. Our production map is exactly affine, so that caveat is already resolved.

The practically observable map is Y=O∘E, with Jacobian JY=JO JE. Encoder rank and observable rank are different. Small singular values imply ill-conditioning: two value histories differing appreciably can have nearly identical measured states. Report singular values in declared units and noise-normalized form; changing coordinate units changes raw conditioning. Full rank is neither robust capacity nor successful content-addressed retrieval.

For a bounded scalar membrane range of lengthL with adversarial readout error boundedε, distinguishable robust codewords need separation>2ε; their number is at most floor(L/(2ε))+1 (a loose boundary-inclusive upper bound). Finite precision also limits usable states. These simple bounds require an explicit noise/precision model; do not assign a bit capacity to ideal real-valued state without one.

**Primary-source context:** Hermann & Krener, *Nonlinear controllability and observability*, IEEE TAC22(5),728–740(1977), DOI10.1109/TAC.1977.1101601, author-hosted paper: https://www.math.ucdavis.edu/~krener/1-25/10.IEEETAC77.pdf . This provides classical nonlinear-observability context. The finite-horizon conditional-encoder fiber proof here is derived directly above, not claimed as a verbatim application of their observability-rank theorem. Do not conflate their state-observation problem with inversion of the preceding value-encoding map.

## 3. Minimal same-coordinate intervention

Use existing A coordinate rather than append a slow trace. Keep u,A,r state dimensions, all timings, and membrane equation unchanged. Only during active events modify the A driving term:

- Original arm: novelty.
- Matched-offset arm: novelty+3η.
- Content arm: novelty+η(v_i+3).

Prespecify η=.1. Outside events, drive0 as original. Keep clipping to[0,5]. Values lie[-3,3], so new added drive is nonnegative. This changes gating physiology/input coupling and current/energy, even though coordinate count is unchanged. It is an intervention, not the original TAN model.

At base values v=(0,0), matched-offset and content arms have **identical trajectories and drive**. Their different value sensitivities therefore cannot be explained simply by baseline gate gain. Originalη0 remains the rank1 reference.

### Fixed minimal test point

N2, keys(.8,1.8), occupied positions(0,4), values(0,0), default production parameters, no post-encoding delay initially. Use the same100ms timeline already tested. Do not fit η or search positions. Check strict positive threshold/clipping margins and r0 first; if invalid, report branch failure rather than silently retune. A longer-separated schedule may yield a tiny smallest singular value; that is a meaningful limitation, not a reason to optimize the test after seeing it.

### Analytic tangent recurrence

Use pre-step A_n and I_n=wK k_n+wV v_n, exactly as production Euler. For each candidate coordinate i, in the content arm away from clipping:

`DA_i,next = bA*DA_i + (h/τA)*η*1[event_i]`, bA=1−h/τA.
`Du_i,next = a*Du_i + κ*(I_n*DA_i + A_n*wV*1[event_i])`.

For matched-offset and original arms the added DA term is0, and DA remains0. During blank slots I_n=0 and indicators0. Initial tangents0. Evaluate J=∂(u_T,A_T)/∂(v1,v2); report entries, determinant, singular values and rank interpretation. Check central differences at a predeclared small step (e.g.1e−4) plus one10×smaller numerical consistency step, not a performance sweep. Threshold/clipping must remain inactive on both perturbed trajectories. Analytic sensitivities have priority over arbitrary SVD rank cutoffs.

## 4. Do not forget cue-time observability

Existing query code sets I_content=0. Therefore A cannot influence query u/spikes/readout; JO has zero A column. Even if the content intervention achieves rank2 in endpoint(u,A), the existing query route still exposes at most one continuous coordinate on a fixed branch. Report this independently. A fitted full-state decoder observing A is a diagnostic extra observer, not proof of successful original query mechanism.

A minimal **separately labeled readout intervention**, if needed later, sets a small known nonzero query-phase content probe I_content=Iprobe and reads analog membrane at two times. In the first subthreshold step:

`[u0,u1]ᵀ = [[1,0],[a,κ Iprobe]] [u0,A0]ᵀ + [0,(h/τm)Iq]ᵀ`.

Its observation determinantκIprobe is nonzero for Iprobe≠0, exposing the second coordinate. This uses an extra analog time sample and a new query-time coupling; count these readout resources explicitly. No need to execute this readout intervention merely to test encoder rank.

## 5. Interpretation limits and stopping rule

- Rank2 forN2 fixed keys/timing would escape this rank1 obstruction without adding a neuron or state coordinate. It would not prove general variable key-value binding, since keys/timing vary in the actual task and require representation too.
- In the same subthreshold two-value-dependent-coordinate regime, N3 still cannot have a differentiable local inverse for all3 independent values. Repurposing refractory dynamics, adding another intrinsic variable, temporal readout memory, or a circuit are distinct possible resources; no network-necessity claim follows.
- If rank2 appears, report conditioning and a prespecified state-noise diagnostic before any strong retrieval claim. No training or full benchmark needed for the first test. If rank remains1 or observability remains1, that is the result; do not automatically escalate architectures.
- Record altered equations and source hash, analytic-vs-finite-difference checks, clipping/spike margins, and original-query observable sensitivity. Stop when this narrowly specified rank/observability question is established.
