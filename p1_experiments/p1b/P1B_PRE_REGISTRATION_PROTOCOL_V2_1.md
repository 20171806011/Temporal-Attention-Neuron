# P1-B Pre-Registration Protocol Amendment 2 (Version 2.1)
## Focused Carrier Audit: Temporal Retention, Content Addressing, and the 10 ms Spike Readout Boundary

**Protocol Version:** 2.1 (Focused Revision Addressing All Codex Round 2 Findings)  
**Date:** September 19, 2026  
**Authors:** Antigravity (Google DeepMind) & Codex (OpenAI, `gpt-6-astra-2`)  
**Protocol Basis:** `RESEARCH_AUDIT_AND_PLAN_2026-09-17.md` (Section 10.1), P1-A Baseline Sign-off (`P1A_FINAL_BASELINE_MANIFEST.md`), and Codex Reviews Rounds 1 & 2  
**Scope Decision:** **H3 (Internal Dynamical Composition) is formally DEFERRED to future multi-neuron circuits (P3/Claim C14). P1-B focuses strictly on H1 (Temporal Retention), H2 (Content Addressing & Transport Delivery), and the 10 ms Spike Readout Channel Boundary.**

---

## 1. Provenance, Scope Narrowing, and Metric Alignment (G1.0)

1. **Explicit Scope Demarcation:**
   - As recommended by Codex, **Hypothesis H3 (Internal Composition within a single dynamical unit) is officially marked as DEFERRED / NOT CLAIMED**. Single-unit dynamics are not asked to perform internal analog arithmetic without an external circuit.
   - P1-B evaluates relational composition exclusively via **Pathway A (Transport Path)**: the neuron routes and delivers the bound operands $\hat{V}_A, \hat{V}_B$, and an explicit, inspectable external linear node evaluates $\hat{y} = \hat{V}_A \pm \hat{V}_B$.
2. **Preliminary Code Status:**
   - The sketch `models_p1b.py` remains non-conforming and barred from all execution. Conforming models will be implemented only after G1-Design approval.
3. **Reference Metric Kernel:**
   - The reference static model (M0) uses the quadratic metric $S(q, k) = -(q - k)^2$ matching frozen P1-A (`models.py:296`).

---

## 2. Dynamically Closed State Space & Physical Equations (G1.1)

### 2.1 Complete Three-Variable State Inventory
To ensure strict dynamical closure including refractory mechanics, the state vector at time $t$ is:
$$\mathbf{z}(t) = \begin{bmatrix} u(t) \\ A(t) \\ r(t) \end{bmatrix} \in \mathbb{R} \times [0, A_{\max}] \times [0, \tau_{\text{ref}}]$$
- $u(t) \in (-\infty, \theta_{\text{th}}]$: Normalized membrane potential (dimensionless, rest $u_{\text{rest}} = 0.0$, threshold $\theta_{\text{th}} = 1.0$, reset $u_{\text{reset}} = 0.0$).
- $A(t) \in [0, A_{\max}]$: Normalized gating conductance ($\tau_A = 15.0\text{ ms}$, coupling $\alpha = 1.0$, $A_{\max} = 5.0$).
- $r(t) \in [0, \tau_{\text{ref}}]$: Remaining refractory time ($\tau_{\text{ref}} = 2.0\text{ ms}$).

### 2.2 Continuous Evolution and Firing Dynamics
For integration step $\Delta t = 0.1\text{ ms}$:
1. **Refractory and Membrane Evolution:**
   - If $r(t) > 0$:
     $$r(t + \Delta t) = \max(0.0, r(t) - \Delta t)$$
     $$u(t + \Delta t) = u_{\text{reset}} = 0.0$$
   - If $r(t) == 0$:
     $$\frac{du}{dt} = \frac{-(u(t) - u_{\text{rest}}) + g_A A(t) I_{\text{content}}(t) + I_{\text{query}}(t) + I_{\text{bias}}}{\tau_m}$$
     $$u(t + \Delta t) = u(t) + \frac{du}{dt} \Delta t$$
2. **Action Potential & Reset:**
   - If $u(t + \Delta t) \ge \theta_{\text{th}}$ and $r(t) == 0$:
     $$s(t + \Delta t) = 1, \quad u(t + \Delta t) \leftarrow u_{\text{reset}}, \quad r(t + \Delta t) \leftarrow \tau_{\text{ref}}$$
   - Otherwise:
     $$s(t + \Delta t) = 0$$
3. **Gating Conductance Evolution:**
   - $A(t)$ evolves continuously at all times (including during refractoriness):
     $$\frac{dA}{dt} = \frac{-A(t) + \alpha \cdot \operatorname{Novelty}(t)}{\tau_A}, \quad A(t + \Delta t) = \min(A_{\max}, \max(0.0, A(t) + \frac{dA}{dt} \Delta t))$$
   - Initial conditions at $t=0$: $u(0) = 0.0$, $A(0) = 0.0$, $r(0) = 0.0$.

### 2.3 Explicit Input Drive Encodings
- $w_K = 1.0$, $w_V = 1.0$, $g_A = 1.0$, $\tau_m = 10.0\text{ ms}$, $I_{\text{bias}} = 0.0$.
- Fixed temporal background baseline: $\mu_{\text{bg}} = 1.4$ (midpoint of calibration key interval $[0.3, 2.5]$).
- During event presentation ($t \in [t_{\text{start}}, t_{\text{end}}]$):
  $$I_{\text{content}}(t) = w_K K_i + w_V V_i, \quad \operatorname{Novelty}(t) = \max(0.0, |K_i| - \mu_{\text{bg}})$$
- During ISI, inter-cue delays, and query presentation:
  $$I_{\text{content}}(t) = 0.0, \quad \operatorname{Novelty}(t) = 0.0$$
- Query drive $\phi(q) = 1.0 \cdot q$.

---

## 3. Streaming Presentation Schedule & Strict 10 ms Cue-to-Delivery Window (G1.2)

### 3.1 Mapping to $W=5$ Slot History
The stream uses the frozen `TrialData` candidate sequences:
- The history buffer consists of $W=5$ chronological slots.
- For $N \in \{2, 3\}$, active events occupy the initial $N$ slots, followed by $(5 - N)$ blank slots (or randomly placed with active event flags, preserved identically from `TrialData`).
- Each slot has duration $T_{\text{slot}} = T_{\text{event}} + T_{\text{isi}} = 10.0\text{ ms} + 10.0\text{ ms} = 20.0\text{ ms}$.
- Total encoding duration: $T_{\text{enc}} = 5 \times 20.0\text{ ms} = 100.0\text{ ms}$.

### 3.2 Strict 10 ms Cue-to-Delivery Window
To resolve all timing ambiguities:
- Query cue $q_A$ arrives at $T_{\text{cue}, A} = T_{\text{enc}} + \tau_{\text{delay}}$ ($\tau_{\text{delay}} \in \{0.0, 5.0, 10.0\}\text{ ms}$).
- Query cue $q_A$ is active for $10.0\text{ ms}$, spanning the half-open interval $[T_{\text{cue}, A}, T_{\text{cue}, A} + 10.0\text{ ms})$.
- **Delivery Deadline:** The readout decoder observes the neural output **strictly during this same 10.0 ms window** $[T_{\text{cue}, A}, T_{\text{cue}, A} + 10.0\text{ ms})$.
- No subsequent extra window is added. Cue onset to operand delivery is **strictly 10.0 ms**.
- Query $q_B$ is presented in an identical independent trial or sequentially at $T_{\text{cue}, B} = T_{\text{cue}, A} + 10.0\text{ ms}$ with an identical 10.0 ms decode window.

---

## 4. Complete $2 \times 2$ Factorial Design & Causal Controls (G1.4)

The primary causal experiment is a balanced $2 \times 2$ factorial crossing membrane potential treatment with gating treatment:

| | Dynamic Gating $A(t)$ | Mean-Gate Matched Constant $A_0$ |
|---|---|---|
| **Retained Membrane $u(t)$** | **Model M1** (Intact TAN) | **Model M2b** (Gate Ablation) |
| **Post-Event Clamped Membrane $u(t)$** | **Model M2a** (Membrane Ablation) | **Model M2d** (Dual Ablation) |

### Definitions of Interventions:
1. **Retained Membrane:** $u(t)$ evolves continuously across events and ISIs.
2. **Post-Event Clamped Membrane (M2a, M2d):** Immediately at the offset of each active event slot, $u(t) \leftarrow u_{\text{rest}} = 0.0$ and $r(t) \leftarrow 0.0$.
3. **Dynamic Gating (M1, M2a):** $A(t)$ evolves adaptively via novelty drive.
4. **Mean-Gate Matched Constant (M2b, M2d):** $A(t) \equiv A_0$, where $A_0$ is the independent calibration mean $\bar{A} = \frac{1}{T_{\text{enc}}} \int_0^{T_{\text{enc}}} A_{\text{M1}}(t) dt$ measured on 500 calibration trials. The residual driving force difference is logged in G1.4.
5. **Full-State Reset Control (M2c - Auxiliary Control):** At each event offset, all three states are clamped: $u(t) \leftarrow 0.0$, $A(t) \leftarrow 0.0$, $r(t) \leftarrow 0.0$.
6. **Strict No-Buffer Access (Applies to all M1, M2a-d):** Models M1 and M2a-d have zero access to candidate arrays, historical voltage logs, or evaluator ground truth.

---

## 5. Discrete Spike Channel Contract & Analytical Codec Benchmarks (G1.6)

### 5.1 Half-Open Window & Exact Spike Budget
- Window: $[0.0, 10.0)\text{ ms}$ (half-open, length exactly $10.0\text{ ms}$).
- Refractory constraint: $\Delta t_{\text{isi}} \ge \tau_{\text{ref}} = 2.0\text{ ms}$.
- Possible spike timestamps: $\{0.0, 2.0, 4.0, 6.0, 8.0\}\text{ ms}$.
- **Maximum Spike Budget:** Exactly **5 spikes** ($\{0, 1, 2, 3, 4, 5\}$).

### 5.2 Registered Codebooks
1. **Count / Rate Codebook ($M=6$ levels):**
   - 6 equal-width bins of width $1.0$ partitioning $[-3.0, 3.0]$.
   - Midpoint reconstructions: $\mathcal{R}_{\text{count}} = \{-2.5, -1.5, -0.5, +0.5, +1.5, +2.5\}$.
2. **Latency / Time-to-First-Spike Codebook ($M=20$ levels):**
   - First-spike latency $t_{\text{first}} \in [0.0, 10.0)\text{ ms}$, quantized into 20 bins of width $0.5\text{ ms}$.
   - Bin midpoints $t_k = 0.25 + k \times 0.5\text{ ms}$ ($k \in \{0, \dots, 19\}$).
   - Linear mapping to 20 value reconstructions over $[-3.0, 3.0]$ (bin width $0.30$):
     $$\hat{V}_k = -2.85 + k \times 0.30$$
   - Silence handling: if zero spikes occur within $[0.0, 10.0)\text{ ms}$, decoder outputs $\hat{V} = 0.0$ and logs an abstention/silence flag.

### 5.3 Pre-Simulation Analytical Codec Benchmarks
For independent $V_A, V_B \sim \mathcal{U}[-3, 3]$, independent midpoint quantization with $M$ bins yields:
$$\mathbb{E}[|e_A \pm e_B|] = \frac{2}{M}$$
- **For $M=6$ (Count Code):** Expected error $\mathbb{E}[E_{\text{comp}}] = \frac{2}{6} = 0.3333$.
- **For $M=20$ (Latency Code):** Expected error $\mathbb{E}[E_{\text{comp}}] = \frac{2}{20} = 0.1000$.

**Pre-Registered Analytical Conclusion:**  
Both the $M=6$ and $M=20$ codecs mathematically cannot achieve $E_{\text{comp}} < 0.05$ under the 10.0 ms biological constraint. This is registered as a known mathematical property of the coarse discretization, **not** an empirical failure of the neuron's dynamical integration.

---

## 6. Revised Statistically Feasible Decision Rules (G1.8)

### 6.1 Endpoints Defined
1. **Ordered Address Accuracy (`AddrAcc`):** Proportion of trials where the decoded operand satisfies $|\hat{V} - V_{\text{target}}| < \text{Tol}_{\text{addr}}$, with $\text{Tol}_{\text{addr}} = 0.50$ (half of minimum key-value spacing).
2. **Joint Address Accuracy (`JointAddrAcc`):** Both operands $A$ and $B$ satisfy ordered address accuracy.
3. **Composition Error ($E_{\text{comp}}$):** Mean absolute error $|\hat{y} - (V_A \pm V_B)|$ in original value units.

### 6.2 Feasible Baseline Validity Condition
- To replace the impossible $\text{Chance} + 0.50$ condition for $N=2$:
  - For $N=2$ (Chance = $0.50$): Model M1 baseline must achieve $\text{JointAddrAcc}(\text{M1}) \ge 0.80$ (80% accuracy).
  - For $N=3$ (Chance = $1/6 \approx 0.1667$): Model M1 baseline must achieve $\text{JointAddrAcc}(\text{M1}) \ge 0.40$ ($> 2 \times \text{Chance}$).
  - If M1 fails this baseline, declare `DYNAMIC_CARRIER_INSUFFICIENT` without evaluating equivalence.

### 6.3 Paired Equivalence Testing via TOST (Newcombe Paired Method)
To test if membrane clamping degrades accuracy:
1. Define paired difference: $d = \text{JointAddrAcc}(\text{M1}) - \text{JointAddrAcc}(\text{M2a})$.
2. Compute the 90% confidence interval for paired proportions using the **Newcombe method for paired differences** (accounting for discordant pairs).
3. **Equivalence Declared:** If $\text{CI}_{90\%}(d) \subset (-0.005, +0.005)$ (within $\pm 0.5$ percentage points).
   - **Permitted Conclusion:** *“Within the registered tasks, delays, and 0.5% equivalence margin, membrane-potential reset had no practically meaningful effect on operand delivery.”*

### 6.4 Spike Channel Decision Rule (Central 95% Confidence Interval)
For Model M4 composition error $E_{\text{comp}}$:
Compute the central 95% bootstrap confidence interval $[\text{LCB}_{95\%}, \text{UCB}_{95\%}]$:
- **Demonstrated Channel Success:** $\text{UCB}_{95\%}(E_{\text{comp}}) < 0.05$.
- **Demonstrated Channel Limitation:** $\text{LCB}_{95\%}(E_{\text{comp}}) \ge 0.05$.
- **Inconclusive:** Interval spans $0.05$.

---

## 7. Executable G1 Gates Specification (G1.0 – G1.9)

| Gate ID | Acceptance Criteria & Automated Assertions |
|---|---|
| **G1.0** | Version 2.1 frozen; H3 explicitly marked deferred; reference kernel locked to $-(q-k)^2$. |
| **G1.1** | State closure assertion: $\mathbf{z}(t) = [u(t), A(t), r(t)]^T$; refractoriness enforced during ODE steps. |
| **G1.2** | Streaming generator check: $W=5$ slots, query cue strictly post-enc at $T_{\text{cue}} \ge 100\text{ ms}$, decode window strictly 10.0 ms. |
| **G1.3** | Leakage test: Query-blind control achieves empirical accuracy within binomial 95% CI of chance ($1/[N(N-1)]$). Oracle achieves 100% on clean trials. |
| **G1.4** | Factorial check: M1, M2a, M2b, M2d, and M2c verified in trace assertions. M2b uses independently calibrated $A_0$. |
| **G1.5** | State collision fixture: With fixed keys, perturbations satisfying $\Delta V_1 = \delta, \Delta V_2 = -(w_1/w_2)\delta$ produce identical $[u(T), A(T), r(T)]^T$ within machine tolerance ($< 10^{-12}$), proving the single-neuron linear subthreshold state collision boundary. |
| **G1.6** | Spike channel fixture: Confirms 10 ms half-open window enforces $\le 5$ spikes; confirms $M=6$ and $M=20$ analytical error benchmarks ($0.3333$ and $0.1000$). |
| **G1.7** | Numerical ODE check: 4th-order Runge-Kutta vs Euler at $\Delta t = 0.1\text{ms}$ vs $0.05\text{ms}$ maintains subthreshold voltage $L_2$ difference $< 10^{-5}$ outside reset times; exact spike times match within $\pm 0.1\text{ms}$. |
| **G1.8** | Statistical code verification: Newcombe paired difference CI and central 95% bootstrap verified on synthetic edge-case vectors (0%, 100%, and discordant pairs). |
| **G1.9** | Budget lock: Runtime budget $\le 2\text{ CPU hours}$, peak RAM $\le 2\text{ GB}$, immutable SHA-256 output tracking. |
