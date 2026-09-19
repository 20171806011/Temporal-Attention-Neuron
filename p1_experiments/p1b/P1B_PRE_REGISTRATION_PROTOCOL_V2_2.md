# P1-B Pre-Registration Protocol Amendment 3 (Version 2.2)
## Focused Carrier Audit: Temporal Retention, Content Addressing, and the Information-Erasure Boundary

**Protocol Version:** 2.2 (Complete Specification Resolving All Codex Round 3 Review Findings)  
**Date:** September 19, 2026  
**Authors:** Antigravity (Google DeepMind) & Codex (OpenAI, `gpt-6-astra-2`)  
**Protocol Basis:** `RESEARCH_AUDIT_AND_PLAN_2026-09-17.md` (Section 10.1), P1-A Baseline Sign-off (`P1A_FINAL_BASELINE_MANIFEST.md`), and Codex Reviews Rounds 1–3  
**Execution Policy:** **G1-Design Submission with Concrete Mathematical and Algorithmic Annex.**

---

## 1. Scope, Provenance, and Core Demarcation

1. **Scope Freeze:**
   - **Hypothesis H3 (Internal Dynamical Composition):** Officially marked as **DEFERRED / NOT CLAIMED**. Single-unit dynamics are not required to perform internal algebraic composition.
   - **Focus of P1-B:** Rigorous evaluation of **H1 (Temporal Retention)** and **H2 (Content Addressing & Transport Delivery)**. Composition is evaluated strictly on the **Transport Pathway (Pathway A)**:
     $$\hat{y} = \hat{V}_A \pm \hat{V}_B$$
     via an external, inspectable linear node.
2. **Prior Scaffold Status:**
   - The sketch `models_p1b.py` remains non-conforming and barred from execution.
3. **Reference Model Metric:**
   - M0 reference static model uses $S(q, k) = -(q - k)^2$ matching frozen P1-A (`models.py:296`).

---

## 2. Dynamically Closed State Space & Physical Equations (G1.1)

### 2.1 State Space Vector $\mathbf{z}(t)$
At every timestep $t$, each physical neuron is completely described by:
$$\mathbf{z}(t) = \begin{bmatrix} u(t) \\ A(t) \\ r(t) \end{bmatrix} \in \mathbb{R} \times [0, A_{\max}] \times [0, \tau_{\text{ref}}]$$
- $u(t) \in (-\infty, \theta_{\text{th}}]$: Normalized membrane potential ($u_{\text{rest}} = 0.0$, $\theta_{\text{th}} = 1.0$, $u_{\text{reset}} = 0.0$).
- $A(t) \in [0, A_{\max}]$: Normalized gating conductance ($\tau_A = 15.0\text{ ms}$, $\alpha = 1.0$, $A_{\max} = 5.0$, initial $A(0) = 0.0$).
- $r(t) \in [0, \tau_{\text{ref}}]$: Remaining refractory time ($\tau_{\text{ref}} = 2.0\text{ ms}$, initial $r(0) = 0.0$).

### 2.2 Continuous Update Equations ($\Delta t = 0.1\text{ ms}$)
1. **Gating Conductance Evolution:**
   $$\frac{dA}{dt} = \frac{-A(t) + \alpha \cdot \operatorname{Novelty}(t)}{\tau_A}$$
   $$A(t + \Delta t) = \min(A_{\max}, \max(0.0, A(t) + \frac{dA}{dt} \Delta t))$$
   $A(t)$ evolves continuously across all states, including during refractoriness.
2. **Membrane & Refractory Evolution:**
   - If $r(t) > 0$:
     $$r(t + \Delta t) = \max(0.0, r(t) - \Delta t), \quad u(t + \Delta t) = u_{\text{reset}} = 0.0, \quad s(t + \Delta t) = 0$$
   - If $r(t) == 0$:
     $$\frac{du}{dt} = \frac{-(u(t) - u_{\text{rest}}) + g_A A(t) I_{\text{content}}(t) + I_{\text{query}}(t) + I_{\text{bias}}}{\tau_m}$$
     $$u_{\text{cand}} = u(t) + \frac{du}{dt} \Delta t$$
     - If $u_{\text{cand}} \ge \theta_{\text{th}}$:
       $$s(t + \Delta t) = 1, \quad u(t + \Delta t) = u_{\text{reset}} = 0.0, \quad r(t + \Delta t) = \tau_{\text{ref}} = 2.0\text{ ms}$$
     - Else:
       $$s(t + \Delta t) = 0, \quad u(t + \Delta t) = u_{\text{cand}}, \quad r(t + \Delta t) = 0.0$$

### 2.3 Input Currents & Parameter Values
- Time constants: $\tau_m = 10.0\text{ ms}$, $\tau_A = 15.0\text{ ms}$, $\tau_{\text{ref}} = 2.0\text{ ms}$.
- Weights: $w_K = 1.0$, $w_V = 1.0$, $g_A = 1.0$, $I_{\text{bias}} = 0.0$.
- Background baseline: $\mu_{\text{bg}} = 1.4$ (fixed midpoint of key range $[0.3, 2.5]$).
- During event slot:
  $$I_{\text{content}}(t) = w_K K_i + w_V V_i, \quad \operatorname{Novelty}(t) = \max(0.0, |K_i| - \mu_{\text{bg}})$$
- During ISIs and cue presentation:
  $$I_{\text{content}}(t) = 0.0, \quad \operatorname{Novelty}(t) = 0.0$$
- Query drive:
  $$I_{\text{query}}(t) = \begin{cases} 1.0 \cdot q, & \text{if } t \in [T_{\text{cue}}, T_{\text{cue}} + 10.0\text{ ms}) \\ 0.0, & \text{otherwise} \end{cases}$$

---

## 3. Streaming Task Schedule & Frozen Episode Mapping (G1.2)

1. **Fixed Episode History ($W=5$):**
   - Strictly preserves `TrialData.event_positions` and `mask` from the frozen P1-A generator (`task_generator.py`).
   - Slot duration: $T_{\text{slot}} = T_{\text{event}} + T_{\text{isi}} = 10.0\text{ ms} + 10.0\text{ ms} = 20.0\text{ ms}$.
   - Total encoding duration: $T_{\text{enc}} = 5 \times 20.0\text{ ms} = 100.0\text{ ms}$.
2. **Independent Repeated Probing:**
   - Role A and Role B queries ($q_A, q_B$) are evaluated as **independent repeated probes** of the identical encoded history episode.
   - For query cue $q$, cue onset is $T_{\text{cue}} = T_{\text{enc}} + \tau_{\text{delay}}$ ($\tau_{\text{delay}} \in \{0.0, 5.0, 10.0\}\text{ ms}$).
3. **Strict 10.0 ms Cue-to-Delivery Window:**
   - Half-open window: $[T_{\text{cue}}, T_{\text{cue}} + 10.0\text{ ms})$.
   - The readout decoder observes neural output strictly during this 10.0 ms window. No subsequent readout period is added.
   - Spikes recorded at $t + \Delta t$: a spike at exactly $T_{\text{cue}} + 10.0\text{ ms}$ is excluded from the window.

---

## 4. Balanced $2 \times 2$ Factorial Design & Causal Controls (G1.4)

| | Dynamic Gating $A(t)$ | Mean-Gate Matched Constant $A_0$ |
|---|---|---|
| **Retained Membrane $u(t)$** | **Model M1** (Intact TAN) | **Model M2b** (Gate Ablation) |
| **Post-Event Clamped Membrane $u(t)$** | **Model M2a** (Membrane Ablation) | **Model M2d** (Dual Ablation) |

### Intervention Rules:
- **Retained Membrane (M1, M2b):** State $\mathbf{z}(t)$ evolves without clamp.
- **Post-Event Clamped Membrane (M2a, M2d):** At each active event offset ($t = t_{\text{end}}$), $u(t) \leftarrow 0.0$ and $r(t) \leftarrow 0.0$. Gating state $A(t)$ is untouched.
- **Dynamic Gating (M1, M2a):** $A(t)$ evolves with novelty drive.
- **Mean-Gate Matched Constant (M2b, M2d):** $A(t) \equiv A_0$, where $A_0$ is the independent calibration mean conductance across 500 calibration trials ($A_0 = \bar{A}_{\text{calib}}$). Residual driving force differences are logged in G1.4.
- **Full-State Reset Control (M2c - Auxiliary):** Clamps $u(t) \leftarrow 0.0$, $A(t) \leftarrow 0.0$, and $r(t) \leftarrow 0.0$ at each event offset.
- **Strict No-Buffer Access:** M1, M2a-d, and M4 have zero access to candidate arrays, historical voltage logs, or evaluator ground truth.

---

## 5. Frozen Analog & Spiking Decoder Specifications

### 5.1 Continuous Analog Decoder for M1 and M2a–d
- The decoder reads the instantaneous pre-spike membrane potential $u(T_{\text{readout}})$ at $T_{\text{readout}} = T_{\text{cue}} + 10.0\text{ ms}$.
- Decoder function:
  $$\hat{V} = g_{\text{dec}} \cdot u(T_{\text{readout}}) + b_{\text{dec}}$$
- Calibration: Linear regression coefficients $(g_{\text{dec}}, b_{\text{dec}})$ are fitted on the 500 calibration trials mapping $u(T_{\text{readout}})$ to $V_{\text{target}}$ and frozen identically across M1 and M2a–d.
- Permitted inputs: Exactly the scalar $u(T_{\text{readout}})$ and query $q$. No historical voltage traces, no candidate arrays.

### 5.2 Discrete Spiking Decoders for Model M4
- Within the half-open window $[T_{\text{cue}}, T_{\text{cue}} + 10.0\text{ ms})$ with $\tau_{\text{ref}} = 2.0\text{ ms}$, maximum spike budget is **5 spikes**.
- **Count Codebook ($M=6$ bins):** 6 equal bins of width $1.0$ over $[-3.0, 3.0]$ with midpoints $\{-2.5, -1.5, -0.5, +0.5, +1.5, +2.5\}$.
- **Latency Codebook ($M=20$ bins):** First-spike latency $t_{\text{first}} \in [0.0, 10.0)\text{ ms}$ quantized into 20 bins of width $0.5\text{ ms}$ (midpoints $0.25 + k \times 0.5\text{ ms}$), mapped to 20 value reconstructions spaced by $0.30$ over $[-3.0, 3.0]$.
- **Silence Handling:** Zero spikes emits $\hat{V} = 0.0$ and logs `SILENCE_ABSTENTION = True`. Silence counts as delivery failure in both `DeliveryAcc` and unconditional $E_{\text{comp}}$.

### 5.3 Pre-Simulation Analytical Codec Benchmarks
For independent uniform operands $V_A, V_B \sim \mathcal{U}[-3, 3]$ under an ideal independent midpoint quantizer:
$$\mathbb{E}[|e_A \pm e_B|] = \frac{2}{M}$$
- Reference prediction for $M=6$ (Count): $\mathbb{E}[E_{\text{comp}}] = 2/6 = 0.3333$.
- Reference prediction for $M=20$ (Latency): $\mathbb{E}[E_{\text{comp}}] = 2/20 = 0.1000$.
- **Pre-Registered Conclusion:** *“For the ideal independent midpoint-quantizer reference under independent uniform operands, population composition MAE is $1/3$ for six bins and $0.10$ for twenty bins. These are reference predictions, not universal lower bounds for the neuronal implementation or its finite-sample results.”*

---

## 6. Mathematical Proposition: Gating-Induced Information Erasure

### Proposition 6.1 (Gating-Induced Information Erasure in M1)
*Under the specified M1 dynamics, initial condition $A(0)=0$, and static baseline $\mu_{\text{bg}} = 1.4$, any active event with key $K_i \le 1.4$ produces $\operatorname{Novelty}(t) = 0$, yielding $A(t) = 0$ and $I_{\text{syn}}(t) = 0$. Consequently, the value $V_i$ never enters the membrane state.*

*For the in-domain clean $N=2$ generator with keys $K \sim \mathcal{U}[0.3, 2.5]$, $P(K_1 \le 1.4) = 0.5$. Since both events are requested, an entirely unobserved independent $V \sim \mathcal{U}[-3, 3]$ can be delivered within tolerance $\pm 0.50$ with probability at most $1/6$. Therefore, the population joint delivery success of M1 is theoretically bounded by:*
$$P(\text{JointDeliveryAcc}_{\text{M1}}) \le \frac{1}{2} \cdot 1.0 + \frac{1}{2} \cdot \frac{1}{6} = \frac{7}{12} \approx 0.5833$$

**Scientific Commitment:**  
This gating-induced information-erasure boundary is pre-registered as a formal mathematical prediction of Model M1. We do not modify the gating function or artificially lower thresholds to hide this result.

---

## 7. Precise Endpoints, Null Models, and Decision Rules (G1.8)

### 7.1 Endpoints
1. **Delivery Accuracy (`DeliveryAcc`):**
   $$\operatorname{DeliverySuccess}_i = \mathbb{I}(|\hat{V}_i - V_{\text{target}, i}| < 0.50) \land \neg \operatorname{Abstained}_i$$
   (Declared practical tolerance $\Delta V = 0.50$).
2. **Joint Delivery Accuracy (`JointDeliveryAcc`):**
   $$\operatorname{JointDeliverySuccess} = \operatorname{DeliverySuccess}_A \land \operatorname{DeliverySuccess}_B$$
3. **Unconditional Composition Error ($E_{\text{comp}}$):**
   $$E_{\text{comp}} = \frac{1}{M} \sum_{i=1}^M |\hat{y}_i - (V_{A, i} \pm V_{B, i})|$$

### 7.2 Registered Null Behavior (Query-Blind Control)
For the registered query-blind control returning two candidate values in fixed order under $N=2$:
$$P(\text{both within } 0.50) = \frac{1}{2} + \frac{1}{2} P(|V_1 - V_2| < 0.50) = \frac{167}{288} \approx 0.5799$$
For $N=3$: $\text{Chance}_{\text{blind}} \approx 0.1667$.

### 7.3 Feasible Baseline Validity Condition
- For $N=2$: $\text{JointDeliveryAcc}(\text{M1}) \ge 0.50$ (validating above-chance operation, acknowledging the 0.5833 erasure ceiling).
- For $N=3$: $\text{JointDeliveryAcc}(\text{M1}) \ge 0.25$ ($> 1.5 \times \text{Chance}$).
- If M1 fails these baselines, declare `DYNAMIC_CARRIER_INSUFFICIENT` and halt without evaluating equivalence.

### 7.4 TOST Equivalence via Newcombe Method 10
To test if membrane clamping degrades accuracy ($d = \text{JointDeliveryAcc}(\text{M1}) - \text{JointDeliveryAcc}(\text{M2a})$):
1. Compute the $90\%$ confidence interval $[\theta_L, \theta_U]$ using **Newcombe (1998) Method 10** for paired proportions with continuity correction:
   $$\theta_L = \hat{d} - \sqrt{(\hat{p}_1 - l_1)^2 + (u_2 - \hat{p}_2)^2 - 2 \hat{\phi} (\hat{p}_1 - l_1)(u_2 - \hat{p}_2)}$$
   $$\theta_U = \hat{d} + \sqrt{(u_1 - \hat{p}_1)^2 + (\hat{p}_2 - l_2)^2 - 2 \hat{\phi} (u_1 - \hat{p}_1)(\hat{p}_2 - l_2)}$$
   where $(l_i, u_i)$ are Wilson score intervals for marginal proportions, and $\hat{\phi}$ is the discordant pair correlation.
2. **Equivalence Declared:** If $[\theta_L, \theta_U] \subset (-0.005, +0.005)$ ($\pm 0.5$ percentage points).
   - **Permitted Conclusion:** *“Within the registered tasks, delays, and 0.5% equivalence margin, membrane-potential reset had no practically meaningful effect on operand delivery.”*

### 7.5 Spike Channel Decision Rule (Central 95% Bootstrap)
Compute central 95% bootstrap confidence interval ($B=2,000$, trial-resampling, seed $2026091901$):
- **Demonstrated Channel Success:** $\text{UCB}_{95\%}(E_{\text{comp}}) < 0.05$.
- **Demonstrated Channel Limitation:** $\text{LCB}_{95\%}(E_{\text{comp}}) \ge 0.05$.
- **Inconclusive:** Interval spans $0.05$.

---

## 8. Executable G1 Gates Specification & Concrete Annex (G1.0 – G1.9)

### 8.1 G1.5 Linear Subthreshold State Collision Fixture
- In subthreshold linear regime without spikes, the effective discrete recurrence weight for slot $i$ is:
  $$c_i = e^{-(T_{\text{enc}} - t_i)/\tau_m} \cdot \bar{A}_i \cdot w_V \cdot \Delta t_{\text{event}}$$
- **Test Case:** Fixed keys $K_1 = 1.8, K_2 = 2.2$. Baseline values $V_1 = -1.5, V_2 = 1.5$ ($|V_1 - V_2| = 3.0 > 1.0$).
- Perturbation: $\Delta V_1 = \delta = 0.5$, $\Delta V_2 = -(c_1 / c_2) \cdot \delta$.
- **Assertion:** Full state vectors $\mathbf{z}(T_{\text{enc}})$ match within machine precision ($||\mathbf{z}_{\text{pert}} - \mathbf{z}_{\text{base}}||_{\infty} < 10^{-10}$), yet target operand differs by $\Delta V_1 = 0.5$, proving non-injectivity of the scalar linear subthreshold state.

### 8.2 G1.7 Numerical Accuracy and Solver Refinement
- **Analytical Benchmark:** Smooth subthreshold decay $\dot{u} = -(u - I)/\tau_m$ with $I=0.5, \tau_m=10.0\text{ ms}, u(0)=0.0$.
  - At $t=5.0\text{ ms}$, $u_{\text{exact}} = 0.5(1 - e^{-0.5}) \approx 0.19673467$.
  - **Tolerance for RK4 ($\Delta t = 0.1\text{ ms}$):** $|u_{\text{RK4}} - u_{\text{exact}}| < 10^{-6}$.
  - **Convergence Requirement:** Step refinement from $0.1$ to $0.05\text{ ms}$ reduces error by at least factor of 10.
- **Event Boundary Assertion:** Spike recorded at $t + \Delta t$; any spike with timestamp exactly $T_{\text{cue}} + 10.0\text{ ms}$ is excluded from $[T_{\text{cue}}, T_{\text{cue}} + 10.0\text{ ms})$.

### 8.3 G1 Summary Table

| Gate ID | Check | Acceptance Criteria |
|---|---|---|
| **G1.0** | Version 2.2 Lock | H3 deferred; Reference M0 locked to $-(q-k)^2$. |
| **G1.1** | Dimensional Closure | $\mathbf{z}(t) = [u, A, r]^T$; refractoriness enforced during ODE steps. |
| **G1.2** | Streaming Timeline | Frozen $W=5$ positions preserved; cue-to-delivery strictly 10.0 ms half-open. |
| **G1.3** | Leakage Audit | Query-blind control delivery matches $167/288 \pm 0.05$; oracle achieves 100%. |
| **G1.4** | Factorial Fidelity | M1, M2a, M2b, M2d, M2c verified in trace assertions. M2b uses calibrated $A_0$. |
| **G1.5** | State Collision | Non-overlapping target values ($|V_1 - V_2| \ge 2.0$) with identical state to $< 10^{-10}$. |
| **G1.6** | Codec Benchmarks | $M=6$ and $M=20$ reference benchmarks ($0.3333$ and $0.1000$) verified analytically. |
| **G1.7** | Numerical Accuracy | RK4 error $< 10^{-6}$ vs exact analytical solution; window boundary checked. |
| **G1.8** | Statistical Methods | Newcombe Method 10 and bootstrap verified on synthetic edge-case tables. |
| **G1.9** | Budget & Hashes | Execution budget $\le 2\text{ CPU hours}$, peak RAM $\le 2\text{ GB}$, SHA-256 manifest. |
