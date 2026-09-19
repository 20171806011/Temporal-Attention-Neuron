# P1-B Pre-Registration Protocol Amendment 4 (Version 2.3)
## Focused Carrier Audit: Temporal Retention, Content Addressing, and the Information-Erasure Boundary

**Protocol Version:** 2.3 (Comprehensive Revision Resolving All Codex Round 4 Findings)  
**Date:** September 19, 2026  
**Authors:** Antigravity (Google DeepMind) & Codex (OpenAI, `gpt-6-astra-2`)  
**Protocol Basis:** `RESEARCH_AUDIT_AND_PLAN_2026-09-17.md` (Section 10.1), P1-A Baseline Sign-off (`P1A_FINAL_BASELINE_MANIFEST.md`), and Codex Reviews Rounds 1–4  
**Scope Decision:** **H3 (Internal Dynamical Composition) is formally DEFERRED. P1-B focuses strictly on H1 (Temporal Retention), H2 (Content Addressing & Transport Delivery), and the 10 ms Readout Channel Boundary.**

---

## 1. Scope, Provenance, and Core Demarcation (G1.0)

1. **Scope:**
   - **Hypothesis H3 (Internal Spiking Composition within a single unit) is officially DEFERRED / NOT CLAIMED.** Single-unit dynamics are not asked to perform internal algebraic arithmetic without an external circuit.
   - P1-B evaluates relational composition exclusively via **Pathway A (Transport Path)**: the neuron routes and delivers the bound operands $\hat{V}_A, \hat{V}_B$, and an explicit, inspectable external linear node evaluates $\hat{y} = \hat{V}_A \pm \hat{V}_B$.
2. **Preliminary Code Status:**
   - The sketch `models_p1b.py` remains non-conforming and barred from execution.
3. **Reference Model Metric:**
   - M0 reference static model uses quadratic metric $S(q, k) = -(q - k)^2$ matching frozen P1-A (`models.py:296`).

---

## 2. Dynamically Closed State Space & Physical Euler Carrier (G1.1)

### 2.1 State Space Vector $\mathbf{z}(t)$
At every integration step $t$, the physical neuron is completely described by:
$$\mathbf{z}(t) = \begin{bmatrix} u(t) \\ A(t) \\ r(t) \end{bmatrix} \in \mathbb{R} \times [0, A_{\max}] \times [0, \tau_{\text{ref}}]$$
- $u(t) \in (-\infty, \theta_{\text{th}}]$: Normalized membrane potential ($u_{\text{rest}} = 0.0$, $\theta_{\text{th}} = 1.0$, $u_{\text{reset}} = 0.0$).
- $A(t) \in [0, A_{\max}]$: Normalized gating conductance ($\tau_A = 15.0\text{ ms}$, $\alpha = 1.0$, $A_{\max} = 5.0$, initial $A(0) = 0.0$).
- $r(t) \in [0, \tau_{\text{ref}}]$: Remaining refractory time ($\tau_{\text{ref}} = 2.0\text{ ms}$, initial $r(0) = 0.0$).

### 2.2 Continuous Euler Integration Equations ($h = 0.1\text{ ms}$)
1. **Gating Conductance Evolution:**
   $$A_{n+1} = \min\left(A_{\max}, \max\left(0.0, A_n + \frac{-A_n + \alpha \cdot \operatorname{Novelty}_n}{\tau_A} h\right)\right)$$
   $A$ evolves continuously across all states, including during refractoriness.
2. **Membrane & Refractory Evolution:**
   - If $r_n > 0$:
     $$r_{n+1} = \max(0.0, r_n - h), \quad u_{n+1} = u_{\text{reset}} = 0.0, \quad s_{n+1} = 0$$
   - If $r_n == 0$:
     $$u_{\text{cand}} = u_n + \frac{-(u_n - u_{\text{rest}}) + g_A A_n I_{\text{content}, n} + I_{\text{query}, n} + I_{\text{bias}}}{\tau_m} h$$
     - If $u_{\text{cand}} \ge \theta_{\text{th}}$:
       $$s_{n+1} = 1, \quad u_{n+1} = u_{\text{reset}} = 0.0, \quad r_{n+1} = \tau_{\text{ref}} = 2.0\text{ ms}$$
     - Else:
       $$s_{n+1} = 0, \quad u_{n+1} = u_{\text{cand}}, \quad r_{n+1} = 0.0$$

### 2.3 Input Currents & Parameter Values
- Time constants: $\tau_m = 10.0\text{ ms}$, $\tau_A = 15.0\text{ ms}$, $\tau_{\text{ref}} = 2.0\text{ ms}$.
- Weights: $w_K = 1.0$, $w_V = 1.0$, $g_A = 1.0$, $I_{\text{bias}} = 0.0$.
- Fixed temporal background baseline: $\mu_{\text{bg}} = 1.4$ (midpoint of key range $[0.3, 2.5]$).
- During active event presentation:
  $$I_{\text{content}}(t) = w_K K_i + w_V V_i, \quad \operatorname{Novelty}(t) = \max(0.0, |K_i| - \mu_{\text{bg}})$$
- During ISIs and query presentation:
  $$I_{\text{content}}(t) = 0.0, \quad \operatorname{Novelty}(t) = 0.0$$
- Query drive:
  $$I_{\text{query}}(t) = \begin{cases} 1.0 \cdot q, & \text{if } t \in [T_{\text{cue}}, T_{\text{cue}} + 10.0\text{ ms}) \\ 0.0, & \text{otherwise} \end{cases}$$

---

## 3. Streaming Schedule & Frozen Episode Mapping (G1.2)

1. **Fixed Episode History ($W=5$):**
   - Strictly preserves `TrialData.event_positions` and `mask` from the frozen P1-A generator (`task_generator.py`).
   - Slot duration: $T_{\text{slot}} = T_{\text{event}} + T_{\text{isi}} = 10.0\text{ ms} + 10.0\text{ ms} = 20.0\text{ ms}$.
   - Total encoding duration: $T_{\text{enc}} = 5 \times 20.0\text{ ms} = 100.0\text{ ms}$.
2. **Independent Repeated Probing:**
   - Role A and Role B queries ($q_A, q_B$) are evaluated as **independent repeated probes** of the identical encoded history episode.
   - For query cue $q$, cue onset is $T_{\text{cue}} = T_{\text{enc}} + \tau_{\text{delay}}$ ($\tau_{\text{delay}} \in \{0.0, 5.0, 10.0\}\text{ ms}$).
3. **Strict 10.0 ms Cue-to-Delivery Window:**
   - Half-open window: $[T_{\text{cue}}, T_{\text{cue}} + 10.0\text{ ms})$.
   - The readout decoder observes neural output strictly during this 10.0 ms window. No subsequent extra readout window is added.
   - Spikes recorded at $t + h$: a spike at exactly $T_{\text{cue}} + 10.0\text{ ms}$ is excluded from the window.

---

## 4. Balanced $2 \times 2$ Factorial Design & Causal Controls (G1.4)

| | Dynamic Gating $A(t)$ | Mean-Gate Matched Constant $A_0$ |
|---|---|---|
| **Retained Membrane $u(t)$** | **Model M1** (Intact TAN) | **Model M2b** (Gate Ablation) |
| **Post-Event Clamped Membrane $u(t)$** | **Model M2a** (Membrane Ablation) | **Model M2d** (Dual Ablation) |

### Intervention Rules:
- **Retained Membrane (M1, M2b):** State $\mathbf{z}(t)$ evolves without clamp.
- **Post-Event Clamped Membrane (M2a, M2d):** Immediately at each active event offset ($t = t_{\text{end}}$), $u(t) \leftarrow 0.0$ and $r(t) \leftarrow 0.0$. Gating state $A(t)$ is untouched.
- **Dynamic Gating (M1, M2a):** $A(t)$ evolves with novelty drive.
- **Mean-Gate Matched Constant (M2b, M2d):** $A(t) \equiv A_0$, where $A_0$ is the independent calibration mean conductance across 500 calibration trials ($A_0 = \bar{A}_{\text{calib}}$). Residual driving force differences are logged in G1.4.
- **Full-State Reset Control (M2c - Auxiliary):** Clamps $u(t) \leftarrow 0.0$, $A(t) \leftarrow 0.0$, and $r(t) \leftarrow 0.0$ at each event offset.
- **Strict No-Buffer Access:** M1, M2a-d, and M4 have zero access to candidate arrays, historical voltage logs, or evaluator ground truth.

---

## 5. Frozen Decoder Specifications & Codec Benchmarks

### 5.1 Continuous Analog Decoder for M1 and M2a–d
- The decoder reads the **left-limit / pre-reset analog membrane potential** at the deadline:
  $$u_{\text{readout}} = u_{\text{cand}}(T_{\text{cue}} + 10.0\text{ ms})$$
  (the unreset candidate potential evaluated at the 100th step of the cue window).
- Decoder function:
  $$\hat{V} = g_{\text{dec}} \cdot u_{\text{readout}} + b_{\text{dec}}$$
- Calibration: Linear regression coefficients $(g_{\text{dec}}, b_{\text{dec}})$ are fitted on the 500 calibration trials (seed `2026091999`) mapping $u_{\text{readout}}$ to $V_{\text{target}}$ and frozen identically across M1 and M2a–d.
- Permitted inputs: Exactly the scalar $u_{\text{readout}}$. Query $q$ is NOT an input to the decoder. Zero access to candidate arrays or history logs.

### 5.2 Discrete Spiking Decoders for Model M4
- Within the half-open window $[T_{\text{cue}}, T_{\text{cue}} + 10.0\text{ ms})$ with $\tau_{\text{ref}} = 2.0\text{ ms}$, maximum spike count is **5 spikes**.
- **Count Codebook (5 valid delivery symbols + abstention):**
  - Count $\in \{1, 2, 3, 4, 5\}$ maps to midpoints $\{-2.0, -1.0, 0.0, +1.0, +2.0\}$.
  - Count $0$ = Silence / Abstention: logs `Abstained = True`, sets fallback $\hat{V} = 0.0$.
- **Latency Codebook (20 bins of width 0.5 ms):**
  - First spike latency $t_{\text{first}} \in [0.0, 10.0)\text{ ms}$ quantized into 20 bins (midpoints $0.25 + k \times 0.5\text{ ms}$), mapped to 20 value reconstructions over $[-3.0, 3.0]$ spaced by $0.30$.
  - Silence (no spikes in 10 ms): logs `Abstained = True`, sets fallback $\hat{V} = 0.0$.
- **Silence & Abstention Accounting:**
  - Abstention is unsuccessful delivery: $\operatorname{DeliverySuccess} = \text{False}$.
  - In unconditional composition MAE ($E_{\text{comp}}$), the fallback $\hat{V} = 0.0$ is retained without extra penalty.
  - Abstention rate is reported separately.

### 5.3 Pre-Simulation Analytical Codec Benchmarks
For independent uniform operands $V_A, V_B \sim \mathcal{U}[-3, 3]$ under an ideal independent midpoint quantizer:
$$\mathbb{E}[|e_A \pm e_B|] = \frac{2}{M}$$
- Reference prediction for $M=6$ (Count): $\mathbb{E}[E_{\text{comp}}] = 2/6 = 0.3333$.
- Reference prediction for $M=20$ (Latency): $\mathbb{E}[E_{\text{comp}}] = 2/20 = 0.1000$.
- **Pre-Registered Conclusion:** *“For the ideal independent midpoint-quantizer reference under independent uniform operands, population composition MAE is $1/3$ for six bins and $0.10$ for twenty bins. These are reference predictions, not universal lower bounds for the neuronal implementation or its finite-sample results.”*

---

## 6. Model-Specific Information-Erasure Boundary of M1

### Proposition 6.1 (Model-Specific Information-Erasure Boundary of M1)
*Under M1’s initialization ($A(0)=0$) and no-bypass input contract, the first active event is erased whenever its key is at or below 1.4 ($K_1 \le 1.4$). More generally, this applies to an event that begins with $A(t_i)=0$ and receives zero novelty throughout its presentation.*

*For the in-domain clean $N=2$ generator with keys $K \sim \mathcal{U}[0.3, 2.5]$, reflection symmetry about 1.4 yields $P(K_1 \le 1.4) = 0.5$. Since both events are requested under $N=2$, an entirely unobserved independent $V \sim \mathcal{U}[-3, 3]$ can be delivered within tolerance $\pm 0.50$ with probability at most $1/6$. Therefore, the population joint delivery success of M1 is theoretically bounded by:*
$$P(\text{JointDeliveryAcc}_{\text{M1}}) \le \frac{1}{2} \cdot 1.0 + \frac{1}{2} \cdot \frac{1}{6} = \frac{7}{12} \approx 0.5833$$

**Scientific Prediction:**  
Because M1 is mathematically bounded by $\le 0.5833$ on $N=2$, Model M1 is pre-registered to **FAIL** the high-fidelity baseline target ($\ge 0.80$). This is registered as a formal, predicted negative result demonstrating that the unadapted surprise-gated leaky membrane carrier is insufficient for multi-slot relational memory delivery.

---

## 7. Endpoints, Null Models, and Decision Rules (G1.8)

### 7.1 Endpoints
1. **Delivery Accuracy (`DeliveryAcc`):**
   $$\operatorname{DeliverySuccess}_i = \mathbb{I}(|\hat{V}_i - V_{\text{target}, i}| < 0.50) \land \neg \operatorname{Abstained}_i$$
   (Tolerance $\Delta V = 0.50$).
2. **Joint Delivery Accuracy (`JointDeliveryAcc`):**
   $$\operatorname{JointDeliverySuccess} = \operatorname{DeliverySuccess}_A \land \operatorname{DeliverySuccess}_B$$
3. **Unconditional Composition Error ($E_{\text{comp}}$):**
   $$E_{\text{comp}} = \frac{1}{M} \sum_{i=1}^M |\hat{y}_i - (V_{A, i} \pm V_{B, i})|$$

### 7.2 Registered Null Models (Query-Blind Controls)
For the fixed-order, distinct-candidate query-blind control:
- For $N=2$: $P_{\text{blind}, 2} = \frac{167}{288} \approx 0.57986$.
- For $N=3$: $P_{\text{blind}, 3} = \frac{62}{243} \approx 0.25514$.

### 7.3 Feasible Baseline Performance Criteria
- Target benchmarks: $\text{JointDeliveryAcc} \ge 0.80$ (for $N=2$) and $\ge 0.40$ (for $N=3$).
- Pre-registered outcome for M1: By Proposition 6.1, M1 will fail the $N=2$ criterion ($\le 0.5833 < 0.80$), formally registering `M1_CARRIER_INSUFFICIENT`.

### 7.4 TOST Equivalence via Newcombe Method 10
To test if joint membrane and refractory reset degrades delivery accuracy ($d = \text{JointDeliveryAcc}(\text{M1}) - \text{JointDeliveryAcc}(\text{M2a})$):
1. **Formula (Newcombe 1998, Method 10 / CRAN `Newcombe_square_and_add_CI_paired_2x2`):**
   Let $2 \times 2$ table counts be $n_{11}, n_{10}, n_{01}, n_{00}$ with marginals $n_{1+} = n_{11} + n_{10}$, $n_{2+} = n_{01} + n_{00}$, $n_{+1} = n_{11} + n_{01}$, $n_{+2} = n_{10} + n_{00}$, and total $N$.
   Marginal proportions: $\hat{p}_1 = n_{1+} / N$, $\hat{p}_2 = n_{+1} / N$, $\hat{d} = \hat{p}_1 - \hat{p}_2$.
   Compute Wilson score intervals for $p_1$: $[l_1, u_1]$, for $p_2$: $[l_2, u_2]$ (at $\alpha = 0.10$ for 90% CI).
   Correlation coefficient $\psi$:
   - If $n_{1+} = 0$ or $n_{2+} = 0$ or $n_{+1} = 0$ or $n_{+2} = 0$: $\psi = 0.0$.
   - Else, with $n_{\text{prod}} = n_{1+} n_{2+} n_{+1} n_{+2}$ and $A = n_{11} n_{00} - n_{10} n_{01}$:
     $$\psi = \begin{cases} (A - N/2) / \sqrt{n_{\text{prod}}}, & \text{if } A > N/2 \\ 0.0, & \text{if } 0 \le A \le N/2 \\ A / \sqrt{n_{\text{prod}}}, & \text{if } A < 0 \end{cases}$$
   Square-and-add confidence limits:
   $$\theta_L = \hat{d} - \sqrt{(\hat{p}_1 - l_1)^2 + (u_2 - \hat{p}_2)^2 - 2 \psi (\hat{p}_1 - l_1)(u_2 - \hat{p}_2)}$$
   $$\theta_U = \hat{d} + \sqrt{(\hat{p}_2 - l_2)^2 + (u_1 - \hat{p}_1)^2 - 2 \psi (\hat{p}_2 - l_2)(u_1 - \hat{p}_1)}$$
2. **Equivalence Declared:** If $[\theta_L, \theta_U] \subset (-0.005, +0.005)$.
   - **Permitted Conclusion:** *“Within the registered tasks, delays, and 0.5% equivalence margin, joint membrane-potential and refractory reset had no practically meaningful effect on operand delivery.”*

### 7.5 Spike Channel Decision Rule (Central 95% Bootstrap)
Trial-resampled central 95% bootstrap ($B=2,000$, seed `2026091901`, resampling complete trial tuple containing both probes):
- **Demonstrated Channel Success:** $\text{UCB}_{95\%}(E_{\text{comp}}) < 0.05$.
- **Demonstrated Channel Limitation:** $\text{LCB}_{95\%}(E_{\text{comp}}) \ge 0.05$.
- **Inconclusive:** Interval spans $0.05$.

---

## 8. Executable G1 Gates Specification & Concrete Implementation Annex

### 8.1 G1.5 Exact Linear Subthreshold State Collision Fixture
- **Exact Recurrence Value Coefficient:**
  For Euler step $h = 0.1\text{ ms}$, $\lambda = 1 - h/\tau_m = 0.99$, and $N_T = T_{\text{enc}} / h = 1000$:
  $$c_i = \frac{h g_A w_V}{\tau_m} \sum_{n \in \mathcal{E}_i} A_n \lambda^{N_T - 1 - n}$$
  where $\mathcal{E}_i$ is the exact set of step indices for event slot $i$.
- **Concrete Test Histories:**
  - In subthreshold regime ($w_K = 0.1, w_V = 0.1$, no spikes throughout):
  - Fixed keys: $K_1 = 1.8, K_2 = 2.2$.
  - History $H$: $V_1(H) = -1.5, V_2(H) = 0.0$.
  - History $H'$: $V_1(H') = +1.0$ (separation $\Delta V_1 = 2.5 > 1.0$).
  - Set compensating value: $V_2(H') = V_2(H) - \frac{c_1}{c_2} \Delta V_1$.
- **Assertion:** Full state $\mathbf{z}(T_{\text{enc}})$ matches within machine precision ($||\mathbf{z}(H) - \mathbf{z}(H')||_{\infty} < 10^{-10}$), yet target operand $V_1$ differs by $2.5 > 1.0$ (disjoint $\pm 0.50$ acceptance intervals $[-2.0, -1.0]$ vs $[0.5, 1.5]$), proving that no single scalar readout can deliver $V_1$ accurately on both histories.

### 8.2 G1.7 Euler Accuracy & Refinement Gate
- **Analytical Benchmark:** Smooth decay $\dot{u} = -(u - I)/\tau_m$ with $I = 0.5, \tau_m = 10.0\text{ ms}, u(0) = 0.0$.
  Exact analytical value: $u_{\text{exact}}(5.0\text{ ms}) = 0.5(1 - e^{-0.5})$.
- **Euler Carrier Validation ($h = 0.1\text{ ms}$):**
  $$|u_{\text{Euler}}(5.0\text{ ms}) - u_{\text{exact}}(5.0\text{ ms})| < 1.0 \times 10^{-3}$$
- **First-Order Convergence Check:**
  Halving $h$ from $0.1$ to $0.05\text{ ms}$ reduces absolute error by at least a factor of $1.8$:
  $$\frac{|u_{\text{Euler}, 0.1} - u_{\text{exact}}|}{|u_{\text{Euler}, 0.05} - u_{\text{exact}}|} \ge 1.8$$
- **Independent RK4 Check:** $|u_{\text{RK4}, 0.1} - u_{\text{exact}}| < 1.0 \times 10^{-6}$.
- **Boundary Assertion:** Any spike recorded at exactly $t = T_{\text{cue}} + 10.0\text{ ms}$ is excluded from $[T_{\text{cue}}, T_{\text{cue}} + 10.0\text{ ms})$.

### 8.3 Calibration Parameters (G1.8)
- Calibration seed: `2026091999`.
- Calibration dataset: 500 independent trials, $N=2$ (250 trials) and $N=3$ (250 trials), delays $\tau \in \{0.0, 5.0, 10.0\}\text{ ms}$, roles pooled.
- Global mean conductance $A_0 = \frac{1}{500} \sum_{k=1}^{500} \bar{A}_{\text{M1}, k}$.
- Linear regression $(g_{\text{dec}}, b_{\text{dec}})$ fitted to map $u_{\text{readout}} \to V_{\text{target}}$ on this dataset.
