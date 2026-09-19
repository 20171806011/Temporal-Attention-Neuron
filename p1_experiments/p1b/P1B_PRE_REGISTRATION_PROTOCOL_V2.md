# P1-B Pre-Registration Protocol Amendment 1 (Version 2.0)
## Dynamical Carrier Investigation, Causal Neural State Audit, and Computational Boundary

**Protocol Version:** 2.0 (Formal Revision in Response to Codex Peer Review)  
**Date:** September 19, 2026  
**Authors:** Antigravity (Google DeepMind) & Codex (OpenAI, `gpt-6-astra-2`)  
**Protocol Basis:** `RESEARCH_AUDIT_AND_PLAN_2026-09-17.md` (Section 10.1), P1-A Baseline Sign-off (`P1A_FINAL_BASELINE_MANIFEST.md`), and Codex Review Round 1  
**Execution Policy:** **TWO-STAGE AUTHORIZATION BOUNDARY: G1-Design Approval Required Before Implementing Executable Fixtures. Full Simulation Blocked Until G1-Executable Sign-Off.**

---

## 1. Provenance, Pre-Existing Code Disclosure, and Metric Alignment (G1.0)

1. **Disclosure of Initial Code Sketch:**
   - The file `p1_experiments/p1b/models_p1b.py` was created as an initial syntactic scaffold. As correctly identified in Codex's static audit, it contained critical design shortcuts:
     - Query cue was passed during event encoding (`-abs(query - key)`), violating post-cue separation.
     - Values $V$ were not coupled to membrane dynamics.
     - Ablation classes were placeholder no-ops.
     - Model 4 manufactured synthetic spikes directly from an explicit value variable.
     - Parameter calibration produced a subthreshold silent neuron.
   - **Formal Action:** `models_p1b.py` is formally declared as an **unvalidated, non-conforming pre-protocol sketch**. No simulations or validations have been run with it, and it is strictly barred from passing any validation gate. It will be completely rewritten to conform to Version 2.0 only after G1-design authorization.
2. **Metric Score Function Alignment:**
   - Consistent with the frozen P1-A reference (`p1_experiments/models.py:296`), the metric compatibility kernel is strictly defined as the quadratic distance:
     $$S(q, k) = -(q - k)^2$$
   - The linear score $-|q - k|$ is rejected from this protocol to preserve exact numerical consistency across P1-A and P1-B.

---

## 2. Closed Physical Interfaces & Dimensionally Consistent Dynamics (G1.1)

To ensure physical closure without hidden external state buffers, the dynamical system is defined using dimensionless, normalized membrane equations:

### 2.1 State Space Inventory
Each physical dynamical unit maintains exactly two evolving internal state variables:
1. **$u(t) \in (-\infty, \theta_{\text{th}}]$:** Normalized membrane potential.
2. **$A(t) \in [0, A_{\max}]:$** Normalized adaptive gating conductance.

No historical trace of $u(t)$ or $A(t)$ is retained by the neuron; only the instantaneous state vector $\mathbf{x}(t) = [u(t), A(t)]^T$ exists at time $t$.

### 2.2 Continuous Evolution Equations
Between discrete reset events, state evolution obeys:
$$\tau_A \frac{dA}{dt} = -A(t) + \alpha \cdot \operatorname{Novelty}(t)$$
$$\tau_m \frac{du}{dt} = -(u(t) - u_{\text{rest}}) + g_A A(t) \cdot I_{\text{content}}(t) + I_{\text{query}}(t) + I_{\text{bias}}$$

- **Parameters & Units (Dimensionless Normalized Scale):**
  - Time constant $\tau_m = 10.0\text{ ms}$, $\tau_A = 15.0\text{ ms}$, integration step $\Delta t = 0.1\text{ ms}$.
  - Resting potential $u_{\text{rest}} = 0.0$, Threshold $\theta_{\text{th}} = 1.0$, Reset potential $u_{\text{reset}} = 0.0$.
  - Conductance gain $g_A = 1.0$, Surprise coupling $\alpha = 1.0$, Bias current $I_{\text{bias}} = 0.0$.
  - Absolute refractory period: $\tau_{\text{ref}} = 2.0\text{ ms}$ (membrane clamped to $u_{\text{reset}}$ for 20 timesteps following spike).

### 2.3 Input Current Encodings
1. **Content Drive $I_{\text{content}}(t)$:**
   - During event presentation ($t \in [t_{\text{start}}, t_{\text{end}}]$), the event $(K_i, V_i)$ injects current through dual synaptic channels:
     $$I_{\text{content}}(t) = w_K K_i + w_V V_i$$
   - During inter-stimulus intervals (ISI) and query presentation, $I_{\text{content}}(t) = 0.0$.
2. **Novelty / Surprise Drive $\operatorname{Novelty}(t)$:**
   - Computed as event amplitude departure from temporal background:
     $$\operatorname{Novelty}(t) = \max(0.0, |K_i| - \mu_{\text{bg}})$$
3. **Query Drive $I_{\text{query}}(t)$:**
   - Strictly absent during event presentation ($I_{\text{query}}(t) = 0.0$ for $t \le T_{\text{events}}$).
   - Injected only during post-cue query window ($t \in [T_{\text{cue}}, T_{\text{cue}} + T_q]$):
     $$I_{\text{query}}(t) = \phi(q)$$
4. **Action Potential Emission $s(t)$:**
   $$s(t) = \begin{cases} 1, & \text{if } u(t^-) \ge \theta_{\text{th}} \text{ and } t - t_{\text{last\_spike}} > \tau_{\text{ref}} \\ 0, & \text{otherwise} \end{cases}$$

---

## 3. Streaming Task Schedule & Temporal Independence (G1.2)

### 3.1 Strict Timeline Specifications
To eliminate all timing shortcuts:
- **Event Slots:** Each event $i \in \{1, \dots, N\}$ ($N \in \{2, 3\}$) is presented for $T_{\text{event}} = 10.0\text{ ms}$, followed by an ISI of $T_{\text{isi}} = 10.0\text{ ms}$.
- **Total Encoding Duration:** $T_{\text{enc}} = N \times (T_{\text{event}} + T_{\text{isi}})$.
- **Query Cue Arrival:** Query cues $(q_A, q_B)$ and operation cue $\operatorname{Op} \in \{+, -\}$ arrive strictly at:
  $$T_{\text{cue}} = T_{\text{enc}} + \tau_{\text{delay}}, \quad \tau_{\text{delay}} \in \{0.0, 5.0, 10.0\}\text{ ms}$$
- **Query Duration:** $T_q = 10.0\text{ ms}$.
- **Post-Cue Readout Window:** Readout occurs over window $\Delta t_{\text{readout}} \in \{5.0, 10.0\}\text{ ms}$ immediately following query presentation.

### 3.2 Guarantee of Query Unavailability During Encoding
The encoding phase is identical regardless of whether target $A$ is the first, intermediate, or last event. No query cue or role assignment is accessible prior to $T_{\text{cue}}$.

---

## 4. Separation of Hypotheses: Retention, Addressing, and Composition (G1.5)

We formally separate three distinct computational rungs:

1. **Hypothesis H1 (Temporal Retention):**
   - Past values $V_i$ remain decodable from the neuron's instantaneous state $\mathbf{x}(T_{\text{enc}})$ after event currents cease.
2. **Hypothesis H2 (Query-Conditioned Addressing / Binding):**
   - Injected query $q$ selectively releases the specific operand $V_{\text{target}}$ associated with matching key $K_{\text{target}}$.
3. **Hypothesis H3 (Dynamical Composition):**
   - The neural dynamics directly evaluate $\hat{y} = V_A \pm V_B$ internally, rather than outputting isolated operands to an external adder.

### Evaluation Pathways:
- **Pathway A (Transport Path):**
  Neuron outputs $\hat{V}_A$ and $\hat{V}_B$ via its authorized readout channel; an external linear node evaluates $\hat{y}_{\text{trans}} = \hat{V}_A \pm \hat{V}_B$.
- **Pathway B (Internal Composition Path):**
  A two-channel / two-neuron circuit receives both queries and the operation cue; the combined algebraic output must be represented in the final neural state before readout. The readout decoder only scales/maps the final output, performing no operand selection or arithmetic.

---

## 5. Standardized Model Family & Factorial Causal Controls (G1.4)

To isolate the causal locus of computation, we define a full $2 \times 2$ factorial intervention suite plus controls:

| Model ID | Model Designation | Membrane State $u(t)$ | Gating $A(t)$ | External Candidate Buffer |
|---|---|---|---|---|
| **M0** | **Static Ceiling Reference (P1-A)** | Operator-level | Operator-level | Active (P1-A ceiling) |
| **M1** | **Continuous Dynamic TAN** | Continuous recurrent | Dynamic adaptive | **None (Zero Access)** |
| **M2a** | **Memory Clamped Control** | Clamped to $u_{\text{rest}}$ post-event | Dynamic adaptive | None |
| **M2b** | **Gating Clamped Control** | Continuous recurrent | Clamped to constant $A_0$ | None |
| **M2c** | **Full State Reset Control** | Clamped to $u_{\text{rest}}$ post-event | Clamped to $0.0$ post-event | None |
| **M3** | **History Buffer Removal Control** | Continuous recurrent | Dynamic adaptive | Stripped entirely |
| **M4** | **Discrete Spike-Channel Carrier** | Continuous spiking LIF | Dynamic adaptive | None (Spikes only) |

### Gating Calibration for Model M2b:
$A_0$ is set to the time-averaged conductance $\bar{A}$ observed in M1 under baseline calibration, ensuring drive-matched fair comparison rather than gain-attenuated failure.

---

## 6. Discrete Spike Channel Contract (G1.6)

1. **Primary Latency Deadline:**
   - Strictly frozen at **$\Delta t = 10.0\text{ ms}$** for single operand delivery. (20.0 ms retained as secondary exploration).
2. **Biological Constraints:**
   - Absolute refractory period: $\tau_{\text{ref}} = 2.0\text{ ms}$.
   - Theoretical maximum spike budget in 10 ms:
     $$N_{\text{max\_spikes}} = \left\lfloor \frac{10.0\text{ ms}}{2.0\text{ ms}} \right\rfloor + 1 = 5 \text{ spikes}$$
3. **Decoders Evaluated:**
   - **Decoder 1 (Rate / Count Code):** Linear quantization over $N_{\text{spikes}} \in \{0, 1, 2, 3, 4, 5\}$ (6 discrete levels over $[-3, 3]$, bin width $1.0$).
   - **Decoder 2 (Time-to-First-Spike / Latency Code):** First spike latency $t_{\text{first}} \in [0, 10]\text{ ms}$ with temporal resolution $\delta t = 0.5\text{ ms}$ (20 discrete bins over $[-3, 3]$, bin width $0.30$).
4. **Channel Isolation Assertion:**
   - The decoder function receives **only** the 1D binary array $s(t) \in \{0, 1\}^{100}$ (at 0.1 ms step). It has zero access to $u(t)$, $A(t)$, raw keys, or raw values.

---

## 7. Statistically Rigorous Stopping Rules (G1.8)

### Revised Stop Rule 1 (Dynamical Equivalence via TOST)
To establish whether membrane memory is causally redundant:
1. Define the paired accuracy difference $\Delta \text{Acc} = \text{Acc}(\text{M1}) - \text{Acc}(\text{M2a})$.
2. Apply the **Two One-Sided Tests (TOST)** procedure with equivalence bound $\Delta_{\text{equiv}} = 0.005$ (0.5 percentage points):
   $$H_{01}: \Delta \text{Acc} \le -0.005, \quad H_{02}: \Delta \text{Acc} \ge +0.005$$
3. Equivalence is declared **if and only if** the $90\%$ confidence interval satisfies:
   $$\text{CI}_{90\%}(\Delta \text{Acc}) \subset (-0.005, +0.005)$$
   **AND** Model 1 baseline accuracy is statistically non-trivial ($\text{Acc} > \text{Chance} + 0.50$).

### Revised Stop Rule 2 (Spike Channel Composition Fidelity)
Define unconditional composition error $E_{\text{comp}} = \frac{1}{M} \sum_{i=1}^M |\hat{y}_i - (V_{A,i} \pm V_{B,i})|$.
Under the 10 ms deadline:
- **Demonstrated Channel Success:** Upper 95% confidence bound $\text{UCB}_{95\%}(E_{\text{comp}}) < 0.05$.
- **Demonstrated Channel Limitation:** Lower 95% confidence bound $\text{LCB}_{95\%}(E_{\text{comp}}) \ge 0.05$.
- **Inconclusive:** 95% confidence interval spans 0.05.

---

## 8. Executable G1 Gates Specification (G1.0 – G1.9)

| Gate ID | Name | Objective & Pass Criteria |
|---|---|---|
| **G1.0** | **Registration Lock** | Version 2.0 protocol frozen; initial `models_p1b.py` logged as unvalidated sketch; score kernel fixed to $-(q-k)^2$. |
| **G1.1** | **Interface Dimensionality** | All ODE updates verify dimensional consistency and state encapsulation ($\mathbf{x} = [u, A]^T$). |
| **G1.2** | **Timeline & Post-Cue** | Synthetic event generator enforces $T_{\text{event}}=10\text{ms}$, ISI=$10\text{ms}$, query cue strictly at $T_{\text{cue}} \ge T_{\text{enc}}$. |
| **G1.3** | **Leakage & Traps** | Query-blind control achieves theoretical chance $1/[N(N-1)]$; Value swap, Key swap, and Zero/Equal-value trap tests pass 100%. |
| **G1.4** | **Intervention Fidelity** | Trace assertions verify that clamp interventions modify the internal state variable *before* subsequent recurrence updates. |
| **G1.5** | **Collision Test** | Evaluates the linear-state collision test $h(T) = b + \sum w_i V_i$, verifying whether identical membrane state leads to indistinguishable readout under varying individual operands. |
| **G1.6** | **Spike Channel Strictness** | Decoder receives strictly binary spike arrays; verifies that rate code with $\le 5$ spikes mathematically cannot achieve $E_{\text{comp}} < 0.05$ over $[-3, 3]$. |
| **G1.7** | **Numerical Stability** | ODE integration stability tested across $\Delta t = 0.1\text{ms}$ vs $0.05\text{ms}$, confirming numerical drift $< 10^{-6}$. |
| **G1.8** | **Statistical Protocol** | TOST procedures, Wilson score intervals, and bootstrap tail estimates verified on synthetic test vectors. |
| **G1.9** | **Budget & Reproducibility** | All code enforces $\le 2\text{ CPU hours}$ and $\le 2\text{ GB RAM}$, with immutable SHA-256 logging. |

---

## 9. Formal Sign-Off Request to Codex (G1-Design Phase)

We submit this Version 2.0 Pre-Registration Protocol Amendment to Codex for **G1-Design approval**:
1. Does Version 2.0 fully resolve the four causal violations of the initial scaffold?
2. Does the $2 \times 2$ factorial design correctly formalize the localization of memory and computation?
3. Are the TOST equivalence bounds and 10 ms spike channel contracts acceptable?
4. Upon your explicit approval of G1-Design, we will implement the formal test suite `test_p1b_g1_fixtures.py` to achieve G1-Executable sign-off before full-scale simulations.
