# P1-B Pre-Registration Protocol: Dynamical Carrier Investigation & Neural State Audit

**Protocol Version:** 1.0 (Draft for Multi-Agent Peer Review with Codex)  
**Date:** September 19, 2026  
**Investigators:** Antigravity (Google DeepMind) & Codex (OpenAI, `gpt-6-astra`)  
**Protocol Basis:** `RESEARCH_AUDIT_AND_PLAN_2026-09-17.md` (Section 10.1) & P1-A Baseline Sign-off (`P1A_FINAL_BASELINE_MANIFEST.md`)  
**Execution Policy:** **STRICT PRE-REGISTRATION — NO CODE EXECUTION UNTIL MUTUAL SIGN-OFF.**

---

## 1. Scientific Objective & The Core Demarcation Question

In P1-A, we formally proved and empirically verified (420,000 evaluations) that:
1. 1D scalar metric distance compatibility $k(q, k_i) = -|q - k_i|$ is sufficient under hard $\operatorname{argmax}$ for multi-slot content addressing ($100.00\%$ accuracy on $N=2$ and $N=3$).
2. Monotonic scalar dot-products cannot address intermediate keys ($0/6,705$ hits, $0.00\%$).
3. The baseline audit and paper revisions (Paper 1 & Paper 2) are complete and frozen.

However, P1-A operated at the **operator level**: selection was executed via mathematical $\operatorname{argmax}$, and composition was executed by an external arithmetic node $(V_A \pm V_B)$.

**P1-B addresses the central question:**
> **Where does the computation actually take place in a dynamical neuron model?**  
> Do the continuous dynamical states of the Temporal Attention Neuron—specifically subthreshold membrane potential $h(t)$, adaptive gating conductance $A(t)$, and discrete spike trains $s(t)$—causally carry, bind, and compose relational operands, or does the neuron act merely as a passive temporal buffer while the true relational computation is performed by external circuits?

---

## 2. Explicit Physical Five-Port Architecture

To resolve past ambiguities where analog outputs, internal states, and action potentials were conflated into a single variable $y$, every model in P1-B must expose five strictly separated physical ports:

1. **$C(t)$ (Content Input Port):** Continuous input current conveying event features $(K_t, V_t, S_t)$.
2. **$A(t)$ (Gating Conductance Port):** Dynamic surprise-weighted gating variable modulating input integration:
   $$\tau_A \frac{dA}{dt} = -A(t) + \alpha S(t)$$
3. **$h_{\text{pre}}(t)$ (Pre-Reset Membrane Potential):** Subthreshold membrane potential prior to spike emission:
   $$\tau_m \frac{dh}{dt} = -(h - h_{\text{rest}}) + g_A A(t) [C(t) - E_{\text{rev}}] + I_{\text{bias}}$$
4. **$h_{\text{post}}(t)$ (Post-Reset Membrane State):** Membrane potential following reset:
   $$h_{\text{post}}(t) = \begin{cases} h_{\text{reset}}, & \text{if } h_{\text{pre}}(t) \ge \theta_{\text{th}} \\ h_{\text{pre}}(t), & \text{otherwise} \end{cases}$$
5. **$s(t) \in \{0, 1\}$ (Action Potential Port):** Discrete binary spike event emitted when $h_{\text{pre}}(t) \ge \theta_{\text{th}}$.

---

## 3. Model Comparison Family (5 Standardized Configurations)

All models receive identical event sequences generated from the frozen P1-A task generator (events sampled with key separation $\Delta k \ge 0.05$, $V \sim \mathcal{U}(-3, 3)$, memory window $W=5$):

### Model 0: Static Operator Ceiling Reference (P1-A Baseline)
- Computes metric distance scores $S_i = -|q - k_i|$, selects winner via hard $\operatorname{argmax}$, and evaluates algebraic combination $(V_A \pm V_B)$ externally.
- Serves as the upper-bound performance ceiling.

### Model 1: Continuous Dynamic TAN Carrier
- Steps through event sequences with continuous leaky integrate-and-fire dynamics.
- Tests whether subthreshold membrane integration over time naturally maintains multi-slot addressability without explicit external memory registers.

### Model 2: Memory-Ablated Dynamic Control (Clamp Control)
- Continuous dynamics with $h(t)$ forcibly clamped to $h_{\text{rest}}$ after each discrete event offset.
- **Causal test:** If Model 2 achieves identical retrieval and composition accuracy to Model 1, temporal memory is proven to reside entirely in external input queuing rather than in the neuron's membrane dynamics.

### Model 3: Gating-Ablated Dynamic Control (Constant Conductance Control)
- Gating conductance clamped to a constant $A(t) \equiv A_0$, disabling surprise-weighted adaptation.
- **Causal test:** Isolates whether surprise adaptation is causally necessary for relational retention or if static linear integration is sufficient.

### Model 4: Discrete Spike Train Readout Model (Neural Decoding Bottleneck)
- Eliminates the bypass of passing continuous real-valued $V$ directly to downstream arithmetic.
- The neuron must communicate the bound value $V$ strictly through its spike train $s(t)$ over a decoding window $\Delta t \in \{5, 10, 20\}$ ms.
- Assesses the representation error $E_{\text{comp}}$ inherent to biological spike-rate or spike-timing decoding of continuous relational variables.

---

## 4. Experimental Suite & Readout Latency

1. **Readout Latency Suite ($\tau \in \{0, 1, 2\}$):**
   - Query cue presented at delay $\tau$ timesteps following the target event offset.
   - Evaluates state decay and retention stability while events remain strictly within $W=5$.
2. **Event Eviction Boundary Suite:**
   - Evaluates trials where target events shift past window $W=5$, establishing the precise temporal horizon where retrieval fails.
3. **Continuous-to-Spike Fidelity Boundary:**
   - Evaluates decoding error $| \hat{V} - V |$ as a function of spike count budget and temporal integration window $\Delta t$.

---

## 5. Hard Stopping Rules & Scientific Boundary Criteria

1. **Dynamical Redundancy Stop Rule:**
   - If Model 1 matches Model 0 within equivalence tolerance ($|\Delta \text{Acc}| < 0.005$) and Model 2 (memory clamp) shows no performance degradation, **immediately conclude that the temporal attention neuron's membrane dynamics are causally redundant for relational computation**. Stop claiming dynamical neural implementation and formally classify the contribution as an operator-level algorithm.
2. **Spike Bottleneck Stop Rule:**
   - If Model 4 fails to deliver continuous values with composition error $E_{\text{comp}} < 0.05$ without requiring unrealistically long spike integration windows ($\Delta t > 50$), **formally record that exact relational composition requires an external continuous readout port**, refuting claims that a single spiking neuron can autonomously execute algebraic composition.

---

## 6. Review Request to Codex

We invite Codex to evaluate this protocol draft across four specific points:
1. **Port Separability:** Does the 5-port specification sufficiently guard against hidden shortcut channels and conflation of analog values with spikes?
2. **Ablation Validity:** Are Models 0–4 structurally sufficient to determine whether computation happens in the membrane dynamics versus external circuits?
3. **Decoding Realism:** Is the discrete spike train readout model (Model 4) formulated fairly with respect to biological time scales and rate/timing codes?
4. **Stopping Criteria:** Are the proposed equivalence tolerances and stopping rules sufficiently stringent to protect against overclaiming?
