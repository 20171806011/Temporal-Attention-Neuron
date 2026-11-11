> Historical archival document. Original language preserved. Personal paths/host metadata and unavailable media links were sanitized; consult README.md and ARCHIVE_INDEX.json.

# Frozen implementation numerical specification

This description was written during report preparation from the already frozen model.py, data.py, common.py and run.py. Those source hashes, rather than this later explanatory file, establish the pre-data numerical specification. No equations, data distribution or selection rule changed after exploration.

## Native equations

For h hidden somata, d has 2h entries, v and previous hidden spikes p have h entries, and output membrane o has two entries. Every episode starts from zero. Parameters are wi:(6,2h), wr:(h,2h), db:(2h,), vb:(h,), wo:(h,2), ob:(2,), totaling 2h²+17h+2.

```
dc = rd*d + .25*(x@wi + p@wr + db)
DEND: local=max(dc,0); d=dc
POINT: bs=1[dc>.5]; local=bs; d=dc*(1-bs)
vc = rs*v + .5*sum(two local inputs per soma) + vb
hs = 1[vc>.5]; v=vc*(1-hs)
oc = rs*o + p@wo + ob
os = 1[oc>.5]; o=oc*(1-os); p=hs
```

The output receives previous-step hidden spikes. Count os only for exactly eight steps beginning at q=16+D. Unique larger count predicts; a tie or silence abstains and is scored wrong. No analog readout, external spike generator or label argument enters forward dynamics. DEND's mathematical graded branches are not claimed biophysical cells. Both architectures retain subthreshold membrane states.

Initial wi draws are Normal(0,.45); wr draws Normal(0,.25) times sqrt(6/h); wo draws Normal(0,.5) times sqrt(6/h). Biases db=.15, vb=.05, ob=.1. Draw order is wi, wr, db, vb, wo, ob, with fixed constants consuming no RNG. Within a block and size the architectures/candidates share the initial parameter vector. The initializer changes only recurrent/output fan-in scale with h; this does not isolate every effect of size on optimization or activation.

## Task distribution and splits

Pattern p in 0..15 is balanced exactly, bits b_j=(p>>j)&1, and y=(b0 AND b1) OR (b2 AND b3). Seven patterns are positive and nine negative. Balanced accuracy averages the two class recalls including abstentions; ordinary majority accuracy would be 9/16, while majority balanced accuracy is .5.

Each example independently draws four high amplitudes U[.8,1.2] and four low amplitudes U[0,.1], selects them by the latent bits, repeats for eight writing steps, adds independent Gaussian SD .03 to each of the 8×4 writing currents and clips negatives to zero. Four distractor values are independently U[0,.5]. RNG order: shuffle balanced patterns, high, low, writing noise, distractor.

Warmup steps 0–7 have all external currents zero. Writing steps 8–15 carry the four feature currents. Hold steps 16 through 15+D have all six currents exactly zero in clean conditions. For interference conditions, only channel 5 at steps 18–21 receives the four saved distractor values. Query steps 16+D through 23+D have only channel 4 equal to one. There is no noise outside the writing window except the explicitly assigned distractor. Internal learned biases and activity persist during clean hold.

Training has four groups of 32 distinct base examples: (D0,clean), (D8,clean), (D24,clean), (D24,interference). Each group contains two of each Boolean pattern. The full update therefore uses 128 examples, with a uniform mean across four group losses. Tuning uses 128 new base examples per tuning block and averages BA across those same four conditions. Exploration uses 256 new base examples per fresh training block and evaluates seven delay/interference conditions plus five D24-clean ablations. Shared bases across conditions are paired observations, not extra independent data blocks.

There are two tuning blocks and three new exploration training/data blocks. The planned five validation blocks were never generated because every prospective entrance gate failed. Exact executed and unused seeds appear in SEED_REGISTRY.json. D48 is outside training-delay support; D8 interference is an unseen combination, not an unseen latent Boolean pattern.

## Optimization and selection

Each fit has exactly 200 full-batch Adam updates, lr=.01, beta1=.9, beta2=.999, epsilon=1e-8. Average the four raw group gradients, globally clip the resulting Euclidean norm at one, perform Adam bias correction, and clip parameters to [-2,2]. Save the fixed final checkpoint; no restart, checkpoint selection or extension.

RATE uses target .75 for the correct output and .125 for the other, with mean square error on count/8 over examples and two outputs. CE uses softmax logits .5*counts and mean cross-entropy. Both add .0001*mean(theta²). The hard-spike backward surrogate is 1/(1+5*abs(candidate−.5))² with inherited reset derivatives. It is a local proxy learning graph, not the derivative of the discontinuous native map. Four-group averaging leaves one effective regularizer contribution.

Candidates are RATE/CE × (rd,rs)=(.95,.85)/(.99,.95), all equally budgeted. Each architecture/size is selected independently by highest two-block mean validation BA, then coverage, then lowest candidate index. All candidate failures are retained. rd=0 and rs in {.85,.95} were checked only on deterministic fixtures; no scientific restricted-model training occurred.

## Intervention and cost semantics

At D24-clean hold midpoint t=28, zero d, zero v, or zero d/v/o/p once before that step's update. A same-time copy operation is the sham. Recurrence cutting zeros p@wr at every hold update t=16..39, without preventing native hidden spikes or hidden-to-output transmission. These are inference interventions only; loss_grad explicitly rejects transient-intervention training.

Per episode continuous-state updates=(3h+2)*T; h previous-spike registers and two decoder counters are distinct. Dense forward MAC=(2h²+14h)*T and backward MAC=(4h²+16h)*T, excluding input gradients. Dense zero computations are not sparse hardware savings. Loss, scalar nonlinearities/reset, optimizer, audit and file I/O are outside these matrix counts.

All trained recurrent/output weights were checked nonzero before using dense fanout 2h and 2 in delivery formulas. Deliveries use the consumed previous spikes; the final hidden spikes are not counted as delivered beyond the simulated window. During recurrence cuts those recurrent deliveries are removed, while emitted events and other deliveries remain counted. POINT branch spikes each deliver to one soma. DEND has 2h*T scheduled graded local transfers including zero-valued transfers. No conversion between these coordinates and joules, ATP, biological time or hardware energy was made.
