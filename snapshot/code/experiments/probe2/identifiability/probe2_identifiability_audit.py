"""
Phase 3 Probe-2 - Identifiability audit (Query-conditioned binding)
====================================================================
HARD CONSTRAINT: no formal training, no formal outputs, no tuning.
Single lightweight audit script (<10 s, deterministic, prints only).
Does NOT write result files and does NOT modify Phase 1/2/Probe-1 files.

Task (frozen candidate):
  E_A = (K_A, V_A), E_B = (K_B, V_B);  K_amp(A)=1.0, K_amp(B)=2.0,
  V in {+1,-1} iid fair;  composite scalar x(E) = K_amp + V*0.25:
    (A,+1)=1.25 (A,-1)=0.75 (B,+1)=2.25 (B,-1)=1.75
  history positions t-4..t-1: A and B occupy two distinct random positions
  (uniform over the 12 arrangements); other two filled with N(0, 0.05^2).
  query x_t = 3.0 (q=A) | 4.0 (q=B); label y = V_q in {-1,+1}.

Gates:
  Audit A: linear logistic on z_B2=[h_{t-4..t-1}, x_t] must have
           balanced accuracy <= 0.58 (else shortcut: AUDIT A FAILED).
  Audit B: same linear logistic on B3 (no-attention) state features
           must be <= 0.58.
  Audit C: B4 attention at query step over the four history taps:
           C1 hit rate (argmax on target position) >= 0.50 (CI);
           C2 contrast E[a_target]/E[a_distractor] >= 2.0;
           C3 query-conditioned routing: JSD of mean-alpha distributions
           (history-normalised) between q=A and q=B with permutation
           p < 0.01.
  Audit D: counterfactuals (position/value/query permutation; identical
           history with both query codes) reported.
  Generator audit: E[V_q | x_i] conditional structure, single-tap
           predictive accuracy, I(q;y) (must be ~0), position balance.
  Theory audit: linear separability of the idealised noiseless dataset by
           convex-hull intersection LP; explicit witness pairs.

Any gate failure -> STATUS: AUDIT_FAILED (stop; analyse mechanism; no
tuning).  Only an all-PASS yields STATUS: AUDIT_PASSED and a separate
formal Probe-2 task may be proposed.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import natural_collision as nc

SEED = 20260904
N = 1000
W = 5
SIG = 0.05                      # noise sigma on empty history slots
KA, KB, DV = 1.0, 2.0, 0.25
CODE = {(0, 1): KA + DV, (0, -1): KA - DV, (1, 1): KB + DV,
        (1, -1): KB - DV}
QCODE = {0: 3.0, 1: 4.0}
GATE_BA = 0.58
GATE_HIT = 0.50
GATE_CONTRAST = 2.0
GATE_P = 0.01


def gen_trials(rng, n, sigma=SIG):
    """Returns X (n,5) history(4)+query, meta: pA,pB,q,V_A,V_B,y,p_target."""
    hist = np.zeros((n, 4))
    pA = np.empty(n, dtype=int)
    pB = np.empty(n, dtype=int)
    q = rng.integers(0, 2, size=n)
    VA = np.where(rng.integers(0, 2, size=n) == 0, -1.0, 1.0)
    VB = np.where(rng.integers(0, 2, size=n) == 0, -1.0, 1.0)
    for i in range(n):
        pos = rng.choice(4, size=2, replace=False)
        pA[i], pB[i] = pos[0], pos[1]
        hist[i, pos[0]] = CODE[(0, int(VA[i]))]
        hist[i, pos[1]] = CODE[(1, int(VB[i]))]
        hist[i] += rng.normal(0, sigma, 4)
        hist[i, pos[0]] -= 0.0
        hist[i, pos[1]] -= 0.0
    # note: events receive no extra noise in the frozen candidate
    # (undo the noise on occupied slots)
    for i in range(n):
        hist[i, pA[i]] = CODE[(0, int(VA[i]))]
        hist[i, pB[i]] = CODE[(1, int(VB[i]))]
    X = np.column_stack([hist, np.asarray([QCODE[int(qq)] for qq in q])])
    y = np.where(q == 0, VA, VB)
    p_target = np.where(q == 0, pA, pB)
    p_distr = np.where(q == 0, pB, pA)
    return (X, y, dict(pA=pA, pB=pB, q=q, VA=VA, VB=VB, pt=p_target,
                       pd=p_distr))


# ---------------- logistic regression (binary, y in {-1,+1}) --------------
def fit_logistic(Xtr, ytr, l2=1e-4, maxiter=800):
    n, d = Xtr.shape
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = (Xtr - mu) / sd

    def unpack(w):
        return w[0], w[1:]

    def loss(w):
        b, ww = unpack(w)
        s = Xs @ ww + b
        return float(np.mean(np.log1p(np.exp(-ytr * s)))) + \
            0.5 * l2 * float((ww ** 2).sum())

    def grad(w):
        b, ww = unpack(w)
        s = Xs @ ww + b
        g = -ytr / (1.0 + np.exp(ytr * s))          # (n,)
        gb = float(g.mean())
        gw = (g[:, None] * Xs).mean(axis=0) + l2 * ww
        return np.concatenate([[gb], gw])

    res = minimize(loss, np.zeros(d + 1), jac=grad, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-12))
    return mu, sd, res.x


def predict_logistic(mu, sd, w, Xte):
    Xs = (Xte - mu) / sd
    s = Xs @ w[1:] + w[0]
    return np.where(s >= 0, 1.0, -1.0)


def ba_of(pred, y):
    rec = [float((pred[y == k] == k).mean()) for k in (-1.0, 1.0)]
    return float(np.mean(rec))


def ba_ci(X, y, nboot=2000, rng=None):
    """Stratified split 80/20 + trial bootstrap CI of test balanced acc."""
    rng = np.random.default_rng(SEED + 1) if rng is None else rng
    tr, te = [], []
    for k in (-1.0, 1.0):
        idx = np.where(y == k)[0]
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        tr.append(idx[:cut])
        te.append(idx[cut:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    mu, sd, w = fit_logistic(X[tr], y[tr])
    pred = predict_logistic(mu, sd, w, X[te])
    obs = ba_of(pred, y[te])
    boot = []
    for _ in range(nboot):
        ix = rng.integers(0, len(te), size=len(te))
        boot.append(ba_of(pred[ix], y[te][ix]))
    return obs, (float(np.percentile(boot, 2.5)),
                 float(np.percentile(boot, 97.5)))


def jsd(p, q):
    p = np.asarray(p, float) / np.asarray(p, float).sum()
    q = np.asarray(q, float) / np.asarray(q, float).sum()
    m = 0.5 * (p + q)
    e = 1e-300
    kl = lambda a, b: float((a * np.log(np.maximum(a, e) /
                                        np.maximum(b, e))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    X, y, meta = gen_trials(rng, N)
    q = meta["q"]
    print("=== Probe-2 Identifiability Audit ===")
    print(f"seed={SEED} N={N} W={W} sigma_noise={SIG}")

    # ---------------- generator audit ----------------
    print("\nGenerator audit\n----------------")
    # position balance
    cnt = np.zeros(4, dtype=float)
    for p in (meta["pA"], meta["pB"]):
        for i in range(4):
            cnt[i] += float((p == i).sum())
    expect = N * 2 / 4
    chi2 = float((((cnt - expect) ** 2) / expect).sum())
    from scipy.stats import chi2 as chi2d
    p_bal = float(1 - chi2d.cdf(chi2, 3))
    print(f"position counts per slot (A+B): {cnt.astype(int).tolist()} "
          f"(chi2 p={p_bal:.3f})")
    # I(q;y) and chi2: y in {-1,1}, q in {0,1}
    tab = np.zeros((2, 2))
    for i in range(N):
        tab[q[i], 0 if y[i] < 0 else 1] += 1
    chi2q, pq, _, _ = chi2_contingency(tab)
    def mi_from_tab(T):
        T = T / T.sum()
        m = 0.0
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if T[i, j] > 0:
                    m += T[i, j] * np.log(T[i, j] / (T.sum(1)[i] *
                                                      T.sum(0)[j]))
        return float(m)
    print(f"I(q;y) = {mi_from_tab(tab):.2e}  (chi2 p={pq:.3f}, must be ~0)")
    # single-tap structure: E[V_q | x_i] binned, and single-tap BA
    print("per-slot conditional structure (max |E[V_q|x_i]| over 8 bins) "
          "and single-tap balanced accuracy:")
    worst_cond = 0.0
    best_1tap = (0.0, None)
    for i in range(4):
        xi = X[:, i]
        lo, hi = xi.min(), xi.max()
        edges = np.linspace(lo, hi, 9)
        mx = 0.0
        for b in range(8):
            m = (xi >= edges[b]) & (xi < edges[b + 1])
            if m.sum() >= 10:
                mx = max(mx, abs(float(y[m].mean())))
        worst_cond = max(worst_cond, mx)
        obs, ci = ba_ci(xi[:, None], y)
        if obs > best_1tap[0]:
            best_1tap = (obs, i)
        print(f"  slot {i}: max|E[Vq|xi-bin]|={mx:.3f}  single-tap BA="
              f"{obs:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
    # full-history single-tap test (all four slots jointly? no: spec is
    # single-position observation)
    print(f"worst per-slot conditional bias: {worst_cond:.3f}; best single-"
          f"slot BA: {best_1tap[0]:.3f} (slot {best_1tap[1]})")
    gen_ok = (best_1tap[0] <= GATE_BA) and (pq > 0.05)

    # ---------------- theory audit ----------------
    print("\nTheory audit\n------------")
    # idealised noiseless discrete set: 12 positions x 4 value combos x 2 q
    P = []
    import itertools
    for (pa, pb) in itertools.permutations(range(4), 2):
        for va in (-1.0, 1.0):
            for vb in (-1.0, 1.0):
                for qq in (0, 1):
                    h = np.zeros(4)
                    h[pa] = CODE[(0, int(va))]
                    h[pb] = CODE[(1, int(vb))]
                    z = np.concatenate([h, [QCODE[qq]]])
                    yy = va if qq == 0 else vb
                    P.append((z, yy))
    Zp = np.array([p[0] for p in P])
    Yp = np.array([p[1] for p in P])
    nplus = Pplus = Zp[Yp > 0]
    nminus = Pminus = Zp[Yp < 0]
    # convex hull intersection LP
    n1, n2 = len(Pplus), len(Pminus)
    c = np.zeros(n1 + n2)
    Aeq = np.vstack([np.hstack([Pplus.T, -Pminus.T]),
                     np.hstack([np.ones((1, n1)), np.zeros((1, n2))]),
                     np.hstack([np.zeros((1, n1)), np.ones((1, n2))])])
    beq = np.concatenate([np.zeros(5), [1.0, 1.0]])
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    hull_overlap = bool(res.success)
    # explicit close witness pair (min-distance opposite-label pair)
    bestd, wp = 1e9, None
    for a in Pplus:
        for b in Pminus:
            d = float(np.linalg.norm(a - b))
            if d < bestd:
                bestd, wp = d, (a, b)
    print(f"idealised discrete dataset: {len(P)} points; convex-hull "
          f"overlap: {hull_overlap} (=> linearly inseparable: "
          f"{hull_overlap})")
    print(f"closest opposite-label pair distance: {bestd:.3f}")
    if wp is not None:
        print(f"  witness +: {np.round(wp[0],2).tolist()}  witness -: "
              f"{np.round(wp[1],2).tolist()}")

    # ---------------- Audit A (B2 linear shortcut) ----------------
    print("\nAudit A - B2\n------------")
    obsA, ciA = ba_ci(X, y)
    print(f"balanced_accuracy: {obsA:.4f}")
    print(f"95% CI: [{ciA[0]:.4f}, {ciA[1]:.4f}]")
    stA = "PASS" if obsA <= GATE_BA else "FAIL"
    print(f"status: {stA}  (red line BA <= {GATE_BA})")

    # ---------------- Audit B (B3 uniform integration) ----------------
    print("\nAudit B - B3\n------------")
    su3 = nc.simulate(X, "noattn")
    F3 = np.column_stack([su3["u"], (su3["u"] > nc.THETA).astype(float)])
    obsB, ciB = ba_ci(F3, y)
    print(f"balanced_accuracy: {obsB:.4f}")
    print(f"95% CI: [{ciB[0]:.4f}, {ciB[1]:.4f}]")
    stB = "PASS" if obsB <= GATE_BA else "FAIL"
    print(f"status: {stB}  (red line BA <= {GATE_BA})")

    # ---------------- Audit C (B4 attention routing) ----------------
    print("\nAudit C - B4\n------------")
    su4 = nc.simulate(X, "tan", need_alpha=True)
    a5 = su4["alpha"][:, 4, :]                 # query-step alpha (N,5)
    a_hist = a5[:, :4]                         # four history taps
    am = a_hist.argmax(axis=1)
    hit = am == meta["pt"]
    hr = float(hit.mean())
    rng_b = np.random.default_rng(SEED + 2)
    hb = np.array([float((am[ix] == meta["pt"][ix]).mean())
                   for ix in (rng_b.integers(0, N, size=N)
                              for _ in range(2000))])
    hr_ci = (float(np.percentile(hb, 2.5)), float(np.percentile(hb, 97.5)))
    print(f"attention hit rate: {hr:.4f}  95% CI: "
          f"[{hr_ci[0]:.4f}, {hr_ci[1]:.4f}]  (random = 0.25)")
    stC1 = "PASS" if hr_ci[0] >= GATE_HIT else "FAIL"
    print(f"C1 status: {stC1}  (requirement >= {GATE_HIT})")
    a_t = a_hist[np.arange(N), meta["pt"]]
    a_d = a_hist[np.arange(N), meta["pd"]]
    R = float(a_t.mean() / max(a_d.mean(), 1e-12))
    print(f"target/distractor contrast R = {R:.3f}")
    stC2 = "PASS" if R >= GATE_CONTRAST else "FAIL"
    print(f"C2 status: {stC2}  (requirement >= {GATE_CONTRAST})")
    # C3 query-conditioned JSD on history-normalised alpha
    an = a_hist / np.maximum(a_hist.sum(axis=1, keepdims=True), 1e-12)
    meanA = an[q == 0].mean(axis=0)
    meanB = an[q == 1].mean(axis=0)
    Dobs = jsd(meanA, meanB)
    perm = []
    for _ in range(1000):
        qp = rng_b.permutation(q)
        mA = an[qp == 0].mean(axis=0)
        mB = an[qp == 1].mean(axis=0)
        perm.append(jsd(mA, mB))
    p_perm = float((1 + np.sum(np.asarray(perm) >= Dobs)) / (1 + 1000))
    print(f"query-conditioned JSD(mean alpha|q=A, |q=B) = {Dobs:.4f} "
          f"(permutation p = {p_perm:.4f})")
    stC3 = "PASS" if (Dobs > 0 and p_perm < GATE_P) else "FAIL"
    print(f"C3 status: {stC3}  (requirement p < {GATE_P})")
    # per-q hit rates (diagnostic)
    print(f"  hit rate | q=A: {float(hit[q == 0].mean()):.3f}   "
          f"hit rate | q=B: {float(hit[q == 1].mean()):.3f}")
    # where does attention point (diagnostic over 5 taps incl. query)?
    five_am = a5.argmax(axis=1)
    print(f"  argmax over full 5-tap window falls on current/query tap in "
          f"{float((five_am == 4).mean()):.3f} of trials")

    # ---------------- Audit D (counterfactuals) ----------------
    print("\nAudit D - counterfactuals\n-------------------------")
    # D3: query permutation (histories fixed)
    qp_all = rng_b.permutation(q)
    pt_p = np.where(qp_all == 0, meta["pA"], meta["pB"])
    hr_p = float((am == pt_p).mean())
    # D2: value permutation (swap codes within key)
    histp = X[:, :4].copy()
    Xp = X.copy()
    for i in range(N):
        histp[i, meta["pA"][i]] = CODE[(0, -int(meta["VA"][i]))]
        histp[i, meta["pB"][i]] = CODE[(1, -int(meta["VB"][i]))]
    Xp[:, :4] = histp
    sup = nc.simulate(Xp, "tan", need_alpha=True)
    a5p = sup["alpha"][:, 4, :4]
    hr_v = float((a5p.argmax(axis=1) == meta["pt"]).mean())
    # D1: position label permutation
    rand_pos = rng_b.integers(0, 4, size=N)
    other = (rand_pos + 1 + (meta["pd"] == rand_pos + 1).astype(int)) % 4
    hr_l = float((am == rand_pos).mean())
    # D4: identical history, both query codes
    Xa, Xb = X.copy(), X.copy()
    Xa[:, 4] = QCODE[0]
    Xb[:, 4] = QCODE[1]
    aa = nc.simulate(Xa, "tan", need_alpha=True)["alpha"][:, 4, :4]
    ab = nc.simulate(Xb, "tan", need_alpha=True)["alpha"][:, 4, :4]
    aan = aa / np.maximum(aa.sum(1, keepdims=True), 1e-12)
    abn = ab / np.maximum(ab.sum(1, keepdims=True), 1e-12)
    Dpair = float(np.mean([jsd(aan[i], abn[i]) for i in range(0, N, 10)]))
    flip = float((aan.argmax(1) != abn.argmax(1)).mean())
    print(f"query permutation:      hit rate = {hr_p:.3f} "
          f"(unpermuted {hr:.3f})")
    print(f"value permutation:      hit rate = {hr_v:.3f}")
    print(f"position-label permut.: hit rate = {hr_l:.3f} (expect ~0.25)")
    print(f"same history, q=A vs q=B: mean per-trial JSD = {Dpair:.4f}, "
          f"history-argmax flip fraction = {flip:.3f}")
    d_ok = (hr_p <= hr + 0.05) and (hr_l < 0.40)

    # ---------------- final ----------------
    print("\nFinal\n-----")
    print(f"Audit A (B2 shortcut): {stA}")
    print(f"Audit B (B3 shortcut): {stB}")
    print(f"Audit C1 (hit rate):   {stC1}")
    print(f"Audit C2 (contrast):   {stC2}")
    print(f"Audit C3 (routing):    {stC3}")
    print(f"generator audit:       {'PASS' if gen_ok else 'FAIL'}")
    passed = (stA == stB == stC1 == stC2 == stC3 == "PASS" and gen_ok)
    print(f"\nSTATUS: {'AUDIT_PASSED' if passed else 'AUDIT_FAILED'}")
    print(f"(runtime {time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
