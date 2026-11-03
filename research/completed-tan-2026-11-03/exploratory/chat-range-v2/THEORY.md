# Range consistency versus robust latest-value addressing

This note concerns the new conceptual associative architecture only. It neither changes original TAN/P1-B results nor v11. All norms below are Euclidean/operator 2-norms. These are conditional linear-algebra derivations, not novelty claims or biological claims.

## 1. Exact range invariant

Let F have independent columns, C=FF^T, K=C^dagger and Q=CK, the orthogonal projector onto S=range(F). For any scalar c,

    b+ = Q b + c Q x = Q(b+c x)

satisfies Q b+=b+. This remains true whether c is an ordinary residual or a leverage-normalized residual, and even if b initially has pollution. If b is initially in S, projecting only the update increment also preserves the invariant; it does NOT remove pre-existing pollution. The tested P/PN/TK rules project the whole vector.

For a genuine new feature g outside S, C+=C+gg^T has range S+=S+span(g). If b is in S, then b+=b+v*g belongs to S+. Induction proves the range invariant for an exact, correctly routed stream.

This does NOT prove b=F*t_latest. A wrong value-address association may lie entirely inside S.

## 2. Exact latent-pollution activation identity

Put p=(I-Q)g != 0. Under exact clean geometry, the dual for newly inserted g is

    (C+gg^T)^dagger g = p / ||p||^2.

For arbitrary prior b and new update b+=b+v*g,

    g^T (C+gg^T)^dagger b+ - v = p^T b / ||p||^2.

Consequently true-null leakage beta=||(I-Q)b|| produces new-key error at most beta/||p||. Exact range repair makes this particular error zero. The identity need not equal an actual output obtained from a faulty C; the main results label a clean-geometry evaluator counterfactual separately.

## 3. Fixed estimated geometry: projection is not present-time decoding repair

Construct Qhat=U_r U_r^T and Khat=U_r diag(1/lambda_j) U_r^T from eigenvalues strictly above the cutoff. Then Khat Qhat=Khat. Whole-b projection and increment projection therefore leave the instantaneous Khat readout unchanged relative to the same unprojected update, as long as C and the routing branch are fixed.

For a fixed repeated observation x, define ell=x^T Khat x. U and P obey

    r_next=(1-ell)r.

They converge at that observation only for 0<ell<2 (with the usual trivial zero-residual exception). At ell=2 the residual oscillates; ell>2 amplifies it. PN gives r_next=0 in exact arithmetic for ell>0. None of these statements proves correct reading at a canonical key f != x. The retained earlier example has dual coordinates (-1,2), ell=5, normalized true-key outputs (-1/5,2/5), but observed-address output 1.

## 4. Approximate projector leakage and content deletion

Let Qhat be an orthogonal projector and rho=||Qhat-Q||. Then

    ||(I-Q) Qhat(b+c x)|| <= rho (||b||+|c| ||x||).

For a valid b in S,

    ||Qhat b-b|| <= rho ||b||.

Thus estimated-range invariance does not imply true-range invariance. A truncated valid direction can be permanently deleted, and a rotated subspace can retain pollution. Idempotence Qhat^2=Qhat only validates algebraic projector structure, not the correct range.

The frozen rotated example uses C=diag(1,.1,0,0), q=.8e2+.6e3, Chat=e1e1^T+.1qq^T. The matrix error norm is .06, projector error .6, and Qhat(.001e3)=.00048e2+.00036e3.

## 5. Spectral-gap certificate and its assumptions

Suppose true C is PSD, has rank N, and smallest nonzero eigenvalue alpha>0. Let Chat=C+Delta be symmetric and ||Delta||<=delta. If

    delta < cutoff < alpha-delta,

then the retained rank is N. In particular delta<alpha/2 is necessary to have such a cutoff. Let Uhat contain the retained eigenvectors with eigenvalue matrix Lambda_hat. Multiplication by P=I-Q gives

    P Uhat Lambda_hat = P Delta Uhat.

Since Lambda_hat >= (alpha-delta)I,

    ||Qhat-Q|| = ||P Uhat|| <= delta/(alpha-delta).

This is a direct special-case eigenspace perturbation argument. If top-N rank retention is otherwise independently certified, the same argument applies without the above sufficient cutoff interval. Do not infer the rank certificate merely from approximate idempotence. The admission count must itself have valid provenance.

A useful conservative inverse bound is obtained from

    Khat-K = -Khat Delta K + Khat(I-Q) - (I-Qhat)K.

With mu=alpha-delta>0,

    kappa := ||Khat-K|| <= delta/(mu*alpha) + rho*(1/mu+1/alpha).

A known upper bound on delta is an external/numerical reliability assumption, not something Chat alone can infer from the faulty matrix. Weak-rank truncation violates the same-rank premise and can give rho=1 even at delta=0.

## 6. A local conditional latest-value error certificate

Assume true distinct-key geometry F, G=F^TF has eigenvalues in [alpha,beta]. Every key has norm <=R, current/latest values and the requested value have magnitude <=V. The requested observation is x=f_i+e with ||e||<=zeta. This assumes an authenticated old-key event or the previously stated old-versus-new-direction promise; no membership theorem is supplied.

Let b*=F*t be the correct latest statistic and assume a certified stored error ||b-b*||<=epsilon_b. Put

    B0=sqrt(beta)*sqrt(N)*V,
    W=sqrt(N)*V/sqrt(alpha),
    e_w=kappa*B0 + epsilon_b/mu,
    D_r=zeta*W + (R+zeta)*e_w,
    D_d=zeta/sqrt(alpha) + sqrt(beta)*kappa*(R+zeta),
    L_ell=(2R*zeta+zeta^2)/alpha + (R+zeta)^2*kappa.

Then current true-key prediction error is <=sqrt(beta)e_w, residual error is <=D_r, the dual coefficient vector differs from e_i by <=D_d, and |ell-1|<=L_ell.

For U/P the post-write vector of true-key errors has the sufficient bound

    E_P = sqrt(beta)e_w + D_r + (2V+D_r)D_d.

For PN, if L_ell<1,

    E_PN = sqrt(beta)e_w + D_r
           + (2V+D_r)(D_d+L_ell)/(1-L_ell).

Requiring the relevant E<=epsilon guarantees each key's error is <=epsilon for THIS write. This is conservative and conditional, not an unconditional lifetime guarantee. The proof is the decomposition d=F^T Khat x, residual=r0+dr and, for PN, d/ell-e_i=(d-e_i+(1-ell)e_i)/ell.

An error budget must be updated after each write. For projected normalized updates, one conservative stored-statistic recurrence is

    epsilon_b_next <= epsilon_b + rho B0
       + 2V(rho R+zeta)
       + (R+zeta)(D_r+2V L_ell)/(1-L_ell).

This need not contract. Repeated fixed bias is not independent noise and does not disappear just because more writes are made.

## 7. Address and admission contracts

For noisy old-versus-new routing, if new canonical features are at distance >=gamma from true S, define E=zeta+rho(R+zeta). A sufficient threshold margin is

    E <= tau < gamma-E.

This does not recognize a genuinely new key inside S. C,b are not a key-set certificate: two rotated orthonormal dictionaries with zero values have identical C,b and disagree about membership.

Within one fixed dictionary, if ||f_i-f_j||<=2zeta, the same observed x can request either old key. Starting from all zero values and requesting value 1 produces incompatible latest targets for tolerance <1/2. Even a full key table would not resolve that adversarial observation without extra address information. Separation >2zeta removes this particular overlap but does not alone certify correct geometry or membership.

Practical conditional safeguards would require: a valid address promise or extra authenticated key information; independently justified matrix/state error bounds; spectral/rank/count consistency; leverage lower bounds; bounded new-direction increments and capacity admission; and refusal when certificates fail. Such guards are proposed constraints, not a fifth tested algorithm in this package. They can require scalar provenance/error budgets, additional state for address certificates, and verification time. Extra repeated measurements help only under a noise model; a fixed adversarial bias cannot be averaged away.

## 8. Sources and provenance

- The experimental numbers in this package were generated by range_consistency_audit_v2.py, not retrieved from the TAN repository.
- NumPy 2.3 eigh documentation: https://numpy.org/doc/2.3/reference/generated/numpy.linalg.eigh.html
- General eigenspace perturbation background: Yi Yu, Tengyao Wang, Richard J. Samworth, A useful variant of the Davis--Kahan theorem for statisticians, https://arxiv.org/abs/1405.0680 . The simple bound above is derived here for this problem's assumptions, not claimed novel.
