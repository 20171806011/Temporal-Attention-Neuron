"""Build source-correction proposals; never write to the frozen repository."""
from pathlib import Path
import difflib,hashlib,json

OUT=Path(__file__).resolve().parent
REPO=OUT.parent.parent/'Temporal-Attention-Neuron'
EXPECTED={
 'snapshot/paper2/sections/s8_composition.tex':'ed692809df97f6f627d33ba421c74ed30f1aa192e7fca5af7bc04071dc869a95',
 'snapshot/paper2/main.tex':'b10865a3ecec8f2b48897dfab13d1a01062fcbb41ff26c7a8e6a447b8dfdb575',
 'README.md':'bb195ff9334b003490e045815ef7c9969763d9d610119c1da418f2588bbf80e8',
}
def sha(x):return hashlib.sha256(x).hexdigest()

MODEL_TEXT=r'''\paragraph{Registered hybrid carrier dynamics and subthreshold state-collision fixture.}
Protocol~V2.6 defines the complete carrier state as
$\mathbf{z}_n=(u_n,A_n,r_n)$, with $r_n$ the remaining refractory time,
not a surprise-weighted accumulator. The registered implementation uses
Euler steps $h=0.1\,\text{ms}$, $\tau_m=10\,\text{ms}$,
$\tau_A=15\,\text{ms}$, $\tau_{\mathrm{ref}}=2\,\text{ms}$,
threshold $1$, and rest/reset potential $0$. For M1,
$\mathbf{z}_0=(0,0,0)$, and
\begin{equation}
\begin{aligned}
 A_{n+1}&=\min\!\left(5,\max\!\left(0,
 A_n+\frac{h}{\tau_A}(-A_n+\nu_n)\right)\right),\\
 u_n^\star&=u_n+\frac{h}{\tau_m}
 \left(-u_n+A_n I_{\mathrm{content},n}+I_{\mathrm{query},n}\right)
 \quad (r_n=0).
\end{aligned}
\label{eq:p1b_carrier}
\end{equation}
The membrane update uses the \emph{old} gate $A_n$; the gate evolves even
during refractoriness. The remaining updates are
\[
(u_{n+1},r_{n+1},s_{n+1})=
\begin{cases}
 (0,\max(0,r_n-h),0),&r_n>0,\\
 (0,\tau_{\mathrm{ref}},1),&r_n=0,\ u_n^\star\ge1,\\
 (u_n^\star,0,0),&r_n=0,\ u_n^\star<1.
\end{cases}
\]
For an active event $(K_i,V_i)$, the registered unit weights give
$I_{\mathrm{content}}=K_i+V_i$ and
$\nu=\max(0,|K_i|-1.4)$; both drives are zero during blanks, ISIs and
query presentation. A query supplies $I_{\mathrm{query}}=q$ for
$10\,\text{ms}$, and is zero otherwise. The two roles independently probe
the same encoded and delayed checkpoint.

For the analog decoder, use cue-local indices $0,\ldots,100$.
At the deadline, $u_{\mathrm{readout}}=0$ if $r_{99}>0$; otherwise
$u_{\mathrm{readout}}=u_{99}^\star$, including when that last step itself
crosses threshold and resets the stored voltage. The frozen readout is
$\hat v=g_{\mathrm{dec}}u_{\mathrm{readout}}+b_{\mathrm{dec}}$,
with displayed rounded coefficients $g_{\mathrm{dec}}=0.132630817$ and
$b_{\mathrm{dec}}=-0.026816484$. The query affects the preceding dynamics
but is not a separate argument of this affine decoder. The coefficients
were calibrated once on 500 independent histories and shared across
the registered analog arms.

Fixture~G1.5 is a specific compensated-value construction on a verified
subthreshold branch, not a claim that all subthreshold inputs are erased.
It fixes positive-novelty keys $K_1=1.8$, $K_2=2.2$ in slots 0 and 1
and uses the fixture-specific content weights $w_K=w_V=0.1$ rather than
the main experiment's unit weights. Since $A_n$ depends on the fixed keys
and timing and no spikes occur, $u_T=b+c_1V_1+c_2V_2$ and $r_T=0$.
Choosing $\Delta V_1=2.5$ and
$\Delta V_2=-(c_1/c_2)\Delta V_1$ gives equal ideal terminal states;
the executable checks legal values, no spikes and a floating residual
$<10^{-10}$. The common query $q=1.8$ requests different first values,
giving a pairwise worst-case ideal-state delivery error of at least
$1.25>0.50$. This is separate from Protocol~V2.6's first-event
zero-novelty erasure result under $A(0)=0$.
'''

FLOOR_TEXT=r'''\paragraph{Confirmatory evaluation: floor-limited comparison and unestablished redundancy.}
The confirmatory audit spanned 4,000 unique histories across 3 post-sequence silence
delays ($\Delta t_{\text{silence}}\in\{0,5,10\}\,\text{ms}$) and 8 model variants,
totaling 96,000 model-episodes (192,000 role queries). Operand delivery requires
$|\hat V-V_{\mathrm{target}}|<0.50$ without abstention; joint delivery requires
both roles to succeed. Protocol~V2.6 \S7.3 registered prerequisite joint
accuracy floors of $0.80$ for $N=2$ and $0.40$ for $N=3$.
M1 achieved only $0.0290$ ($58/2{,}000$) for $N=2$ across all delays, and
$0.0275$--$0.0280$ ($55$--$56/2{,}000$) for $N=3$.
Every cell therefore remains \texttt{FLOOR\_LIMITED\_COMPARISON}.

M2a preserves the dynamic novelty-driven gate and resets $u,r$ to zero
at every active-event offset; it is not the constant-gate arm.
M2b and M2d use the frozen mean gate $A_0=0.061886726$ (displayed rounded),
with M2d also resetting $u,r$. The cell-specific TOST comparison of
M1 and M2a uses equivalence margin $\pm0.005$. The reported $90\%$
confidence intervals lie numerically
inside that margin ($N=2$: $[-0.001515,0.001515]$;
$N=3$: $[-0.002978,0.000841]$ at $0\,\text{ms}$ and
$[-0.002262,0.001186]$ at $5,10\,\text{ms}$).
However, the failed baseline prerequisite prevents a confirmatory
equivalence or functioning-memory-redundancy conclusion in every cell
(\texttt{redundancy\_inferred = False}).
'''

SPIKE_TEXT=r'''\paragraph{Spiking channel limitation.}
M4\_Count and M4\_Latency use the same physical carrier and query simulation
as M1. Their decoders consume only the spike indicators at cue steps
1--99; the deadline spike at step 100 is excluded, and no analog state
or input-current bypass is provided. Silence is abstention with the
registered zero fallback. The separate composition-risk criterion for
\texttt{DEMONSTRATED\_CHANNEL\_LIMITATION} is
$\mathrm{LCB}_{95\%}(E_{\mathrm{comp}})\ge0.05$, not the operand-delivery
tolerance. Both tested end-to-end readouts meet that criterion across
all cells ($E_{\mathrm{comp}}\in[2.21,2.57]$).
Delay-pooled $95\%$ bootstrap confidence intervals were:
$N=2$: M4\_Count $[2.393533,2.502027]$, M4\_Latency $[2.243582,2.332606]$;
$N=3$: M4\_Count $[2.473506,2.571918]$, M4\_Latency $[2.212398,2.296450]$.
Episode-level abstention rates were $82.1\%$--$83.0\%$
(single-query silence $57.15\%$--$58.60\%$).
These are limitations of the specified carrier-plus-readout implementations;
they do not isolate spike discretization as the sole bottleneck or establish
a universal requirement for population coding.
'''

def main():
    originals={p:(REPO/p).read_bytes() for p in EXPECTED}
    for p,b in originals.items():assert sha(b)==EXPECTED[p]
    texts={p:b.decode() for p,b in originals.items()}
    section='snapshot/paper2/sections/s8_composition.tex';s=texts[section]
    a=s.index(r'\paragraph{Continuous carrier dynamics and subthreshold state collisions.}')
    b=s.index(r'\subsection{Scope of the rung}',a)
    replacement=MODEL_TEXT+'\n'+FLOOR_TEXT+'\n'+SPIKE_TEXT+'\n'
    s=s[:a]+replacement+s[b:]
    s=s.replace('for the continuous ODE carrier and frozen linear readout','for the registered hybrid carrier and frozen linear readout')
    mainpath='snapshot/paper2/main.tex';maintext=texts[mainpath]
    old='certifies that subthreshold inputs undergo analytical state collisions erasing history, baseline dynamical carriers fail prerequisite delivery floors'
    new='includes a specified subthreshold state-collision fixture, shows that the registered dynamical carrier fails prerequisite delivery floors'
    assert maintext.count(old)==1;maintext=maintext.replace(old,new)
    changes={section:s,mainpath:maintext}
    readme=texts['README.md']
    old='Constructed the conceptual ladder charting the necessary architectural transitions from temporal saliency filtering to opponent E-I competition, vector-space metric addressing, Winner-Take-All symbol selection, and algebraic composition.'
    new='Constructed a ladder of sufficient mechanisms within the tested model families, spanning temporal saliency filtering, opponent E-I competition, metric addressing, Winner-Take-All symbol selection, and algebraic composition.'
    assert readme.count(old)==1;readme=readme.replace(old,new)
    old='empirically characterizing the capacity limits of single-unit channels and establishing the necessity of distributed multi-channel circuits for multi-item binding.'
    new='empirically characterizing the limitations of the specified dynamical carrier and single-unit readouts for multi-item binding; the closure does not establish a universal requirement for distributed or population coding.'
    assert readme.count(old)==1;readme=readme.replace(old,new)
    optional={'README.md':readme}
    for patchname,group in [('manuscript_corrections.patch',changes),('readme_scope_corrections.patch',optional)]:
        chunks=[]
        for rel,newtext in group.items():
            chunks.append(f'diff --git a/{rel} b/{rel}\n')
            chunks.extend(difflib.unified_diff(texts[rel].splitlines(keepends=True),newtext.splitlines(keepends=True),fromfile='a/'+rel,tofile='b/'+rel))
            dest=OUT/'manuscript_preview'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(newtext)
        (OUT/patchname).write_text(''.join(chunks))
    # All historical count/accuracy/CI quantities in the replaced paragraphs remain.
    preserved=['4,000','96,000','192,000','0.80','0.40','0.0290','58/2{,}000','0.0275','0.0280','55','56/2{,}000','0.132630817','-0.026816484','0.061886726',
               '-0.001515','0.001515','-0.002978','0.000841','-0.002262','0.001186','2.393533','2.502027','2.243582','2.332606','2.473506','2.571918','2.212398','2.296450','82.1','83.0','57.15','58.60',
               'FLOOR\\_LIMITED\\_COMPARISON','DEMONSTRATED\\_CHANNEL\\_LIMITATION','redundancy\\_inferred = False']
    assert all(x in s for x in preserved)
    assert all((REPO/p).read_bytes()==originals[p] for p in EXPECTED)
    manifest=dict(frozen_commit='85213f0d417d54e4f48b301ed1c0e4200ef3c421',source_sha256=EXPECTED,
        patches={p:sha((OUT/p).read_bytes()) for p in ('manuscript_corrections.patch','readme_scope_corrections.patch')},
        preview_sha256={p:sha((OUT/'manuscript_preview'/p).read_bytes()) for p in [*changes,*optional]},
        preserved_historical_strings=preserved,repository_bytes_unchanged=True,
        scope='Unapplied corrections. Main patch is source-contract fidelity; optional README patch narrows interpretation. Release PDF/layout not changed.')
    (OUT/'PATCH_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('patches','repository_bytes_unchanged')},indent=2))

if __name__=='__main__':main()
