"""Independent exact block-polynomial proof and saved-ledger checks.

No imports from supplied timescale executable and no edits to its outputs.
"""
from fractions import Fraction as Q
from pathlib import Path
import csv, hashlib, itertools, json
import numpy as np

ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'prior_sources/timescale-heldout'
result=json.loads((SRC/'results.json').read_text())

def exact_polynomial(tm,tA):
    am=1-Q(1,10)/tm; ag=1-Q(1,10)/tA;bm=1-am
    am100=am**100;ag100=ag**100
    cross=(am100-ag100)/(am-ag);tot=(1-am100)/(1-am)
    def run(v):
        u=A=Q(0);upper=Q(0);gateupper=Q(0)
        for slot in range(5):
            j=0 if slot==0 else 1 if slot==4 else None
            k=Q(0) if j is None else [Q(4,5),Q(9,5)][j]
            s=Q(0) if j is None else k+v[j]
            D=Q(0) if j is None else max(Q(0),k-Q(7,5))+Q(1,10)*(v[j]+3)
            for dd,ss in [(D,s),(Q(0),Q(0))]:
                next_A=ag100*A+(1-ag100)*dd
                next_u=am100*u+bm*ss*(A*cross+dd*(tot-cross))
                lo,hi=min(A,next_A),max(A,next_A)
                upper=max(upper,u,ss*hi if ss>=0 else ss*lo)
                gateupper=max(gateupper,hi)
                u,A=next_u,next_A
        return u,A,upper,gateupper
    p00=run([Q(0),Q(0)]);px=run([Q(1),Q(0)]);nx=run([Q(-1),Q(0)])
    py=run([Q(0),Q(1)]);ny=run([Q(0),Q(-1)]);p11=run([Q(1),Q(1)])
    U0=p00[0];Ux=(px[0]-nx[0])/2;Uy=(py[0]-ny[0])/2
    Uxx=(px[0]+nx[0])/2-U0;Uyy=(py[0]+ny[0])/2-U0
    Uxy=p11[0]-U0-Ux-Uy-Uxx-Uyy
    U=[U0,Ux,Uy,Uxx,Uxy,Uyy];A=[p00[1],px[1]-p00[1],py[1]-p00[1]]
    assert A[1]>0 and A[2]>0
    corners=[]
    for x,y in itertools.product([Q(-1,2),Q(1,2)],repeat=2):
        det=(Ux+2*Uxx*x+Uxy*y)*A[2]-(Uy+Uxy*x+2*Uyy*y)*A[1]
        assert det<0
        corners.append({'point':[float(x),float(y)],'exact_sign':-1,'value':float(det)})
    upper=run([Q(1,2),Q(1,2)])
    assert upper[2]<1 and upper[3]<5
    # At query A decreases, and each u update is a convex combination of u and
    # q+.1*A. This upper bound therefore covers the entire noiseless query box.
    query_bound=max(upper[0],Q(4,5)+Q(1,10)*upper[1])
    assert query_bound<1
    return U,A,dict(corner_determinants=corners,gate_gradient_positive=True,
        conservative_encoding_membrane_upper=float(upper[2]),gate_upper=float(upper[3]),
        conservative_query_membrane_upper=float(query_bound))

checks={};table=[]
targets=np.loadtxt(SRC/'heldout_targets.csv',delimiter=',',skiprows=1)
assert targets.shape==(4096,2)
assert np.array_equal(targets,np.random.RandomState(20261105).uniform(-.5,.5,size=(4096,2)))
assert hashlib.sha256((SRC/'heldout_targets.csv').read_bytes()).hexdigest()==result['targets_sha256']
for name,arm in result['arms'].items():
    U,A,proof=exact_polynomial(arm['tau_m'],arm['tau_A'])
    du=max(abs(float(x)-y) for x,y in zip(U,arm['polynomial_U']))
    da=max(abs(float(x)-y) for x,y in zip(A,arm['polynomial_A']))
    assert du<1e-15 and da<1e-15
    proof.update(independent_U_match=du,independent_A_match=da)
    proof['ledger_cells_checked']=[]
    for cell in arm['cells']:
        scenario=cell['scenario']
        with (SRC/f'{name}_{scenario}_predictions.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==4096 and [int(r['id']) for r in rows]==list(range(4096))
        assert np.array_equal(np.array([[float(r['v1']),float(r['v2'])] for r in rows]),targets)
        pred=np.array([[float(r['pred1']),float(r['pred2'])] for r in rows])
        valid=np.array([r['status']=='VALID' for r in rows]);errors=np.abs(pred-targets)
        assert np.all(pred[~valid]==0)
        met=cell['metrics']
        assert np.max(abs(np.sqrt(np.mean(errors**2,axis=0))-met['rmse_per_role']))<1e-14
        assert np.max(abs(np.mean(errors,axis=0)-met['mae_per_role']))<1e-14
        assert abs(errors.max(axis=1).mean()-met['mean_e_comp'])<1e-14
        assert valid.mean()==met['coverage']
        assert np.mean(valid&np.all(errors<.05,axis=1))==met['local_valid_joint_success_0_05']
        counts={s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)}
        assert counts==cell['status_counts']
        assert cell['physical_branch_departures']==0 and cell['numerical_boundary_snaps']==0
        proof['ledger_cells_checked'].append(scenario)
        table.append({'arm':name,'scenario':scenario,**met,'status_counts':counts})
    checks[name]=proof
report=dict(exact_fraction_block_checks=checks,ledger_rows_checked=4096*15,
            target_seed_bytes_verified=True,all_metrics_verified=True,
            source_script_sha256=hashlib.sha256((SRC/'run_timescales.py').read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256((SRC/'PROTOCOL.md').read_bytes()).hexdigest())
(ROOT/'independent_checks.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['| Arm | Noise | Ecomp | RMSE (v1,v2) | MAE (v1,v2) | Coverage | Local joint | Out of domain |',
       '| --- | --- | ---: | --- | --- | ---: | ---: | ---: |']
for r in table:
    fm=lambda v:'/'.join(f'{x:.6g}' for x in v)
    lines.append(f"| {r['arm']} | {r['scenario']} | {r['mean_e_comp']:.9g} | {fm(r['rmse_per_role'])} | {fm(r['mae_per_role'])} | {100*r['coverage']:.4f}% | {100*r['local_valid_joint_success_0_05']:.4f}% | {r['status_counts'].get('OUT_OF_DOMAIN',0)} |")
(ROOT/'all15cells.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report,indent=2))
