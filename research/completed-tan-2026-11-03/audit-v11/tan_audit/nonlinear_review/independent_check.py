"""Independent source review checks; no imports from supplied executable."""
from fractions import Fraction as Q
from pathlib import Path
import csv,hashlib,itertools,json
import numpy as np
OUT=Path(__file__).resolve().parent;SRC=OUT.parent/'prior_sources/nonlinear-heldout'
r=json.loads((SRC/'results.json').read_text())
targets=np.loadtxt(SRC/'heldout_targets.csv',delimiter=',',skiprows=1)
assert targets.shape==(4096,2)
assert np.array_equal(targets,np.random.RandomState(20261103).uniform(-.5,.5,(4096,2)))
assert hashlib.sha256((SRC/'heldout_targets.csv').read_bytes()).hexdigest()==r['targets_sha256']

am=Q(99,100);ag=Q(149,150);bm=1-am
M=am**100;G=ag**100;C=(M-G)/(am-ag);S=(1-M)/(1-am)
def exact_endpoint(v):
    u=A=Q(0);bound=gate_bound=Q(0)
    for slot in range(5):
        j=0 if slot==0 else 1 if slot==4 else None
        k=Q(0) if j is None else [Q(4,5),Q(9,5)][j]
        s=Q(0) if j is None else k+v[j]
        d=Q(0) if j is None else max(Q(0),k-Q(7,5))+Q(1,10)*(v[j]+3)
        for ss,dd in [(s,d),(Q(0),Q(0))]:
            An=G*A+(1-G)*dd;un=M*u+bm*ss*(A*C+dd*(S-C))
            lo,hi=min(A,An),max(A,An)
            bound=max(bound,u,ss*hi if ss>=0 else ss*lo);gate_bound=max(gate_bound,hi)
            u,A=un,An
    return u,A,bound,gate_bound

z=exact_endpoint([Q(0),Q(0)]);xp=exact_endpoint([Q(1),Q(0)]);xm=exact_endpoint([Q(-1),Q(0)])
yp=exact_endpoint([Q(0),Q(1)]);ym=exact_endpoint([Q(0),Q(-1)]);xy=exact_endpoint([Q(1),Q(1)])
u0=z[0];ux=(xp[0]-xm[0])/2;uy=(yp[0]-ym[0])/2
uxx=(xp[0]+xm[0])/2-u0;uyy=(yp[0]+ym[0])/2-u0;uxy=xy[0]-u0-ux-uy-uxx-uyy
U=[u0,ux,uy,uxx,uxy,uyy];A=[z[1],xp[1]-z[1],yp[1]-z[1]]
assert A[1]>0 and A[2]>0
dets=[]
for x,y in itertools.product([Q(-1,2),Q(1,2)],repeat=2):
    det=(ux+2*uxx*x+uxy*y)*A[2]-(uy+uxy*x+2*uyy*y)*A[1]
    assert det<0;dets.append(float(det))
corner=exact_endpoint([Q(1,2),Q(1,2)])
assert corner[2]<1 and corner[3]<5 and Q(4,5)+Q(1,10)*corner[1]<1
assert max(abs(float(a)-b) for a,b in zip(U,r['polynomial_U']))<1e-15
assert max(abs(float(a)-b) for a,b in zip(A,r['polynomial_A']))<1e-15
U=np.array(list(map(float,U)));A=np.array(list(map(float,A)))
x,y=targets.T
states=np.column_stack([U[0]+U[1]*x+U[2]*y+U[3]*x*x+U[4]*x*y+U[5]*y*y,A[0]+A[1]*x+A[2]*y])
J=np.array([[U[1],U[2]],[A[1],A[2]]]);base=np.array([U[0],A[0]])
nrng=np.random.RandomState(20261104);Zu=nrng.standard_normal((4096,100));ZA=nrng.standard_normal((4096,100));Zs=nrng.standard_normal((4096,3))
reconstructed={}
for scenario,ss,su,sA in [('noiseless',0.,0.,0.),('sensor_only',1e-6,0.,0.),('membrane_process_only',0.,1e-6,0.),('sensor_and_membrane',1e-6,1e-6,0.),('gate_process_only',0.,0.,1e-6)]:
    u=states[:,0].copy();g=states[:,1].copy();samples={}
    for n in range(100):
        un=.99*u+.001*g+.008;gn=(149/150)*g
        u=un+su*Zu[:,n];g=gn+sA*ZA[:,n]
        assert np.all(u<1) and np.all(g>0) and np.all(g<5)
        if n in (0,99):samples[n+1]=u.copy()
    for m in (1,100):
        d=.001*((149/150)**m-.99**m)/((149/150)-.99)
        y0=states[:,0]+ss*Zs[:,0];yt=samples[m]+ss*Zs[:,1 if m==1 else 2]
        reconstructed[scenario,m]=np.column_stack([y0,(yt-.8*(1-.99**m)-.99**m*y0)/d])

table=[];max_observation_error=0.;post=json.loads((SRC/'paired_uncertainty.json').read_text())
for c in r['cells']:
    scenario=c['scenario'];m=c['m']
    rows=list(csv.DictReader((SRC/f'{scenario}_m{m}_predictions.csv').open()))
    assert len(rows)==4096 and [int(x['id']) for x in rows]==list(range(4096))
    true=np.array([[float(x['v1']),float(x['v2'])] for x in rows]);assert np.array_equal(true,targets)
    pred=np.array([[float(x['pred1']),float(x['pred2'])] for x in rows]);valid=np.array([x['status']=='VALID' for x in rows]);error=abs(pred-targets)
    state=np.array([[float(x['u_hat']),float(x['A_hat'])] for x in rows]);obs_err=float(np.max(abs(state-reconstructed[scenario,m])))
    max_observation_error=max(max_observation_error,obs_err);assert obs_err<1e-10
    met=c['bounded_exact_inverse'];assert np.all(pred[~valid]==0)
    assert abs(error.max(axis=1).mean()-met['mean_e_comp'])<1e-14
    assert np.max(abs(error.mean(axis=0)-met['mae_per_role']))<1e-14
    assert np.max(abs(np.sqrt((error**2).mean(axis=0))-met['rmse_per_role']))<1e-14
    assert valid.mean()==met['coverage'] and np.mean(valid&np.all(error<.05,axis=1))==met['validity_aware_local_joint_success_0_05']
    if valid.any():
        cond=met['conditional_valid_only']
        assert int(valid.sum())==cond['n']
        assert abs(error[valid].max(axis=1).mean()-cond['mean_e_comp'])<1e-14
        assert np.max(abs(error[valid].mean(axis=0)-cond['mae_per_role']))<1e-14
        assert np.max(abs(np.sqrt((error[valid]**2).mean(axis=0))-cond['rmse_per_role']))<1e-14
    counts={s:sum(x['status']==s for x in rows) for s in set(x['status'] for x in rows)};assert counts==c['status_counts']
    assert c['numerical_boundary_snaps']==c['physical_branch_departures']==0
    linear=(state-base)@np.linalg.inv(J).T;le=abs(linear-targets);lm=c['unbounded_linear_diagnostic']
    assert abs(le.max(axis=1).mean()-lm['mean_e_comp'])<1e-12
    assert np.max(abs(np.sqrt((le**2).mean(axis=0))-lm['rmse_per_role']))<1e-12
    delta=error.max(axis=1)-abs(targets).max(axis=1);se=delta.std(ddof=1)/64
    interval=next(a for a in post['comparisons'] if a['scenario']==scenario and a['m']==m)
    assert abs(delta.mean()-interval['mean_paired_ecomp_difference_vs_zero'])<1e-14
    assert np.max(abs(np.array([delta.mean()-1.96*se,delta.mean()+1.96*se])-interval['approx_normal_95_ci']))<1e-14
    table.append(dict(scenario=scenario,m=m,Ecomp=met['mean_e_comp'],RMSE=met['rmse_per_role'],coverage=met['coverage'],local_joint=met['validity_aware_local_joint_success_0_05'],linear_Ecomp=lm['mean_e_comp'],status_counts=counts,observation_reconstruction_error=obs_err))
assert len(table)==10
report=dict(ledger_rows_checked=40960,all_bounded_metrics=True,all_linear_diagnostics=True,all_posthoc_intervals=True,
    exact_block_coefficient_checks=True,corner_determinants=dets,positive_gate_f2=True,
    conservative_wholebox_u_bound=float(corner[2]),conservative_wholebox_A_bound=float(corner[3]),
    conservative_query_u_bound=float(Q(4,5)+Q(1,10)*corner[1]),max_observation_reconstruction_error=max_observation_error,
    target_hash=r['targets_sha256'],source_sha256=hashlib.sha256((SRC/'run_heldout.py').read_bytes()).hexdigest(),cells=table)
(OUT/'independent_checks.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['| Noise | Sample m | Exact Ecomp | RMSE roles | Coverage | Local joint .05 | Linear Ecomp | Status counts |','|---|---:|---:|---|---:|---:|---:|---|']
for x in table:
    rm='/'.join(f'{v:.7g}' for v in x['RMSE'])
    lines.append(f"| {x['scenario']} | {x['m']} | {x['Ecomp']:.10g} | {rm} | {x['coverage']:.8f} | {x['local_joint']:.8f} | {x['linear_Ecomp']:.10g} | {x['status_counts']} |")
(OUT/'all10cells.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='cells'},indent=2))
