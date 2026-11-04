#!/usr/bin/env python3
"""Frozen, bounded conceptual range-repair audit. No TAN/ODE/Codex execution.

Main grid: 140 deterministic 4-D scenarios x 4 arms = 560 trajectories.
Only evaluator owns canonical keys and latest targets. TK receives the noiseless
CURRENT feature as a declared oracle resource; it receives no clean inverse.
All projected arms reproject the WHOLE b on an old write.
Run: python range_consistency_audit_v2.py --out results_v2.json
"""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, platform, sys
from pathlib import Path
from typing import Any
import numpy as np

SPEC_HASH = 'dc73c12e5ccceed4f51efca859365930f8756c8515a579b764abe69bbce130a8'
ARMS = ('U', 'P', 'PN', 'TK')

def norm(x):
    return float(np.linalg.norm(x, 2))

def spec_ops(C, cutoff):
    C = (C + C.T) / 2
    ev, U = np.linalg.eigh(C)
    take = ev > cutoff
    V = U[:, take]
    Q = V @ V.T
    K = (V / ev[take]) @ V.T if np.any(take) else np.zeros_like(C)
    return K, Q, ev, int(np.sum(take))

def true_ops(F):
    # Evaluator ONLY. Formula avoids cutting off a true, weak positive mode.
    G = F.T @ F
    dual = F @ np.linalg.inv(G)
    K = dual @ dual.T
    Q = F @ np.linalg.solve(G, F.T)
    return K, (Q+Q.T)/2

class Memory:
    """No keys/targets/history table. Only C,b,count and declared numerical rules."""
    def __init__(self, d, arm, cutoff, tau, leverage_min=1e-12, cap=1e12):
        self.C = np.zeros((d,d), dtype=np.float64)
        self.b = np.zeros(d, dtype=np.float64)
        self.count = 0
        self.arm = arm
        self.cutoff, self.tau = cutoff, tau
        self.leverage_min, self.cap = leverage_min, cap
        self.stopped = False

    def ops(self):
        return spec_ops(self.C, self.cutoff)

    def write(self, x, value):
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == self.b.shape and np.all(np.isfinite(x)) and np.isfinite(value)
        if self.stopped:
            return {'branch':'stopped'}
        K,Q,ev,rank = self.ops()
        score = norm(x-Q@x)
        base = {'score':score,'retained_rank':rank,'admissions':self.count}
        if score > self.tau:
            if self.count >= len(self.b):
                self.stopped = True
                return dict(base,branch='reject_capacity')
            self.C += np.outer(x,x)
            self.b += value*x
            self.count += 1
            ans = dict(base, branch='new')
        else:
            b0 = self.b.copy() if self.arm == 'U' else Q@self.b
            removed = norm(self.b-b0)
            w0 = K@b0
            residual = float(value-x@w0)
            ell = float(x@K@x)
            if self.arm in ('PN','TK') and ell <= self.leverage_min:
                self.stopped = True
                return dict(base,branch='reject_leverage',leverage=ell)
            coefficient = residual if self.arm in ('U','P') else residual/ell
            step = x if self.arm == 'U' else Q@x
            self.b = b0+coefficient*step
            residual_after = float(value-x@(K@self.b))
            if self.arm in ('PN','TK'):
                expected = 0.
            else:
                expected = (1-ell)*residual
            ans = dict(base,branch='old',leverage=ell,coefficient=coefficient,
                       residual_before=residual,residual_after=residual_after,
                       recurrence_error=abs(residual_after-expected),
                       reprojection_removed=removed,
                       estimated_range_defect=norm(self.b-Q@self.b))
        Kn,_,_,_=self.ops()
        if not np.all(np.isfinite(self.b)) or max(norm(self.b),norm(Kn@self.b))>self.cap:
            self.stopped=True
            ans['stop_reason']='state_norm_cap'
        return ans

    def read(self, features):
        K,_,_,_=self.ops()
        return np.asarray(features).T@(K@self.b)

def signed_weak(C):
    e,U=np.linalg.eigh(C)
    idx=np.flatnonzero(e>1e-12)[0]
    u=U[:,idx]
    j=int(np.argmax(np.abs(u)))
    if u[j]<0: u=-u
    return float(e[idx]),u

def make_fault(name,ratio,alpha,u,e3):
    d=alpha*ratio
    if name=='none': return np.zeros((len(u),len(u)))
    if name=='weak_null_symmetric_mix': return d*(np.outer(u,e3)+np.outer(e3,u))
    if name=='weak_erosion': return -d*np.outer(u,u)
    if name=='null_lift': return d*np.outer(e3,e3)
    raise ValueError(name)

def metrics(mem,F,targets,x_noisy,x_presented,desired):
    K,Q,ev,rank=mem.ops(); Ktrue,Qtrue=true_ops(F)
    w=K@mem.b; pred=F.T@w
    return {'true_predictions':pred.tolist(),
      'max_latest_error':float(np.max(np.abs(pred-targets))),
      'true_null_leak':norm(mem.b-Qtrue@mem.b),
      'estimated_range_defect':norm(mem.b-Q@mem.b),
      'projector_error':norm(Q-Qtrue),
      'idempotence_error':norm(Q@Q-Q),
      'retained_rank':rank,'admissions':mem.count,
      'b_norm':norm(mem.b),'w_norm':norm(w),
      'residual_at_noisy':float(desired-x_noisy@w),
      'residual_at_presented':float(desired-x_presented@w),
      'eigenvalues':ev.tolist()}

def run_grid(spec):
    d=spec['dimension']; E=np.eye(d); e1,e2,e3=E[:,0],E[:,1],E[:,2]
    records=[]; certificate_pass=0; exact_control_pass=0
    recurrence_abs=0.; recurrence_scaled=0.
    projected_range_scaled=0.
    for a,cut,fault,noise in itertools.product(spec['a_values'],spec['spectral_cutoffs_absolute'],spec['faults'],spec['noise_cases']):
        F=np.column_stack([e1,e1+a*e2]); Ctrue=F@F.T
        alpha,u=signed_weak(Ctrue); Delta=make_fault(fault['name'],fault['ratio'],alpha,u,e3)
        delta=norm(Delta)
        ndir=np.zeros(d) if noise['direction']=='zero' else (e3 if noise['direction']=='e3' else .8*e2+.6*e3)
        x=e1+noise['zeta']*ndir
        case=f'a={a:g}|cut={cut:g}|fault={fault["name"]}:{fault["ratio"]:g}|noise={noise["name"]}'
        for arm in ARMS:
            m=Memory(d,arm,1e-12,1e-9,spec['leverage_min'],spec['state_norm_cap'])
            for j,val in enumerate(spec['initial_latest']):
                assert m.write(F[:,j],val)['branch']=='new'
            assert np.max(np.abs(m.read(F)-spec['initial_latest'])) < 1e-9
            m.C += Delta
            m.cutoff,m.tau=cut,spec['tau_runtime']
            xp=e1 if arm=='TK' else x
            K0,Q0,ev0,rank0=m.ops(); _,Qt=true_ops(F)
            deltaQ=norm(Q0-Qt)
            certified= bool(delta < cut < alpha-delta)
            bound=delta/(alpha-delta) if alpha>delta else None
            if certified:
                assert rank0==2
                assert deltaQ <= bound+1e-9
                certificate_pass+=1
            initial=metrics(m,F,np.array(spec['initial_latest']),x,xp,spec['repeat_target'])
            traj=[]; checkpoints={}; branches=[]
            t=np.array([spec['repeat_target'],spec['initial_latest'][1]])
            for rep in range(1,spec['repetitions']+1):
                event=m.write(xp,spec['repeat_target']); branches.append(event['branch'])
                obs=metrics(m,F,t,x,xp,spec['repeat_target'])
                if event['branch']=='old':
                    er=event['recurrence_error']; recurrence_abs=max(recurrence_abs,er)
                    scaled=er/(1+abs(event['residual_before'])+abs(event['residual_after']))
                    recurrence_scaled=max(recurrence_scaled,scaled)
                    assert scaled<1e-7, (case,arm,rep,event)
                    if arm!='U':
                        scaledr=event['estimated_range_defect']/(1+norm(m.b))
                        projected_range_scaled=max(projected_range_scaled,scaledr)
                        assert scaledr<1e-9
                traj.append({'step':rep,'event':event,'metrics':obs})
                if rep in spec['checkpoints']: checkpoints[str(rep)]=obs
            b_before=m.b.copy()
            pre=metrics(m,F,t,x,xp,1.)
            expanded=np.column_stack([F,e3]); Ke,Qe=true_ops(expanded)
            ghost=float(e3@Ke@b_before)
            # The exact clean expanded geometry identity is evaluator-only.
            ghost_formula=float(e3@b_before)
            assert abs(ghost-ghost_formula)<1e-8*(1+abs(ghost_formula))
            newevent=m.write(e3,spec['later_new_value'])
            t3=np.r_[t,spec['later_new_value']]
            after=metrics(m,expanded,t3,e3,e3,0.)
            old_drift=float(np.max(np.abs(np.array(after['true_predictions'][:2])-np.array(pre['true_predictions']))))
            finalevent=m.write(F[:,1],spec['final_clean_f2_value'])
            t4=np.array([1.,spec['final_clean_f2_value'],0.])
            final=metrics(m,expanded,t4,F[:,1],F[:,1],spec['final_clean_f2_value'])
            r={
              'case':case,'a':a,'cutoff':cut,'fault':fault,'noise':noise,'arm':arm,
              'alpha_initial':alpha,'delta_norm':delta,'initial_rank':rank0,
              'projector_error_initial':deltaQ,'projector_idempotence_initial':norm(Q0@Q0-Q0),
              'gap_cutoff_certified':certified,'projector_bound_if_gap':bound,
              'initial':initial,'repeat_branch_counts':{b:branches.count(b) for b in sorted(set(branches))},
              'checkpoints':checkpoints,'repeat_trajectory':traj,
              'pre_new':pre,'clean_geometry_ghost_scorer_only':ghost,
              'weak_b_component_before_new':float(u@b_before),
              'new_event':newevent,'after_new':after,'old_output_drift_on_new':old_drift,
              'final_f2_event':finalevent,'after_final_f2':final,
              'max_b_norm_during_repeat':max(z['metrics']['b_norm'] for z in traj),
              'max_w_norm_during_repeat':max(z['metrics']['w_norm'] for z in traj),
              'stopped':m.stopped}
            if fault['name']=='none' and noise['name']=='clean' and cut<alpha:
                assert max(pre['max_latest_error'],after['max_latest_error'],final['max_latest_error'])<1e-8
                exact_control_pass+=1
            records.append(r)
    assert len(records)==560
    return {'records':records,'checks':{'trajectories':len(records),
       'scenarios':len(records)//4,'gap_certified_trajectories':certificate_pass,
       'clean_exact_control_trajectories':exact_control_pass,
       'max_recurrence_absolute_error':recurrence_abs,
       'max_recurrence_scaled_error':recurrence_scaled,
       'max_projected_range_relative_defect':projected_range_scaled}}

def fixtures():
    d=4; E=np.eye(d); e1,e2,e3=E[:,0],E[:,1],E[:,2]
    out={}
    # Preserve prior normalized wrong-address example, now embedded in R^4.
    a=.001; F=np.column_stack([e1,e1+a*e2]); C=F@F.T
    K,Q=true_ops(F); x=e1+2*a*e2; ell=float(x@K@x)
    arms={}
    for arm in ARMS:
        m=Memory(d,arm,1e-12,.25)
        m.C=C.copy();m.b=np.zeros(d);m.count=2
        inp=e1 if arm=='TK' else x
        seq=[]
        for j in range(6):
            ev=m.write(inp,1.);seq.append({'event':ev,'true_predictions':m.read(F).tolist(),'noisy_prediction':float(x@(m.ops()[0]@m.b))})
        arms[arm]=seq
    assert np.allclose(arms['PN'][-1]['true_predictions'],[-.2,.4],atol=1e-8)
    assert abs(arms['PN'][-1]['noisy_prediction']-1)<1e-8
    out['normalized_wrong_address']={'a':a,'leverage':ell,'arms':arms}
    # An absent eigenmode is valid stored content. Full b projection deletes it.
    a=.01; F=np.column_stack([e1,a*e2]); C=F@F.T;b=F@np.array([0.,1.])
    rows={}
    for arm in ARMS:
        m=Memory(d,arm,2e-4,.25);m.C=C.copy();m.b=b.copy();m.count=2
        before=m.b.copy(); ev=m.write(e1,0.)
        after=m.b.copy();m.cutoff=1e-10
        rows[arm]={'event':ev,'b_before':before.tolist(),'b_after':after.tolist(),
           'readout_after_threshold_recovery':m.read(F).tolist()}
    assert np.allclose(rows['U']['readout_after_threshold_recovery'],[0,1])
    for arm in ('P','PN','TK'): assert np.allclose(rows[arm]['readout_after_threshold_recovery'],[0,0])
    out['valid_weak_content_deletion']={'weak_eigenvalue':a*a,'destructive_cutoff':2e-4,'recovery_cutoff':1e-10,'arms':rows}
    # Estimated projector can be idempotent, rotate true content, and keep pollution.
    alpha=.1; q=.8*e2+.6*e3
    Ctrue=np.outer(e1,e1)+alpha*np.outer(e2,e2)
    Chat=np.outer(e1,e1)+alpha*np.outer(q,q)
    Khat,Qhat,_,_=spec_ops(Chat,1e-10)
    Qtrue=np.outer(e1,e1)+np.outer(e2,e2)
    b=.001*e3;bp=Qhat@b
    assert np.allclose(bp,[0,.00048,.00036,0])
    out['rotated_projector']={'delta_C':norm(Chat-Ctrue),'alpha':alpha,
      'projector_error':norm(Qhat-Qtrue),'idempotence':norm(Qhat@Qhat-Qhat),
      'b_before':b.tolist(),'b_projected':bp.tolist(),
      'true_null_leak_after':norm(bp-Qtrue@bp),
      'claim':'range(Qhat) is not the true range; this is not proof of actual post-new ghost under unchanged faulty geometry'}
    # Same statistics, different key membership. All values zero.
    F1=np.column_stack([e1,e2]); F2=np.column_stack([(e1+e2)/np.sqrt(2),(-e1+e2)/np.sqrt(2)])
    assert np.allclose(F1@F1.T,F2@F2.T)
    out['span_inside_new_key_ambiguity']={'dictionary_A':F1.tolist(),'dictionary_B':F2.tolist(),
      'C_difference':norm(F1@F1.T-F2@F2.T),'b_both_zero':True,
      'incoming':e1.tolist(),'old_in_A':True,'new_but_inside_span_in_B':True,
      'note':'outside the promise in world B; no membership guarantee follows'}
    # Exact-Q invariant and analytic new-mode ghost, using an initially valid b.
    F=np.column_stack([e1,e2]);K,Q=true_ops(F); z=.001
    rows={}
    for arm in ARMS:
        m=Memory(d,arm,1e-10,.25);m.C=F@F.T;m.b=np.zeros(d);m.count=2
        inp=e1 if arm=='TK' else e1+z*e3
        ev=m.write(inp,1.); leak=norm(m.b-Q@m.b); old=m.read(F).tolist()
        ne=m.write(e3,0.);newread=float(m.read(e3))
        rows[arm]={'leak_before_new':leak,'old_outputs_before_new':old,'new_output':newread}
    assert abs(rows['U']['new_output']-z)<1e-12
    for arm in ('P','PN','TK'):assert abs(rows[arm]['new_output'])<1e-12
    out['exact_Q_ghost_and_invariant']=rows
    return out

def summarize(grid):
    rows=grid['records']; summary={}
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm]
        summary[arm]={
           'trajectories':len(rr),
           'repeat_false_new_trajectories':sum(r['repeat_branch_counts'].get('new',0)>0 for r in rr),
           'repeat_stopped_or_rejected_trajectories':sum(any(k not in ('old','new') for k in r['repeat_branch_counts']) for r in rr),
           'later_new_not_admitted_as_new':sum(r['new_event']['branch']!='new' for r in rr),
           'later_new_routed_old':sum(r['new_event']['branch']=='old' for r in rr),
           'later_new_stopped_or_rejected':sum(r['new_event']['branch'] not in ('old','new') for r in rr),
           'stopped':sum(r['stopped'] for r in rr),
           'post_new_max_error_above_1e-6':sum(r['after_new']['max_latest_error']>1e-6 for r in rr),
           'max_post_new_latest_error_including_stopped_diagnostics':max(r['after_new']['max_latest_error'] for r in rr),
           'max_post_new_latest_error_without_stopped':max(r['after_new']['max_latest_error'] for r in rr if not r['stopped']),
           'max_true_null_leak_before_new':max(r['pre_new']['true_null_leak'] for r in rr),
           'max_clean_geometry_ghost_abs':max(abs(r['clean_geometry_ghost_scorer_only']) for r in rr),
           'max_noisy_query_residual_after_repeats':max(abs(r['pre_new']['residual_at_noisy']) for r in rr)}
    selected={}
    # Predetermined slices, no ranking by observed performance.
    selections=[
      (1.,1e-10,'none',0.,'null_0.02'),
      (.01,1e-10,'none',0.,'mixed_0.02'),
      (.01,1e-4,'none',0.,'clean'),
      (1.,1e-10,'weak_null_symmetric_mix',.1,'null_0.02'),
      (.01,1e-10,'null_lift',.1,'clean'),
      (.01,1e-4,'weak_erosion',.1,'mixed_0.001')]
    for s in selections:
        rr=[r for r in rows if (r['a'],r['cutoff'],r['fault']['name'],r['fault']['ratio'],r['noise']['name'])==s]
        selected[str(s)]={r['arm']:{k:r[k] for k in ['initial_rank','projector_error_initial','gap_cutoff_certified','repeat_branch_counts','pre_new','clean_geometry_ghost_scorer_only','new_event','after_new','old_output_drift_on_new','after_final_f2']} for r in rr}
    return {'by_arm':summary,'fixed_slices':selected}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('results_v2.json'))
    ap.add_argument('--spec',type=Path,default=Path(__file__).with_name('frozen_spec.json'))
    args=ap.parse_args()
    specbytes=args.spec.read_bytes();h=hashlib.sha256(specbytes).hexdigest()
    if h!=SPEC_HASH:raise RuntimeError('Frozen spec hash mismatch; use a new version for any changes.')
    spec=json.loads(specbytes)
    grid=run_grid(spec);fx=fixtures();su=summarize(grid)
    result={'version':spec['version'],'scope':spec['scope'],
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},
      'rng':'none','spec_sha256':h,
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'checks':grid['checks'],'summary':su,'fixtures':fx,'grid':grid['records']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False,ensure_ascii=False);f.write('\n')
    print(json.dumps({'checks':grid['checks'],'by_arm':su['by_arm']},indent=2))

if __name__=='__main__':main()
