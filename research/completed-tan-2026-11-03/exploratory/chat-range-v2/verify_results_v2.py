#!/usr/bin/env python3
"""Separate result verifier: arithmetic/contracts, not external experimental replication.
Uses stored records, exact rational counterexample checks, and source/spec hashes.
Run: python verify_results_v2.py [--results results_v2.json] [--out verification_v2.json]
"""
from __future__ import annotations
import argparse, hashlib, json
from fractions import Fraction as F
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,default=root/'results_v2.json')
    ap.add_argument('--out',type=Path,default=root/'verification_v2.json')
    a=ap.parse_args();r=json.loads(a.results.read_text())
    assert r['spec_sha256']==sha(root/'frozen_spec.json')
    assert r['script_sha256']==sha(root/'range_consistency_audit_v2.py')
    grid=r['grid'];assert len(grid)==560
    identities=set((z['case'],z['arm']) for z in grid);assert len(identities)==560
    counters={'rank_matched_gap_projector_checks':0,'rank_matched_gap_nonzero_fault_checks':0,
              'PN_presented_input_exact_within_1e-9':0,'fixed_C_recurrence_checks':0}
    comparisons=[]
    for z in grid:
        alpha,delta=z['alpha_initial'],z['delta_norm']
        # Supplemental algebraic check, not a newly selected experimental configuration.
        # In the stored grid, retained rank=2 means top-two eigenvectors. With delta<alpha/2,
        # ||Qhat-Q|| <= delta/(alpha-delta), regardless of a sufficient cutoff placement.
        if z['initial_rank']==2 and delta<alpha/2:
            assert z['projector_error_initial']<=delta/(alpha-delta)+2e-9
            counters['rank_matched_gap_projector_checks']+=1
            counters['rank_matched_gap_nonzero_fault_checks']+=int(delta>0)
        for step in z['repeat_trajectory']:
            ev=step['event']
            if ev['branch']=='old':
                ell=ev['leverage'];rb=ev['residual_before'];ra=ev['residual_after']
                expected=0. if z['arm'] in ('PN','TK') else (1-ell)*rb
                assert abs(ra-expected)<=1e-7*(1+abs(rb)+abs(ra))
                counters['fixed_C_recurrence_checks']+=1
        if z['arm']=='PN' and not z['stopped']:
            assert abs(z['pre_new']['residual_at_presented'])<1e-9
            counters['PN_presented_input_exact_within_1e-9']+=1
    # Whole-b Q projection leaves the current K readout unchanged (before C changes).
    for case in sorted(set(z['case'] for z in grid)):
        a0={z['arm']:z for z in grid if z['case']==case}
        u,p=a0['U'],a0['P']
        yu=np.array(u['pre_new']['true_predictions']);yp=np.array(p['pre_new']['true_predictions'])
        err=float(np.max(np.abs(yu-yp)));scale=1+float(np.max(np.abs(yu)))
        assert err/scale<1e-7
        comparisons.append(err/scale)
    # Exact rational check of the preserved normalized-address counterexample:
    # in true-key coordinates the update direction is (-1,2), leverage=5.
    coord=[F(-1),F(2)];ell=sum(x*x for x in coord)
    y=[x/ell for x in coord]
    assert y==[F(-1,5),F(2,5)] and sum(c*z for c,z in zip(coord,y))==1
    # Exact rational rotated projection of z*e3 onto .8e2+.6e3.
    co,si,z=F(4,5),F(3,5),F(1,1000)
    projected=[co*si*z,si*si*z]
    assert projected==[F(3,6250),F(9,25000)]
    fx=r['fixtures']['valid_weak_content_deletion']['arms']
    assert fx['U']['readout_after_threshold_recovery']==[0.0,1.0]
    for arm in ['P','PN','TK']:assert fx[arm]['readout_after_threshold_recovery']==[0.0,0.0]
    output={'all_checks_passed':True,'scope':'Independent verification path within same Chat environment, not third-party replay',
      'results_sha256':sha(a.results),'counters':counters,
      'max_relative_U_P_pre_new_readout_difference':max(comparisons),
      'exact_rational_wrong_address':[str(v) for v in y],
      'exact_rational_rotated_projection':[str(v) for v in projected],
      'supplement_note':'Rank-matched perturbation bound checks were added after examining main-grid coverage; no grid or model parameters were changed. The frozen sufficient cutoff certificate covered only zero-fault cases.'}
    a.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
