"""Exact algebraic certificates only; no IID trials or parameter sweep."""
from fractions import Fraction as Q
from pathlib import Path
import json, math

OUT=Path(__file__).resolve().parent
delta=Q(1,20); D=Q(11,5); h=Q(1,10); T=Q(10); tol=Q(1,2)
alpha=2/(D*D+delta*delta); rho=(D*D-delta*delta)/(D*D+delta*delta)
assert alpha==Q(800,1937) and rho==Q(1935,1937)
assert 1-alpha*delta*delta==rho and 1-alpha*D*D==-rho
assert 6*rho**100>tol
assert 6*rho**2405>tol and 6*rho**2406<tol
lam=Q(99,100)**100
k1=Q(2); k2=(1-lam)*k1
assert Q(3,10)<=k2<k1<Q(5,2) and k1-k2>=delta
assert k2-(1-lam)*k1==0
# Closure maximum of converged first-target error; lambda threshold is exact.
threshold=Q(1,644)
assert Q(297,20)*threshold/(delta-Q(5,2)*threshold)==tol
gamma_cont=math.log(12)/(float(T)*float(delta*delta))
capture_gain=1/(1-float(lam))
gamma_fast=math.log(24)/(float(T)*float(delta*delta))
n_fast=math.ceil(float(T)*gamma_fast*float(D*D))
h_fast=float(T)/n_fast
assert n_fast==6153
assert 0<1-h_fast*gamma_fast*float(D*D)<1
fast_bound=6*(1-h_fast*gamma_fast*float(delta*delta))**n_fast
assert fast_bound<.25
out={
 'kind':'Exact analytic certificate; no IID data and no candidate simulation',
 'delta':str(delta),'D':str(D),'condition_ratio':str((D/delta)**2),
 'alpha_minimax':str(alpha),'gamma_minimax':str(alpha/h),'rho_minimax':str(rho),
 'worst_closure_residual_100':float(6*rho**100),
 'required_updates':2406,
 'certificate_previous_error':float(6*rho**2405),
 'certificate_required_error':float(6*rho**2406),
 'time_at_registered_h_ms':float(2406*h),
 'max_h_for_10ms_with_2406_updates':float(T/2406),
 'continuous_gamma_required_nonstrict':gamma_cont,
 'euler_gamma_stability_supremum':float(2/(h*D*D)),
 'first_tracking_lambda_native':float(lam),
 'effective_gap_collision_keys':[float(k1),float(k2)],
 'effective_gap_collision_exact':True,
 'first_tracking_lambda_threshold':str(threshold),
 'first_tracking_tau_euler_threshold_ms':float(h)/(1-float(threshold)**.01),
 'first_tracking_tau_continuous_threshold_ms':10/math.log(644),
 'total_abs_slope_variation_maxgap_100':float(5280*(1-rho**100)),
 'total_abs_slope_variation_maxgap_2406':float(5280*(1-rho**2406)),
 'leak_10ms_necessary_gamma':float(11*Q(1,10)/(delta*delta)),
 'leak_tau_necessary_at_minimax_gamma_ms':float(11/((alpha/h)*delta*delta)),
 'selected_calibrated_capture_gain':capture_gain,
 'selected_fast_gamma':gamma_fast,
 'selected_fast_updates':n_fast,
 'selected_fast_h_ms':h_fast,
 'selected_fast_min_rate_per_ms':gamma_fast*float(delta*delta),
 'selected_fast_max_rate_per_ms':gamma_fast*float(D*D),
 'selected_fast_euler_closure_error_bound':fast_bound,
 'selected_fast_continuous_closure_error_bound':.25,
 'selected_A100_oscillatory_slope_bound':float(6*alpha*D),
 'selected_global_slope_bound':120,
 'capture_relative_error_99_steps':capture_gain*(1-.99**99)-1,
 'capture_relative_error_101_steps':capture_gain*(1-.99**101)-1,
 'limitations':['Candidate-specific constant-gain bound','Perfect-anchor bound is favorable','Strict tolerances require strict parameter margins','No energy or all-neuron impossibility claim']
}
(OUT/'THEORY_CERTIFICATES.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
