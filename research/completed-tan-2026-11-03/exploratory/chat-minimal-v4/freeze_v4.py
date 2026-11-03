from pathlib import Path
import json,hashlib,datetime
p=Path(__file__).resolve().parent
v3=json.loads((p/'v3_reference_frozen_spec.json').read_text())
spec={
'version':'minimal-scalar-moment-v4.0',
'scope':'One bounded conceptual Chat sandbox comparison; not TAN/P1-B/v11 execution. No Codex, no training, no tuning.',
'freeze_stage':'This specification and hash were written before implementing or executing v4 learners.',
'Nmax':v3['Nmax'], 'key_range':v3['key_range'], 'value_bound':v3['value_bound'],
'required_value_error':v3['required_value_error'],
'ordinary_sensor_error_bound':v3['ordinary_sensor_error_bound'],
'stored_word_bits':64, 'rounding':v3['rounding'],
'conditions':v3['conditions'], 'events':v3['events'],
'methods':['redundant_common_certificate','minimal_common_certificate','explicit_table_same_observations'],
'payloads':{
 'redundant_common_certificate':{'m_exponents':'0..2L','b_exponents':'0..L','payload_words':'3L+2','dynamic_aux_words':'count,mu,beta,sigma: 4'},
 'minimal_common_certificate':{'m_exponents':'1..L','b_exponents':'0..L-1','payload_words':'2L','dynamic_aux_words':'count,mu,beta,sigma: 4'},
 'explicit_table_same_observations':{'payload_words':'L anchors + L values = 2L','dynamic_aux_words':'count,sigma: 2'}},
'common_certificate':{
 'root_inputs':'count N, m1..mN and common componentwise moment error mu only; m0=N constructed transiently for minimal.',
 'root_proposal':v3['root_proposal'],
 'root_search':v3['root_certificate'],
 'value_inputs':'first N b moments, certified root intervals, beta; interval Lagrange inversion',
 'gates':['integer shape/domain checks','certified disjoint root intervals','same v3 OLD/NEW/ambiguous address rule','capacity N<L for NEW','candidate root certificate','all candidate value errors <= epsilon'],
 'removed_for_both_moment_methods':['Hankel alpha_lower > 4*(L+1)*mu gate','redundant/high-order moment consistency gate'],
 'no_surrogate_spectral_gate':True,
 'old_update_bound':'beta_new = ceil_h(beta + E_i + abs(r)*L*root_radius + h/2); L used for BOTH representations, never L-1 in this comparison.',
 'new_update_bounds':'mu_new=ceil_h(mu+h/2), beta_new=ceil_h(beta+h/2)',
 'new_anchor':'round_h(observation); sigma_new=max(sigma,ceil_h(zeta+abs(anchor-observation)))',
 'no_projection_repair_or_key_replacement':True},
'canonical_key_semantics':v3['canonical_key_semantics'],
'external_contract':v3['external_contract'],
'holding_fault_model':'No unmodelled drift, counter corruption, bit flips, or lost transactions; initialized zero; fixed-point updates are the only storage perturbations. No continuous-time holding experiment.',
'evaluation':v3['evaluation'],
'fairness':'Same observations/sensor bounds/value range/capacity/64-bit persistent word format. Not equal total bits. Exact-reference algebra is not fixed-bit hardware.',
'primary_isolation_prediction':'Given identical used prefixes and common budgets, redundant and minimal methods must have identical roots, decisions, predictions and coverage at every request. Extra stored moments are not read by common certification.',
'ideal_reference':'Small exact-rational minimal fixture for N=0..L, zero-key/count, repeat, full-capacity new refusal and unknown query. Prior v3 exact results referenced, not rerun as a stream competitor.',
'prior_v3_policy':'Copy existing summary as report only. Do not execute v3 learner or overwrite v3 files. A source audit comparison may compare reports.',
'measurements':['same 12 requests x 3 conditions x 3 methods','accepted latest error','all other committed-memory retention error','completed versus requested targets','write and read coverage','exact snapshot bytes for every refusal including count/budgets','root reconstruction and value-certification wall times by method','instrumented rational bit sizes and structural workspace counts'],
'runtime_policy':'Timing is descriptive host-wall-clock cost, not energy/hardware complexity; written separately so correctness results can be byte reproducible.',
'failure_policy':'Record all failed certificates; no threshold/precision/sequence search or post-result tightening. Fail-closed, original persistent bytes unchanged on refusal.',
'randomness':'None. No seed; observations and all offsets inherited exactly from v3.',
'inputs_sha256':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['v3_reference_scalar_moment_v3.py','v3_reference_frozen_spec.json','v3_reference_summary_v3.json']}}
b=json.dumps(spec,ensure_ascii=False,indent=2,sort_keys=True).encode()+b'\n'
(p/'frozen_spec_v4.json').write_bytes(b)
(p/'SPEC_INITIAL_SHA256.txt').write_text(hashlib.sha256(b).hexdigest()+'  frozen_spec_v4.json\n')
(p/'freeze_receipt.json').write_text(json.dumps({'spec_sha256':hashlib.sha256(b).hexdigest(),'utc_written':datetime.datetime.now(datetime.timezone.utc).isoformat(),'note':'No v4 learner or v4 trials existed before this file. System wall clock is provenance only.'},indent=2)+'\n')
print('Frozen spec SHA256',hashlib.sha256(b).hexdigest())
