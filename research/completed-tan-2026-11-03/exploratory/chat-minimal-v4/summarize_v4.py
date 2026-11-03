#!/usr/bin/env python3
"""Summarize saved records; never calls a learner or alters previous packages."""
from pathlib import Path
from fractions import Fraction as F
import argparse,csv,json,hashlib

def number(x):return F(x['exact']) if isinstance(x,dict) and 'exact' in x else F(x)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',default=str(Path(__file__).resolve().parent));a=ap.parse_args();p=Path(a.directory)
    r=json.loads((p/'results_v4.json').read_text());old=json.loads((p/'v3_reference_summary_v3.json').read_text());timing=json.loads((p/'timing_v4.json').read_text())
    rows=[];refusals=[];prior=[]
    for run in r['runs']:
        s=run['summary'];rows.append({'condition':run['condition']['name'],'method':run['method'],**s})
        for rec in run['records']:
            if not rec['decision']['accepted']:
                refusals.append({'condition':run['condition']['name'],'method':run['method'],'event':rec['event'],
                    'reason':rec['decision']['reason'],'before_hex':rec['snapshot_hex_before'],'after_hex':rec['snapshot_hex_after'],
                    'candidate_certificate':rec['decision'].get('detail',{}).get('candidate_certificate'),
                    'candidate_state':rec['decision'].get('detail',{}).get('candidate_state_for_external_audit'),
                    'current_committed_latest':rec['evaluator_only_committed_latest'],
                    'rejected_request_gap_if_readable':rec['rejected_request_gap_if_readable'],
                    'rejected_new_request_absent':rec['rejected_new_request_absent']})
        if run['method']=='redundant_common_certificate':
            oo=next(z for z in old['conditions'] if z['condition']==run['condition']['name'] and z['method']=='certified_moments')
            prior.append({'condition':run['condition']['name'],'v3_accepted_reported':oo['accepted'],'v4_common_accepted':s['accepted'],
               'v3_refusal_reasons_reported':oo['rejected_events'],'v4_refusal_reasons':s['rejected_events'],
               'accepted_count_same':oo['accepted']==s['accepted'],
               'source':'existing v3 summary copied byte-for-byte; no v3 re-execution'})
    summary={'scope':r['scope'],'environment':r['environment'],'rows':rows,
       'transactions':sum(z['attempts'] for z in rows),'accepted':sum(z['accepted'] for z in rows),
       'rejected':len(refusals),'all_refusals_bitwise_identical':all(z['before_hex']==z['after_hex'] for z in refusals),
       'common_certificate_equivalence':r['common_certificate_equivalence'],
       'prior_v3_report_comparison':prior,'timing_first_execution':timing,
       'results_sha256':hashlib.sha256((p/'results_v4.json').read_bytes()).hexdigest(),
       'caution':'Identical common-certificate coverage does not imply identical fault-detection abilities outside the no-drift error model.'}
    (p/'summary_v4.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (p/'refusals_v4.json').write_text(json.dumps({'scope':'Refusal is an incomplete requested write, not success.','records':refusals},indent=2,sort_keys=True)+'\n')
    with (p/'comparison_v4.csv').open('w',newline='') as f:
        names=['condition','method','accepted','attempts','persistent_words','max_accepted_latest_error','max_other_retention_error','answered_reads_requested','read_attempts_requested','requested_target_reads_within_epsilon']
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for z in rows:w.writerow({k:float(number(z[k])) if isinstance(z[k],dict) and 'exact' in z[k] else z[k] for k in names})
    print('wrote summary, refusal ledger, CSV; no new trials')
if __name__=='__main__':main()
