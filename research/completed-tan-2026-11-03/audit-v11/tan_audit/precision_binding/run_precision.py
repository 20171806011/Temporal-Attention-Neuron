"""One fixed persistent-bit budget, with prefix precision explicitly retained."""
from pathlib import Path
import sys,json,hashlib,time,resource,itertools
from fractions import Fraction as F
import numpy as np
OUT=Path(__file__).resolve().parent;REPO=OUT.parent.parent/'Temporal-Attention-Neuron'
sys.path.insert(0,str(REPO/'p1_experiments'))
from task_generator import TaskGenerator
ARMS=['RAW32','PHASE32','CACHE32','ENDPOINT_ONLY']
RANGES={'slope':(-128.,128.,16),'intercept':(-320.,320.,16),'key16':(.3,2.5,16),'value16':(-3.,3.,16),'key8':(.3,2.5,8),'value8':(-3.,3.,8)}

def decode_word(code,spec):
    lo,hi,bits=spec;assert isinstance(code,int) and 0<=code<2**bits
    return lo+code*(hi-lo)/(2**bits-1)

def quantize(x,spec,label):
    lo,hi,bits=spec
    if not np.isfinite(x) or x<lo-1e-12 or x>hi+1e-12:raise ValueError('RANGE_EXCEEDED')
    scaled=(x-lo)/(hi-lo)*(2**bits-1);code=int(np.rint(scaled))
    if not 0<=code<2**bits:raise ValueError('CODE_OUT_OF_RANGE')
    decoded=decode_word(code,spec);error=decoded-x;half=(hi-lo)/(2*(2**bits-1))
    assert abs(error)<=half+1e-10
    return code,dict(label=label,raw=float(x),range=[lo,hi],bits=bits,scaled=float(scaled),code=code,decoded=decoded,
                     error=error,half_step=half,numerical_boundary_band=bool(x<lo or x>hi))

class Writer:
    __slots__=('arm','phase','words')
    def __init__(self,arm):self.arm=arm;self.phase=0;self.words=[]
    def write(self,k,v):
        assert self.phase in (0,1);logs=[];extra={}
        if self.arm=='CACHE32':
            ck,rk=quantize(k,RANGES['key8'],'key');cv,rv=quantize(v,RANGES['value8'],'value')
            self.words.extend([ck,cv]);logs.extend([rk,rv]);self.phase+=1
        elif self.phase==0:
            if self.arm=='ENDPOINT_ONLY':self.words=[float(k),float(v)]
            else:
                specs=[RANGES['slope'],RANGES['intercept']] if self.arm=='RAW32' else [RANGES['key16'],RANGES['value16']]
                codek,rk=quantize(k,specs[0],'first_key');codev,rv=quantize(v,specs[1],'first_value');self.words=[codek,codev];logs.extend([rk,rv])
            self.phase=1
        else:
            if self.arm=='ENDPOINT_ONLY':oldk,oldv=self.words
            else:
                specs=[RANGES['slope'],RANGES['intercept']] if self.arm=='RAW32' else [RANGES['key16'],RANGES['value16']]
                oldk,oldv=[decode_word(c,s) for c,s in zip(self.words,specs)]
            denominator=float(k-oldk)
            if abs(denominator)<=1e-12:raise ValueError('WRITE_SINGULAR')
            a=(v-oldv)/denominator;b=v-a*k
            ca,ra=quantize(a,RANGES['slope'],'final_slope');cb,rb=quantize(b,RANGES['intercept'],'final_intercept')
            extra=dict(first_record_decoded=[oldk,oldv],effective_denominator=denominator,unquantized_written_coefficients=[a,b])
            self.words=[ca,cb];self.phase=2;logs.extend([ra,rb])
        # Payload snapshot is returned to the evaluator only, never retained by the writer.
        return dict(phase=self.phase,payload=list(self.words),quantization=logs,**extra)
    def query(self,q):
        assert self.phase==2
        if self.arm=='CACHE32':
            k=[decode_word(self.words[i],RANGES['key8']) for i in [0,2]]
            vals=[decode_word(self.words[i],RANGES['value8']) for i in [1,3]]
            return vals[int(np.argmin((np.array(k)-q)**2))]
        a=decode_word(self.words[0],RANGES['slope']);b=decode_word(self.words[1],RANGES['intercept'])
        return a*q+b

def bound_values():
    f=F;epsa=f(128,65535);epsb=f(320,65535);final=f(5,2)*epsa+epsb
    result={}
    for arm,ek,ev in [('RAW32',epsa,epsb),('PHASE32',f(11,10*65535),f(3,65535))]:
        gap=f(1,20)-ek;slope=(6+ev)/gap;intercept=3+f(5,2)*slope;prefix=ev+slope*ek
        assert gap>0 and slope<128 and intercept<320
        result[arm]=dict(eps_key=float(ek),eps_value=float(ev),gap_lower=float(gap),write_slope_upper=float(slope),write_intercept_upper=float(intercept),prefix_error_upper=float(prefix),final_error_upper=float(final),total_error_upper=float(prefix+final),total_exact=str(prefix+final))
    result['CACHE32']=dict(key_error_upper=float(f(11,2550)),total_error_upper=float(f(3,255)),total_exact=str(f(3,255)))
    result['ENDPOINT_ONLY']=dict(total_error_upper=float(final),total_exact=str(final))
    return result

def serialize(t,variant):
    pos=t.event_positions
    return dict(id=f'iid_{t.trial_id}_{variant}',base_id=f'iid_{t.trial_id}',group='iid',variant=variant,
        positions=list(pos),keys=t.keys[pos].tolist(),values=t.values[pos].tolist(),queries=list(t.query_cues),targets=list(t.target_values),seed_metadata=t.seed)

def variants(rec):
    a=dict(rec,variant='original',base_id=rec['id'])
    q=dict(a,id=rec['id']+'_query_swap',variant='query_swap',queries=rec['queries'][::-1],targets=rec['targets'][::-1])
    v=dict(a,id=rec['id']+'_value_swap',variant='value_swap',values=rec['values'][::-1])
    v['targets']=[v['values'][rec['keys'].index(qv)] for qv in rec['queries']]
    return [a,q,v]

def evaluate(rec,arm,bounds):
    w=Writer(arm);trace=[];status='VALID'
    try:
        for key,value in zip(rec['keys'],rec['values']):trace.append(w.write(key,value))
        pred=[w.query(q) for q in rec['queries']]
    except ValueError as e:
        status=str(e);w.phase=3;pred=[0.,0.]
    if status=='VALID':
        assert all(type(x) is int for x in w.words)
        assert len(w.words)==(4 if arm=='CACHE32' else 2)
        bits=8 if arm=='CACHE32' else 16;assert all(0<=x<2**bits for x in w.words)
        for step in trace:
            if arm!='ENDPOINT_ONLY' or step['phase']==2:assert all(type(x) is int for x in step['payload'])
    t1,t2=[20*x for x in rec['positions']]
    first_bits=128 if arm=='ENDPOINT_ONLY' else 16 if arm=='CACHE32' else 32
    peak=128 if arm=='ENDPOINT_ONLY' else 32
    resources=dict(first_phase_payload_bits=first_bits,final_payload_bits=32,peak_payload_bits=peak,controller_bits=2,
        allocated_payload_bit_ms=peak*100,occupied_payload_bit_ms=first_bits*(t2-t1)+32*(100-t2),controller_bit_ms=200,
        first_to_second_ms=t2-t1,first_to_readout_ms=100-t1,payload_write_events=2,
        phase_coding_switch=arm=='PHASE32',division_count=0 if arm=='CACHE32' else 1,static_ranges_are_shared_parameters=True,
        transient_arithmetic_bits=64,incoming_event_and_query_bits=64,physical_write_latency_simulated=False,biophysical_energy_estimate=None)
    row=dict(input_id=rec['id'],base_id=rec['base_id'],group=rec['group'],variant=rec['variant'],arm=arm,targets=rec['targets'],queries=rec['queries'],
        predictions=pred,status=status,valid=status=='VALID',payload=list(w.words),phase=w.phase,write_trace=trace,resources=resources,
        query_input_fields=['persistent_integer_codes','q','fixed_arm_constants'])
    if status=='VALID':
        error=float(max(abs(np.array(pred)-rec['targets'])));assert error<=bounds[arm]['total_error_upper']+1e-10
        row['observed_max_error']=error;row['analytic_error_bound']=bounds[arm]['total_error_upper'];row['bound_pass']=True
        if arm!='CACHE32':
            aa,bb=trace[1]['unquantized_written_coefficients'];a=decode_word(w.words[0],RANGES['slope']);b=decode_word(w.words[1],RANGES['intercept'])
            prefix=np.array([aa*k+bb-v for k,v in zip(rec['keys'],rec['values'])]);contribution=np.array([(a-aa)*k+(b-bb) for k in rec['keys']])
            actual=np.array([w.query(k)-v for k,v in zip(rec['keys'],rec['values'])]);assert max(abs(actual-prefix-contribution))<1e-10
            assert abs(prefix[1])<1e-10
            row['error_decomposition']=dict(chronological_pre_final_errors=prefix.tolist(),chronological_final_quant_contribution=contribution.tolist(),final_decoded_coefficients=[a,b])
        else:
            kdecoded=[decode_word(w.words[j],RANGES['key8']) for j in (0,2)]
            for q in rec['queries']:assert int(np.argmin((np.array(kdecoded)-q)**2))==rec['keys'].index(q)
            row['cache_identity_preserved']=True
    return row

def metrics(rows):
    e=abs(np.array([r['predictions'] for r in rows])-np.array([r['targets'] for r in rows]));valid=np.array([r['valid'] for r in rows])
    return dict(n=len(rows),coverage=float(valid.mean()),mean_e_comp=float(e.max(axis=1).mean()),max_error=float(e.max()),mae_per_role=e.mean(axis=0).tolist(),
        rmse_per_role=np.sqrt((e*e).mean(axis=0)).tolist(),valid_joint_0_5=float(np.mean(valid&np.all(e<.5,axis=1))),local_valid_joint_0_05=float(np.mean(valid&np.all(e<.05,axis=1))),
        status_counts={s:sum(r['status']==s for r in rows) for s in sorted(set(r['status'] for r in rows))})

def main():
    started=time.perf_counter();bounds=bound_values();gen=TaskGenerator();rng=np.random.RandomState(2026110215);inputs=[]
    for i in range(512):
        t=gen.generate_single_trial(rng,i,2026110215,N=2)
        inputs.extend([serialize(t,'original'),serialize(gen.create_query_swap_variant(t),'query_swap'),serialize(gen.create_value_swap_variant(t),'value_swap')])
    ap=OUT/'PREDEFINED_ADVERSARIES.json';assert hashlib.sha256(ap.read_bytes()).hexdigest()=='fb1e0af0e44e9352d2dedde84ca3f0c7343a2eb04f3aa6a2e5447e07d05f3023'
    for rec in json.loads(ap.read_text()):inputs.extend(variants(rec))
    ip=OUT/'inputs.jsonl';ip.write_text(''.join(json.dumps(x)+'\n' for x in inputs));ih=hashlib.sha256(ip.read_bytes()).hexdigest();(OUT/'INPUTS_SHA256').write_text(ih+'  inputs.jsonl\n')
    print('frozen inputs before metrics',ih,flush=True)
    rows=[evaluate(rec,arm,bounds) for rec in inputs for arm in ARMS]
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    assert len(inputs)==1590 and len(rows)==6360
    assert all(r['valid'] for r in rows), 'Unexpected saturation/singularity; retained ledger, halt interpretation'
    by={(r['base_id'],r['variant'],r['arm']):r for r in rows}
    for base in set(r['base_id'] for r in rows):
        for arm in ARMS:
            a=by[(base,'original',arm)];q=by[(base,'query_swap',arm)]
            assert a['payload']==q['payload'] and a['predictions'][::-1]==q['predictions']
    cells=[]
    for group,variant,arm in itertools.product(['iid','adversarial'],['original','query_swap','value_swap'],ARMS):
        rr=[r for r in rows if r['group']==group and r['variant']==variant and r['arm']==arm]
        cells.append(dict(group=group,variant=variant,arm=arm,**metrics(rr)))
    losses={a:np.array([r['observed_max_error'] for r in rows if r['group']=='iid' and r['variant']=='original' and r['arm']==a]) for a in ARMS}
    diffs={'RAW32_minus_ENDPOINT_ONLY':losses['RAW32']-losses['ENDPOINT_ONLY'],'PHASE32_minus_RAW32':losses['PHASE32']-losses['RAW32'],'PHASE32_minus_CACHE32':losses['PHASE32']-losses['CACHE32']}
    brng=np.random.RandomState(2026110216);indices=brng.randint(0,512,size=(2000,512))
    contrast={k:dict(mean=float(d.mean()),percentile95_ci=np.quantile(d[indices].mean(axis=1),[.025,.975]).tolist(),replicates=2000) for k,d in diffs.items()}
    resources={}
    for arm in ARMS:
        rr=[r for r in rows if r['group']=='iid' and r['variant']=='original' and r['arm']==arm]
        resources[arm]={k:dict(mean=float(np.mean([r['resources'][k] for r in rr])),min=min(r['resources'][k] for r in rr),max=max(r['resources'][k] for r in rr)) for k in ['peak_payload_bits','allocated_payload_bit_ms','occupied_payload_bit_ms','first_to_second_ms']}
    result=dict(protocol_sha256=hashlib.sha256((OUT/'PREREGISTRATION.md').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256=ih,
        input_count=len(inputs),prediction_rows=len(rows),role_predictions=2*len(rows),fresh_histories=512,reused_adversarial=18,bounds=bounds,metrics=cells,
        paired_fresh_original_contrasts=contrast,resources=resources,quantization_events=sum(len(s['quantization']) for r in rows for s in r['write_trace']),
        numerical_boundary_bands=sum(q['numerical_boundary_band'] for r in rows for s in r['write_trace'] for q in s['quantization']),
        validation=dict(integer_payload_budget=True,cache_query_identity=True,no_saturation_or_singularity=True,all_raw_error_bounds=True,query_swap_exact=True,first_error_decomposition=True),
        wall_seconds=time.perf_counter()-started,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitation='32bits denotes persistentpayload only, plus2controllerbits; exactinputs64bitarithmetic/timing/division/gains/readout outsidebudget; endpointonly peak128bits; notTANneuron.' )
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'metrics':cells,'contrasts':contrast,'resources':resources,'wall_seconds':result['wall_seconds']},indent=2))

if __name__=='__main__':main()
