"""Independent full-ledger review; never imports the experimental implementation.

Reconstructs generator draw order, online code words, query interface, exact bounds,
metrics and paired bootstrap. Outputs only separate reviewer artifacts.
"""
from pathlib import Path
from fractions import Fraction as Q
import ast, collections, hashlib, json, math, time
import numpy as np

P=Path(__file__).resolve().parent
ARMS=('RAW32','PHASE32','CACHE32','ENDPOINT_ONLY')
SPECS={'a':(-128.,128.,16),'b':(-320.,320.,16),
       'k16':(.3,2.5,16),'v16':(-3.,3.,16),
       'k8':(.3,2.5,8),'v8':(-3.,3.,8)}

def close(x,y,tol=1e-12):
    assert np.allclose(x,y,rtol=0,atol=tol),(x,y)

def expand(base):
    q=dict(base,id=base['id'].replace('_original','')+'_query_swap',variant='query_swap',
           queries=base['queries'][::-1],targets=base['targets'][::-1])
    v=dict(base,id=base['id'].replace('_original','')+'_value_swap',variant='value_swap',values=base['values'][::-1])
    v['targets']=[v['values'][base['keys'].index(x)] for x in base['queries']]
    return [base,q,v]

def reconstruct_inputs():
    rng=np.random.RandomState(2026110215);out=[];reject=0
    for i in range(512):
        while True:
            keys=np.sort(rng.uniform(.3,2.5,2))
            if keys[1]-keys[0]>=.05:break
            reject+=1
        keys=rng.permutation(keys).tolist();values=rng.uniform(-3.,3.,2).tolist()
        positions=sorted(rng.choice(5,2,replace=False).tolist())
        target_positions=rng.choice(positions,2,replace=False).tolist()
        inds=[positions.index(x) for x in target_positions]
        out.extend(expand(dict(id=f'iid_{i}_original',base_id=f'iid_{i}',group='iid',variant='original',
            positions=positions,keys=keys,values=values,queries=[keys[j] for j in inds],
            targets=[values[j] for j in inds],seed_metadata=2026110215)))
    for x in json.loads((P/'PREDEFINED_ADVERSARIES.json').read_text()):
        out.extend(expand(dict(x,base_id=x['id'])))
    return out,reject

def dequant(c,s):
    lo,hi,bits=s
    return lo+c*(hi-lo)/(2**bits-1)

def round_code(x,s):
    lo,hi,bits=s
    assert lo<=x<=hi
    # Built-in round is nearest-even for a binary64 scaled argument.
    c=round((x-lo)/(hi-lo)*(2**bits-1))
    assert 0<=c<2**bits
    return c

def exact_nearest_binary_input(x,s):
    lo,hi,bits=s
    scaled=(Q(x)-Q(lo))/(Q(hi)-Q(lo))*(2**bits-1)
    n=scaled.numerator//scaled.denominator;remainder=scaled-n
    return n+(remainder>Q(1,2) or (remainder==Q(1,2) and n%2==1))

def check_quantization(entry):
    s=(*entry['range'],entry['bits']);x=entry['raw'];code=round_code(x,s)
    assert code==entry['code'];decoded=dequant(code,s)
    close(decoded,entry['decoded']);close(decoded-x,entry['error'])
    half=(s[1]-s[0])/(2*(2**s[2]-1));close(half,entry['half_step'])
    assert abs(decoded-x)<=half+1e-10
    assert not entry['numerical_boundary_band']
    exact=exact_nearest_binary_input(x,s)
    return int(exact!=code),abs(float((Q(s[0])+code*(Q(s[1])-Q(s[0]))/(2**s[2]-1))-decoded))

def online(c,arm):
    k1,k2=c['keys'];v1,v2=c['values']
    if arm=='CACHE32':
        first=[round_code(k1,SPECS['k8']),round_code(v1,SPECS['v8'])]
        words=first+[round_code(k2,SPECS['k8']),round_code(v2,SPECS['v8'])]
        return first,words,None
    if arm=='ENDPOINT_ONLY':first=[k1,v1];kh,vh=first
    else:
        pair=('a','b') if arm=='RAW32' else ('k16','v16')
        first=[round_code(k1,SPECS[pair[0]]),round_code(v1,SPECS[pair[1]])]
        kh,vh=[dequant(x,SPECS[n]) for x,n in zip(first,pair)]
    a=(v2-vh)/(k2-kh);b=v2-a*k2
    return first,[round_code(a,SPECS['a']),round_code(b,SPECS['b'])],(kh,vh,a,b)

def query(words,arm,q):
    # Deliberately accepts no trial, targets, first record, or input history.
    if arm=='CACHE32':
        keys=[dequant(words[j],SPECS['k8']) for j in (0,2)]
        winner=min(range(2),key=lambda j:abs(keys[j]-q))
        return dequant(words[2*winner+1],SPECS['v8']),winner
    return dequant(words[0],SPECS['a'])*q+dequant(words[1],SPECS['b']),None

def rational_online(c,arm):
    # Exact rational arithmetic on the binary64 input values and binary64
    # declared range endpoints. Only an audit oracle, never a fitted decoder.
    def quant(x,name):
        lo,hi,bits=SPECS[name];z=(x-Q(lo))/(Q(hi)-Q(lo))*(2**bits-1)
        n=z.numerator//z.denominator;rem=z-n
        return n+int(rem>Q(1,2) or (rem==Q(1,2) and n%2))
    def deq(code,name):
        lo,hi,bits=SPECS[name];return Q(lo)+code*(Q(hi)-Q(lo))/(2**bits-1)
    k1,k2=map(Q,c['keys']);v1,v2=map(Q,c['values'])
    if arm=='CACHE32':return [quant(k1,'k8'),quant(v1,'v8'),quant(k2,'k8'),quant(v2,'v8')]
    if arm=='ENDPOINT_ONLY':kh,vh=k1,v1
    else:
        kn,vn=('a','b') if arm=='RAW32' else ('k16','v16')
        kh,vh=deq(quant(k1,kn),kn),deq(quant(v1,vn),vn)
    a=(v2-vh)/(k2-kh);b=v2-a*k2
    return [quant(a,'a'),quant(b,'b')]

def main():
    start=time.perf_counter();r=json.loads((P/'results.json').read_text())
    inputs=[json.loads(x) for x in (P/'inputs.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (P/'predictions.jsonl').read_text().splitlines()]
    expected,rejections=reconstruct_inputs();assert expected==inputs
    assert len(inputs)==1590 and len(rows)==6360 and 2*len(rows)==12720
    assert len({x['id'] for x in inputs})==1590
    source={x['id']:x for x in inputs};idx={}
    f=Q(640,65535);bounds={};details={}
    for arm,ek,ev in [('RAW32',Q(128,65535),Q(320,65535)),('PHASE32',Q(11,655350),Q(3,65535))]:
        gap=Q(1,20)-ek;ab=(6+ev)/gap;bb=3+Q(5,2)*ab;prefix=ev+ab*ek
        bounds[arm]=prefix+f;details[arm]=(gap,ab,bb,prefix)
        assert gap>0 and ab<128 and bb<320
    bounds['CACHE32']=Q(3,255);bounds['ENDPOINT_ONLY']=f
    assert 2*Q(11,2550)<Q(1,20)
    for a,v in bounds.items():
        assert str(v)==r['bounds'][a]['total_exact'];close(float(v),r['bounds'][a]['total_error_upper'])
    qcount=0;tie_differences=[];oracle_differences=[];max_dequant_residual=0.;max_prefix={a:[0.,0.] for a in ARMS if a!='CACHE32'}
    max_decomp=0.;first_slot_counts=collections.Counter();group_counts=collections.Counter()
    for x in inputs:
        assert all(.3<=k<2.5 for k in x['keys']) and abs(x['keys'][1]-x['keys'][0])>=.05
        assert all(-3<=v<3 for v in x['values'])
        assert len(set(x['positions']))==2 and x['positions']==sorted(x['positions']) and all(0<=p<5 for p in x['positions'])
        assert x['targets']==[x['values'][x['keys'].index(q)] for q in x['queries']]
        if x['group']=='iid' and x['variant']=='original':first_slot_counts[x['positions'][0]]+=1
    for row in rows:
        x=source[row['input_id']];a=row['arm'];group_counts[(x['group'],x['variant'],a)]+=1
        assert row['targets']==x['targets'] and row['queries']==x['queries']
        assert row['group']==x['group'] and row['variant']==x['variant'] and row['base_id']==x['base_id']
        assert row['status']=='VALID' and row['valid'] and row['phase']==2 and row['bound_pass']
        first,words,temp=online(x,a);tr=row['write_trace'];assert len(tr)==2
        exact_words=rational_online(x,a)
        if words!=exact_words:oracle_differences.append(dict(input_id=x['id'],arm=a,binary_payload=words,exact_payload=exact_words))
        assert tr[0]['payload']==first and tr[1]['payload']==words and row['payload']==words
        assert [t['phase'] for t in tr]==[1,2]
        assert all(type(w) is int and 0<=w<2**(8 if a=='CACHE32' else 16) for w in words)
        if a!='ENDPOINT_ONLY':assert all(type(w) is int for w in first)
        if a=='CACHE32':
            for j,t in enumerate(tr):
                assert [e['label'] for e in t['quantization']]==['key','value']
                close([e['raw'] for e in t['quantization']],[x['keys'][j],x['values'][j]])
        else:
            if a=='ENDPOINT_ONLY':assert tr[0]['quantization']==[]
            else:
                assert [e['label'] for e in tr[0]['quantization']]==['first_key','first_value']
                close([e['raw'] for e in tr[0]['quantization']],[x['keys'][0],x['values'][0]])
            assert [e['label'] for e in tr[1]['quantization']]==['final_slope','final_intercept']
            close([e['raw'] for e in tr[1]['quantization']],temp[2:])
        for step in tr:
            for event in step['quantization']:
                different,res=check_quantization(event);qcount+=1;max_dequant_residual=max(max_dequant_residual,res)
                if different:tie_differences.append(dict(input_id=x['id'],arm=a,label=event['label'],binary_code=event['code'],exact_binary_input_code=exact_nearest_binary_input(event['raw'],(*event['range'],event['bits']))))
        pred=[query(words,a,q)[0] for q in x['queries']];close(pred,row['predictions'])
        e=max(abs(p-v) for p,v in zip(pred,x['targets']));close(e,row['observed_max_error'])
        assert e<=float(bounds[a])+1e-10
        if a=='CACHE32':
            assert row['cache_identity_preserved']
            for q in x['queries']:assert query(words,a,q)[1]==x['keys'].index(q)
        else:
            kh,vh,aa,bb=temp;close(tr[1]['first_record_decoded'],[kh,vh]);close(tr[1]['effective_denominator'],x['keys'][1]-kh)
            close(tr[1]['unquantized_written_coefficients'],[aa,bb])
            if a in details:
                gap,ab,bbound,pb=details[a]
                assert abs(x['keys'][1]-kh)>=float(gap)-1e-13 and abs(aa)<=float(ab)+1e-10 and abs(bb)<=float(bbound)+1e-10
            d=row['error_decomposition'];an,bn=[dequant(w,SPECS[n]) for w,n in zip(words,('a','b'))]
            pre=[aa*k+bb-v for k,v in zip(x['keys'],x['values'])]
            fin=[(an-aa)*k+(bn-bb) for k in x['keys']]
            close(pre,d['chronological_pre_final_errors']);close(fin,d['chronological_final_quant_contribution']);close([an,bn],d['final_decoded_coefficients'])
            close(pre[0],vh-x['values'][0]-aa*(kh-x['keys'][0]),1e-10);assert abs(pre[1])<1e-10
            for j in range(2):
                max_prefix[a][j]=max(max_prefix[a][j],abs(pre[j]));actual=query(words,a,x['keys'][j])[0]-x['values'][j]
                max_decomp=max(max_decomp,abs(actual-pre[j]-fin[j]));assert abs(fin[j])<=float(f)+1e-10
        t1,t2=[20*p for p in x['positions']];fb=128 if a=='ENDPOINT_ONLY' else 16 if a=='CACHE32' else 32;peak=max(fb,32)
        resources=dict(first_phase_payload_bits=fb,final_payload_bits=32,peak_payload_bits=peak,controller_bits=2,
            allocated_payload_bit_ms=100*peak,occupied_payload_bit_ms=fb*(t2-t1)+32*(100-t2),controller_bit_ms=200,
            first_to_second_ms=t2-t1,first_to_readout_ms=100-t1,payload_write_events=2,phase_coding_switch=a=='PHASE32',
            division_count=0 if a=='CACHE32' else 1,static_ranges_are_shared_parameters=True,transient_arithmetic_bits=64,
            incoming_event_and_query_bits=64,physical_write_latency_simulated=False,biophysical_energy_estimate=None)
        assert resources==row['resources'];idx[(x['base_id'],x['variant'],a)]=row
    assert qcount==22260==r['quantization_events'];assert max_decomp<1e-10
    for (base,var,a),row in idx.items():
        if var=='original':
            swapped=idx[(base,'query_swap',a)];assert row['payload']==swapped['payload'] and row['predictions'][::-1]==swapped['predictions']
    for cell in r['metrics']:
        rr=[x for x in rows if (x['group'],x['variant'],x['arm'])==(cell['group'],cell['variant'],cell['arm'])]
        errors=np.abs(np.array([x['predictions'] for x in rr])-np.array([x['targets'] for x in rr]));n=len(rr)
        assert n==cell['n']==(512 if cell['group']=='iid' else 18) and cell['coverage']==1
        close(errors.max(axis=1).mean(),cell['mean_e_comp']);close(errors.max(),cell['max_error']);close(errors.mean(axis=0),cell['mae_per_role'])
        close(np.sqrt((errors**2).mean(axis=0)),cell['rmse_per_role'])
        for delta,key in [(.5,'valid_joint_0_5'),(.05,'local_valid_joint_0_05')]:close(np.all(errors<delta,axis=1).mean(),cell[key])
        assert cell['status_counts']=={'VALID':n}
    loss={a:np.array([idx[(f'iid_{i}','original',a)]['observed_max_error'] for i in range(512)]) for a in ARMS}
    bi=np.random.RandomState(2026110216).randint(0,512,size=(2000,512))
    for label,c in r['paired_fresh_original_contrasts'].items():
        a,b=label.split('_minus_');diff=loss[a]-loss[b]
        close(diff.mean(),c['mean']);close(np.quantile(diff[bi].mean(axis=1),[.025,.975]),c['percentile95_ci']);assert c['replicates']==2000
    for a,stats in r['resources'].items():
        for key,val in stats.items():
            series=[idx[(f'iid_{i}','original',a)]['resources'][key] for i in range(512)]
            close([np.mean(series),min(series),max(series)],[val['mean'],val['min'],val['max']])
    for field,name in [('protocol_sha256','PREREGISTRATION.md'),('script_sha256','run_precision.py'),('inputs_sha256','inputs.jsonl')]:assert hashlib.sha256((P/name).read_bytes()).hexdigest()==r[field]
    tree=ast.parse((P/'run_precision.py').read_text());writer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Writer')
    slot=next(n for n in writer.body if isinstance(n,ast.Assign));assert ast.literal_eval(slot.value)==('arm','phase','words')
    querynode=next(n for n in writer.body if isinstance(n,ast.FunctionDef) and n.name=='query');assert [a.arg for a in querynode.args.args]==['self','q']
    globals_read={n.id for n in ast.walk(querynode) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
    assert globals_read<={'self','q','k','vals','i','a','b','decode_word','RANGES','int','np'}
    report=dict(status='PASS',independent_generator=True,generator_rejections=rejections,input_count=len(inputs),ledger_rows=len(rows),role_outputs=2*len(rows),metrics_cells=len(r['metrics']),quantization_events=qcount,
        exact_bounds={a:str(b) for a,b in bounds.items()},all_query_outputs_independently_reconstructed=True,all_resources_verified=True,bootstrap_replicates=2000,
        binary_scaled_vs_exact_input_rounding_disagreements=tie_differences,max_exact_dequant_float_residual=max_dequant_residual,
        full_rational_online_oracle_rows=len(rows),full_rational_online_payload_differences=oracle_differences,
        max_chronological_pre_final_errors=max_prefix,max_error_decomposition_residual=max_decomp,first_slot_counts=dict(first_slot_counts),
        writer_slots=['arm','phase','words'],query_arguments=['self','q'],original_sources_preserved=True,seconds=time.perf_counter()-start,
        limitation='Persistent payload bits are logical code capacity, not Python memory, total circuit state, transient workspace, latency, energy or neuronal implementation.')
    (P/'INDEPENDENT_REVIEW_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
