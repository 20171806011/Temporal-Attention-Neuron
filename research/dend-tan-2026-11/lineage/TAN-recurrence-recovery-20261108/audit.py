"""Independent native replay, optimizer reconstruction and selection/accounting audit."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from independent import replay,make_base,assemble,initial
R=Path(__file__).resolve().parent;START=time.perf_counter();CHECKS=0;MAXERR=0.
PARTS={'d':(0,),'vp':(1,3),'dvp':(0,1,3),'o':(2,),'all':(0,1,2,3)}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(v,msg):
    global CHECKS
    CHECKS+=1
    if not bool(v):raise AssertionError(msg)
    if time.perf_counter()-START>300:raise TimeoutError('300 second independent audit cap')
def close(a,b,msg):
    global MAXERR
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) if np.size(a) else 0.;MAXERR=max(MAXERR,err);check(err<1e-8,msg+f' {err}')
def verify_base(p,n,seed):
    got=dict(np.load(p));b=make_base(n,seed)
    for k,v in b.items():check(np.array_equal(got[k],v),'data regeneration '+k)
    return got
def prediction(c):
    p=np.full(len(c),-1);p[c[:,0]>c[:,1]]=0;p[c[:,1]>c[:,0]]=1;return p
def metric(c,y,row):
    p=prediction(c);rec=[np.mean(p[y==j]==j) for j in (0,1)]
    vals=dict(balanced_accuracy=sum(rec)/2,accuracy=np.mean(p==y),coverage=np.mean(p>=0),silence=np.mean(c.sum(1)==0),recalls=rec,prediction_counts=[sum(p==j) for j in (-1,0,1)])
    for k,v in vals.items():close(v,row[k],k)
    return vals
def objective(c,y,loss,w):
    c=c.astype(float)
    if loss=='RATE':
        target=np.full(c.shape,.125);target[np.arange(len(y)),y]=.75;value=np.mean((c/8-target)**2)
    else:
        z=.5*c;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);value=-np.log(p[np.arange(len(y)),y]).mean()
    return value+.0001*np.mean(w*w)
def accounting(v,row,h,arm,T,suffix=False):
    r=v['resources'];keys=('branch_events','hidden_events','output_events','recurrent_deliveries','output_deliveries','hold_branch_events','hold_hidden_events','hold_output_events')
    vals={('suffix_' if suffix else '')+k:r[:,j].mean() for j,k in enumerate(keys[:5] if suffix else keys)}
    if suffix:vals.update(suffix_steps=T,suffix_forward_macs=T*(2*h*h+14*h),suffix_state_updates=T*(3*h+2),suffix_graded_local_transfers=T*2*h if arm=='DEND' else 0)
    else:
        vals.update(branch_deliveries=vals['branch_events'],internal_deliveries=vals['branch_events']+vals['recurrent_deliveries']+vals['output_deliveries'],
          state_updates=T*(3*h+2),graded_local_transfers=T*2*h if arm=='DEND' else 0,forward_macs=T*(2*h*h+14*h),parameters=2*h*h+17*h+2,
          continuous_slots=3*h+2,persistent_continuous=3*h+2,previous_spike_registers=h,output_counters=2,spiking_cells=h+2 if arm=='DEND' else 3*h+2,graded_branches=2*h if arm=='DEND' else 0)
    for k,value in vals.items():close(value,row[k],'resource '+k)
def planned_cases():
    rows=[dict(condition='baseline',delay=d) for d in (16,24,28,36,48,64,72,84,96)]
    for duration in (4,12,24):
        for recovery in (0,12,48):
            d=12+duration+recovery
            rows.append(dict(condition=f'cut_L{duration}_R{recovery}',delay=d,cut=[28,28+duration],L=duration,recovery=recovery))
            if recovery>0:rows.append(dict(condition=f'keepoff_L{duration}_R{recovery}',delay=d,cut=[28,16+d],L=duration,recovery=recovery))
    for d in (24,48,96):
        for part in ('d','vp','all'):rows.append(dict(condition='zero_'+part,delay=d,zero_tick=28,zero_part=part))
        rows.append(dict(condition='identity',delay=d,sham_tick=28))
    return rows
def main():
    for n,h in read(R/'FREEZE.json')['files'].items():check(sha(R/n)==h,'scientific freeze '+n)
    source=read(R/'SOURCE_LOCK.json');old=(R/source['directory']).resolve();check(sha(old/'MANIFEST.json')==source['manifest_sha256'],'source manifest');check(sha(old/'AUDIT.json')==source['audit_sha256'],'source independent audit')
    rows=read(R/'evaluate/ROWS.json');check(len(rows)==864,'864 preplanned evaluations');done=0;total_macs=0;total_nominal=0
    for s in range(3):
        b=verify_base(R/'evaluate'/f'test_s{s}.npz',256,3941020000+100*s)
        for tr in [r for r in source['cohort'] if r['seed']==s]:
            ident=tr['id'];h=tr['h'];arm=tr['arm'];path=old/source['weights'][ident]['path'];check(sha(path)==source['weights'][ident]['sha256'],'locked source weights');w=np.load(path)['theta'];rr=[r for r in rows if r['id']==ident];check(len(rr)==36,'complete per-model cases');baseline={}
            for row,case in zip(rr,planned_cases()):
                for k,v in case.items():check(row[k]==v,'independent case timing '+k)
                D=case['delay'];x,q=assemble(b,D,'clean');check(np.all(x[:,16:q]==0),'strict zero external hold');ref=replay(x,w,arm,h,q,case=case);saved=dict(np.load(R/'evaluate'/f'{ident}_d{D}_{case["condition"]}.npz'))
                for k in ('counts','spikes'):check(np.array_equal(ref[k],saved[k]),'all native '+k)
                for k in ('states','resources'):close(ref[k],saved[k],'all native '+k)
                metric(ref['counts'],b['y'],row);accounting(ref,row,h,arm,q+8)
                bounds=case.get('cut',(-1,-1));cutsteps=sum(bounds[0]<=t<bounds[1] for t in range(q+8));executed=(q+8)*(2*h*h+14*h)-cutsteps*2*h*h
                check(row['cut_steps']==cutsteps and row['executed_forward_macs']==executed,'removed recurrent matrix operation accounting');reset=dict(d=2*h,vp=2*h,all=4*h+2).get(case.get('zero_part'),0);check(row['reset_values']==reset,'zeroed state/register values')
                if case['condition']=='baseline':baseline[D]=ref
                if case['condition']=='identity':
                    for k in ('counts','spikes','states','resources'):check(np.array_equal(ref[k],baseline[D][k]),'identity sham '+k)
                total_macs+=256*executed;total_nominal+=256*row['forward_macs'];done+=1
            print('audit',ident,'36 conditions complete',flush=True)
    ss=read(R/'SCIENCE_SUMMARY.json');check(ss['executed_forward_macs']==total_macs and ss['nominal_forward_macs']==total_nominal,'aggregate MACs');check(ss['wall_seconds']<600 and ss['new_training']==0,'no retraining and bounded science')
    summary=dict(status='PASS',checks=CHECKS,evaluations=done,new_training=0,max_numeric_error=MAXERR,wall_seconds=time.perf_counter()-START,scope='All native conditions and retained traces/resources replayed by independent equations; source weights and previous audit locked; original training is not repeated.')
    (R/'AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (R/'AUDIT_FAILURE.json').write_text(json.dumps(dict(reason=str(e),checks=CHECKS,wall_seconds=time.perf_counter()-START),indent=2)+'\n');raise
