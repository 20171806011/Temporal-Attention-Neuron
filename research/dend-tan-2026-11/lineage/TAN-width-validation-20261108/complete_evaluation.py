"""Append only missing predeclared evaluations; preserve all existing raw files."""
from pathlib import Path
import time,sys,json
import numpy as np
import model,engine,base_data
from common import R,RET,PARTS,CONDITIONS,sha,read,dump,memory,assemble,basic,utc
C=R/'completion'
EVENTS=('branch_events','hidden_events','output_events','hold_branch_events','hold_hidden_events','hold_output_events','recurrent_deliveries','output_deliveries','branch_deliveries','internal_deliveries')
def recover_state(x,w,arm,h,tick):
    W,B,bd,bv,O,bo=model.unpack(w,h);parts=[[],[],[],[]]
    for first in range(0,len(x),64):
        xx=x[first:first+64];n=len(xx);d=np.zeros((n,2*h));v=np.zeros((n,h));o=np.zeros((n,2));p=np.zeros((n,h))
        for t in range(tick):
            dc=RET[0]*d+.25*(xx[:,t]@W+p@B+bd)
            if arm=='DEND':local=np.maximum(dc,0);d=dc
            else:bs=(dc>.5).astype(float);local=bs;d=dc*(1-bs)
            vc=RET[1]*v+.5*local.reshape(n,h,2).sum(2)+bv;hs=(vc>.5).astype(float);v=vc*(1-hs)
            oc=RET[1]*o+p@O+bo;os=(oc>.5).astype(float);o=oc*(1-os);p=hs
        for j,a in enumerate((d,v,o,p)):parts[j].append(a)
    return [np.concatenate(a) for a in parts]
def static_cost(h,arm,T):
    return dict(parameters=2*h*h+17*h+2,continuous_slots=3*h+2,persistent_continuous=3*h+2,previous_spike_registers=h,output_counters=2,
      spiking_cells=h+2 if arm=='DEND' else 3*h+2,graded_branches=2*h if arm=='DEND' else 0,state_updates=(3*h+2)*T,forward_macs=(2*h*h+14*h)*T,graded_local_transfers=2*h*T if arm=='DEND' else 0)
def fixture():
    b=base_data.make_base(80,7993700);x,q=assemble(b,24,'clean');w=model.initialize(7993701,48);_,_,a=engine.evaluate(x,w,'POINT',48,q,28);b=recover_state(x,w,'POINT',48,28)
    assert all(np.array_equal(u,v) for u,v in zip(a,b))
    dump(R/'COMPLETION_PRECHECK.json',dict(status='PASS',checks=4,bit_exact_boundary=True,scientific_data_generated=False))
def main():
    assert not (C/'SUMMARY.json').exists() and not (C/'FAILURE.json').exists()
    frozen=read(R/'COMPLETION_FREEZE.json')
    for p,h in frozen['files'].items():assert sha(R/p)==h,'completion freeze '+p
    for p,meta in frozen['original_raw_files'].items():assert sha(R/p)==meta['sha256'],'original raw changed '+p
    for p,h in read(R/'FREEZE.json')['files'].items():assert sha(R/p)==h,'original source freeze'
    tic=time.perf_counter();cpu=time.process_time();newcount=0;newmacs=0;recoverymacs=0;recoveryseconds=0
    rows=read(C/'original_json'/'ROWS.json');trs=read(C/'original_json'/'TRANSFERS.json');over=read(C/'original_json'/'OVERHEAD.json');b=dict(np.load(R/'evaluate'/'test_s4.npz'));mapping=dict(np.load(R/'evaluate'/'donors_s4.npz'))
    def guard():
        assert time.perf_counter()-tic<90,'90 second completion budget'
        assert memory()['current_working_set_bytes']<1024**3,'memory budget'
    def save(path,arrays):
        nonlocal newcount
        assert not path.exists(),'refusing to overwrite existing result';np.savez_compressed(path,**arrays);newcount+=1
    for tr in [r for r in read(R/'train'/'ROWS.json') if r['id'] in ('s4_POINT_h48','s4_DEND_h96','s4_POINT_h96')]:
        ident=tr['id'];h=tr['h'];arm=tr['arm'];w=np.load(R/'train'/f'{ident}.npz')['theta'];meta={k:tr[k] for k in ('id','seed','h','arm','loss','ret')};baselines={};construction=0.;suffixsec=0.
        for D,kind in CONDITIONS:
            guard();x,q=assemble(b,D,kind);tick=16+D//2 if kind=='clean' and D in (24,96) else None;path=R/'evaluate'/f'{ident}_d{D}_{kind}_baseline.npz'
            if path.exists():
                assert ident=='s4_POINT_h48' and kind=='clean' and D in (0,24,48),'unplanned preexisting condition'
                saved=dict(np.load(path));cost={**static_cost(h,arm,x.shape[1]),**{key:None for key in EVENTS}};state=None;origin='original_unflushed'
                if tick:
                    start=time.perf_counter();state=recover_state(x,w,arm,h,tick);recoveryseconds+=time.perf_counter()-start;recoverymacs+=len(x)*tick*(2*h*h+14*h)
            else:
                saved,cost,state=engine.evaluate(x,w,arm,h,q,tick);save(path,saved);newmacs+=len(x)*x.shape[1]*(2*h*h+14*h);origin='supplemental_new'
            metric,_=basic(saved['counts'],b['y']);rows.append(dict(**meta,delay=D,kind=kind,condition='baseline',origin=origin,**metric,**cost))
            if tick:baselines[D]=(saved['counts'],state)
        for D in (24,96):
            x,q=assemble(b,D,'clean');tick=16+D//2;baseline,state=baselines[D];counts={}
            for kind,part in [('identity','all')]+[(k,p) for k in ('opposite','same') for p in PARTS]:
                guard();start=time.perf_counter();idx=np.arange(len(x)) if kind=='identity' else mapping[kind];mixed=[a[idx].copy() if j in PARTS[part] else a.copy() for j,a in enumerate(state)];construction+=time.perf_counter()-start;start=time.perf_counter()
                saved,cost=engine.suffix(x,w,arm,h,q,tick,mixed);suffixsec+=time.perf_counter()-start;newmacs+=len(x)*(x.shape[1]-tick)*(2*h*h+14*h);name=kind+'_'+part
                if part=='all':assert np.array_equal(saved['counts'],baseline[idx]),'full donor output'
                save(R/'evaluate'/f'{ident}_d{D}_clean_{name}.npz',saved);counts[name]=saved['counts'];metric,_=basic(saved['counts'],b['y'])
                rows.append(dict(**meta,delay=D,kind='clean',condition=name,origin='supplemental_new',**metric,**cost,suffix_steps=x.shape[1]-tick,suffix_forward_macs=(x.shape[1]-tick)*(2*h*h+14*h),suffix_state_updates=(x.shape[1]-tick)*(3*h+2),suffix_graded_local_transfers=(x.shape[1]-tick)*2*h if arm=='DEND' else 0))
            for part in PARTS:trs.append(dict(**meta,delay=D,part=part,origin='supplemental_new',**engine.transfer(counts['opposite_'+part],counts['same_'+part],baseline,b['y'],mapping)))
        over.append(dict(id=ident,transplant_construction_seconds=construction,suffix_inference_seconds=suffixsec,origin='supplemental_new'))
        for name,obj in [('ROWS',rows),('TRANSFERS',trs),('OVERHEAD',over)]:dump(C/(name+'.json'),obj)
        print('supplement',ident,'complete',flush=True)
    guard();assert newcount==81 and len(rows)==840 and len(trs)==300
    assert rows[:756]==read(C/'original_json'/'ROWS.json') and trs[:270]==read(C/'original_json'/'TRANSFERS.json')
    # Canonical summaries are extended only after snapshots and prefix equality checks.
    for name in ('ROWS','TRANSFERS','OVERHEAD'):dump(R/'evaluate'/(name+'.json'),read(C/(name+'.json')))
    dump(C/'SUMMARY.json',dict(status='COMPLETE_WITH_DOCUMENTED_BUDGET_DEVIATION',utc=utc(),wall_seconds=time.perf_counter()-tic,cpu_seconds=time.process_time()-cpu,**memory(),new_evaluations=81,existing_unflushed_conditions=3,original_complete_conditions=756,forward_macs_new_conditions=newmacs,state_recovery_macs=recoverymacs,state_recovery_seconds=recoveryseconds))
    dump(R/'COMPLETION_STATE.json',dict(status='complete',utc=utc(),original_evaluation_failed=True,raw_files_overwritten=0,new_evaluations=81))
    print(read(C/'SUMMARY.json'))
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='fixture':fixture()
    else:
        try:main()
        except Exception as e:dump(C/'FAILURE.json',dict(utc=utc(),reason=str(e)));raise
