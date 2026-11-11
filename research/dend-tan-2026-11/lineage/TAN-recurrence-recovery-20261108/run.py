import sys,time,platform,os,numpy as np
import base_data,engine
from common import *
def freeze():
    assert not (R/'FREEZE.json').exists();assert read(R/'PRECHECK.json')['status']=='PASS'
    dump(R/'FREEZE.json',dict(utc=utc(),new_scientific_data_generated=False,previous_results_seen=True,files={p.name:sha(p) for p in sorted(R.iterdir()) if p.suffix in ('.py','.md','.json')}))
    dump(R/'ENVIRONMENT.json',dict(machine=platform.node(),python=sys.version,numpy=np.__version__,executable=sys.executable,threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}));print('FROZEN',sha(R/'FREEZE.json'),flush=True)
def science():
    B=Budget()
    try:
        source=read(R/'SOURCE_LOCK.json');old=(R/source['directory']).resolve();assert sha(old/'MANIFEST.json')==source['manifest_sha256'];assert sha(old/'AUDIT.json')==source['audit_sha256']
        for s in range(3):
            b=base_data.make_base(256,3941020000+100*s);np.savez_compressed(B.out/f'test_s{s}.npz',**b);inputs={D:assemble(b,D,'clean') for D in (16,24,28,36,48,64,72,84,96)}
            for tr in [r for r in source['cohort'] if r['seed']==s]:
                ident=tr['id'];path=old/source['weights'][ident]['path'];assert sha(path)==source['weights'][ident]['sha256'];w=np.load(path)['theta'];h=tr['h'];arm=tr['arm'];meta={k:tr[k] for k in ('id','seed','h','arm','lr','loss','ret')};baseline={}
                for case in cases():
                    B.guard();D=case['delay'];x,q=inputs[D];saved,cost=engine.evaluate(x,w,arm,h,q,case);metric,_=basic(saved['counts'],b['y'])
                    np.savez_compressed(B.out/f'{ident}_d{D}_{case["condition"]}.npz',**saved)
                    if case['condition']=='baseline':baseline[D]=saved
                    if case['condition']=='identity':
                        for k in ('counts','states','spikes','resources'):assert np.array_equal(saved[k],baseline[D][k]),'identity sham '+k
                    B.rows.append(dict(**meta,**case,**metric,**cost));B.macs+=len(x)*cost['executed_forward_macs'];B.nominal_macs+=len(x)*cost['forward_macs'];B.sample_steps+=len(x)*x.shape[1]
                B.flush();B.guard(disk=True);print('evaluate',ident,'36 conditions complete',flush=True)
        B.finish()
    except Exception as e:B.fail(e);raise
if __name__=='__main__':{'freeze':freeze,'science':science}[sys.argv[1]]()
