import time,os,sys,platform,numpy as np
import model,core,population,base_data,original_tan
from common import memory,basic,structural_cost,utc
from design import *
class Run:
    def __init__(self):
        for n,h in read(R/'FREEZE.json')['files'].items():assert sha(R/n)==h,n
        assert not (R/'RUN_STATE.json').exists();self.tic=time.perf_counter();self.cpu=time.process_time();self.stage='start';self.rows=[];self.steps=0;self.mark('running')
    def mark(self,status,**more):dump(R/'RUN_STATE.json',dict(status=status,stage=self.stage,pid=os.getpid(),utc=utc(),wall_seconds=time.perf_counter()-self.tic,**more))
    def guard(self):
        if time.perf_counter()-self.tic>840:raise TimeoutError('840-second total scientific cap')
        if memory()['current_working_set_bytes']>1024**3:raise MemoryError('1GiB working set')
    def begin(self,stage):self.guard();self.stage=stage;self.out=R/stage;self.out.mkdir();self.rows=[];self.stic=time.perf_counter();self.mark('running')
    def flush(self):dump(self.out/'ROWS.json',self.rows)
    def finish_stage(self):
        self.guard();self.flush();dump(self.out/'SUMMARY.json',dict(rows=len(self.rows),wall_seconds=time.perf_counter()-self.stic,**memory()))
        if sum(p.stat().st_size for p in R.rglob('*') if p.is_file())>1.5*1024**3:raise OSError('1.5GiB artifact cap')
def units(B):
    rng=np.random.default_rng(3964020000);waves={'zero':np.zeros(24),'constant_1':np.ones(24),'constant_4':np.full(24,4.),'step_4':np.r_[np.zeros(8),np.full(8,4.),np.zeros(8)],'pulse_4':np.r_[np.zeros(8),4.,np.zeros(15)],'alternating_4':4*(np.arange(24)%2),'variable':rng.uniform(0,4,24)};rows=[]
    for arm,name in itertools.product(ARMS,waves):
        B.guard();inp=waves[name];m=np.zeros((1,1));H=np.zeros((1,1,5));ref=original_tan.make_variant('TAN-full' if arm=='TAN_FULL' else 'TAN-no-attention');states=[];hist=[];spikes=[];cand=[];surp=[];contexts=[]
        for current in inp:
            m,H,z,a=core.cell(np.array([[current]]),m,H,arm=='TAN_FULL');zr=ref.forward(current);assert z.item()==zr and abs(m.item()-ref.h)<1e-10
            states.append(m.item());hist.append(H[0,0].copy());spikes.append(z.item());cand.append(a['candidate'].item());surp.append(a['surprise'].item());contexts.append(a['context'].item())
        ident=arm+'_'+name;np.savez_compressed(B.out/(ident+'.npz'),input=inp,states=states,histories=hist,spikes=spikes,candidates=cand,surprises=surp,contexts=contexts);rows.append(dict(id=ident,arm=arm,probe=name,events=sum(spikes),late_events=sum(spikes[5:]),max_candidate=max(cand)))
    dump(B.out/'UNIT_ROWS.json',rows)
def calibrate(B):
    B.begin('calibration');units(B);xx=probes();np.savez_compressed(B.out/'inputs.npz',**xx)
    for s,a,g,l in itertools.product(range(3),range(2),range(2),range(2)):
        arm=ARMS[a];G=GS[g];L=LS[l];w=population.initialize(3964000000+s,6)
        for name,x in xx.items():
            B.guard();z=model.evaluate(x,w,arm,G,L);ident=f'c{s}_a{a}_g{g}_l{l}_{name}';np.savez_compressed(B.out/(ident+'.npz'),**z);B.steps+=len(x)*24
            B.rows.append(dict(id=ident,seed=s,arm=arm,G=G,L=L,probe=name,mean_resources=z['resources'].mean(0).tolist(),layer_candidate_max_first8=[float(z['candidates'][:,:,sl].max()) for sl in (slice(0,12),slice(12,18),slice(18,20))],**structural_cost(arm,6,24)))
        B.flush()
    B.finish_stage();dump(B.out/'LOCK.json',dict(utc=utc(),labels_read=False,selection='none; all G/L settings retained',files={p.name:sha(p) for p in sorted(B.out.iterdir()) if p.is_file()}));print('CALIBRATION COMPLETE 144 population conditions and 14 unit probes',flush=True)
def train(B):
    B.begin('train');U=read(R/'DESIGN.json')['updates'];bases={}
    for s in range(3):bases[s]=base_data.make_base(64,3961010000+100*s);np.savez_compressed(B.out/f'base_s{s}.npz',**bases[s])
    for cfg in conditions():
        tic=time.perf_counter();initial=population.initialize(3963010000+cfg['seed'],6);w=initial.copy();m=w*0;v=w*0;gs=[];cs=[];hs=[];b=bases[cfg['seed']];x=assemble(b)
        try:
            for k in range(U):
                B.guard();loss,g,c=model.loss_grad(x,b['y'],w,cfg['arm'],cfg['G'],cfg['L'],cfg['reg']);rn=2*cfg['reg']*w/176;norm=float(np.linalg.norm(g));assert np.isfinite(g).all() and np.isfinite(loss)
                hs.append([loss,norm,float(np.linalg.norm(g-rn)),float(np.linalg.norm(rn))]);g/=max(1,norm);gs.append(g.copy());cs.append(c.astype(np.uint8));B.steps+=64*24
                m=.9*m+.1*g;v=.999*v+.001*g*g;w=np.clip(w-LR*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
        except Exception:
            np.savez_compressed(B.out/(cfg['id']+'_PARTIAL.npz'),initial=initial,theta=w,gradients=gs,counts=cs,history=hs);raise
        np.savez_compressed(B.out/(cfg['id']+'.npz'),initial=initial,theta=w,gradients=gs,counts=cs,history=hs)
        B.rows.append(dict(**cfg,updates=U,seconds=time.perf_counter()-tic,dense_forward_macs=U*64*24*156,dense_backward_macs=U*64*24*240,tan_backward_unit_steps=U*64*24*20,attention_backward_unit_steps=U*64*24*20 if cfg['arm']=='TAN_FULL' else 0,history_gradient_accumulations=U*64*(5*24-10)*20,weight_gain_multiplications=U*156,gradient_gain_multiplications=U*156,cost_per_episode=structural_cost(cfg['arm'],6,24)));B.flush();print('TRAIN',cfg['id'],'complete',flush=True)
    B.finish_stage();dump(B.out/'TRAIN_LOCK.json',dict(utc=utc(),test_generated=False,files={p.name:sha(p) for p in sorted(B.out.iterdir()) if p.is_file()}))
def evaluate(B):
    for name,h in read(R/'train/TRAIN_LOCK.json')['files'].items():assert sha(R/'train'/name)==h
    B.begin('eval');bases={}
    for s in range(3):bases[s]=base_data.make_base(256,3962010000+100*s);np.savez_compressed(B.out/f'test_s{s}.npz',**bases[s])
    for cfg in conditions():
        B.guard();b=bases[cfg['seed']];w=np.load(R/'train'/(cfg['id']+'.npz'))['theta'];z=model.evaluate(assemble(b),w,cfg['arm'],cfg['G'],cfg['L']);np.savez_compressed(B.out/(cfg['id']+'.npz'),**z);B.steps+=256*24
        B.rows.append(dict(**cfg,**basic(z['counts'],b['y']),mean_resources=z['resources'].mean(0).tolist(),weight_gain_multiplications=8*156,**structural_cost(cfg['arm'],6,24)));B.flush()
    B.finish_stage()
def freeze():
    assert not (R/'FREEZE.json').exists() and read(R/'PRECHECK.json')['status']=='PASS' and read(R/'TIMING.json')['status']=='PASS'
    files={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.suffix in ('.py','.md','.json')};dump(R/'FREEZE.json',dict(utc=utc(),scientific_data_generated=False,files=files));print('FROZEN',flush=True)
def science():
    B=Run();dump(R/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,machine=platform.node(),threads={n:os.environ.get(n) for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    try:
        calibrate(B);train(B);evaluate(B);B.guard();dump(R/'SCIENCE_SUMMARY.json',dict(status='COMPLETE',utc=utc(),wall_seconds=time.perf_counter()-B.tic,cpu_seconds=time.process_time()-B.cpu,**memory(),fits=48,updates=48*read(R/'DESIGN.json')['updates'],evaluations=48,calibration_population_conditions=144,isolated_unit_probes=14,population_sample_steps=B.steps,dense_forward_macs=B.steps*156));B.mark('complete')
    except Exception as e:
        B.flush();B.mark('stopped',reason=str(e));dump(R/'SCIENCE_FAILURE.json',dict(stage=B.stage,reason=str(e),wall_seconds=time.perf_counter()-B.tic));raise
if __name__=='__main__':{'freeze':freeze,'science':science}[sys.argv[1]]()
