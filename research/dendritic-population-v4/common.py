from pathlib import Path
import hashlib,json,importlib.util,time
import numpy as np
import model
R=Path(__file__).resolve().parent;OUT=R/'results'
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mask():
    m=np.ones(176);m[72:144]=(np.arange(6)[:,None]==np.arange(12)[None,:]//2).ravel();return m

def stats(x,y,t,arm):
    v=model.metrics(x,y,t,arm)
    return {**{k:v[k] for k in ('accuracy','coverage','abstentions','events_mean')},'abstention_rate':v['abstentions']/len(y),'silent_fraction':float(np.all(v['counts']==0,1).mean()),'both_saturated_fraction':float(np.all(v['counts']==8,1).mean()),'rate_error':model.loss_counts(v['counts'],y,'RATE')[0]}

def parity(stage):
    spec=importlib.util.spec_from_file_location('preserved_v2',R.parent/'dendritic-population-v2/model.py');v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
    x=np.random.default_rng(9).uniform(0,1,(3,24,6));t=model.initialize(19)
    for arm in ('DEND','POINT'):
        for cut in (False,True):
            theta=t*mask() if cut else t;c,cache=model.forward(x,theta,arm);v=v2.forward(x,t,arm,cut_cross=cut,trace=True)
            assert np.array_equal(c,v['counts']) and np.array_equal(np.stack([z['states'] for z in cache],1),v['states'])
    dump(OUT/f'parity_{stage}.json',{'status':'PASS','stage':stage,'intact_and_cross_mask':True,'v2_sha256':sha(R.parent/'dendritic-population-v2/model.py')})

def choice(rows,budget):
    configs=[]
    for loss in ('CE','RATE'):
        for lr in (.01,.03):
            g=[r for r in rows if r['stage']=='A' and r['method']=='ADAM' and r['steps']==budget and r['loss']==loss and r['lr']==lr]
            assert len(g)==10
            configs.append({'loss':loss,'lr':lr,'min_accuracy':min(r['validation']['accuracy'] for r in g),'mean_accuracy':float(np.mean([r['validation']['accuracy'] for r in g])),'mean_rate_error':float(np.mean([r['validation']['rate_error'] for r in g]))})
    best=min(configs,key=lambda z:(-z['min_accuracy'],-z['mean_accuracy'],z['mean_rate_error'],z['lr'],0 if z['loss']=='RATE' else 1))
    return {'budget':budget,'candidates':configs,'selected':best}

def check_freeze():
    f=json.loads((R/'FREEZE.json').read_text())
    for name,h in f['sha256'].items():assert sha(R/name)==h,name
    for name,h in f['dependencies'].items():assert sha(R.parent/name)==h,name
