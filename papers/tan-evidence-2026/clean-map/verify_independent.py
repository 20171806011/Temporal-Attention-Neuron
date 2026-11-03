"""New stdlib-only check of clean-map evidence. No archived audit code imported.
Exact rational exp bounds use range reduction, 40-term Taylor tail, and squaring.
Checks endpoint signs and branches; full scalar recurrence checks saved fixtures.
This is computational verification, not a proof assistant or external replication.
"""
from fractions import Fraction as Q
from pathlib import Path
import json,math,hashlib,subprocess,platform
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
HALF=Q(1,2);EPS=Q(1,10**9);checks=[]
def check(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def point(x):return Q(x),Q(x)
def add(x,y):return x[0]+y[0],x[1]+y[1]
def sub(x,y):return x[0]-y[1],x[1]-y[0]
def mul(x,y):
 z=[a*b for a in x for b in y];return min(z),max(z)
def div(x,y):
 assert y[0]>0
 return mul(x,(1/y[1],1/y[0]))
def scale(x,s):return mul(x,point(s))
def rounded(x,n=90):
 d=10**n
 return Q((x[0]*d).__floor__(),d),Q((x[1]*d).__ceil__(),d)
def exp_bound(x):
 x=Q(x);assert 0<=x<=16
 y=x/16;t=s=Q(1)
 for k in range(1,41):t=t*y/k;s+=t
 first=t*y/41;tail=first/(1-y/42)
 b=rounded((s,s+tail))
 for _ in range(4):b=rounded(mul(b,b))
 return b
def tanh_bound(x):
 e=exp_bound(2*x)
 # tanh is monotone in exp(2x); endpoints give a tighter direct bound.
 return (e[0]-1)/(e[0]+1),(e[1]-1)/(e[1]+1)
def d5(a,mode):
 e=exp_bound(Q(8,5)*a*a)
 ep=point(EPS) if mode=='historical_task_unshifted' else scale(e,EPS)
 return div(scale(mul(tanh_bound(Q(4,5)*a),e),a),add(add(point(4),e),ep))
def T(a,b,mode):
 d=d5(a,mode)
 if d[1]<=HALF:h=d
 elif d[0]>HALF:h=point(0)
 else:raise AssertionError('Unresolved reset branch')
 s=(4*b-a)/5;assert s>0;q=2*s
 ea,eb=exp_bound(q*a),exp_bound(q*b)
 ep=point(EPS) if mode=='historical_task_unshifted' else scale(exp_bound(q*max(a,b)),EPS)
 c=div(add(scale(ea,a),scale(eb,b)),add(add(add(point(3),ea),eb),ep))
 return add(scale(h,Q(1,16)),mul(tanh_bound(s),c))
def compact(x):
 y=rounded(x,50);return {'lower':str(y[0]),'upper':str(y[1]),'lower_float':float(y[0]),'upper_float':float(y[1])}
def scalar_run(a,b,mode):
 window=[0.]*5;h=0.;rows=[]
 for t in range(1,15):
  x=a if t==5 else b if t==9 else 0.
  window=window[1:]+[x];S=max(0.,x-sum(window)/5);scores=[2*S*z for z in window]
  shift=max(scores) if mode=='native_stabilized' else 0.
  weights=[math.exp(z-shift) for z in scores]
  C=sum(w*z for w,z in zip(weights,window))/(sum(weights)+1e-9)
  A=math.tanh(S)*C;u=h/2+A;y=int(u>.5);h=0. if y else u
  rows.append({'u':u,'h':h,'spike':y,'S':S,'C':C,'A':A})
 return rows

def main():
 data=json.loads((P/'results.json').read_text());spec=json.loads((P/'frozen_spec.json').read_text())
 provenance=json.loads((P/'provenance.json').read_text())
 for f in provenance['selected_files']:
  b=(P/f['path']).read_bytes();check(len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256'],'selected bytes '+f['path'])
 source_checks=[]
 for kind in ['native','simulator','task']:
  path=spec['source'][kind+'_path'];b=subprocess.check_output(['git','show',spec['source']['commit']+':'+path],cwd=ROOT)
  blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
  check(blob==spec['source'][kind+'_blob'],'pinned original Git blob '+kind)
  source_checks.append({'path':path,'git_blob':blob})
 # Supports and complete response factorization were checked against these sources.
 check(Q(11,20)<Q(3,4)<Q(19,20),'reference target exclusive low support')
 check(Q(21,20)<Q(17,16)<Q(29,20),'no-reset target exclusive high support')
 check(Q(21,20)<Q(5,4)<Q(29,20),'reset target exclusive high support')
 check((4*Q(11,20)-Q(29,20))/5==Q(3,20),'strict positive cue surprise on full present support')
 certificates=[]
 for c in data['collisions']:
  mode=c['mode'];a=Q(c['high_a']);t0=T(Q(3,4),Q(1),mode)
  check(t0[0]>0 and t0[1]<HALF,'common query below reset '+mode+c['high_a'])
  dlow=d5(Q(3,4),mode);dhigh=d5(a,mode)
  check(dlow[1]<HALF,'low target no reset '+mode+c['high_a'])
  check((dhigh[1]<HALF) if a==Q(17,16) else (dhigh[0]>HALF),'high target reset branch '+mode+c['high_a'])
  cert={'mode':mode,'high_a':str(a),'common_T':compact(t0),'D5_high':compact(dhigh),'brackets':[]}
  for name,lo,hi in [('outer',Q(9,10),Q(1)),('saved_tight',Q(c['root_certificate']['lower']),Q(c['root_certificate']['upper']))]:
   l=sub(T(a,lo,mode),t0);h=sub(T(a,hi,mode),t0)
   check(Q(11,20)<lo<hi<Q(29,20),'legal cue bracket '+mode+c['high_a']+name)
   check(l[1]<0<h[0],'exact endpoint signs '+mode+c['high_a']+name)
   cert['brackets'].append({'kind':name,'lower':str(lo),'upper':str(hi),'lower_residual':compact(l),'upper_residual':compact(h)})
  certificates.append(cert)
 boundary=[]
 for x in data['reset_boundaries']:
  mode=x['mode'];lo=Q(x['certificate']['lower']);hi=Q(x['certificate']['upper'])
  l=sub(d5(lo,mode),point(HALF));h=sub(d5(hi,mode),point(HALF))
  check(l[1]<0<h[0],'t5 reset boundary signs '+mode)
  boundary.append({'mode':mode,'lower_residual':compact(l),'upper_residual':compact(h)})
 maxerr=0.;steps=0;histories=0
 for block in data['formula_checks']:
  mode=block['mode']
  for trial in block['trajectories']:
   histories+=1;rows=scalar_run(float(Q(trial['a'])),float(Q(trial['b'])),mode)
   for row,saved in zip(rows,trial['rows']):
    steps+=1
    for k in ['u','h','S','C','A']:
     e=abs(row[k]-saved[k]);maxerr=max(maxerr,e);check(e<2e-14,'scalar saved recurrence '+str(steps)+k)
    check(row['spike']==saved['spike'],'scalar spike '+str(steps))
   f=[x['u'] for x in rows[8:13]]+[x['spike'] for x in rows[8:13]]
   t=f[0];post=0. if t>.5 else t
   expected=[t]+[post/2**k for k in range(1,5)]+[int(t>.5),0,0,0,0]
   check(f==expected,'exact floating tail factorization '+str(histories))
 check(histories==64 and steps==896,'complete archived fixed fixture coverage')
 out={'status':'PASS','checks':len(checks),'source_git_blobs':source_checks,'python':platform.python_version(),'exact_collision_certificates':certificates,'reset_boundary_checks':boundary,'fixed_histories':histories,'scalar_updates':steps,'max_saved_scalar_difference':maxerr,'scope':'New implementation checks four exact-real collision brackets in both epsilon conventions, reset branches, and all 64 saved fixed trajectories. No archived primary/checker code imported. Not external human replication or formal proof-assistant verification. Local derivative rectangle and global monotonicity certificates were not independently regenerated. No classifier training, Bayes risk, new noise or finite-precision theorem.'}
 (P/'verification_independent.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['status','checks','fixed_histories','scalar_updates','max_saved_scalar_difference']}))
if __name__=='__main__':main()
