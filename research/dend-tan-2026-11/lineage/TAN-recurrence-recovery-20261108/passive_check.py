"""Post hoc algebraic check on saved traces only; no new network evaluation.

With zero external input and recurrent drive disabled, native DEND obeys
d_end = .95**L d_start + .25*db*(1-.95**L)/(1-.95).
This is an equation identity, not an additional predictive endpoint.
"""
from pathlib import Path
import json,time
import numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    tic=time.perf_counter();source=read(R/'SOURCE_LOCK.json');old=(R/source['directory']).resolve();rows=read(R/'evaluate/ROWS.json');out=[];weights={}
    for r in rows:
        if r['arm']!='DEND' or 'cut' not in r:continue
        h=r['h'];ident=r['id']
        if ident not in weights:weights[ident]=np.load(old/source['weights'][ident]['path'])['theta']
        w=weights[ident];db=w[12*h+2*h*h:14*h+2*h*h];start,end=r['cut'];L=end-start
        trace=np.load(R/'evaluate'/f'{ident}_d{r["delay"]}_{r["condition"]}.npz')['states'];d0=trace[:,start-1,:2*h];d1=trace[:,end-1,:2*h];factor=.95**L
        expected=factor*d0+.25*db*(1-factor)/(1-.95);err=float(np.max(abs(d1-expected)));assert err<1e-8
        centered0=d0-d0.mean(0);centered1=d1-d1.mean(0);center_err=float(np.max(abs(centered1-factor*centered0)));assert center_err<1e-8
        norm=float(np.linalg.norm(centered0));ratio=float(np.linalg.norm(centered1)/norm) if norm>1e-10 else None
        out.append(dict(id=ident,h=h,lr=r['lr'],seed=r['seed'],delay=r['delay'],condition=r['condition'],cut_steps=L,expected_contraction=factor,observed_centered_contraction=ratio,max_affine_error=err,max_centered_error=center_err,retained_examples=8))
    assert len(out)==180
    record=dict(status='PASS',post_hoc=True,new_scientific_network_runs=0,scope='Algebraic identity on the already saved eight-example DEND traces. Does not establish task decodability or a new independent replicate.',conditions=len(out),wall_seconds=time.perf_counter()-tic,max_numeric_error=max(max(r['max_affine_error'],r['max_centered_error']) for r in out),contraction_24=.95**24,contraction_72=.95**72,rows=out)
    (R/'PASSIVE_CHECK.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in record.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
