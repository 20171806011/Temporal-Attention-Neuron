from pathlib import Path
import json,csv,numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def ci(v):
    a=np.array(v,float);ix=np.random.default_rng(3941900001).integers(0,len(a),(20000,len(a)))
    return dict(values=a.tolist(),mean=float(a.mean()),min=float(a.min()),max=float(a.max()),n=len(a),ci95=np.quantile(a[ix].mean(1),[.025,.975]).tolist())
def main():
    assert read(R/'AUDIT.json')['status']=='PASS';rows=read(R/'evaluate/ROWS.json');groups={};contrasts=[]
    for a in ('DEND','POINT'):
        for h in (24,96):
            for lr in (.01,.0025):
                rr=[r for r in rows if (r['arm'],r['h'],r['lr'])==(a,h,lr)];g=dict(baselines={},cuts={},zeros={});groups[f'{a}_h{h}_lr{lr}']=g
                def vals(D,c,k='balanced_accuracy'):
                    xx=[r for r in rr if r['delay']==D and r['condition']==c];assert [r['seed'] for r in xx]==[0,1,2];return np.array([r[k] for r in xx])
                for D in (16,24,28,36,48,64,72,84,96):g['baselines'][str(D)]={k:ci(vals(D,'baseline',k)) for k in ('balanced_accuracy','coverage','silence','hold_hidden_events')}
                for L in (4,12,24):
                    for recovery in (0,12,48):
                        D=12+L+recovery;name=f'L{L}_R{recovery}';base=vals(D,'baseline');cut=vals(D,'cut_'+name);continued=vals(D,'keepoff_'+name) if recovery else cut
                        g['cuts'][name]=dict(delay=D,duration=L,recovery=recovery,cut_ba=ci(cut),continued_ba=ci(continued),cut_minus_baseline=ci(cut-base),resumption_benefit=ci(cut-continued))
                        if L==24 and recovery==48:contrasts.append(dict(arm=a,h=h,lr=lr,comparison='resumption_benefit_L24_R48',primary=h==96 and lr==.0025,**ci(cut-continued)))
                for D in (24,48,96):
                    g['zeros'][str(D)]={p:dict(ba=ci(vals(D,'zero_'+p)),minus_baseline=ci(vals(D,'zero_'+p)-vals(D,'baseline'))) for p in ('d','vp','all')}
                deficit=lambda recovery:np.array(g['cuts'][f'L24_R{recovery}']['cut_minus_baseline']['values'])
                g['normalized_recovery_L24_R48_minus_R0']=ci(deficit(48)-deficit(0))
    result=dict(groups=groups,restoration_contrasts=contrasts,inference='Three original fitted-model blocks, new stimuli. Two primary contrasts at h96/lr.0025 chosen prospectively for this new mechanism assay after observing the prior fit study. All 24 fitted models retained. Pointwise unadjusted block bootstrap.')
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (R/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader()
        for r in rows:wr.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    print(json.dumps(dict(primary=[r for r in contrasts if r['primary']],all_restoration=contrasts),indent=2))
if __name__=='__main__':main()
