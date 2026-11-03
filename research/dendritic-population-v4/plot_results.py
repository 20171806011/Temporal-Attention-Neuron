from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'results';dev=json.loads((P/'development_log.json').read_text());test=json.loads((P/'TEST_SUMMARY.json').read_text())
fig,ax=plt.subplots(1,3,figsize=(16,4.8));colors={'DEND':'#2466a8','POINT':'#bd610d'}
for arm,shift in [('DEND',-.14),('POINT',.14)]:
    for level in range(4):
        v=np.array([r['validation']['accuracy']*100 for r in dev if r['stage']=='B' and r['level']==level and r['arm']==arm]);ax[0].scatter(np.full(len(v),level+shift),v,color=colors[arm],alpha=.8,s=28,label=arm if level==0 else None)
    rows=sorted([r for r in test['rows'] if r['level']==3 and r['arm']==arm],key=lambda r:r['seed'])
    ax[1].plot(range(5),[r['accuracy']*100 for r in rows],'o-',color=colors[arm],label=arm)
    vals=[np.mean([r['accuracy']*100 for r in test['rows'] if r['level']==2 and r['arm']==arm and r['condition']==c]) for c in ['intact','retrained','cut']]
    ax[2].bar(np.arange(3)+shift,vals,width=.27,color=colors[arm],label=arm)
    for k,v in enumerate(vals):ax[2].text(k+shift,v+1,f'{v:.1f}',ha='center',fontsize=9)
ax[0].set(xticks=range(4),xticklabels=['L0','L1','L2','L3'],ylabel='Unconditional accuracy (%)',title='Development ladder: five seeds')
ax[1].set(xticks=range(5),xlabel='Fresh initialization/data seed',title='L3 fresh test: 512 histories per seed')
ax[2].set(xticks=range(3),xticklabels=['Intact','Retrained\nno cross links','Inference\ncut'],title='L2 fresh test: recurrence controls')
for a in ax:a.set_ylim(0,110);a.axhline(90,color='gray',ls=':',linewidth=1);a.legend(fontsize=9,loc='lower left')
d=test['differences']['3'];fig.suptitle(f"V4 fixed RATE/.01, 200 updates: L3 paired difference {100*d['difference']:.2f} pp, exploratory 95% CI [{100*d['percentile95'][0]:.2f}, {100*d['percentile95'][1]:.2f}]",fontsize=12)
fig.tight_layout();fig.savefig(P/'comparison.png',dpi=160);fig.savefig(P/'comparison.pdf');plt.close(fig)
