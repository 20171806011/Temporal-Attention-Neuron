from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parent/'results';s=json.loads((R/'summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
colors={'DEND':'#2466a8','POINT':'#bd610d'}
for a,shift in [('DEND',-.17),('POINT',.17)]:
    vals=[next(r['accuracy']*100 for r in s['rows'] if (r['seed'],r['arm'],r['mode'])==(k,a,'intact')) for k in range(3)]
    axes[0].bar(np.arange(3)+shift,vals,width=.32,label=a,color=colors[a])
    for k,v in enumerate(vals):axes[0].text(k+shift,v+1,f'{v:.1f}',ha='center',fontsize=10)
axes[0].axhline(50,color='gray',linestyle=':',label='Balanced forced-guess reference')
axes[0].set(xticks=range(3),xticklabels=['Seed 0','Seed 1','Seed 2'],ylim=(0,65),ylabel='Unconditional accuracy (%)',title='512 fresh test histories per seed; ties incorrect')
axes[0].legend(fontsize=9)
entries=[('DEND - POINT',s['primary']),('DEND intact - cut',s['interaction']['DEND']),('POINT intact - cut',s['interaction']['POINT'])]
for k,(label,r) in enumerate(entries):
    m=r['difference']*100;lo,hi=np.array(r['percentile95'])*100
    axes[1].errorbar(m,k,xerr=[[m-lo],[hi-m]],fmt='o',color='#2466a8',capsize=5)
axes[1].axvline(0,color='black',linewidth=1);axes[1].set(yticks=range(3),yticklabels=[x[0] for x in entries],xlabel='Accuracy difference (percentage points)',title='Exploratory paired hierarchical 95% intervals')
axes[1].invert_yaxis();fig.suptitle('V2: training/readout failure; no demonstrated dendritic benefit',fontsize=14)
fig.tight_layout();fig.savefig(R/'comparison.png',dpi=160);fig.savefig(R/'comparison.pdf');plt.close(fig)
