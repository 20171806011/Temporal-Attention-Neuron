"""Render saved test results; no simulation or fitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
D=json.loads((P/'results/results.json').read_text())
regimes=['synchronous_clean','jitter4_noise03','jitter8_noise10']
labels=['Synchronous\nclean','Jitter 0–4\nnoise SD .03','Jitter 0–8\nnoise SD .10']
fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
x=np.arange(3);width=.25
for j,(arm,title,color) in enumerate([('DEND','Dendritic: 6 cells','#2166ac'),('POINT_NETWORK','Point circuit: 12 cells','#b35806'),('POOLED','Pooled ablation','#888888')]):
 vals=[D['splits']['test'][r][arm]['accuracy']*100 for r in regimes]
 ax[0].bar(x+(j-1)*width,vals,width,label=title,color=color)
 for xx,v in zip(x+(j-1)*width,vals):ax[0].text(xx,v+1,f'{v:.1f}',ha='center',fontsize=8)
ax[0].set(xticks=x,xticklabels=labels,ylabel='Unconditional accuracy (%)',ylim=(0,114),title='600 paired test episodes per regime')
ax[0].legend(fontsize=8,loc='lower left')
diffs=[D['test_comparisons'][r]['POINT_NETWORK']['difference']*100 for r in regimes]
ci=np.array([D['test_comparisons'][r]['POINT_NETWORK']['percentile95'] for r in regimes])*100
ax[1].errorbar(x,diffs,yerr=np.stack([np.array(diffs)-ci[:,0],ci[:,1]-diffs]),fmt='o',capsize=5,color='#2166ac')
ax[1].axhline(0,color='black',lw=.8)
ax[1].set(xticks=x,xticklabels=labels,ylabel='Dendritic − point circuit (percentage points)',title='Paired stratified bootstrap, 95% intervals')
fig.suptitle('Local segregation helps; this dendritic implementation does not beat the point circuit',fontsize=11)
fig.savefig(P/'results/comparison.png',dpi=180);fig.savefig(P/'results/comparison.pdf')
