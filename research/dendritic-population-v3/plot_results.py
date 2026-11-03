from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/'results';s=json.loads((R/'summary.json').read_text());sel=json.loads((R/'SELECTION.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(11.5,4.5))
for offset,a,color in [(-.18,'DEND','#2466a8'),(.18,'POINT','#bd610d')]:
    vals=[np.mean([r['accuracy'] for r in s['rows'] if r['arm']==a and r['mode']==m])*100 for m in ('initial','trained')]
    ax[0].bar(np.arange(2)+offset,vals,width=.34,color=color,label=a)
    for k,v in enumerate(vals):ax[0].text(k+offset,v+2,f'{v:.2f}',ha='center')
ax[0].set(xticks=[0,1],xticklabels=['Initial','Trained'],ylim=(0,115),ylabel='Mean fresh-test accuracy (%)',title='Both circuits learn sustained channel identity')
ax[0].legend();configs=sel['configurations'];vals=[r['mean_rate_error'] for r in configs]
ax[1].bar(range(4),vals,color=['#777777','#999999','#2466a8','#6495bd'])
ax[1].set(xticks=range(4),xticklabels=['CE .01','CE .03','RATE .01','RATE .03'],ylabel='Validation rate-target MSE',title='All 24 candidates: 100% validation accuracy')
for k,v in enumerate(vals):ax[1].text(k,v+.0006,f'{v:.4f}',ha='center',fontsize=10)
ax[1].set_ylim(0,.047);fig.suptitle('V3 learnability sanity check — no architecture advantage established',fontsize=13)
fig.tight_layout();fig.savefig(R/'comparison.png',dpi=160);fig.savefig(R/'comparison.pdf');plt.close(fig)
