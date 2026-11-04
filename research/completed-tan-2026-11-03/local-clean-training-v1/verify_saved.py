import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;j=json.loads((R/'results.json').read_text());checks=0
for row in j['runs']:
 d=np.load(R/f"predictions_{row['seed']}.npz");mask=d['test_clean'];y=d['test_y']
 c=set(d['clean_only_indices'].tolist());m=set(d['mixed_indices'].tolist());assert len(c&m)==750;checks+=1
 for arm in ['clean_only','mixed']:
  p=d[arm+'_prediction'];cm=[[sum(int(t==a and v==b) for t,v in zip(y[mask],p[mask])) for b in range(3)] for a in range(3)]
  assert cm==row['arms'][arm]['confusion'];score=sum(cm[k][k]/sum(cm[k]) for k in range(3))/3
  assert abs(score-row['arms'][arm]['balanced3'])<1e-15;assert np.array_equal(p,d[arm+'_probability'].argmax(1));checks+=3
  print(row['seed'],arm,'recalls',row['arms'][arm]['recalls'],'predicted_counts',np.array(cm).sum(0).tolist())
assert abs(sum(x['difference'] for x in j['runs'])/3-j['primary_difference'])<1e-15;checks+=1
print('Checks passed',checks)
