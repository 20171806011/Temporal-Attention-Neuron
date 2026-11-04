#!/usr/bin/env python3
"""Post-execution descriptive paired intervals; not preregistered tests."""
import csv,json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
r=json.loads((p/'results.json').read_text());intervals=[]
for c in r['cells']:
 rows=list(csv.DictReader((p/f"{c['scenario']}_m{c['m']}_predictions.csv").open()))
 target=np.array([[float(x['v1']),float(x['v2'])] for x in rows])
 pred=np.array([[float(x['pred1']),float(x['pred2'])] for x in rows])
 d=np.max(abs(pred-target),axis=1)-np.max(abs(target),axis=1)
 se=d.std(ddof=1)/np.sqrt(len(d))
 intervals.append({'scenario':c['scenario'],'m':c['m'],'mean_paired_ecomp_difference_vs_zero':float(d.mean()),'approx_normal_95_ci':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)]})
(p/'paired_uncertainty.json').write_text(json.dumps({'note':'Post-execution descriptive uncertainty, not preregistered testing; all cells included; normal approximation over iid histories, no multiple-comparison adjustment.','comparisons':intervals},indent=2)+'\n')
