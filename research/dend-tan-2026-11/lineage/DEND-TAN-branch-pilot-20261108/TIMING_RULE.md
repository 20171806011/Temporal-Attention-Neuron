> Historical archival document. Original language preserved. Personal paths/host metadata and unavailable media links were sanitized; consult README.md and ARCHIVE_INDEX.json.

# Before scientific data

Fixed24fits:4arms(DEND,FULL,NOATT,GAIN2) x2learning rates(.01,.0025) x3 seeds;200updates each,64D0+64D24. No configuration selection. Fixture8updates/arm,32total,seed79981001/2(data),79982001(initial). Budget check:2*200*6*sum(per-arm max last4 step times)+100seconds evaluation allowance must be<840. Otherwise stop before science, do not choose from observed performance. Scientific training/evaluation share840s, audit300s. No scientific seeds or results used in feasibility check.
