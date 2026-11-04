# Prespecified minimal mechanism fixture

Recorded before execution, 2026-11-02. Not a new confirmatory population study.

Run frozen M1 and existing M2b; two active events at slots 0 and 4, keys 0.8 and 1.8, second value 0; first value alternates -0.5/+0.5. Probe q=0.8 only. Use saved full-precision calibration A0/g/b, no refitting. Timing variation is exactly added silence delays {0,5,10} ms, fixed before results, 12 individual executions total. No timing/parameter search.

Instrument euler_step returns without modifying recurrence. Save complete step traces and boundary states. Verify zero spikes throughout; check pairwise exact byte identity for M1 and analytic M2b delta-u = A0*(1-0.99^100)*0.99^(steps_since_first_offset), delta-v=1; readout delta prediction g*delta-u. Analytical comparison absolute tolerance 1e-15, report residuals separately. M1 exact collision is expected because first event novelty=0 and initial A=0. M2b nonzero delta means local information survives on this pair, not robust retrieval.

Do not run production runner, calibration, training or full benchmark; read runner only for protocol. No external writes, installs or spending.
