# Prespecified fixed-physics two-sample timing comparison

2026-11-02, before execution; follows reviewer two-sample-timing-protocol.md. Use prior content-A encoder η=.1, same keys(.8,1.8),slots(0,4),9value grid±.5/0 and localδ=(.001,-.001). No training or timing search.

Both schedules apply Iprobe=.1 and q=.8 for100steps. Compare analog samples(u0,u1) vs(u0,u100), same physical path, current/duration and sample count2, different decision latency0.1ms vs10ms. Check actualA0<2 andu0<1 for all9histories; otherwise stop subthreshold claim. Validate analytic O_m and exact current/trajectory match using frozen euler_step. Deadline m100 is analog, not a spike-code bin.

Predeclare4096paired noise paths, seed20261102. Four scenarios: sensor-onlyσs1e-6; membrane-process-onlyσu1e-6; bothσs=σu1e-6; gate-process-onlyσA1e-6. Independent innovations added AFTER each deterministic query drift into updatedu,A; no initial physical uncertainty. Generate each100step path once, sample at1/100. ε0 shared, ε1 and ε100 independent. Reuse standardized draws across scenarios. Log all physical threshold/clipping departures; if any, report them and do not apply linear covariance claims to those paths. Gaussian noise is not bounded; observed branch safety is not universal safety.

Compare empirical state-error covariance against analytic covariance; propagate through fixed prior base-J inverse (external local diagnostic). Record noise-free nonlinearization bias separately. Units differ for sensor, membrane and gate innovations: same numericSD is not matched physical energy. No added σ,m grid after results. Stop once fixed comparison and caveat established. No expanded training.
