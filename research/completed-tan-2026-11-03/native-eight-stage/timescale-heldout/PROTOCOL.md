# Final bounded three-arm timescale test

2026-11-02 before execution. Adopt reviewer timescale-retention-protocol.md. Exactly baseline(τm10,τA15), gate_only(10,150), uniform(100,150) ms. Both leak AND forcing h/τ change. All other contentη.1, keys(.8,1.8),slots(0,4),h.1,encoding100ms,queryq.8/Iprobe.1/10ms,m100twoanalog samples fixed. Start states0, no fitting.

4096 fresh iid uniform[-.5,.5]^2 targets RandomState20261105; save/hash before metrics. Noise20261106 draws membrane(4096,100),gate(4096,100),sensor(4096,3); reuse identical disturbances across arms. Same5noise cases: noiseless,sensor1e-6,membrane1e-6,both,gate1e-6. After-drift state innovations, no initial state uncertainty; equal raw per-step disturbances across τ are not equal physical current noise.

Regenerate exact Fraction polynomial per arm; detcorner constantnegative + connectedA-fibers =>globalinjectivity onbox. Monotone uppercorner bound verifies wholebox branch. Vector actualrecurrence all4096 vs polynomial; scalar frozen euler first8/corners with explicitτ; query analytical O100 verification. Solver rules identical prior: stablequadraticroots, filter bothcoordsbox, boundaryroundoff1e-10 separatelycounted,residual1e-10, no real/valid/multiple roots orphysicalbranchdeparture=>abstain,fallback0. No truth-based selection or broadclipping.

Report all15arm/noise cells unconditionalRMSE/MAE/Ecomp,coverage,statuscounts,conditionalerrors,newlocalδ.05validjointsuccess,zero baseline. Primary sensor-only Ecomp comparisons: eacharm-vs-zero,gate-vs-baseline,uniform-vs-baseline, fixed2000 pairedhistorypercentilebootstraps seed20261107 (no significance stopping). Save perhistoryledgers.

Resources: perhistory∑A h,∑|Isyn|h,∑u²h,peaks,endpointranges forencoding/query; dimensionless proxies notATP. Same3coordinates but changedintrinsicretention/charging/current costs. No reset/carryover simulation. Stopafterthisonecomparison and consolidate prior series, no further τ/domain/noisesweep, no installs/publicwrites/spending.
