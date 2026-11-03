# Prespecified nonlinear held-out fixed-domain test

2026-11-02 before execution. Adopt reviewer nonlinear-heldout-protocol.md with explicit seeds: NumPyRandomState20261103 for4096 iid targets uniformD=[-.5,.5]^2; RandomState20261104 draws membrane innovations(4096,100), gateinnovations(4096,100), then sensorerrors(4096,3). No data fitting.

Regenerate endpoint polynomial from exact Fraction recurrence with η=.1, fixedkeys(.8,1.8),slots(0,4), same3coordinate carrier. Verify affine detJ strictlynegative atall4corners and positive f2 =&gt;connectedA-fiber globalinjectivity onD. Verify monotone uppercorner completeencoding branch; actual vector recurrence all4096 vs polynomial, frozen scalar euler first8 andcorners. Prior9pointgrid andboundarysolver checks separatefromheldout.

Exactinverse: recover state using O_m fromsame100stepIprobe.1 q.8 trajectory,m1/100; stablequadratic roots, exactlinear fallback onlyB2==0. Negative discriminant=&gt;NO_REAL_ROOT withoutclamp; filter bothcoordinatesD. Boundarytolerance1e-10 permits onlyroundoffbandsnaps,countseparately; residualtolerance1e-10stateunits. Exactly1validroot=&gt;decode; none/multiple/physicalbranchdeparture=&gt;abstain withfixed(0,0)fallback. No truth-basedrootchoice, broadclip, projection,orparametersearch.

Noise: noiseless,sensor-only1e-6,membrane-process-only1e-6,both1e-6,gate-process-only1e-6. Innovationsafterdrift;sharedphysical100steppathbetweenschedules;ε0shared ε1/100independent. Invalidphysicalbranch retainedindenominator,fallback0.

Primary unconditionalRMSE/MAE eachrole andmeanmaxerror Ecomp;coverage/abstentioncounts;conditionalerrors separately. Localjointsuccessrequiresvaliddecode andbotherrors&lt;.05 (NEWlocalcriterion,notP1B). Mandatoryzero-baseline andunboundedfrozenlinearinversediagnostic. Originaltolerance.5istrivialonD andnotreportedassuccess. Saveperhistorytargets/predictions/statuses forpairedreanalysis. Stopafterfixedtable/repeatchecks,nofits/domainexpansion/installs/publicwrites/spending.
