# One bounded timescale intervention: retention versus generic slowing

Proposed2026-11-02 before new model runs. Highest-value question: is the fragile fixed-two-value decoder mainly a consequence of matching an80ms-old event to10/15ms decay constants, or does selective slowing fail even after verified rank2 and observability? This is a mechanistic test of the modified single-unit carrier, not yet a general-addressing benchmark. Perform this one comparison; do not launch an optimization sweep if it fails.

## 1. Frozen arms and rationale

| Arm | tau_m ms | tau_A ms | Rationale |
|---|---:|---:|---|
|Baseline|10|15|Existing content-A model|
|Gate-only slow|10|150|One decade longer retained content trace, spanning the100ms history|
|Uniform slow|100|150|One decade slowing of both coordinates; separates selective retention from generic slowing|

These factors are selected from the fixed100ms task horizon, not fitted to held-out outcomes. No intermediate tau values, gain retuning, added neurons, variables, slots or sample counts. η=.1, keys(.8,1.8), positions(0,4), h=.1ms,100ms encoding, queryq=.8 andIprobe=.1 for10ms unchanged. Use **only the established deadline two-sample schedulem100**, not another readout-time search. Earlierm1 was a diagnostic; the deadline schedule has already been fixed for the next question.

Changing tau changes both leak and normalized forcing: a=1−h/tau_m, bA=1−h/tau_A, membrane input coefficienth/tau_m and A drive coefficienth/tau_A. Do NOT change only the decay factor while silently retaining the old input coefficient. The equilibrium gains remain fixed under this definition, but finite-window charging amplitude and currents change. Allarmsstartu=A=r=0, notmatchedlearnedendpointstates. This is a jointretention/chargingintervention, not an exclusivecausaltestofdecay. A futurearmholdinginstantaneousdrivegainfixed whilealteringleak couldseparate thoseeffects, but is explicitly NOT addednow.

## 2. Analytic prechecks and predictions

Recompute the same affine-A/quadratic-u map for each arm using polynomial propagation, not fitted samples. PriorD=[−.5,.5]², fixed known keys/timing. Check maximumcorner(.5,.5) complete trajectory, all gate/current positivity and threshold/clipping margins. Thiscornerbounds thewholebox, notjustsampledcorners: a,bAarepositive; eachactivedriveandcontentcurrentisnonnegativeandcoordinatewisemonotone (firstcurrent.3..1.3,second1.3..2.3); startingstateszero, inductionmakesAandtheunthresholdedu coordinatewisemonotone at everystep. Therefore maxcorner'sstrictsubthresholdmargin proves allinteriorhistoriesfollowthatbranch. Gateclippinginactive follows itsmaxcornerbound aswell. Compute all4corners of affine detJ. Constant sign plus connected affineA fibers proves globalinjectivity onD as before. Do not extrapolate polynomial outside this branch/domain.

Independent equation arithmetic predicts:

| Arm | smallest singular valueJ | smallest singular valueO100J | conditionO100J | max encodingu onD | max encodingA onD |
|---|---:|---:|---:|---:|---:|
|Baseline|5.89060e−5|2.59528e−6|26042.6|.3384253|.3666053|
|Gate-only slow|.00216623|.000145021|78.1985|.0609920|.0616294|
|Uniform slow|.000739401|.0000107399|464.1902|.00902905|.0616294|

All4detJ corners are negative: baseline approximately[−4.11045e−6,−3.88598e−6]; gate-only[−2.89148e−5,−2.49947e−5]; uniform[−6.00061e−6,−5.62869e−6]. These are analytical recurrence predictions, **not simulation results**. Regenerate full precision independently and confirm actual implementation. No constants may be changed in response to these predicted numbers or later empirical outcomes.

A retention ratiof1/f2 rises fromabout.0047425 toabout.58654 because the two events are80ms apart. This supports the mechanistic hypothesis; absolute amplitudes fall because slowerA charges less during10ms events. Uniform slowing also reduces query transduction becauseh/tau_m is tenfold smaller. Thus improved retention, baseline amplitude and observation coupling must be reported separately, not summarized as a free increase in capacity.

For each arm O100 has rows[1,0] and[a^100,d100], withd100=(h/tau_m)Iprobe*(bA^100−a^100)/(bA−a). d100 values:.0438720 baseline,.0610129 gate-only,.00920821 uniform. Knownquery offsetq*(1−a^100). Check noiseless predictions against actual recurrence.

## 3. Held-out execution contract

Reuse exact nonlinear solver specification: eacharm's own frozen coefficients, realroot count, Dfilter, numericalboundary tolerance1e−10, state residual tolerance1e−10; no posthoc clipping/projection; no validroot=&gt;abstain with fixedzero fallback. Originaldelta.5 remains invalid as a successmetric onthisprior. Use localdelta.05, coverage, unconditional errors and identical zero predictor.

New held-out seeds proposed20261105 targets and20261106 noise,4096 independent uniformD pairs. Publish/hash targets before computing model metrics. No reuse of this set for subsequent parameter choice. Keep prior5noise cases unchanged: noiseless, sensor-only1e−6, membrane-stateprocess-only1e−6, both, gate-stateprocess-only1e−6. Same100-step processnoise innovations across arms and sharedsensorerrors acrossarms for paired comparisons; this is matching normalized-state disturbances perEulerupdate, **not** a claim of identical physical current/noise energy across differenttaus. Noise injection location stays as priorprotocol. A physical input-current-noise model would require scaling stateinnovations byh/tau and is a differentfuturequestion, not silently mixed here.

Actualbranchdepartures/clipping must be logged and receive preregisteredabstention accounting, not dropped. Verify polynomial against all4096 deterministic encodingendpoints and scalarrecurrence onfirst8+corners. Record solverfailure categories, validcoverage, bothroleRMSE/MAE, unconditionalEcomp, localjointsuccess and zero-baseline metrics. Primarycomparison is unconditionalEcomp versuszero and baseline in sensor-onlycase; roleRMSE andcoverage expose tradeoffs. Remainingnoise cases diagnose robustness, not newparameterselection.

Pairedbootstrap confidence intervals over4096histories may quantify fixed-arm riskdifferences. No significance-driven stopping, armaddition or noisetuning. No training.

## 4. Resource and mechanism accounting

Statecoordinatecount remains3 and physicalevent/readoutduration/samplecount stays fixed, but longerintrinsicmemory is a changed biological/dynamical resource. Report tauvalues, endpointstate ranges, maxgate, voltage andsynapticcurrent, sum(A*h), sum(abs(Isyn)*h), and sum(u²*h) overencoding/query as dimensionless/proxy integrals. Do not call these ATP or measuredphysicalenergy. Report actualcurrent/time traces or summaries; same externalprobe amplitude/duration does not make internalgatedcurrent identical.

The fixed externaldeadline does not increase, but intrinsic relaxationtime beyond thetrial does. If circuitreset/carryover is not simulated, explicitlyexclude itsenergy/throughputcost. No claim that samecoordinatecount means unchangedphysicalresourcecost.

## 5. What to conclude and when to stop

If gate-onlyslowing yields robust gains with exactheldoutinverse, conclude that the existingfailedparameterization's timescalemismatch was a remediable bottleneck for thisfixedtwo-value task. Do not infer learned/generalcontentaddressing; decoder still knowskeys/timing and exactequations. If neitherarmbeatsnullrobustly, close thisconditionalcarrier/readoutseries as an informativeboundary rather than tuning moreknobs. If uniformslowing differs fromselectiveA, interpret with the separatelyreportedencoding andobservationconditioning, not a broadbestarchitectureclaim.

The next genuineresearchboundary after thisonecomparison would be unknown/randomkeys+timing orN3 under declaredresourcebudget—not furtherretuning onthecurrenttwofixedoperands. Advancing there needs a newpreregisteredtask, not rebranding the presentconditionaldecoder.
