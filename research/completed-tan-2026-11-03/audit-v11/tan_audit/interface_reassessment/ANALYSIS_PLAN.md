# Registered-interface audit and one contract-preserving diagnostic

Frozen2026-11-02 before executing the new interface fixtures or novelty-stratified outcomes. Prior whole-cell M1 outcomes are already known. This is a prospective plan for a retrospective diagnostic replay, not a new confirmatory registration or replacement of original P1B labels.

## Authority and scope

Use unchanged P1B ProtocolV2.6, models_p1b_v2.py, runner_p1b.py and task_generator.py at frozen commit85213f0d417d54e4f48b301ed1c0e4200ef3c421. Manuscript paraphrases do not override these equations. Distinguish generator/runner/raw-array wrapper access from the registered state-transition boundary. The latter receives content=k+v, novelty=max(0,abs(k)-1.4), querycurrent and fixed parameters, while masks/timing drive presentation and interventions. The upstream Python wrapper retains raw keys/values; M0 explicitly uses that cache. No claim that the entire software API or original window-based TAN loses those inputs.

All original weights1, timeconstants, threshold/reset/clipping, timing, initial conditions, calibration and readout stay fixed. No new states, side channels, fitted decoder, tuning or narrowed training population. No interpolation/enriched write arm.

## Exact boundary fixtures

Six histories form three pairs, all admissible under the original generator support and min spacing. Query order is held equal within each pair; queries reuse stored keys as specified by the task.

1. N2 low-flat pair atslots(0,4): keys/values[(.8,1),(1,-1)] versus[(1,.8),(.8,-.8)], commonqueries(.8,1), targets(1,-1) versus(-.8,.8). Same complete current/mask/timing/query path, not merely the same content projection. Bothnoveltieszero. TargetLinf gap1.8 gives minimax/pairedmean Ecomp lowerbound.9 for any common outputs; no prevalenceclaim.
2. N3 primed-low pair atslots(0,2,4): prependcommon(2,0) to the two low assignments, samequeries(.8,1). Samefullpath again, but positiveprimer creates strictly positive A duringbothambiguousevents in nativeM1. This tests projectionambiguity separatelyfrom known first-eventA=0erasure. Initialconditionsremainregisteredzero, notmanuallyprecharged.
3. N2 both-positive projection control atslots(0,4): [(1.6,1),(2,-1)] versus[(2,.6),(1.6,-.6)], commonqueries(1.6,2). Contentwaveformmatches, but noveltydiffers. Fullwaveformcollision must be rejected. No searchforanotherpositivepair: exactinversek=novelty+1.4, v=content-k provesidealbranchinjectivity. Finitebinary64rounding canmergeonlyULP-scalevalues; reportseparately, notanoperand-toleranceambiguity.

Validate exactFraction portwaveforms andtargetdifferences, numericalgeneratoradmissibility, eventmask/slot/onset/offset, actualeuler_stepcalls, statebefore/after, spikeflags/candidates andoutputpath. Instrument a reversiblein-memorywrapper aroundthe importedfrozen euler_step toobserveactualcalls; nevereditrepository. Checkencoding atall8registeredmodels throughrunner, andall3delays withbothrolequeries; M0 isexpectedtodistinguishrawrecords. For5distinctphysicalcarriers(M1,M2a,M2b,M2c,M2d), includeinitialA0differenceandposteventinterventions; M4usesM1encoding. Sharedfullprecisioncalibration comesfromfrozen summary_p1b.json, norefit. Independentrepeatedrolesprobe thesamecheckpoint, noqueryrecurrencecarryover. Savefullactualcallrecords incompressedNPZ andJSONmetadata, pluspermodel/roleoutputs andfixedparameters.

## Correct query-aware information statement

On positive novelty eachactiveevent'srawkey/valueisrecoverablefromthefullports. Morestrongly, whenatmostoneactiveeventhaszeronovelty, theentirewaveformplusanexactvalidquery identifiesitsrequestedvalue: matchqagainstdecodedpositivekeys; ifnomatch, theuniqueloweventhasvaluecontent-q. Distinctkeyspacingpreventsmultiplematches. Thisisafavorableofflineinformationoracle, notmemory/readoutaccessprovidedtoM1. Historieswithtwoormoreloweventsarenotautomaticallyambiguousoneveryquery; thecollisionprovespossibility, notprevalenceorpointwiseclassification.

## One minimal next experiment: frozen M1 novelty poststratification

Reconstructexactlytheoriginal4000histories:2000N2usingcontinuousRandomState2026091920,2000N3usingcontinuousRandomState2026091930. Metadata trialseedsarenotRNGresets. Save/hashcompleteunroundedinputs beforemodelmetrics. Historicaltrialledgerbytesareabsent, so thisisconditionalreproductionunderinstalledNumPy2.3.5, with comparisonagainstthefrozenaggregateJSON; donotclaimhistoricalbyteidentity.

RunonlyunchangedM1withsavedg_dec,b_dec,A0atdelays0,5,10ms. Encodingcanbeevaluatedonceperhistoryandreusedasanevaluatorcheckpointbecausequery/delayfunctionsdonotmutateit; thisdoesnotaddmodelmemory. Validate selectedepisodesagainstoriginalrunner.evaluate_episode. No M2a/M1 equivalenceornewspikingdecisioniscomputed. M0source/reference andfixtureoutputsremainascontractcontrols.

Beforeseeingstratumoutcomes, fixallorderednoveltypatterns:4N2patternsand8N3patterns, where1meansk>1.4. Reporteachpattern'sn, full-datasetfraction, jointstrict.5accuracy, roleMAE/RMSE, unconditionalEcomp, pluswholecellandthreepartitionsALL_POSITIVE,EXACTLY_ONE_ZERO,AT_LEAST_TWO_ZERO. Also pooledsufficient-waveform-identifiable subsetzero_count<=1. Saveall12000pairedM1episode rows/24000roleoutputs, retainallhistoriesinwholecelldenominators, andreportzero-countgroupswhenempty. No subgroupoptimization, hypothesispvalues,bootstrapornewpasslabels.

Theofflinewaveformoracleiscomputedonlyforthezero_count<=1subset, withnoaccess totruevaluesduringdecoding. Itreceivesactualeventports,maskandq, reconstructspositivekeys, orusesuniquezeronoveltyevent. Verifyitagainstheldouttargets butneverfeeditsanswer/historytoM1. Itdemonstratesinputavailabilityforanunrestrictedobserver; itisnotanaddedneuralbaselineorclaimedphysicallyfreecache.

Comparethe6unstratifiedM1cellsagainstfrozenmodel_summaries toensuretheanalysispreservestheoriginalbehavior. Conditionalrisks aredescriptive, notcausaleffectsofnovelty; noveltygroupsalsodifferinkeyamplitude,querydriveanddynamics. IfM1stillfailsinthefull-port-identifiablegroup, theflatinputambiguitycannotbethesoleexplanationofitsfailures. Donotassertthatthisisolatesmemory,querytransductionorreadout individually.

Stop after this audit, diagnostic and separate computational review. Separate known facts, replicated results and new diagnostic claims. Any proposed injective front-end change is explicitly a new contract, not implemented here. Preserve all prior artifacts and original labels.
