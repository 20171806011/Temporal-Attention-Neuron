# V5 failures inspected before V6: post hoc descriptive analysis

**This is a post hoc audit of already observed V5 results.** It generated no new data, trained no new models, and makes no causal claim. It uses all ten final B DEND seeds for CE/.03 and their matched RATE/.01 controls, including successful seeds; it is not an independently tuned new comparison.

The 20 complete Adam trajectories were reconstructed from saved clipped gradients and exactly reproduce every final weight array. Native forward replay uses the frozen V5 implementation on the saved 128-episode training sets at updates 0, 1, 5, 10, 20, 50, 100, 150 and 200. Final 512-episode test outputs are read from the saved V5 arrays, not regenerated or reevaluated. Thus these observations do not create additional independent experiments.

## Final training versus held-out behavior

All accuracies count ties and silence as errors. A failure is accuracy <90% or coverage <90%. The table reports each training seed, not episode-pooled pseudo-replicates.

| Seed | RATE train acc | RATE test acc | CE train acc | CE test acc | CE train coverage | CE train silence | CE test silence |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100.0000% | 100.0000% | 100.0000% | 99.4141% | 100.0000% | 0.0000% | 0.0000% |
| 1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 2 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 3 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 4 | 100.0000% | 99.8047% | 100.0000% | 99.6094% | 100.0000% | 0.0000% | 0.0000% |
| 5 | 98.4375% | 99.4141% | 50.0000% | 50.0000% | 50.0000% | 50.0000% | 50.0000% |
| 6 | 100.0000% | 100.0000% | 100.0000% | 99.8047% | 100.0000% | 0.0000% | 0.0000% |
| 7 | 75.0000% | 75.0000% | 0.0000% | 0.0000% | 0.0000% | 100.0000% | 100.0000% |
| 8 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |
| 9 | 100.0000% | 99.6094% | 0.0000% | 0.0000% | 0.0000% | 100.0000% | 100.0000% |

## Failed CE/.03 seeds and observed checkpoints

Counts below concern the native output readout at steps16–23. Candidate margins are prethreshold output values minus .5, so a nonpositive margin cannot emit an output spike at that step. Hidden activity is separately reported; output silence must not be called whole-network silence.

| Seed | Updates | Train acc | Coverage | Silence | Mean signed count margin | Hidden spikes / episode, all24 | Output spikes / episode, all24 | Max output candidate margin, readout |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0 | 47.66% | 95.31% | 1.56% | -0.2344 | 72.4453 | 4.4531 | 1.01912 |
| 5 | 1 | 50.00% | 100.00% | 0.00% | -0.0234 | 74.2891 | 8.8672 | 0.834895 |
| 5 | 5 | 4.69% | 21.09% | 78.91% | -0.1172 | 80.7969 | 0.2422 | 0.075083 |
| 5 | 10 | 0.00% | 0.00% | 100.00% | 0.0000 | 79.6953 | 0.0000 | -0.882143 |
| 5 | 20 | 0.00% | 0.00% | 100.00% | 0.0000 | 80.3281 | 0.0000 | -2.00602 |
| 5 | 50 | 0.00% | 0.00% | 100.00% | 0.0000 | 82.0703 | 0.0000 | -2.41226 |
| 5 | 100 | 0.00% | 0.00% | 100.00% | 0.0000 | 81.2422 | 0.0000 | -1.82593 |
| 5 | 150 | 25.00% | 25.00% | 75.00% | 0.8672 | 70.1250 | 1.2266 | 0.164607 |
| 5 | 200 | 50.00% | 50.00% | 50.00% | 3.0000 | 75.5859 | 5.4688 | 0.301065 |
| 7 | 0 | 50.00% | 100.00% | 0.00% | 0.0078 | 58.9688 | 11.3828 | 1.22595 |
| 7 | 1 | 50.00% | 100.00% | 0.00% | -0.0234 | 52.4062 | 3.9766 | 0.975351 |
| 7 | 5 | 0.00% | 0.00% | 100.00% | 0.0000 | 43.2109 | 0.0000 | -0.0463652 |
| 7 | 10 | 0.00% | 0.00% | 100.00% | 0.0000 | 45.5391 | 0.0000 | -0.337088 |
| 7 | 20 | 0.00% | 0.00% | 100.00% | 0.0000 | 46.0000 | 0.0000 | -0.502825 |
| 7 | 50 | 0.00% | 0.00% | 100.00% | 0.0000 | 46.0000 | 0.0000 | -0.600738 |
| 7 | 100 | 0.00% | 0.00% | 100.00% | 0.0000 | 46.0000 | 0.0000 | -0.606686 |
| 7 | 150 | 0.00% | 0.00% | 100.00% | 0.0000 | 46.0000 | 0.0000 | -0.606723 |
| 7 | 200 | 0.00% | 0.00% | 100.00% | 0.0000 | 46.0000 | 0.0000 | -0.606716 |
| 9 | 0 | 50.00% | 100.00% | 0.00% | 0.1250 | 76.5234 | 12.8906 | 1.13961 |
| 9 | 1 | 50.00% | 100.00% | 0.00% | 0.5234 | 79.5391 | 15.8047 | 1.11529 |
| 9 | 5 | 50.00% | 100.00% | 0.00% | 0.0000 | 81.1328 | 1.4609 | 0.231401 |
| 9 | 10 | 0.00% | 0.00% | 100.00% | 0.0000 | 82.3828 | 0.0469 | -0.852744 |
| 9 | 20 | 0.00% | 0.00% | 100.00% | 0.0000 | 82.0938 | 0.0000 | -1.57335 |
| 9 | 50 | 0.00% | 0.00% | 100.00% | 0.0000 | 83.4609 | 0.0000 | -2.01952 |
| 9 | 100 | 0.00% | 0.00% | 100.00% | 0.0000 | 83.6406 | 0.0000 | -2.06923 |
| 9 | 150 | 0.00% | 0.00% | 100.00% | 0.0000 | 83.6250 | 0.0000 | -2.07312 |
| 9 | 200 | 0.00% | 0.00% | 100.00% | 0.0000 | 83.5859 | 0.0000 | -2.0798 |

## Final native margins and activity for all CE/.03 seeds

| Seed | Train count margin mean [min,max] | Test count margin mean [min,max] | Train hidden spikes / episode, readout8 | Test hidden spikes / episode, readout8 | Train output candidate margin max | Test output candidate margin max | Last20 data surrogate gradient norm median |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | 8.0000 [8,8] | 7.9219 [-6,8] | 25.9766 | 26.0020 | 0.669023 | 1.46471 | 0.00329434 |
| 1 | 8.0000 [8,8] | 8.0000 [8,8] | 19.3203 | 19.3633 | 1.7585 | 1.7585 | 0.00268072 |
| 2 | 7.9922 [7,8] | 7.9785 [6,8] | 16.0078 | 16.0215 | 1.6443 | 1.6443 | 0.00159009 |
| 3 | 8.0000 [8,8] | 8.0000 [8,8] | 22.5703 | 22.5859 | 1.75272 | 1.75272 | 0.00194057 |
| 4 | 7.9844 [7,8] | 7.8945 [-6,8] | 13.5391 | 13.5742 | 0.955898 | 1.09863 | 0.00456953 |
| 5 | 3.0000 [0,8] | 3.0000 [0,8] | 28.0000 | 28.0000 | 0.301065 | 0.434828 | 0.0934253 |
| 6 | 8.0000 [8,8] | 7.9844 [0,8] | 6.5078 | 6.5977 | 1.29723 | 1.29723 | 0.00570915 |
| 7 | 0.0000 [0,0] | 0.0000 [0,0] | 16.0000 | 16.0000 | -0.606716 | -0.606716 | 9.12603e-06 |
| 8 | 7.9922 [7,8] | 7.9844 [7,8] | 22.0000 | 22.0020 | 1.37275 | 1.63152 | 0.00147567 |
| 9 | 0.0000 [0,0] | 0.0000 [0,0] | 30.1016 | 30.1562 | -2.0798 | -2.07954 | 5.08171e-05 |

## Interpretation and unavailable evidence

For CE/.03 seeds 5, 7 and 9, the final training accuracies are respectively 50%, 0% and 0%, exactly matching their held-out accuracies. Thus the three observed failures already exist on the training examples; they are not a collapse occurring only on held-out examples. The other seven CE seeds fit all training examples and score 99.4141–100% on their saved tests. These descriptions concern this observed cohort, not a universal generalization guarantee.

Seed 7 is fully output-silent by the sampled checkpoint after 5 updates and at every later sampled checkpoint. Seed 9 is fully readout-silent by checkpoint 10 and at all later sampled checkpoints; at checkpoint 10 it still has a small amount of output activity outside the readout window. Seed 5 is also fully readout-silent at checkpoints 10, 20, 50 and 100, then partially recovers to 25% training accuracy at 150 and 50% at 200. This temporary collapse would be missed by looking only at its final 50% score. Sparse activity checkpoints do not locate the exact transition update or exclude unsampled excursions.

The silent final seeds 7/9 retain hidden activity: respectively 16.0/30.10 hidden spikes per training episode during the eight readout steps. Their maximum output candidate-minus-threshold margins are −0.6067/−2.0798 on training data, with similar held-out margins. Their last 20 median data-surrogate gradient norms are approximately 9.13e-6/5.08e-5, small but nonzero; this does not prove a zero-gradient absorbing state. In contrast, RATE/.01 reaches 98.4375% training and 99.4141% test accuracy for seed 5, and 100% training / 99.6094% test for seed 9. Its matched seed 7 also fails at 75% training and test accuracy, so RATE does not rescue every problematic initialization in this cohort.

The five development seeds used to select DEND CE/.03 all had 100% validation accuracy. The B seed cohort was retrained from new data and initializations. Failure to transfer that development-selected recipe across training seeds is distinct from a train/test generalization gap within one fitted seed. No final test was used for retraining or selection in this audit.

Both loss and learning rate differ between CE/.03 and RATE/.01. Their paired observational contrast cannot assign a failure to CE alone, learning rate alone, a specific recurrent connection, the surrogate gradient, a particular threshold margin or representational capacity. Even a temporally preceding change is not a causal intervention. The gradient summaries describe the prescribed surrogate rule. They are not exact hard-spike derivatives.

Loss and gradient/update norms are available at all 200 preupdate steps. Activity and margins here are available at only nine reconstructed training checkpoints; statements about an onset must say “by the sampled checkpoint” and must not assert an exact collapse update. Intermediate held-out activity was not recorded and is not recreated. The JSON retains all sampled metrics, per-class margins and recalls, full saved loss/norm sequences, source hashes and exact-reconstruction checks.

This audit is internal to the same AI-assisted workflow. It is not external independent replication or peer review, and it does not add to V5’s ten independently trained seed pairs.

Runtime for reconstruction and replay: 7.659 seconds. 180 training-batch forward replays, 23040 training episode replays; 20 saved final test arrays inspected. Full numerical details: [posthoc_v5.json](posthoc_v5.json).
