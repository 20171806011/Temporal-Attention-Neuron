# V9 complete result tables

Abstentions are counts; accuracy, balanced accuracy, coverage, silence, nonzero ties and class recalls are percentages. Every failed fit is retained. A pass requires both accuracy and coverage at least 90%. Class recalls include abstentions as errors. Confusion rows are true class 0/1 and columns predict 0, predict 1, abstain. All datasets have balanced class counts.

## Training at update 200 (20 fits)

| ID | Accuracy | Balanced acc. | Coverage | Abstentions | Silence | Nonzero ties | Recall 0 / 1 | Pass |
|---|---:|---:|---:|---:|---:|---:|---|---|
| s0_DEND_intact | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s1_DEND_intact | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s2_DEND_intact | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s3_DEND_intact | 53.91 | 53.91 | 63.28 | 47 | 0.00 | 36.72 | 29.69 / 78.12 | no |
| s4_DEND_intact | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s0_DEND_retrained | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s1_DEND_retrained | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s2_DEND_retrained | 99.22 | 99.22 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 98.44 | yes |
| s3_DEND_retrained | 96.88 | 96.88 | 97.66 | 3 | 0.00 | 2.34 | 100.00 / 93.75 | yes |
| s4_DEND_retrained | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s0_POINT_intact | 86.72 | 86.72 | 96.88 | 4 | 0.00 | 3.12 | 92.19 / 81.25 | no |
| s1_POINT_intact | 75.00 | 75.00 | 85.94 | 18 | 0.00 | 14.06 | 75.00 / 75.00 | no |
| s2_POINT_intact | 57.03 | 57.03 | 69.53 | 39 | 0.00 | 30.47 | 54.69 / 59.38 | no |
| s3_POINT_intact | 30.47 | 30.47 | 57.81 | 54 | 0.00 | 42.19 | 57.81 / 3.12 | no |
| s4_POINT_intact | 22.66 | 22.66 | 41.41 | 75 | 0.00 | 58.59 | 25.00 / 20.31 | no |
| s0_POINT_retrained | 78.91 | 78.91 | 92.97 | 9 | 0.00 | 7.03 | 57.81 / 100.00 | no |
| s1_POINT_retrained | 35.94 | 35.94 | 56.25 | 56 | 0.00 | 43.75 | 60.94 / 10.94 | no |
| s2_POINT_retrained | 47.66 | 47.66 | 69.53 | 39 | 0.00 | 30.47 | 57.81 / 37.50 | no |
| s3_POINT_retrained | 27.34 | 27.34 | 57.81 | 54 | 0.00 | 42.19 | 42.19 / 12.50 | no |
| s4_POINT_retrained | 28.12 | 28.12 | 55.47 | 57 | 0.00 | 44.53 | 51.56 / 4.69 | no |

## Fresh held-out evaluation (30 models)

| ID | Accuracy | Balanced acc. | Coverage | Abstentions | Silence | Nonzero ties | Recall 0 / 1 | Pass |
|---|---:|---:|---:|---:|---:|---:|---|---|
| s0_DEND_intact | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s1_DEND_intact | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s2_DEND_intact | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s3_DEND_intact | 51.76 | 51.76 | 60.35 | 203 | 0.00 | 39.65 | 27.73 / 75.78 | no |
| s4_DEND_intact | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s0_DEND_retrained | 99.02 | 99.02 | 99.02 | 5 | 0.00 | 0.98 | 98.44 / 99.61 | yes |
| s1_DEND_retrained | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s2_DEND_retrained | 98.63 | 98.63 | 99.61 | 2 | 0.00 | 0.39 | 100.00 / 97.27 | yes |
| s3_DEND_retrained | 96.68 | 96.68 | 98.63 | 7 | 0.00 | 1.37 | 100.00 / 93.36 | yes |
| s4_DEND_retrained | 100.00 | 100.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 100.00 | yes |
| s0_DEND_cut | 50.00 | 50.00 | 100.00 | 0 | 0.00 | 0.00 | 100.00 / 0.00 | no |
| s1_DEND_cut | 63.28 | 63.28 | 88.87 | 57 | 10.16 | 0.98 | 32.42 / 94.14 | no |
| s2_DEND_cut | 49.41 | 49.41 | 84.18 | 81 | 0.00 | 15.82 | 3.91 / 94.92 | no |
| s3_DEND_cut | 50.00 | 50.00 | 95.90 | 21 | 0.00 | 4.10 | 100.00 / 0.00 | no |
| s4_DEND_cut | 66.80 | 66.80 | 91.80 | 42 | 0.00 | 8.20 | 100.00 / 33.59 | no |
| s0_POINT_intact | 79.88 | 79.88 | 92.19 | 40 | 0.00 | 7.81 | 82.42 / 77.34 | no |
| s1_POINT_intact | 74.02 | 74.02 | 83.59 | 84 | 0.00 | 16.41 | 72.66 / 75.39 | no |
| s2_POINT_intact | 52.15 | 52.15 | 67.97 | 164 | 0.00 | 32.03 | 44.92 / 59.38 | no |
| s3_POINT_intact | 32.81 | 32.81 | 59.77 | 206 | 0.00 | 40.23 | 58.98 / 6.64 | no |
| s4_POINT_intact | 29.10 | 29.10 | 49.80 | 257 | 0.00 | 50.20 | 30.08 / 28.12 | no |
| s0_POINT_retrained | 78.32 | 78.32 | 92.19 | 40 | 0.00 | 7.81 | 57.03 / 99.61 | no |
| s1_POINT_retrained | 37.50 | 37.50 | 59.96 | 205 | 0.00 | 40.04 | 55.47 / 19.53 | no |
| s2_POINT_retrained | 51.17 | 51.17 | 74.02 | 133 | 0.00 | 25.98 | 51.56 / 50.78 | no |
| s3_POINT_retrained | 28.12 | 28.12 | 58.98 | 210 | 0.00 | 41.02 | 42.97 / 13.28 | no |
| s4_POINT_retrained | 26.56 | 26.56 | 52.34 | 244 | 0.00 | 47.66 | 47.66 / 5.47 | no |
| s0_POINT_cut | 59.18 | 59.18 | 81.84 | 93 | 0.00 | 18.16 | 25.78 / 92.58 | no |
| s1_POINT_cut | 55.08 | 55.08 | 87.11 | 66 | 0.00 | 12.89 | 21.09 / 89.06 | no |
| s2_POINT_cut | 55.86 | 55.86 | 77.93 | 113 | 0.00 | 22.07 | 49.22 / 62.50 | no |
| s3_POINT_cut | 47.27 | 47.27 | 85.35 | 75 | 0.00 | 14.65 | 93.36 / 1.17 | no |
| s4_POINT_cut | 38.48 | 38.48 | 68.55 | 161 | 0.00 | 31.45 | 49.61 / 27.34 | no |

## Held-out confusion and signed count margin

Margin = true-class count minus other-class count, over all episodes. The CSV retains all 17 histogram bins from −8 through 8.

| ID | Confusion | Mean margin | Median margin | Both saturated (%) |
|---|---|---:|---:|---:|
| s0_DEND_intact | `[[256, 0, 0], [0, 256, 0]]` | 5.7324 | 6.0000 | 0.00 |
| s0_DEND_retrained | `[[252, 0, 4], [0, 255, 1]]` | 4.5391 | 5.0000 | 0.00 |
| s0_POINT_intact | `[[211, 23, 22], [40, 198, 18]]` | 2.0879 | 2.0000 | 0.00 |
| s0_POINT_retrained | `[[146, 71, 39], [0, 255, 1]]` | 2.0391 | 2.0000 | 0.00 |
| s0_DEND_cut | `[[256, 0, 0], [256, 0, 0]]` | 0.9355 | 0.5000 | 0.00 |
| s0_POINT_cut | `[[66, 109, 81], [7, 237, 12]]` | 0.7773 | 1.0000 | 0.00 |
| s1_DEND_intact | `[[256, 0, 0], [0, 256, 0]]` | 5.1973 | 5.0000 | 0.00 |
| s1_DEND_retrained | `[[256, 0, 0], [0, 256, 0]]` | 5.6348 | 6.0000 | 0.00 |
| s1_POINT_intact | `[[186, 21, 49], [28, 193, 35]]` | 2.7734 | 2.0000 | 0.00 |
| s1_POINT_retrained | `[[142, 28, 86], [87, 50, 119]]` | 0.2441 | 0.0000 | 0.00 |
| s1_DEND_cut | `[[83, 130, 43], [1, 241, 14]]` | 1.0410 | 1.0000 | 0.00 |
| s1_POINT_cut | `[[54, 156, 46], [8, 228, 20]]` | 0.7617 | 1.0000 | 0.00 |
| s2_DEND_intact | `[[256, 0, 0], [0, 256, 0]]` | 4.7734 | 5.0000 | 0.00 |
| s2_DEND_retrained | `[[256, 0, 0], [5, 249, 2]]` | 5.1055 | 5.0000 | 0.00 |
| s2_POINT_intact | `[[115, 45, 96], [36, 152, 68]]` | 0.6562 | 1.0000 | 0.00 |
| s2_POINT_retrained | `[[132, 48, 76], [69, 130, 57]]` | 0.3516 | 1.0000 | 0.00 |
| s2_DEND_cut | `[[10, 178, 68], [0, 243, 13]]` | 0.6602 | 0.0000 | 0.00 |
| s2_POINT_cut | `[[126, 48, 82], [65, 160, 31]]` | 0.6953 | 1.0000 | 0.00 |
| s3_DEND_intact | `[[71, 40, 145], [4, 194, 58]]` | 0.6582 | 1.0000 | 0.00 |
| s3_DEND_retrained | `[[256, 0, 0], [10, 239, 7]]` | 3.3496 | 3.0000 | 0.00 |
| s3_POINT_intact | `[[151, 33, 72], [105, 17, 134]]` | 0.1152 | 0.0000 | 0.00 |
| s3_POINT_retrained | `[[110, 19, 127], [139, 34, 83]]` | -0.0645 | 0.0000 | 0.00 |
| s3_DEND_cut | `[[256, 0, 0], [235, 0, 21]]` | 0.5664 | 1.0000 | 0.00 |
| s3_POINT_cut | `[[239, 1, 16], [194, 3, 59]]` | 0.1543 | 0.0000 | 0.00 |
| s4_DEND_intact | `[[256, 0, 0], [0, 256, 0]]` | 6.2441 | 7.0000 | 0.00 |
| s4_DEND_retrained | `[[256, 0, 0], [0, 256, 0]]` | 5.6348 | 5.0000 | 0.00 |
| s4_POINT_intact | `[[77, 51, 128], [55, 72, 129]]` | 0.1074 | 0.0000 | 0.00 |
| s4_POINT_retrained | `[[122, 26, 108], [106, 14, 136]]` | 0.1055 | 0.0000 | 0.00 |
| s4_DEND_cut | `[[256, 0, 0], [128, 86, 42]]` | 1.8789 | 4.0000 | 0.00 |
| s4_POINT_cut | `[[127, 33, 96], [121, 70, 65]]` | 0.1367 | 0.0000 | 0.00 |

## Held-out activity and retrospective settling

Events are means per episode over all 24 steps, including failed predictions. State variation includes the transition from zero initial state. Curves list correct-and-settled percentages at readout steps 1–8; they use future prefixes and are not online stopping times or energy measurements.

| ID | Branch events | Soma events | Output events | Total events | State L1 change | Correct-settled curve (%) |
|---|---:|---:|---:|---:|---:|---|
| s0_DEND_intact | 0.000 | 47.227 | 16.244 | 63.471 | 46.341 | 69.34, 76.17, 92.97, 99.22, 99.61, 100.00, 100.00, 100.00 |
| s0_DEND_retrained | 0.000 | 35.869 | 16.225 | 52.094 | 42.902 | 64.84, 69.34, 79.10, 87.70, 93.75, 98.24, 98.63, 99.02 |
| s0_POINT_intact | 20.148 | 13.926 | 20.832 | 54.906 | 57.315 | 33.79, 37.50, 51.37, 60.35, 69.14, 72.07, 74.41, 79.88 |
| s0_POINT_retrained | 13.613 | 15.039 | 18.750 | 47.402 | 53.618 | 17.97, 22.85, 32.23, 45.31, 56.45, 71.09, 74.22, 78.32 |
| s0_DEND_cut | 0.000 | 50.473 | 15.877 | 66.350 | 44.346 | 19.73, 24.41, 27.54, 37.89, 48.44, 50.00, 50.00, 50.00 |
| s0_POINT_cut | 16.789 | 12.549 | 19.582 | 48.920 | 52.315 | 13.48, 18.36, 28.12, 36.33, 46.09, 49.22, 52.15, 59.18 |
| s1_DEND_intact | 0.000 | 60.727 | 17.006 | 77.732 | 55.656 | 72.46, 95.31, 97.66, 98.83, 99.22, 99.41, 99.61, 100.00 |
| s1_DEND_retrained | 0.000 | 72.320 | 18.762 | 91.082 | 59.209 | 73.44, 87.50, 95.90, 98.24, 98.83, 100.00, 100.00, 100.00 |
| s1_POINT_intact | 16.424 | 15.088 | 19.945 | 51.457 | 57.118 | 41.02, 58.01, 58.79, 61.52, 63.87, 66.60, 70.31, 74.02 |
| s1_POINT_retrained | 17.596 | 12.090 | 16.385 | 46.070 | 47.689 | 13.87, 16.02, 17.97, 21.29, 22.66, 23.63, 31.05, 37.50 |
| s1_DEND_cut | 0.000 | 45.508 | 9.320 | 54.828 | 58.726 | 28.32, 48.24, 58.59, 62.30, 62.30, 62.70, 62.89, 63.28 |
| s1_POINT_cut | 19.326 | 17.904 | 17.436 | 54.666 | 56.451 | 22.27, 32.03, 35.16, 36.33, 41.41, 42.38, 42.97, 55.08 |
| s2_DEND_intact | 0.000 | 36.393 | 15.332 | 51.725 | 61.655 | 54.30, 91.99, 97.07, 100.00, 100.00, 100.00, 100.00, 100.00 |
| s2_DEND_retrained | 0.000 | 27.496 | 17.180 | 44.676 | 54.570 | 62.50, 97.07, 97.66, 98.44, 98.63, 98.63, 98.63, 98.63 |
| s2_POINT_intact | 14.562 | 10.473 | 15.883 | 40.918 | 50.409 | 13.87, 25.00, 25.59, 33.98, 34.57, 41.80, 43.55, 52.15 |
| s2_POINT_retrained | 12.611 | 10.516 | 17.045 | 40.172 | 47.106 | 10.55, 15.82, 16.02, 23.24, 27.73, 37.50, 41.41, 51.17 |
| s2_DEND_cut | 0.000 | 35.807 | 20.117 | 55.924 | 46.645 | 14.26, 41.02, 45.90, 49.41, 49.41, 49.41, 49.41, 49.41 |
| s2_POINT_cut | 12.570 | 8.973 | 16.314 | 37.857 | 46.197 | 15.43, 21.29, 21.48, 39.84, 42.77, 48.24, 48.44, 55.86 |
| s3_DEND_intact | 0.000 | 57.967 | 14.084 | 72.051 | 51.496 | 12.89, 16.99, 25.00, 29.88, 32.62, 38.67, 46.29, 51.76 |
| s3_DEND_retrained | 0.000 | 47.219 | 13.436 | 60.654 | 47.814 | 39.84, 83.01, 84.57, 88.87, 92.19, 93.16, 94.53, 96.68 |
| s3_POINT_intact | 18.305 | 14.371 | 16.375 | 49.051 | 47.800 | 4.88, 7.03, 9.38, 9.57, 11.33, 11.72, 19.53, 32.81 |
| s3_POINT_retrained | 10.166 | 9.098 | 18.492 | 37.756 | 38.207 | 0.20, 2.15, 7.62, 11.52, 15.82, 19.92, 22.46, 28.12 |
| s3_DEND_cut | 0.000 | 41.801 | 17.756 | 59.557 | 46.560 | 24.02, 41.41, 49.80, 50.00, 50.00, 50.00, 50.00, 50.00 |
| s3_POINT_cut | 13.658 | 11.303 | 14.930 | 39.891 | 41.675 | 12.70, 15.62, 16.21, 23.83, 25.59, 28.52, 38.87, 47.27 |
| s4_DEND_intact | 0.000 | 43.201 | 22.670 | 65.871 | 39.721 | 66.60, 79.69, 89.45, 100.00, 100.00, 100.00, 100.00, 100.00 |
| s4_DEND_retrained | 0.000 | 54.420 | 18.602 | 73.021 | 46.572 | 68.16, 76.17, 96.88, 99.80, 99.80, 99.80, 100.00, 100.00 |
| s4_POINT_intact | 11.986 | 10.963 | 18.602 | 41.551 | 44.931 | 16.21, 18.16, 20.70, 20.70, 20.70, 20.70, 20.90, 29.10 |
| s4_POINT_retrained | 13.441 | 14.121 | 18.014 | 45.576 | 44.488 | 13.67, 15.43, 15.62, 16.41, 17.38, 21.09, 22.27, 26.56 |
| s4_DEND_cut | 0.000 | 50.486 | 22.594 | 73.080 | 33.639 | 47.66, 48.44, 51.95, 51.95, 65.04, 65.43, 65.43, 66.80 |
| s4_POINT_cut | 12.490 | 11.531 | 17.691 | 41.713 | 41.932 | 9.96, 10.16, 11.13, 12.11, 14.65, 22.07, 28.71, 38.48 |

## Paired seed-block contrasts

All intervals are 95% percentile bootstrap intervals from 20,000 resamples of five complete paired blocks. Values below are percentage points. There is no family-wide multiplicity correction; all eight are exploratory. Negative failure differences mean fewer intact failures. A degenerate or zero-crossing interval does not establish equivalence.

| Contrast | Primary descriptive | Mean | Pointwise 95% interval | Five paired values |
|---|---|---:|---|---|
| DEND_intact_minus_retrained_accuracy | yes | -8.52 | [-26.76, 0.98] | 0.98, 0.00, 1.37, -44.92, 0.00 |
| DEND_intact_minus_cut_accuracy | no | 34.45 | [17.70, 47.58] | 50.00, 36.72, 50.59, 1.76, 33.20 |
| DEND_intact_minus_retrained_failed | no | 20.00 | [0.00, 60.00] | 0.00, 0.00, 0.00, 100.00, 0.00 |
| DEND_intact_minus_cut_failed | no | -80.00 | [-100.00, -40.00] | -100.00, -100.00, -100.00, 0.00, -100.00 |
| POINT_intact_minus_retrained_accuracy | yes | 9.26 | [1.52, 23.05] | 1.56, 36.52, 0.98, 4.69, 2.54 |
| POINT_intact_minus_cut_accuracy | no | 2.42 | [-10.27, 15.12] | 20.70, 18.95, -3.71, -14.45, -9.38 |
| POINT_intact_minus_retrained_failed | no | 0.00 | [0.00, 0.00] | 0.00, 0.00, 0.00, 0.00, 0.00 |
| POINT_intact_minus_cut_failed | no | 0.00 | [0.00, 0.00] | 0.00, 0.00, 0.00, 0.00, 0.00 |

## Frozen seed registry

| Block | Training data | Test data | Initialization |
|---|---:|---:|---:|
| 0 | 2090000000 | 2090010000 | 2090020000 |
| 1 | 2090000010 | 2090010010 | 2090020001 |
| 2 | 2090000020 | 2090010020 | 2090020002 |
| 3 | 2090000030 | 2090010030 | 2090020003 |
| 4 | 2090000040 | 2090010040 | 2090020004 |

The analysis RNG seed is 2090030001. Initialization is shared on retained coordinates. The ten cut models have no separate training fit; their provenance is the corresponding intact vector multiplied by the fixed mask. Full train activity and all native traces remain in JSON/NPZ and the CSV.
