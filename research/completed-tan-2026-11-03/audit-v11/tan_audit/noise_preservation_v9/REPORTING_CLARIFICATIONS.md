# Reporting clarifications; frozen experiment unchanged

2026-11-02, before fresh outcome reporting. Independent source/wording review identified two wording clarifications that change no parameter, input, comparison, inference code or protocol bytes.

- The keep probability 529/1013 uses exact means .2645 and .5065 in the frozen saved summary table. The manuscript prints rounded means .265 and .506; the ratio is not computed from those rounded displayed numbers.
- The six bootstrap percentile intervals are pointwise descriptive 95% intervals. They have no simultaneous or familywise coverage guarantee. The protocol's instruction to consider all comparisons descriptively does not convert pointwise intervals into simultaneous confidence intervals.
- scipy.stats.ranksums is a Wilcoxon rank-sum test; that name is valid and appears correctly in the manuscript. The saved CSV field wilcoxon_p must not be interpreted as a paired Wilcoxon signed-rank calculation. The reproducible concern is the independent-sample analysis of paired seed data, already acknowledged in the manuscript, rather than a claim that all Wilcoxon naming was erroneous.

No model/threshold/window change, re-registration, extra endpoint or significance-driven selection is introduced. The original labels and statistics remain historical artifacts. Any all-success or all-failure empirical rate and degenerate empirical-bootstrap interval must not be interpreted as proof that the corresponding population probability is exactly zero or one.
