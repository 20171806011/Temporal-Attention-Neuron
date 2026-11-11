> Historical archival document. Original language preserved. Personal paths/host metadata and unavailable media links were sanitized; consult README.md and ARCHIVE_INDEX.json.

# Timing-only feasibility rule, before scientific data

48 fits: 2 TAN variants x G{1,4} x L{.5,2} x regularizer{.0001,0} x 3 new paired blocks. Start with 150 updates. Choose the largest of [150,100,60] satisfying `1.75 * updates * 24 * (max_full_step_seconds + max_noatt_step_seconds) + 90 <= 840`. Per-arm maximum uses the last four of eight fixture updates for each G/L combination. If none fits, stop before science. No factor, seed, learning rate or performance-dependent selection. Same chosen updates for every fit. Fixture seeds 79971001(data),79972001(initialization),79973001(gradient fixture); no scientific seeds. Fixture timing cap90s. Science includes all calibration/train/test and uses a single 840s clock, audit separately300s. Fixture timing does not guarantee sustained runtime.
