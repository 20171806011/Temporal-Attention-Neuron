> Historical archival document. Original language preserved. Personal paths/host metadata and unavailable media links were sanitized; consult README.md and ARCHIVE_INDEX.json.

# Before formal data

24fits:4arms x2fixed learning rates x3new seeds. 400updates each,64nominal training episodes. Test256episodes/seed under6fixed conditions (144evaluations). No model/condition selection. Twelve independent fixture updates per arm (48total), data seed79991001 and initial seed79992001. Feasible only if2*400*6*sum(per-arm maximum last6 step times)+100s <840s; otherwise stop before science. No fallback selection after performance. Timing cap90s, full scientific train/eval single840s cap, audit300s,1GiB working set,1.5GiB artifacts. All failures retained.
