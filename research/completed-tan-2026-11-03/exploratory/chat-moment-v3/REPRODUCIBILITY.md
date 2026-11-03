# Reproducibility and development record

Frozen specification SHA256: `12969e0af938064554551688f1e2743ef4fc0d91cf92306438b6fa940a574a3a`.
Final results SHA256: `54094d7cd9f24944012d1c3e0266e1961aa94a4deb6663ff247e4a77cb0b24a8`.

The final main script was executed twice in the same reference environment. The
complete JSON outputs were byte-identical. `verify_v3.py` separately checked the
saved result without importing the learner implementation. This is same-agent
internal checking, not third-party independent replication.

The first execution already produced the final accept/refuse pattern. Later
changes added requested-key read denominators, saved root certificates, and
scorer-only diagnostics of rejected candidates. No frozen condition, noise,
threshold, error budget formula, update equation or accept decision was changed.
Rejected candidate diagnostics were not fed back into the learner or its gates.

No random seed is needed: all observations and noise offsets are deterministic
rationals. NumPy root proposals may vary on another numerical backend; they are
never accepted without exact-rational interval validation. Byte identity is
reported only for the two reference-environment executions, not promised for all
platforms. The correctness contract is the interval and transaction assertions.

Persistent memory uses signed int64 fixed-point codes. Certificate computation
uses arbitrary-precision rational workspace, explicitly not a same-64-bit
hardware implementation. The noise model contains bounded input errors and
quantized writes, not uncalibrated autonomous hardware drift.

The v1 and v2 archive hashes were unchanged before and after this work; see
preservation_checks.json. No original TAN repository or v11 data were modified.
