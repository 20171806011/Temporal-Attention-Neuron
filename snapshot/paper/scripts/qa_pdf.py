"""TAN-I PDF QA: text-level checks (no visual inspection possible in
this environment; the QA report records this limitation).

Notes on extraction artifacts handled here:
  * pypdf may split a line at the extraction newline, so checks use
    substrings that do not span layout line breaks where possible.
  * cmtt underscores (AUDIT\_FAILED) can be mapped to spaces by pypdf,
    so each check accepts '_' and ' ' as equivalent separators.
  * banned-phrase hits are only counted as affirmative when they are
    NOT negated within the 100 chars preceding the hit.
"""
import re
from pathlib import Path

from pypdf import PdfReader

pdf = Path("paper/main.pdf")
r = PdfReader(str(pdf))
n = len(r.pages)
text = []
for i, p in enumerate(r.pages):
    try:
        text.append((p.extract_text() or "") + "\n")
    except Exception as e:  # noqa
        text.append(f"[page {i+1} extraction error: {e}]\n")
full = "\n".join(text)

print(f"pages: {n}")
print(f"extracted chars: {len(full)}")


def found(s):
    """Substring check tolerant of '_' being extracted as ' '."""
    if s in full:
        return True
    return s.replace("_", " ") in full


checks = {
    "title": "Boundary of Scalar Temporal",
    "abstract": "Background.",
    "keywords": "mechanistic",
    "phase1 K B4": "2.047",
    "probe1 pooled diff": "0.0570",
    "probe2 B2": "0.779",
    "probe2 hit": "0.511",
    "AUDIT_FAILED": "AUDIT_FAILED",
    "TERMINATED": "TERMINATED",
    "flip rate": "0.000",
    "claim separation box": "Routing",
    "references": "References",
    "supp S1": "Detailed TAN equations",
    "supp S8": "Reproducibility details",
}
for k, s in checks.items():
    print(f"  check '{k}': {'FOUND' if found(s) else 'MISSING'}")

banned = ["intelligent", "superior memory", "universally outperforms",
          "semantic binding is established", "distractor-robust",
          "intrinsic manifold dimension"]
neg = re.compile(r"\b(no|not|without|absence|absent|never|none)\b", re.I)
print("banned-phrase scan (affirmative occurrences must be ABSENT):")
for b in banned:
    hits = [m.start() for m in re.finditer(re.escape(b), full)]
    affirm = [h for h in hits
              if not neg.search(full[max(0, h - 100):h].lower())]
    print(f"  '{b}': {'ABSENT' if not affirm else 'PRESENT x' + str(len(affirm))}"
          + (f" (all {len(hits)} hit(s) negated)" if hits and not affirm else ""))

print(f"literal '??' occurrences: {full.count('??')}")
print(f"page count: {n}")

# bib item count from .bbl
bbl = Path("paper/main.bbl")
if bbl.exists():
    b = bbl.read_text(encoding="utf-8")
    print(f"bibliography entries: {b.count(chr(92)+'bibitem')}")
