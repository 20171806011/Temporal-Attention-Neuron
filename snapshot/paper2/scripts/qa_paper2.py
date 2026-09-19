"""Paper 2 QA: text-level and layout validation."""
import re
import sys
from pathlib import Path
import pypdf

def main():
    pdf_path = Path("main.pdf")
    if not pdf_path.exists():
        print("ERROR: main.pdf not found")
        sys.exit(1)

    reader = pypdf.PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    print(f"Paper 2 page count: {num_pages}")
    assert num_pages == 28, f"Expected 28 pages, got {num_pages}"

    raw_text = ""
    for i, p in enumerate(reader.pages):
        raw_text += f"\n--- PAGE {i+1} ---\n" + p.extract_text()
    full_text = " ".join(raw_text.split())

    required_phrases = [
        "The Minimal Architectural Ladder for Relational Computation",
        "Li Zexu",
        "School of Physics and Astronomy, University of Leeds",
        "Final Revised Manuscript (September 2026)",
        "0/6,705",
        "100.00%",
        "Master claim ledger",
        "Sufficiency of 1D Metric Compatibility",
        "Impossibility of Scalar Intermediate Addressing",
        "Proposition 5.3",
        "Proposition 5.4",
        "Reproducibility statement",
        "References",
    ]

    for req in required_phrases:
        if req in full_text:
            print(f"  check '{req}': FOUND")
        else:
            print(f"  check '{req}': MISSING")
            sys.exit(1)

    # Check for literal '??'
    qq_matches = re.findall(r'\?\?', full_text)
    print(f"literal '??' occurrences: {len(qq_matches)}")
    assert len(qq_matches) == 0, "Found literal '??' in PDF!"

    # Check for references 1..13
    for r in range(1, 14):
        ref_str = f"[{r}]"
        assert ref_str in full_text, f"Reference {ref_str} not found in text!"
    print("All 13 bibliography references [1]-[13] FOUND.")

    print("\nALL PAPER 2 QA CHECKS PASSED SUCCESSFULLY (28 pages, 0 missing refs, 0 ??).")

if __name__ == "__main__":
    main()
