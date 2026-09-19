import pathlib
import re

files = list(pathlib.Path("paper").rglob("*.tex"))
pats = [
    re.compile(r"`"),
    re.compile(r"'\.(pdf|csv|md|py|json|log|npz|tex)"),
    re.compile(r"}(?=[a-z_0-9])"),
    re.compile(r"'[)\.,;:]"),
]
for f in files:
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        for p in pats:
            if p.search(line):
                print(f"{f.name}:{i}: {line.strip()[:130]}")
                break
