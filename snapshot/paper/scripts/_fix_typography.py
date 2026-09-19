import pathlib

P = pathlib.Path("paper")

# 1) preamble additions in main.tex
main = P / "main.tex"
t = main.read_text(encoding="utf-8")
if "emergencystretch" not in t:
    t = t.replace(
        "\\graphicspath{{figures/}}",
        "\\graphicspath{{figures/}}\n"
        "\\tolerance=2000\n"
        "\\emergencystretch=2em\n"
        "\\hfuzz=1.5pt",
    )
main.write_text(t, encoding="utf-8")

# 2) S8: allow breaks in code paths and the hash
s8 = P / "supplementary" / "S8_reproducibility.tex"
t = s8.read_text(encoding="utf-8")
t = t.replace("/", "/\\allowbreak{}")
t = t.replace("\\_", "\\_\\allowbreak{}")
hash_ = "AABF8EBBB81E911E69117591F84E4F836CF59CF8C1B505DE53A8B7C1A88143C7"
broken = "\\allowbreak{}".join([hash_[i:i + 16]
                                for i in range(0, len(hash_), 16)])
t = t.replace(hash_, broken)
s8.write_text(t, encoding="utf-8")

# 3) main.tex: allow breaks at underscores inside the long \texttt paths
t = main.read_text(encoding="utf-8")
t = t.replace("\\_", "\\_\\allowbreak{}")
main.write_text(t, encoding="utf-8")

# 4) supplementary files: allow breaks after underscores in status tokens
for f in P.joinpath("supplementary").glob("*.tex"):
    t = f.read_text(encoding="utf-8")
    t2 = t.replace("\\_", "\\_\\allowbreak{}")
    if t2 != t:
        f.write_text(t2, encoding="utf-8")

# 5) tab_probe1: p-columns for statuses and smaller font
tp = P / "tables" / "tab_probe1.tex"
t = tp.read_text(encoding="utf-8")
t = t.replace("\\begin{tabular}{cccccc}",
              "\\small\\begin{tabular}{lllcp{2.4cm}p{2.4cm}}")
t = t.replace("seed & model & condition & bal3 & coverage status & final "
              "status \\\\",
              "seed & model & condition & bal3 & \\raggedright coverage "
              "status & \\raggedright final status \\\\")
t = t.replace("\\_", "\\_\\allowbreak{}")
tp.write_text(t, encoding="utf-8")
print("typography patch applied")
