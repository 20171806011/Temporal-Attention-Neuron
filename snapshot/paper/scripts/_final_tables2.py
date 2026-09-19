import pathlib

tab = pathlib.Path("paper/tables")

# ---- tab_params: guaranteed-narrow fixed layout ----
p = ("\\begingroup\\setlength{\\tabcolsep}{3pt}\\small"
     "\\begin{tabular}{@{}p{2.3cm}p{1.8cm}p{0.9cm}p{4.4cm}@{}}")
h = "parameter & value & symbol & note \\\\"
rows = [
    "window length & 5 & $W$ & receptive field \\\\",
    "leak & 0.5 & $\\lambda$ & membrane decay \\\\",
    "threshold & 0.5 & $\\theta$ & spike threshold \\\\",
    "query weight & 2.0 & $W_q$ & linear map \\\\",
    "key weight & 1.0 & $W_k$ & linear map \\\\",
    "value weight & 1.0 & $W_v$ & linear map \\\\",
    "inverse temperature & 1.0 & $\\beta$ & softmax \\\\",
    "noise tolerance & 0.0 & $\\varepsilon$ & dead zone \\\\",
    "softmax offset & $10^{-9}$ & --- & denominator \\\\",
    "reset rule & $h_t\\!\\leftarrow\\!0$ & --- & after spike \\\\",
]
content = "\n".join([p, "\\toprule", h, "\\midrule"] + rows +
                    ["\\bottomrule", "\\end{tabular}\\endgroup\n"])
(tab / "tab_params.tex").write_text(content, encoding="utf-8")

# ---- tab_probe1: rebuild rows from the frozen audit md ----
import re
md = (pathlib.Path("audit_v3_amended") / "PER_SEED_AUDIT.md").read_text(
    encoding="utf-8")
pat = re.compile(r"\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*"
                 r"([\d.]+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|")


def esc(s):
    return s.replace("_", "\\_\\allowbreak{}")


rows = []
for line in md.splitlines():
    m = pat.match(line)
    if m:
        seed, model, cond, bal, sh, cov, fin = m.groups()
        rows.append(f"{seed} & {model} & {cond} & {bal} & {esc(cov)} & "
                    f"{esc(fin)} \\\\")
spec = ("\\begingroup\\setlength{\\tabcolsep}{3pt}\\small"
        "\\begin{tabular}{@{}lllcp{2.35cm}p{2.35cm}@{}}")
header = "seed & model & condition & bal3 & coverage status & final "
"status \\\\"
content = "\n".join([spec, "\\toprule", header, "\\midrule"] + rows +
                    ["\\bottomrule", "\\end{tabular}\\endgroup\n"])
(tab / "tab_probe1.tex").write_text(content, encoding="utf-8")
print("params + probe1 rewritten, rows:", len(rows))

# raise hfuzz to 3pt in the preamble
main = pathlib.Path("paper/main.tex")
t = main.read_text(encoding="utf-8")
t = t.replace("\\hfuzz=1.5pt", "\\hfuzz=3pt")
main.write_text(t, encoding="utf-8")
print("hfuzz=3pt")
