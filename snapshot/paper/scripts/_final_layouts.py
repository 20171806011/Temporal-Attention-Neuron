import pathlib

tab = pathlib.Path("paper/tables")
main = pathlib.Path("paper/main.tex")

# phase2: fixed p-column layout with centering
spec2 = ("\\begingroup\\setlength{\\tabcolsep}{4pt}\\small"
         "\\begin{tabular}{>{\\centering\\arraybackslash}p{0.9cm}"
         ">{\\centering\\arraybackslash}p{0.8cm}"
         ">{\\centering\\arraybackslash}p{1.3cm}"
         ">{\\centering\\arraybackslash}p{1.3cm}"
         "p{2.6cm}"
         ">{\\centering\\arraybackslash}p{1.0cm}"
         ">{\\centering\\arraybackslash}p{1.5cm}"
         ">{\\centering\\arraybackslash}p{1.1cm}}")
hdr2 = ("Model & frame & $d_D^{ev}$ & $\\Delta d_D$ & raw/strat/resid & "
        "logTr & $\\tau_{e}$ & $\\tau_{1/2}$ \\\\")
rows2 = [
    "B1 & zC & 1.674 & 0.674 & 1.674/1.732/1.747 & 2.66 & 6.4 & 4 \\\\",
    "B3 & zC & 1.458 & 0.458 & 1.458/1.495/1.490 & 2.38 & 7.9 & 8 \\\\",
    "B3 & zF & 1.484 & 0.484 & 1.484/1.509/1.498 & 2.48 & 8.5 & 3 \\\\",
    "B4 & zC & 1.589 & 0.589 & 1.589/1.701/1.712 & 2.67 & 6.3 & 2 \\\\",
    "B4 & zF & 2.089 & 1.089 & 2.089/2.250/2.280 & 3.58 & 5.4 & 8 \\\\",
    "C0 & zU & 1.055 & -- & 1.055/1.051/1.045 & 1.96 & -- & -- \\\\",
]
content = "\n".join([spec2, "\\toprule", hdr2, "\\midrule"] + rows2 +
                    ["\\bottomrule", "\\end{tabular}\\endgroup\n"])
(tab / "tab_phase2.tex").write_text(content, encoding="utf-8")

# params: reduced fixed widths
spec1 = ("\\begingroup\\setlength{\\tabcolsep}{4pt}\\small"
         "\\begin{tabular}{p{2.6cm}cp{1.1cm}p{5.6cm}}")
hdr1 = "parameter & value & symbol & note \\\\"
rows1 = [
    "window length & 5 & $W$ & receptive field \\\\",
    "leak & 0.5 & $\\lambda$ & membrane decay \\\\",
    "threshold & 0.5 & $\\theta$ & spike threshold \\\\",
    "query weight & 2.0 & $W_q$ & linear map \\\\",
    "key weight & 1.0 & $W_k$ & linear map \\\\",
    "value weight & 1.0 & $W_v$ & linear map \\\\",
    "inverse temperature & 1.0 & $\\beta$ & softmax \\\\",
    "noise tolerance & 0.0 & $\\varepsilon$ & dead zone \\\\",
    "softmax offset & $10^{-9}$ & --- & denominator \\\\",
    "reset & $h_t \\leftarrow 0$ & --- & after spike \\\\",
]
content = "\n".join([spec1, "\\toprule", hdr1, "\\midrule"] + rows1 +
                    ["\\bottomrule", "\\end{tabular}\\endgroup\n"])
(tab / "tab_params.tex").write_text(content, encoding="utf-8")

# main.tex: allow breaks after slashes in prose lines (not in includegraphics)
t = main.read_text(encoding="utf-8")
out = []
for line in t.splitlines(keepends=True):
    if any(s in line for s in ("\\includegraphics", "\\input",
                               "\\graphicspath")):
        out.append(line)
        continue
    if "/" in line and "\\allowbreak" not in line:
        line = line.replace("/", "/\\allowbreak{}")
    out.append(line)
main.write_text("".join(out), encoding="utf-8")
print("phase2/params rebuilt; main.tex slash breaks added")
