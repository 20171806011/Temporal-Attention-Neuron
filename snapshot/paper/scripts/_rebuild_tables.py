import pathlib

tab = pathlib.Path("paper/tables")


def rebuild(name, spec, header):
    f = tab / name
    lines = f.read_text(encoding="utf-8").splitlines()
    rows = [l for l in lines if l.strip() and "\\" in l and
            not l.strip().startswith("\\") and l.strip() != "\\bottomrule"]
    # safer: take rows strictly between midrule and bottomrule
    out = []
    # keep any content rows: rows start with digits or letters & contain &
    data = []
    started = False
    for l in lines:
        if l.strip() == "\\midrule":
            started = True
            continue
        if started and l.strip() == "\\bottomrule":
            break
        if started and l.strip():
            data.append(l)
    out.append("\\small" + spec.replace("\\begin{tabular}", "\\begin{tabular}") if False else spec)
    out.append("\\toprule")
    out.append(header)
    out.append("\\midrule")
    out.extend(data)
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    f.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(name, "rows:", len(data))


rebuild("tab_params.tex",
        "\\begin{tabular}{lp{1.7cm}cp{7.0cm}}",
        "parameter & value & symbol & note \\\\")

rebuild("tab_phase2.tex",
        "\\begin{tabular}{llccp{2.1cm}ccc}",
        "Model & frame & $d_D^{ev}$ & $\\Delta d_D$ & raw/strat/resid & "
        "logTr & $\\tau_{e\\text{-fold}}$ & $\\tau_{1/2}$ \\\\")

rebuild("tab_probe2.tex",
        "\\begin{tabular}{p{4.7cm}p{5.3cm}p{4.9cm}}",
        "audit & measured & requirement / gate \\\\")

rebuild("tab_claims.tex",
        "\\begin{tabular}{p{7.7cm}p{4.3cm}p{2.5cm}}",
        "claim & status & evidence \\\\")

rebuild("tab_phase1.tex",
        "\\begin{tabular}{lcccccc}",
        "Model & $n$ pairs & mean $K$ & 95\\% CI & median $K$ & "
        "\\% $K{>}10^3$ & post-reset viol. \\% \\\\")

rebuild("tab_probe1.tex",
        "\\begin{tabular}{lllcp{2.2cm}p{2.2cm}}",
        "seed & model & condition & bal3 & coverage status & final "
        "status \\\\")
print("done")
