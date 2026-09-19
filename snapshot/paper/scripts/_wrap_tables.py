import pathlib

tab = pathlib.Path("paper/tables")


def wrap(name):
    f = tab / name
    t = f.read_text(encoding="utf-8")
    start = "\\begin{tabular}"
    end = "\\end{tabular}"
    assert t.startswith(start), name
    head, _, rest = t.partition(start)
    assert rest.rstrip().endswith(end), name
    body = rest[: rest.rindex(end)] + end
    new = (head + "\\begingroup\\setlength{\\tabcolsep}{4pt}" + start +
           body + "\\endgroup\n")
    f.write_text(new, encoding="utf-8")
    print("wrapped", name)


for n in ["tab_params.tex", "tab_phase1.tex", "tab_phase2.tex",
          "tab_probe1.tex", "tab_probe2.tex", "tab_claims.tex"]:
    wrap(n)
