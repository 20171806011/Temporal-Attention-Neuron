import pathlib

files = list(pathlib.Path("paper").rglob("*.tex"))
for f in files:
    t = f.read_text(encoding="utf-8")
    orig = t
    # normalize any doubled backslash before underscore
    t = t.replace("\\\\_", "\\_")
    # stray artifacts from the earlier quote conversion
    t = t.replace("}audit\\_v3\\_amended/MACHINE\\_READABLE\\_AUDIT.json'.",
                  "(\\texttt{audit\\_v3\\_amended/MACHINE\\_READABLE\\_AUDIT.json}).")
    t = t.replace("}audit\\_v3\\_amended/MACHINE_READABLE_AUDIT.json'.",
                  "(\\texttt{audit\\_v3\\_amended/MACHINE\\_READABLE\\_AUDIT.json}).")
    if t != orig:
        f.write_text(t, encoding="utf-8")
        print("fixed", f.name)
