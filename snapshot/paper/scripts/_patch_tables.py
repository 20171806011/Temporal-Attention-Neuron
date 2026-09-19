import pathlib

p = pathlib.Path("paper/scripts/make_figs_tables.py")
t = p.read_text(encoding="utf-8")

# tab_params -> small + p column layout
t = t.replace(
    'out = ["\\\\begin{tabular}{lccl}",',
    'out = ["\\\\small\\\\begin{tabular}{lp{1.6cm}cp{7.0cm}}",',
)
# tab_phase1 -> small
t = t.replace(
    'out = ["\\\\begin{tabular}{lcccccc}",',
    'out = ["\\\\small\\\\begin{tabular}{lcccccc}",',
)
# tab_phase2 -> small + p column for the amplitude-control column
t = t.replace(
    'out = ["\\\\begin{tabular}{llcccccc}",',
    'out = ["\\\\small\\\\begin{tabular}{llccp{2.1cm}ccc}",',
)
t = t.replace(
    '"Model & frame & $d_D^{event}$ & $\\\\Delta d_D$ & raw/strat/'
    'resid $d_D$ & event logTr (nats) & $\\\\tau_{e\\\\text{-fold}}$ & '
    '$\\\\tau_{1/2}$ \\\\\\\\",',
    '"Model & frame & $d_D^{ev}$ & $\\\\Delta d_D$ & raw/strat/'
    'resid & logTr & $\\\\tau_{e\\\\text{-fold}}$ & $\\\\tau_{1/2}$ '
    '\\\\\\\\",',
)
# tab_probe2 -> p columns
t = t.replace(
    'out = ["\\\\begin{tabular}{lcc}",',
    'out = ["\\\\small\\\\begin{tabular}{p{4.6cm}p{5.4cm}p{5.0cm}}",',
)
# tab_claims -> p columns
t = t.replace(
    'out = ["\\\\begin{tabular}{lcc}",',
    'out = ["\\\\small\\\\begin{tabular}{p{7.6cm}p{4.4cm}p{2.6cm}}",',
)
# probe1 header for three p-adapted layouts must stay at 3 columns: check
t = t.replace(
    'out = ["\\\\begin{tabular}{lcccccc}",',
    'out = ["\\\\small\\\\begin{tabular}{lcccccc}",',
)
p.write_text(t, encoding="utf-8")
print("generator patched")
