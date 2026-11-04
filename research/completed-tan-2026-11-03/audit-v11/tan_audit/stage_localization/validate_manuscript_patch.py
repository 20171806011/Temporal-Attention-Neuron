"""Read-only apply validation and isolated LaTeX syntax check for unapplied patches."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,tempfile

OUT=Path(__file__).resolve().parent
REPO=OUT.parent.parent/'Temporal-Attention-Neuron'
spec=importlib.util.spec_from_file_location('private_patch_builder',OUT/'prepare_manuscript_patch.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
def run(args,cwd):
    p=subprocess.run(args,cwd=cwd,text=True,capture_output=True)
    return {'argv':args,'cwd':str(cwd),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

before={p:sha(REPO/p) for p in builder.EXPECTED}
checks=[]
for name in ('manuscript_corrections.patch','readme_scope_corrections.patch'):
    checks.append(run(['git','apply','--check','--whitespace=error-all',str(OUT/name)],REPO))
    assert checks[-1]['exit_code']==0,checks[-1]
fragment=OUT/'manuscript_preview'/'p1b_correction_fragment.tex'
fragment.write_text('\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n\\begin{document}\n'+builder.MODEL_TEXT+'\n'+builder.FLOOR_TEXT+'\n'+builder.SPIKE_TEXT+'\n\\end{document}\n')
temp=Path(tempfile.mkdtemp(prefix='tan_patch_latex_',dir='/tmp'))
latex=run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={temp}',str(fragment)],OUT)
# Keep full command output as evidence, but avoid noisy display.
(OUT/'manuscript_latex_validation.log').write_text(latex['stdout']+'\nSTDERR\n'+latex['stderr'])
assert latex['exit_code']==0,latex
status=run(['git','status','--short'],REPO)
after={p:sha(REPO/p) for p in builder.EXPECTED}
assert before==after==builder.EXPECTED
assert status['exit_code']==0 and not status['stdout'],status
manifest=json.loads((OUT/'PATCH_MANIFEST.json').read_text())
result={'frozen_commit':manifest['frozen_commit'],'patch_sha256':manifest['patches'],
 'read_only_apply_checks':checks,'repository_status':status,'source_hashes_before':before,
 'source_hashes_after':after,'repository_bytes_unchanged':True,
 'latex_fragment_check':{'argv':latex['argv'],'exit_code':latex['exit_code'],
 'source_sha256':sha(fragment),'log':'manuscript_latex_validation.log',
 'pdf_sha256':sha(temp/'p1b_correction_fragment.pdf'),
 'scope':'Isolated replacement paragraphs compile; not a full manuscript build or release layout check.'},
 'historical_value_checks':'Builder asserts the selected historical count, coefficient, accuracy, CI and decision-label strings remain. Historical experiments are not rerun by this documentation task.'}
(OUT/'PATCH_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'apply_exit_codes':[x['exit_code'] for x in checks],'latex_fragment_exit_code':latex['exit_code'],'repository_clean':True,'patch_sha256':manifest['patches']},indent=2))
