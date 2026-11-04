"""Exercise independent review in a temporary, source-only archive layout."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
repository = here.parent.parent/'Temporal-Attention-Neuron'
verified = json.loads((here/'INDEPENDENT_REVIEW_CHECKS.json').read_text())
with tempfile.TemporaryDirectory(prefix='tan_stage_review_snapshot_') as temporary:
    root = Path(temporary)
    audit = root/'tan_audit/stage_localization'
    source = root/'Temporal-Attention-Neuron'
    audit.mkdir(parents=True)
    for name in [*verified['primary_sha256'], 'independent_review.py']:
        shutil.copy2(here/name, audit/name)
    for name in ['p1_experiments/task_generator.py',
                 'p1_experiments/p1b/models_p1b_v2.py',
                 'p1_experiments/p1b/summary_p1b.json']:
        destination = source/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository/name, destination)
    assert not (source/'.git').exists()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
    completed = subprocess.run([sys.executable, str(audit/'independent_review.py')],
        env=environment, capture_output=True, text=True, check=False)
    (here/'archive_replay_review.log').write_text(completed.stdout+completed.stderr)
    assert completed.returncode == 0, completed.stderr
    replay = json.loads((audit/'INDEPENDENT_REVIEW_CHECKS.json').read_text())
    assert replay['status'] == 'PASS'
    assert replay['repository_status_verified'] is False
    assert replay['repository_status_unchanged'] is None
    assert replay['repository_porcelain'] is None
    assert replay['primary_sha256'] == verified['primary_sha256']
    assert replay['independent_script_sha256'] == verified['independent_script_sha256']
    assert replay['primary_artifacts_unchanged'] is True
    result = {key: replay[key] for key in [
        'status', 'repository_status_verified', 'repository_status_unchanged',
        'repository_status_availability', 'primary_artifacts_unchanged',
        'independent_script_sha256', 'independently_regenerated_histories',
        'query_rows_checked', 'metric_cells_checked', 'paired_bootstrap_contrasts_checked',
        'metric_max_residual', 'bootstrap_max_residual', 'wall_seconds', 'maxrss_kib']}
    result.update(exit_code=completed.returncode,
        scope='Snapshot-layout replay of independent reviewer only; primary experiment was not rerun.',
        temporary_snapshot_removed_after_test=True)
(here/'ARCHIVE_REPLAY_REVIEW.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
