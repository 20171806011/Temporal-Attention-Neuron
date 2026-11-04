from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parent
REF='85213f0d417d54e4f48b301ed1c0e4200ef3c421'
base=f'https://raw.githubusercontent.com/20171806011/Temporal-Attention-Neuron/{REF}/'
files={'p1_experiments/p1b/models_p1b_v2.py':'1193e5406b93e8b42b3290d7e2b9428beb5916e4', 'p1_experiments/p1b/summary_p1b.json':'ef170c06a65eaac4265fd22eaf8b30008fccd230', 'p1_experiments/p1b/P1B_FINAL_SCIENTIFIC_CLOSURE_REPORT.md':'72a6fd3ff51c4dbaf44f09dba8cb070bcc990140', 'LICENSE':None}
rows=[]
for path,want in files.items():
    data=urllib.request.urlopen(base+path,timeout=25).read()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if want and blob!=want: raise RuntimeError((path,blob,want))
    dest=ROOT/'source'/Path(path).name
    if dest.exists() and dest.read_bytes()!=data: raise RuntimeError('Refusing overwrite with different source')
    dest.write_bytes(data)
    rows.append({'path':path,'url':base+path,'bytes':len(data),'git_blob':blob,'expected_blob':want,'sha256':hashlib.sha256(data).hexdigest()})
(ROOT/'source_manifest.json').write_text(json.dumps({'commit':REF,'files':rows},indent=2)+'\n')
print(json.dumps(rows,indent=2))
