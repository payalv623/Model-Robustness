"""Build the portable project and only the frozen 13,500 source images."""
from pathlib import Path
import csv, hashlib, json, zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

code=HERE/'template2_colab_code.zip'
with zipfile.ZipFile(code,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted((HERE/'project').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.csv','.txt'):
            z.write(p,p.relative_to(HERE/'project'))
rows=list(csv.DictReader((ROOT/'project_metadata/source_splits.csv').open()))
assert len(rows)==13500 and len({r['class_id'] for r in rows})==10
source=HERE/'template2_imagenet10_sources.zip'
if not source.exists():
    tmp=source.with_suffix('.zip.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED) as z:
        for r in rows:
            data=(ROOT/r['path']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError(r['path'])
            z.writestr(r['path'],data)
    tmp.replace(source)
with zipfile.ZipFile(source) as z:
    assert set(z.namelist())=={r['path'] for r in rows}
    for r in rows:
        if hashlib.sha256(z.read(r['path'])).hexdigest()!=r['sha256']:raise ValueError(r['path'])
manifest={'source_images':13500,'classes':10,'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (code,source)}}
(HERE/'transfer_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
