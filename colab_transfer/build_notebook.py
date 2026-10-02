from pathlib import Path
import json, textwrap, ast
H=Path(__file__).resolve().parent
m=json.loads((H/'transfer_manifest.json').read_text())
cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip()+'\n'})
def code(s):
 s=textwrap.dedent(s).strip()+'\n';ast.parse(s)
 cells.append({'cell_type':'code','metadata':{},'source':s,'execution_count':None,'outputs':[]})
md('''# Template 2 — background bias, 10 frozen classes
This notebook runs the class detector and SAM on NVIDIA CUDA, preserving the frozen 13,500 images, train/validation/test splits, seed 42, 90/10 assignments, annotations and quality gates.

**Before starting:** select Runtime → Change runtime type → T4 GPU. Upload `template2_colab_code.zip` and `template2_imagenet10_sources.zip` to `My Drive/Template2-Colab`.

Run cells in order. First a 100-image benchmark and pixel audit run; then full generation. Google Drive authorization is needed for inputs and resumable checkpoints. No model training starts here. Generated masks remain pending visual review.

CUDA output is separate from the running Mac/MPS output. After a runtime reset, rerun these cells with the same bundles to restore saved CUDA checkpoints. Up to 99 newly generated images may need reprocessing after an abrupt disconnection.''')
code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
DRIVE = Path('/content/drive/MyDrive/Template2-Colab')
assert DRIVE.is_dir(), 'Create My Drive/Template2-Colab and upload the two ZIP files.'
''')
code('''import hashlib, json, os, shutil, subprocess, sys, zipfile, time
ROOT = Path('/content/template2_project')
ROOT.mkdir(exist_ok=True)
def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def safe_extract(path, root):
    with zipfile.ZipFile(path) as z:
        for member in z.infolist():
            assert (root/member.filename).resolve().is_relative_to(root.resolve()), 'Unsafe archive path'
            assert (member.external_attr >> 16) & 0o170000 != 0o120000, 'Symlink rejected'
        z.extractall(root)
'''+f'EXPECTED = {repr(m["files"])}\n'+'''
for name, info in EXPECTED.items():
    source=DRIVE/name
    deadline=time.monotonic()+3600
    while not source.exists():
        if time.monotonic()>deadline: raise TimeoutError(f'Upload still missing after one hour: {source}. Finish the Drive upload and rerun this cell.')
        print(f'Waiting for Drive upload: {name}',flush=True)
        time.sleep(30)
    local=Path('/content')/name
    if not local.exists() or digest(local)!=info['sha256']:
        shutil.copyfile(source,local)
    assert local.stat().st_size==info['bytes'] and digest(local)==info['sha256'], f'Bundle damaged: {name}'
    safe_extract(local,ROOT)
print('Both bundles verified and extracted to local VM storage.')
''')
md('''## Install the pinned environment
An isolated Python 3.11 environment keeps Colab's preinstalled packages untouched. Weights download from their original official URLs and are checked against the recorded SHA-256 digests.''')
code('''subprocess.run([sys.executable,'-m','pip','install','-q','uv==0.8.22'],check=True)
UV=[sys.executable,'-m','uv']
ENV=Path('/content/template2_env')
if not (ENV/'bin/python').exists():
    subprocess.run(UV+['venv','--python','3.11',str(ENV)],check=True)
PY=str(ENV/'bin/python')
subprocess.run(UV+['pip','install','--python',PY,'torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu124'],check=True)
subprocess.run(UV+['pip','install','--python',PY,'-r',str(ROOT/'requirements-colab.txt')],check=True)
subprocess.run([PY,'-c',"import torch; assert torch.cuda.is_available(), 'Select a T4 GPU runtime'; print(torch.__version__,torch.cuda.get_device_name(0))"],check=True)
subprocess.run([PY,'download_checkpoints.py'],cwd=ROOT,check=True)
''')
md('''## Restore saved results and validate the frozen sources
All source hashes are checked. Resume only accepts records whose configuration and output hashes match. No images are dropped because they are difficult to segment.''')
code('''os.environ['TEMPLATE2_BACKUP_DIR']=str(DRIVE/'checkpoints')
subprocess.run([PY,'-c',"import os; from colab_checkpoint import restore; from color_dataset import ROOT,load_sources,sha; restore(os.environ['TEMPLATE2_BACKUP_DIR'],ROOT); rows,_,_=load_sources(); assert len(rows)==13500; assert all(sha((ROOT/r['path']).read_bytes())==r['sha256'] for r in rows); print('Verified all 13,500 source images in 10 frozen classes')"],cwd=ROOT,check=True)
''')
md('''## Benchmark 100 images (10 from each class)
Both model-device fields must say `cuda:0`, and the independent pixel audit must pass. The printed runtime estimate excludes setup, transfer and manual mask review. A resumed benchmark is not a fresh wall-time measurement.''')
code('''def run_job(full=False):
    name='production_v3_cuda' if full else 'refined_pilot_v3_cuda'
    report=ROOT/'reports/background_bias_current'/name
    report.mkdir(parents=True,exist_ok=True)
    args=[PY,'-u','colab_run.py']+(['--full'] if full else [])
    with (report/'generation.log').open('a',buffering=1) as log:
        process=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='');log.write(line)
            if process.wait()!=0: raise RuntimeError('Worker stopped; see the log above. Saved checkpoints are retained.')
        except BaseException:
            if process.poll() is None:
                process.terminate();process.wait(timeout=30)
            raise
run_job(False)
''')
code('''from IPython.display import display, Image
pilot=ROOT/'reports/background_bias_current/refined_pilot_v3_cuda'
print((pilot/'timing.json').read_text())
display(Image(filename=str(pilot/'overview_0.png')))
print('Inspect remaining overview_1.png through overview_9.png in the pilot report folder. Numeric success is not semantic approval.')
''')
md('''## Generate all Template 2 candidates
This retains all 13,500 sources and writes original, nature, urban and neutral variants. Candidate manifests contain 108,000 rows across the eight conditions. Checkpoints go to Drive every 100 processed images and on normal completion/errors. The final audit checks pixels, source hashes, ratios and alignment with Template 1. **Flagged masks and the fixed 150-image visual sample still require review before training manifests are released.**''')
code('''pilot_audit=json.loads((pilot/'audit.json').read_text())
assert pilot_audit['status']=='numeric_audit_passed' and pilot_audit['source_images_verified']==100
run_job(True)
''')
code('''report=ROOT/'reports/background_bias_current/production_v3_cuda'
result=json.loads((report/'audit.json').read_text())
print(json.dumps({k:v for k,v in result.items() if k not in ('unresolved_sources','issues')},indent=2))
print('Persistent results:',DRIVE/'checkpoints/production_v3_cuda')
print('Review queue:',report/'review_queue.json')
print('Required visual sample:',report/'required_visual_sample.json')
''')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Template2_Colab.ipynb','provenance':[]},'kernelspec':{'display_name':'Python 3','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'},'cells':cells}
for i,c in enumerate(cells):c['id']=f'template2-{i:02d}'
(H/'Template2_Colab.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
print('Notebook compiled:',len(cells),'cells')
