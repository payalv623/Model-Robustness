"""Colab-only free CPU continuation. Run after mounting Drive and extracting original bundles."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path('/content/template2_project')
DRIVE=Path('/content/drive/MyDrive/Template2-Colab')
TARGET=Path('/content/template2_cpu_recovery')
SUPPORT=Path(__file__).resolve().parent
assert (DRIVE/'checkpoints/production_v3_cuda/index.json').exists(), 'Original CUDA Drive checkpoint is required'
assert (ROOT/'background_dataset.py').exists(), 'Run the source/code extraction cell first'

def command(args,cwd=None):
    p=subprocess.Popen([str(x) for x in args],cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in p.stdout:print(line,end='',flush=True)
        if p.wait():raise RuntimeError('Command failed: '+str(args[0]))
    except BaseException:
        p.terminate();p.wait();raise

print('FREE CPU RECOVERY: original checkpoints preserved; same 10 classes, 13,500 sources, 90/10 assignments and quality gates.',flush=True)
print('Saved CUDA checkpoint records:',len(json.loads((DRIVE/'checkpoints/production_v3_cuda/index.json').read_text())),flush=True)
command([sys.executable,'-m','pip','install','-q','uv==0.8.22'])
UV=[sys.executable,'-m','uv'];ENV=Path('/content/template2_cpu_env');PY=ENV/'bin/python'
if not PY.exists():command(UV+['venv','--python','3.11',ENV])
command(UV+['pip','install','--python',PY,'torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cpu'])
command(UV+['pip','install','--python',PY,'-r',ROOT/'requirements-colab.txt'])
command([PY,'-c',"import torch;print('CPU environment:',torch.__version__,'threads',torch.get_num_threads());assert torch.ones(1,device='cpu').device.type=='cpu'"])
command([PY,ROOT/'download_checkpoints.py'])
print('Restoring original CUDA checkpoints and verifying archive checksums...',flush=True)
command([PY,'-c',"from colab_checkpoint import restore;from color_dataset import ROOT,load_sources,sha;restore('/content/drive/MyDrive/Template2-Colab/checkpoints',ROOT);rows,_,_=load_sources();assert len(rows)==13500;assert all(sha((ROOT/r['path']).read_bytes())==r['sha256'] for r in rows);print('Verified all 13,500 frozen sources',flush=True)"],ROOT)
command([PY,SUPPORT/'prepare_cpu_recovery.py',ROOT,TARGET,SUPPORT])
os.environ['TEMPLATE2_BACKUP_DIR']=str(DRIVE/'cpu_recovery_checkpoints')
command([PY,'-c',"import os;from colab_checkpoint import restore;from color_dataset import ROOT;restore(os.environ['TEMPLATE2_BACKUP_DIR'],ROOT)"],TARGET)
# Keep the small recovery scripts available after the temporary CPU VM disconnects.
for name in ('start_cpu_recovery.py','prepare_cpu_recovery.py','recovery_provenance.py'):
    destination=DRIVE/'cpu_recovery_code'/name;destination.parent.mkdir(exist_ok=True);shutil.copyfile(SUPPORT/name,destination)
print('Starting CPU continuation. Numerical audit runs after generation; visual quality review remains mandatory.',flush=True)
command([PY,'-u',TARGET/'run_cpu_recovery.py'],TARGET)
