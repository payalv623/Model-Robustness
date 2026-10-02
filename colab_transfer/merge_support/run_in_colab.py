# Run after mounting Drive, extracting original bundles, and creating PY (Python 3.11 + pinned NumPy/Pillow).
from pathlib import Path
import os,sys,hashlib,shutil,subprocess,time
DRIVE=Path('/content/drive/MyDrive/Template2-Colab');CUDA=Path('/content/template2_project')
MAC=Path('/content/template2_mac_remainder');MERGED=Path('/content/template2_merged');SUPPORT=Path('/content/template2_merge_support')
assert DRIVE.is_dir(), 'Mount the project Drive folder first'
sys.path.insert(0,str(CUDA))
from colab_checkpoint import extract,restore
restore(DRIVE/'checkpoints',CUDA)
for name,checksum,dest in [
 ('template2_mac_remainder.zip','8621c14c92659a3fc816e0b422bd91451d210e6ef43172d9ac98612f65853217',MAC),
 ('template2_merge_code.zip','a7b6bbcdaf8907cba178ff08a355c91f08bb0ed1579459ce2de9a4ac00f341f1',SUPPORT)]:
 source=DRIVE/name;deadline=time.monotonic()+1800
 while not source.exists():
  if time.monotonic()>deadline:raise TimeoutError('Complete Drive upload: '+name)
  print('Waiting for completed Drive upload:',name,flush=True);time.sleep(30)
 local=Path('/content')/name;shutil.copyfile(source,local)
 with local.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==checksum, 'Archive hash mismatch'
 dest.mkdir(exist_ok=True);extract(local,dest);print('Verified and extracted',name,flush=True)
env={**os.environ,'TEMPLATE2_BACKUP_DIR':str(DRIVE/'merged_checkpoints')}
report=MERGED/'reports/background_bias_current/production_v3_merged_cuda_mps';report.mkdir(parents=True,exist_ok=True)
with (report/'merge.log').open('a',buffering=1) as log:
 worker=subprocess.Popen([PY,'-u',str(SUPPORT/'merge_results.py'),str(CUDA),str(MAC),str(MERGED)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
 for line in worker.stdout:print(line,end='',flush=True);log.write(line)
 if worker.wait():raise RuntimeError('Merge/audit failed; original inputs remain intact. Inspect the output above.')
print('Results saved to:',DRIVE/'merged_checkpoints/production_v3_merged_cuda_mps',flush=True)
