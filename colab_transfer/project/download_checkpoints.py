"""Download the original pinned weights and verify every recorded SHA-256."""
import json
from pathlib import Path
import urllib.request
from colab_checkpoint import digest
ROOT=Path(__file__).resolve().parent

def fetch(url,path,expected):
    if path.exists() and digest(path)==expected:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    urllib.request.urlretrieve(url,temp)
    if digest(temp)!=expected:
        raise ValueError(f'Download digest mismatch: {path.name}')
    temp.replace(path)
    print('Verified',path.name,flush=True)

if __name__=='__main__':
    sam=json.loads((ROOT/'project_metadata/sam_checkpoint.json').read_text())
    fetch(sam['source_url'],ROOT/'checkpoints/sam_vit_b_01ec64.pth',sam['sha256'])
    dino=json.loads((ROOT/'project_metadata/grounding_dino_checkpoint.json').read_text())
    for f in dino['files']:
        fetch(f['url'],ROOT/'checkpoints/grounding-dino-tiny'/f['file'],f['sha256'])
