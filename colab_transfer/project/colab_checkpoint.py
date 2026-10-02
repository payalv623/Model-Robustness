"""Atomic incremental archives on Drive; computation stays on VM-local disk."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def extract(archive, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root) or item.filename.startswith('/'):
                raise ValueError('Unsafe archive path')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlink rejected')
        z.extractall(root)


class Backup:
    def __init__(self, root, output):
        self.root, self.output = Path(root), Path(output)
        destination = os.environ.get('TEMPLATE2_BACKUP_DIR')
        self.dest = Path(destination) / self.output.name if destination else None
        self.saved = {}
        if self.dest:
            self.dest.mkdir(parents=True, exist_ok=True)
            index = self.dest / 'index.json'
            if index.exists():
                self.saved = json.loads(index.read_text())

    def save(self, force=False):
        if not self.dest:
            return
        changed=[]
        for p in sorted((self.output/'records').glob('*.json')):
            h=digest(p)
            if self.saved.get(p.name)!=h:
                changed.append((p,h))
        if len(changed)<100 and not force:
            return
        if not changed and not force:
            return
        files=set()
        for p,h in changed:
            record=json.loads(p.read_text());files.add(p)
            for relative, expected in record.get('file_hashes',{}).items():
                f=self.output/relative
                if not f.resolve().is_relative_to(self.output.resolve()) or digest(f)!=expected:
                    raise ValueError('Cannot back up corrupted or unsafe output')
                files.add(f)
        files.update(p for p in self.output.glob('*.json'))
        for folder in [self.output/'candidate_manifests',self.output/'manifests',self.root/'reports/background_bias_current'/self.output.name]:
            if folder.exists():
                files.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.html','.png','.log'))
        sequence=len(list(self.dest.glob('chunk-*.zip')))+1
        name=f'chunk-{sequence:06d}.zip'
        temporary=self.dest/(name+'.partial')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_STORED) as z:
            for f in sorted(files):
                z.write(f,f.relative_to(self.root))
        checksum=digest(temporary)
        temporary.replace(self.dest/name)
        (self.dest/(name+'.sha256')).write_text(checksum+'\n')
        self.saved.update({p.name:h for p,h in changed})
        index=self.dest/'index.json.partial'; index.write_text(json.dumps(self.saved,sort_keys=True));index.replace(self.dest/'index.json')
        print(f'Drive checkpoint: {name} ({len(changed)} records)',flush=True)


def restore(destination, root):
    destination=Path(destination)
    for folder in sorted(destination.glob('*_cuda')):
        for p in sorted(folder.glob('chunk-*.zip')):
            h=p.with_suffix('.zip.sha256')
            if not h.exists():
                raise ValueError(f'Incomplete checkpoint: {p}; preserve it and investigate before resuming')
            if digest(p)!=h.read_text().strip():
                raise ValueError(f'Checkpoint hash mismatch: {p}')
            extract(p,root)
    print('Saved CUDA checkpoints restored. Every source/output hash is rechecked on resume.')
