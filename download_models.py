"""Restore the project's pinned model files without importing PyTorch or SAM."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(url, path, expected):
    if path.is_file() and digest(path) == expected:
        print('OK', path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    urllib.request.urlretrieve(url, temporary)
    if digest(temporary) != expected:
        raise ValueError(f'Download digest mismatch: {path.name}; existing file preserved')
    temporary.replace(path)
    print('Verified', path.name)


def model_files(root=ROOT):
    sam = json.loads((root / 'project_metadata/sam_checkpoint.json').read_text())
    yield sam['source_url'], root / 'checkpoints/sam_vit_b_01ec64.pth', sam['sha256']
    dino = json.loads((root / 'project_metadata/grounding_dino_checkpoint.json').read_text())
    for item in dino['files']:
        yield item['url'], root / 'checkpoints/grounding-dino-tiny' / item['file'], item['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify local files without downloading')
    args = parser.parse_args()
    failed = 0
    for url, path, expected in model_files():
        if args.check_only:
            ok = path.is_file() and digest(path) == expected
            print(('OK' if ok else 'MISSING/CHANGED'), path.relative_to(ROOT))
            failed += not ok
        else:
            fetch(url, path, expected)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
