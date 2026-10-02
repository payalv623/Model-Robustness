"""Extract and verify only the frozen classes; never delete the source ZIP.

Run: .venv/bin/python prepare_dataset.py 'archive (1).zip'
Requires Pillow. Original train.X1 and val.X splits are preserved.
"""

import argparse
import csv
import hashlib
import io
import json
import platform
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parent
FROZEN_IDS = (
    'n01440764', 'n01484850', 'n01494475', 'n01531178', 'n01632777',
    'n01665541', 'n01687978', 'n01695060', 'n01749939', 'n01775062',
)
SPLITS = ('train.X1', 'val.X')


def prepare(source):
    metadata_path = ROOT / 'project_metadata' / 'selected_classes.json'
    metadata = json.loads(metadata_path.read_text())
    assert tuple(c['id'] for c in metadata['classes']) == FROZEN_IDS
    target = ROOT / 'archive'
    report_dir = ROOT / 'reports' / 'dataset_preparation'
    report_dir.mkdir(parents=True, exist_ok=True)
    counts, verified, errors = Counter(), Counter(), []
    hashes = defaultdict(list)
    rows = []
    total_bytes = 0
    reused = 0

    with zipfile.ZipFile(source) as archive:
        entries = []
        names = set()
        for entry in archive.infolist():
            parts = PurePosixPath(entry.filename).parts
            if entry.is_dir() or not any(c in parts for c in FROZEN_IDS):
                continue
            if (len(parts) != 3 or parts[0] not in SPLITS
                    or parts[1] not in FROZEN_IDS or '..' in parts
                    or '\\' in entry.filename
                    or Path(parts[2]).suffix.lower() not in ('.jpg', '.jpeg', '.png')):
                raise ValueError(f'Unexpected selected entry: {entry.filename}')
            if entry.filename in names:
                raise ValueError(f'Duplicate ZIP entry: {entry.filename}')
            names.add(entry.filename)
            entries.append(entry)
            counts['/'.join(parts[:2])] += 1
        missing = [f'{s}/{c}' for s in SPLITS for c in FROZEN_IDS if not counts[f'{s}/{c}']]
        if missing:
            raise ValueError(f'Missing source classes/splits: {missing}')
        labels = json.loads(archive.read('Labels.json'))
        selected_labels = {c: labels[c] for c in FROZEN_IDS}

        for index, entry in enumerate(entries, 1):
            destination = target / entry.filename
            row = {'source_member': entry.filename,
                   'path': destination.relative_to(ROOT).as_posix(),
                   'source_split': entry.filename.split('/')[0],
                   'class_id': entry.filename.split('/')[1],
                   'bytes': entry.file_size, 'sha256': '',
                   'width': '', 'height': '', 'mode': '', 'status': 'failed'}
            try:
                data = archive.read(entry)  # zipfile verifies the member CRC.
                digest = hashlib.sha256(data).hexdigest()
                row['sha256'] = digest
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                        raise ValueError('Existing destination differs; refusing to overwrite')
                    reused += 1
                else:
                    temporary = destination.with_suffix(destination.suffix + '.partial')
                    temporary.write_bytes(data)
                    temporary.replace(destination)
                disk_data = destination.read_bytes()
                if hashlib.sha256(disk_data).hexdigest() != digest:
                    raise ValueError('Extracted file does not match ZIP member')
                with Image.open(io.BytesIO(disk_data)) as image:
                    image.verify()
                with Image.open(io.BytesIO(disk_data)) as image:
                    image.load()  # Full decode, not just header validation.
                    image.convert('RGB').load()
                    row.update(width=image.width, height=image.height, mode=image.mode)
                row['status'] = 'verified'
                verified['/'.join(PurePosixPath(entry.filename).parts[:2])] += 1
                hashes[digest].append(entry.filename)
                total_bytes += len(disk_data)
            except Exception as exc:
                errors.append({'file': entry.filename, 'error': str(exc)})
            rows.append(row)
            if index % 500 == 0 or index == len(entries):
                print(f'Checked {index}/{len(entries)}; errors: {len(errors)}', flush=True)

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    cross_split = [paths for paths in duplicates
                   if len({p.split('/')[0] for p in paths}) > 1]
    unexpected = []
    for split in SPLITS:
        actual = {p.relative_to(target).as_posix()
                  for p in (target / split).rglob('*') if p.is_file()}
        expected = {name for name in names if name.startswith(split + '/')}
        unexpected.extend(sorted(actual - expected))
    passed = (not errors and not cross_split and not unexpected and counts == verified)
    report = {
        'status': 'passed' if passed else 'needs_review',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_zip': str(source.resolve()), 'source_zip_bytes': source.stat().st_size,
        'destination': str(target), 'frozen_class_ids': list(FROZEN_IDS),
        'selected_images': len(entries), 'verified_images': sum(verified.values()),
        'verified_bytes': total_bytes, 'reused_files': reused,
        'source_counts': dict(counts), 'verified_counts': dict(verified),
        'errors': errors, 'byte_identical_duplicates': duplicates,
        'cross_split_byte_identical_duplicates': cross_split,
        'unexpected_files': unexpected, 'python': platform.python_version(),
        'pillow': pillow_version,
        'checks': ['ZIP member CRC', 'source-to-output SHA256 equality',
                   'Pillow verify and full RGB decode', 'per-class per-split counts',
                   'byte-identical cross-split duplicates'],
        'limitations': ['Near-duplicate images are not detected.',
                       'Dataset provenance is user-supplied Kaggle ImageNet-100.',
                       'Official source splits are preserved; experiment splits are not assigned.'],
    }
    (report_dir / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    with (report_dir / 'manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if passed:
        (target / 'Labels.json').write_text(json.dumps(selected_labels, indent=2) + '\n')
        for item in metadata['classes']:
            item['name'] = selected_labels[item['id']]
            item['path'] = f"archive/train.X1/{item['id']}"
            item['validation_path'] = f"archive/val.X/{item['id']}"
        metadata_path.write_text(json.dumps(metadata, indent=4) + '\n')
    print(json.dumps({k: report[k] for k in
                     ('status', 'selected_images', 'verified_images', 'verified_bytes', 'errors')}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_zip', type=Path)
    args = parser.parse_args()
    raise SystemExit(prepare(args.source_zip))
