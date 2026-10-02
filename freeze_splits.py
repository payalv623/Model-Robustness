"""Freeze source-image splits once for every template, without copying images."""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42


def main():
    audit_dir = ROOT / 'reports' / 'dataset_preparation'
    verification = json.loads((audit_dir / 'verification.json').read_text())
    if verification['status'] != 'passed':
        raise ValueError('Dataset verification must pass before freezing splits')
    manifest_path = audit_dir / 'manifest.csv'
    with manifest_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    classes = json.loads((ROOT / 'project_metadata' / 'selected_classes.json').read_text())['classes']
    if len(rows) != verification['verified_images'] or any(r['status'] != 'verified' for r in rows):
        raise ValueError('Manifest does not match verification report')
    output = []
    for item in classes:
        class_rows = [r for r in rows if r['class_id'] == item['id']]
        training = [r for r in class_rows if r['source_split'] == 'train.X1']
        heldout = [r for r in class_rows if r['source_split'] == 'val.X']
        if not training or not heldout:
            raise ValueError(f"Missing source split for {item['id']}")
        # SHA256 ordering avoids dependence on filesystem order or RNG versions.
        training.sort(key=lambda r: hashlib.sha256(f"{SEED}|{r['path']}".encode()).hexdigest())
        n_validation = round(len(training) * 0.2)
        for split, selected in (
            ('train', training[n_validation:]),
            ('validation', training[:n_validation]),
            ('test', heldout),
        ):
            for row in sorted(selected, key=lambda r: r['path']):
                output.append({
                    'path': row['path'], 'sha256': row['sha256'],
                    'class_id': item['id'], 'class_index': item['index'],
                    'group': 'A' if item['index'] < 5 else 'B',
                    'split': split, 'source_split': row['source_split'],
                })
    if len(output) != len(rows) or {r['path'] for r in output} != {r['path'] for r in rows}:
        raise ValueError('Split assignment must cover every source image exactly once')
    digest_splits = {}
    for row in output:
        prior = digest_splits.setdefault(row['sha256'], row['split'])
        if prior != row['split']:
            raise ValueError(f"Identical image appears in different splits: {row['path']}")
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(output[0]))
    writer.writeheader()
    writer.writerows(output)
    content = buffer.getvalue().encode()
    destination = ROOT / 'project_metadata' / 'source_splits.csv'
    if destination.exists() and destination.read_bytes() != content:
        raise FileExistsError('Frozen splits differ; refusing to overwrite')
    destination.write_bytes(content)
    config = {
        'seed': SEED,
        'assignment': 'SHA256 ordering of seed|project-relative source path within each class',
        'training_source': '80% of each original train.X1 class',
        'validation_source': '20% of each original train.X1 class',
        'test_source': 'All original val.X images, reserved for final evaluation',
        'counts': dict(Counter(row['split'] for row in output)),
        'per_class_counts': {item['id']: dict(Counter(
            row['split'] for row in output if row['class_id'] == item['id']
        )) for item in classes},
        'split_manifest_sha256': hashlib.sha256(content).hexdigest(),
        'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'near_duplicate_audit': 'not performed',
        'pretraining_overlap': 'unknown; held out from this project, not necessarily from model pretraining',
    }
    (ROOT / 'project_metadata' / 'split_config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(config['counts'], indent=2))


if __name__ == '__main__':
    main()
