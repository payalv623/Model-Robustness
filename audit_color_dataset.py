"""Full completeness, provenance, assignment and pixel audit for Color outputs."""

import csv
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction

import numpy as np
from PIL import Image, ImageOps

from color_dataset import (ROOT, IDS, SPLITS, VARIANTS, MODES, FIELDS, load_sources,
                           make_config, output_path, plan_conditions, manifest_rows,
                           csv_bytes, sha, save_json, ensure_output_location)


def audit(root=ROOT, output=None, report_dir=None, workers=4):
    root = root.resolve()
    output = ensure_output_location(root, output or root / 'color_bias_dataset')
    report_dir = report_dir or root / 'reports/color_bias_current'
    if not (output / 'config.json').is_file():
        raise ValueError('Color dataset has no generation configuration')
    save_json(output / 'status.json', {'status': 'auditing', 'audit_passed': False})
    try:
        return _audit(root, output, report_dir, workers)
    except Exception as exc:
        save_json(output / 'status.json', {'status': 'audit_failed', 'audit_passed': False, 'error': str(exc)})
        raise


def _audit(root, output, report_dir, workers):
    rows, metadata, split_config = load_sources(root)
    config = json.loads((output / 'config.json').read_text())
    if config != make_config(root, config['correlation'], config['seed']):
        raise ValueError('Dataset config or frozen input hashes changed')
    generation = json.loads((output / 'generation.json').read_text())
    issues = []
    total_issues = 0

    def issue(message):
        nonlocal total_issues
        total_issues += 1
        if len(issues) < 200:
            issues.append(message)

    expected_paths = {output_path(row, variant).as_posix() for row in rows for variant in VARIANTS}
    actual_paths = {p.relative_to(output).as_posix() for p in (output / 'images').rglob('*') if p.is_file()}
    for name in sorted(expected_paths - actual_paths):
        issue(f'Missing output: {name}')
    for name in sorted(actual_paths - expected_paths):
        issue(f'Unexpected output: {name}')
    if len(expected_paths) != 4 * len(rows):
        issue('Output path collision')

    def check_source(row):
        errors, digests, checked = [], {}, 0
        source = root / row['path']
        try:
            data = source.read_bytes()
            if sha(data) != row['sha256']:
                raise ValueError('Source SHA256 mismatch')
            # Independent implementation: do not call the generator renderer.
            with Image.open(io.BytesIO(data)) as image:
                original = image.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
            base = np.asarray(original).astype(np.float32)
            red = base * np.float32(0.70)
            blue = base * np.float32(0.70)
            red[:, :, 0] += np.float32(76.5)
            blue[:, :, 2] += np.float32(76.5)
            expected = {
                'original': np.asarray(original),
                'red': np.clip(red, 0, 255).astype(np.uint8),
                'blue': np.clip(blue, 0, 255).astype(np.uint8),
                'grayscale': np.asarray(ImageOps.grayscale(original).convert('RGB')),
            }
            for variant in VARIANTS:
                relative = output_path(row, variant).as_posix()
                path = output / relative
                if not path.is_file():
                    continue  # Already counted by the completeness check.
                try:
                    file_data = path.read_bytes()
                    digests[relative] = sha(file_data)
                    with Image.open(io.BytesIO(file_data)) as image:
                        image.verify()
                    with Image.open(io.BytesIO(file_data)) as image:
                        if image.mode != 'RGB' or image.size != (224, 224) or image.format != 'PNG':
                            raise ValueError('Expected 224x224 RGB PNG')
                        pixels = np.asarray(image)
                        if not np.array_equal(pixels, expected[variant]):
                            raise ValueError('Pixel values differ from independently computed transformation')
                    checked += 1
                except Exception as exc:
                    errors.append(f'{relative}: {exc}')
        except Exception as exc:
            errors.append(f'{source}: {exc}')
        return digests, checked, errors

    hashes, checked = {}, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (digests, count, errors) in enumerate(pool.map(check_source, rows), 1):
            hashes.update(digests)
            checked += count
            for error in errors:
                issue(error)
            if index % 500 == 0 or index == len(rows):
                print(f'Audited {index}/{len(rows)} sources; pixel-verified PNGs={checked}; issues={total_issues}', flush=True)
    # Keep expected manifest construction possible when a PNG is missing.
    for name in expected_paths:
        hashes.setdefault(name, 'MISSING')
    assignments = plan_conditions(rows, config)
    distributions, manifest_hashes = {}, {}
    manifest_count = 0
    ratio = Fraction(config['correlation'])
    expected_manifests = {f'manifests/{split}/{mode}.csv' for split in SPLITS for mode in MODES}
    actual_manifests = {p.relative_to(output).as_posix() for p in (output / 'manifests').rglob('*') if p.is_file()}
    if actual_manifests != expected_manifests:
        issue('Missing or unexpected condition manifest files')
    for split in SPLITS:
        for mode in MODES:
            name = f'manifests/{split}/{mode}.csv'
            path = output / name
            expected_rows = manifest_rows(rows, assignments, hashes, split, mode)
            if not path.is_file():
                continue
            try:
                content = path.read_bytes()
                manifest_hashes[name] = sha(content)
                reader = csv.DictReader(io.StringIO(content.decode()))
                if tuple(reader.fieldnames or ()) != FIELDS:
                    raise ValueError('Manifest columns differ')
                records = list(reader)
                manifest_count += len(records)
                if content != csv_bytes(expected_rows):
                    issue(f'Manifest does not match source assignments and current output hashes: {name}')
                if generation['manifest_hashes'].get(name) != sha(content):
                    issue(f'Manifest changed after generation: {name}')
                if len(records) != split_config['counts'][split]:
                    issue(f'Incorrect number of manifest records: {name}')
                distribution = {}
                for class_id in IDS:
                    selected = [r for r in records if r['class_id'] == class_id]
                    n = split_config['per_class_counts'][class_id][split]
                    aligned = sum(r['aligned'] == 'True' for r in selected)
                    expected_aligned = {'correlated': int(n * ratio), 'randomized': n // 2,
                                        'reversed': n - int(n * ratio), 'counterfactual': 0}
                    if len(selected) != n or (mode in expected_aligned and aligned != expected_aligned[mode]):
                        issue(f'Wrong class count/correlation: {split}/{mode}/{class_id}')
                    distribution[class_id] = {
                        'total': len(selected), 'variants': dict(Counter(r['variant'] for r in selected)),
                        'aligned': aligned if mode not in ('original', 'grayscale') else None,
                    }
                distributions[f'{split}/{mode}'] = distribution
            except Exception as exc:
                issue(f'{name}: {exc}')
    if generation['config_sha256'] != sha((output / 'config.json').read_bytes()):
        issue('Configuration changed after generation')
    if generation['unique_pngs'] != len(expected_paths) or generation['source_images'] != len(rows):
        issue('Generation totals differ from expected totals')
    if manifest_count != len(rows) * len(MODES):
        issue('Total manifest rows differ from expected total')
    if checked != len(expected_paths):
        issue('Not every expected PNG passed the pixel audit')
    result = {
        'status': 'passed' if total_issues == 0 else 'failed',
        'audited_utc': datetime.now(timezone.utc).isoformat(),
        'source_images': len(rows), 'expected_unique_pngs': len(expected_paths),
        'pixel_verified_pngs': checked, 'manifest_files': len(manifest_hashes),
        'manifest_rows': manifest_count, 'source_split_counts': split_config['counts'],
        'issues_count': total_issues, 'issues': issues, 'distributions': distributions,
        'manifest_hashes': manifest_hashes, 'config': config,
        'source_split_manifest_sha256': config['split_manifest_sha256'],
        'checks': ['Frozen class IDs, indices, groups and source counts',
                   'Source SHA256, exact duplicate split leakage and original split preservation',
                   'Complete expected PNG file set; no extras',
                   'Full PNG verification and decoding; 224x224 RGB',
                   'Every pixel compared with independently computed transformation',
                   'Every manifest row checked against frozen source, label, condition, assignment and output hash',
                   'Per-class and per-split condition ratios and output pairing'],
        'limitations': ['No near-duplicate or exhaustive manual label audit',
                       'Pretrained model overlap with ImageNet is unknown',
                       'Direct square resizing can distort aspect ratio, equally in all conditions',
                       'Grayscale removes natural color as well as the artificial cue',
                       'Cue predicts a five-class group, not an individual class',
                       'Dataset verification does not establish model robustness; model experiments are pending'],
    }
    save_json(report_dir / 'audit.json', result)
    save_json(output / 'status.json', {'status': 'verified' if not total_issues else 'audit_failed',
                                     'audit_passed': total_issues == 0,
                                     'audit_report': str((report_dir / 'audit.json').resolve())})
    return result
