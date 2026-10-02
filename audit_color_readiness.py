"""Audit recovered Color implementation without generating the research dataset.

Rehashes every source image and tests legacy generation on temporary copies
of 100 real training images. Writes results to reports/color_readiness/.
Run with .venv/bin/python audit_color_readiness.py.
"""

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'robustness-matplotlib'))

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    result = {'audited_utc': datetime.now(timezone.utc).isoformat(), 'checks': {}}
    checks = result['checks']
    metadata = json.loads((ROOT / 'project_metadata/selected_classes.json').read_text())
    classes = metadata['classes']
    expected = [c['id'] for c in classes]
    historical = json.loads(subprocess.check_output(
        ['git', 'show', 'HEAD:project_metadata/selected_classes.json'], cwd=ROOT))
    assert [(c['index'], c['id']) for c in classes] == [(c['index'], c['id']) for c in historical['classes']]
    assert metadata['group_a'] == historical['group_a']
    assert metadata['group_b'] == historical['group_b']
    assert metadata['group_a']['indices'] == list(range(5))
    assert metadata['group_b']['indices'] == list(range(5, 10))
    checks['historical_class_indices_ids_and_groups_match'] = True
    for filename in ('reports/Template_1_Color_Bias_Report.md',
                     'reports/color_bias/Template_1_Color_Bias_Report.md'):
        text = (ROOT / filename).read_text()
        assert re.findall(r'Class \d+: (n\d+)', text) == expected
    pixel_records = json.loads((ROOT / 'reports/color_pixel_statistics.json').read_text())
    assert [r['class_name'] for r in pixel_records] == expected
    checks['both_historical_reports_and_pixel_samples_match_classes'] = True

    source_rows = read_csv(ROOT / 'reports/dataset_preparation/manifest.csv')
    split_rows = read_csv(ROOT / 'project_metadata/source_splits.csv')
    config = json.loads((ROOT / 'project_metadata/split_config.json').read_text())
    assert hashlib.sha256((ROOT / 'project_metadata/source_splits.csv').read_bytes()).hexdigest() == config['split_manifest_sha256']
    assert hashlib.sha256((ROOT / 'reports/dataset_preparation/manifest.csv').read_bytes()).hexdigest() == config['source_manifest_sha256']
    assert len(source_rows) == len(split_rows) == 13500
    by_path = {r['path']: r for r in source_rows}
    assert len(by_path) == len(source_rows)
    assert {r['path'] for r in split_rows} == set(by_path)
    digests = defaultdict(set)
    for index, row in enumerate(split_rows, 1):
        source = ROOT / row['path']
        assert source.is_relative_to(ROOT / 'archive')
        assert source.parent.name == row['class_id']
        assert classes[int(row['class_index'])]['id'] == row['class_id']
        assert row['group'] == ('A' if int(row['class_index']) < 5 else 'B')
        assert row['sha256'] == by_path[row['path']]['sha256']
        data = source.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['sha256'], str(source)
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            assert image.size == (int(by_path[row['path']]['width']), int(by_path[row['path']]['height']))
        digests[row['sha256']].add(row['split'])
        if index % 3000 == 0:
            print(f'Rehashed and decoded {index}/13500 source images', flush=True)
    assert all(len(splits) == 1 for splits in digests.values())
    for split in ('train.X1', 'val.X'):
        assert {p.name for p in (ROOT / 'archive' / split).iterdir() if p.is_dir()} == set(expected)
    checks['all_13500_source_hashes_and_decodes_passed'] = True
    checks['split_manifest_hashes_paths_groups_and_no_exact_leakage_passed'] = True
    result['split_counts'] = dict(Counter(r['split'] for r in split_rows))

    spec = importlib.util.spec_from_file_location('recovered_color', ROOT / 'legacy/color_bias_template1_legacy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJECT_ROOT = ROOT
    module.SOURCE_DIR = ROOT / 'archive'
    module.METADATA_FILE = ROOT / 'project_metadata/selected_classes.json'
    assert [c['id'] for c in module.load_frozen_classes()] == expected
    result['legacy_source_counts'] = {c['id']: len(module.get_images(module.resolve_class_source(c))) for c in classes}
    result['current_settings'] = {'correlated_ratio': module.CORRELATED_RATIO,
                                  'original_weight': module.ORIGINAL_WEIGHT,
                                  'tint_weight': module.TINT_WEIGHT,
                                  'resolution': list(module.IMAGE_SIZE)}
    result['current_color_dataset_exists'] = (ROOT / 'color_bias_dataset').exists()
    result['assignment_counts_without_writing_images'] = {}
    for split in ('train', 'validation', 'test'):
        per_class = {}
        for item in classes:
            paths = [ROOT / r['path'] for r in split_rows
                     if r['class_id'] == item['id'] and r['split'] == split]
            random.seed(42)
            assigned = module.exact_correlated_assignment(paths, item['index'])
            colors = Counter(a['color'] for a in assigned)
            matched = sum(a['correlated'] for a in assigned)
            assert matched == int(len(paths) * .9)
            assert all(a['color'] == ('red' if item['index'] < 5 else 'blue')
                       for a in assigned if a['correlated'])
            per_class[item['id']] = {'total': len(paths), 'aligned': matched,
                                     'mismatched': len(paths) - matched, **colors}
        result['assignment_counts_without_writing_images'][split] = per_class

    # Generate all four legacy modes only in a temporary directory, using
    # 10 copied source images per class. Production data/reports stay untouched.
    with tempfile.TemporaryDirectory(prefix='color-audit-') as directory:
        temporary = Path(directory)
        module.OUTPUT_DIR = temporary / 'outputs'
        fixture_classes = []
        for item in classes:
            folder = temporary / 'source' / item['id']
            folder.mkdir(parents=True)
            candidates = [ROOT / r['path'] for r in split_rows
                          if r['class_id'] == item['id'] and r['split'] == 'train'][:10]
            for path in candidates:
                shutil.copyfile(path, folder / path.name)
            fixture_classes.append({**item, 'path': str(folder)})
        random.seed(42)
        with contextlib.redirect_stdout(io.StringIO()):
            corr = module.generate_correlated(fixture_classes)
            rand = module.generate_randomized(fixture_classes, corr)
            module.generate_reversed(fixture_classes)
            module.generate_grayscale(fixture_classes)
        detected = {mode: Counter() for mode in ('correlated', 'randomized', 'reversed', 'grayscale')}
        for item in fixture_classes:
            manifest = read_csv(module.OUTPUT_DIR / 'correlated' / item['id'] / 'manifest.csv')
            assignments = {Path(r['source']).name: r for r in manifest}
            for source in sorted(Path(item['path']).glob('*.JPEG')):
                with Image.open(source) as image:
                    original = image.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
                array = np.asarray(original).astype(np.float32)
                red = np.clip(.7 * array + .3 * np.array([255, 0, 0], dtype=np.float32), 0, 255).astype(np.uint8)
                blue = np.clip(.7 * array + .3 * np.array([0, 0, 255], dtype=np.float32), 0, 255).astype(np.uint8)
                for mode in detected:
                    path = module.OUTPUT_DIR / mode / item['id'] / (source.stem + '.png')
                    with Image.open(path) as output:
                        assert output.size == (224, 224) and output.mode == 'RGB' and output.format == 'PNG'
                        actual = np.asarray(output)
                    if mode == 'grayscale':
                        assert np.array_equal(actual, np.asarray(ImageOps.grayscale(original).convert('RGB')))
                        color = 'grayscale'
                    else:
                        if np.array_equal(actual, red):
                            color = 'red'
                        elif np.array_equal(actual, blue):
                            color = 'blue'
                        else:
                            raise AssertionError(f'Pixel formula mismatch: {path}')
                    detected[mode][color] += 1
                    if mode == 'correlated':
                        assert color == assignments[source.name]['color']
                    if mode == 'reversed':
                        assert color == ('blue' if item['index'] < 5 else 'red')
            assert corr[item['id']]['correlated'] == 9 and corr[item['id']]['mismatch'] == 1
        assert detected['randomized'] == detected['correlated']
        result['temporary_smoke_test'] = {'source_images': 100, 'outputs_pixel_verified': 400,
                                         'per_mode_colors': {k: dict(v) for k, v in detected.items()},
                                         'randomized_per_class_counts': rand}
        # Demonstrate that the legacy verifier does not flag a missing output.
        missing = next((module.OUTPUT_DIR / 'correlated').rglob('*.png'))
        missing.unlink()
        missing_audit = module.verify_generated_images(fixture_classes)['correlated']
        result['legacy_missing_file_probe'] = missing_audit
        assert missing_audit['files_checked'] == 99 and missing_audit['bad_files'] == 0
    checks['temporary_400_outputs_pixel_formula_size_channels_and_assignments_passed'] = True

    result['findings'] = [
        {'priority': 'high', 'issue': 'Legacy generation ignores source_splits.csv',
         'detail': 'It reads all 1300 train.X1 images per class, mixing our 1040 train and 260 validation images; val.X test images are omitted.'},
        {'priority': 'high', 'issue': 'Legacy integrity audit cannot establish completeness',
         'detail': 'Removing one temporary output produced files_checked=99, bad_files=0. Missing directories are skipped and image.load() is not called.'},
        {'priority': 'medium', 'issue': 'Reversed means 100% opposite, not 10/90',
         'detail': 'This is a valid fully conflicting stress test but differs from the previously discussed mirrored correlation and needs an explicit name.'},
        {'priority': 'medium', 'issue': 'Randomized is globally shuffled, not exact 50/50 per class',
         'detail': 'Finite-sample class-color associations can remain; especially important for small test sets.'},
        {'priority': 'medium', 'issue': 'Only correlated mode saves image-level manifests',
         'detail': 'Randomized, reversed, and grayscale assignments lack equivalent manifest records.'},
        {'priority': 'medium', 'issue': 'Visual panels illustrate transformations, not actual assignments',
         'detail': 'The CORRELATED panel always uses the group-majority tint, including when a saved sample would belong to the mismatch minority.'},
        {'priority': 'medium', 'issue': 'No untinted standardized control output',
         'detail': 'Grayscale removes natural object colors as well as the tint; use an untinted RGB control to distinguish this effect.'},
        {'priority': 'medium', 'issue': 'Legacy ratio metadata is hard-coded',
         'detail': 'Changing CORRELATED_RATIO alone would leave experiment_metadata.json and report prose claiming 90/10.'},
        {'priority': 'context', 'issue': 'Historical image counts differ',
         'detail': 'Recovered reports state 12973 source images; the fresh training source has 13000. The reason for the old 27-image difference is unknown without its manifest.'},
        {'priority': 'context', 'issue': 'Not a complete robustness experiment yet',
         'detail': 'No current Color dataset or model evaluation exists. No near-duplicate or manual all-image label audit was performed.'},
    ]
    result['status'] = 'source_data_and_core_color_transform_verified; production_generation_needs_updates'
    result['legacy_code_sha256'] = hashlib.sha256((ROOT / 'legacy/color_bias_template1_legacy.py').read_bytes()).hexdigest()
    report_dir = ROOT / 'reports/color_readiness'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'checks': checks,
                      'findings': len(result['findings'])}, indent=2))


if __name__ == '__main__':
    main()
