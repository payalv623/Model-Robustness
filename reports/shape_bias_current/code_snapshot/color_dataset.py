"""Deterministic, split-aware Color dataset generation and manifest planning."""

import csv
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version

ROOT = Path(__file__).resolve().parent
IDS = ('n01440764', 'n01484850', 'n01494475', 'n01531178', 'n01632777',
       'n01665541', 'n01687978', 'n01695060', 'n01749939', 'n01775062')
SPLITS = ('train', 'validation', 'test')
VARIANTS = ('original', 'red', 'blue', 'grayscale')
MODES = ('original', 'correlated', 'randomized', 'reversed',
         'counterfactual', 'grayscale', 'all_red', 'all_blue')
FIELDS = ('source', 'source_sha256', 'output', 'output_sha256', 'class_id',
          'class_index', 'group', 'split', 'condition', 'variant', 'aligned')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_bytes(data)
    temporary.replace(path)


def save_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def csv_bytes(rows, fields=FIELDS):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def load_sources(root=ROOT):
    metadata_file = root / 'project_metadata/selected_classes.json'
    splits_file = root / 'project_metadata/source_splits.csv'
    split_config = json.loads((root / 'project_metadata/split_config.json').read_text())
    metadata = json.loads(metadata_file.read_text())
    classes = metadata['classes']
    if [(c['index'], c['id']) for c in classes] != list(enumerate(IDS)):
        raise ValueError('Frozen class order or IDs changed')
    for name, indices, majority in [('group_a', list(range(5)), 'warm_red'),
                                     ('group_b', list(range(5, 10)), 'cool_blue')]:
        if metadata[name]['indices'] != indices or metadata[name]['correlated_color'] != majority:
            raise ValueError('Frozen color group mapping changed')
    if sha(splits_file.read_bytes()) != split_config['split_manifest_sha256']:
        raise ValueError('Frozen split manifest hash mismatch')
    rows = read_csv(splits_file)
    paths, digest_splits, counts = set(), {}, Counter()
    for row in rows:
        index = int(row['class_index'])
        if index not in range(10) or IDS[index] != row['class_id']:
            raise ValueError('Invalid class label')
        if row['group'] != ('A' if index < 5 else 'B') or row['split'] not in SPLITS:
            raise ValueError('Invalid group or split')
        path = Path(row['path'])
        source_split = 'val.X' if row['split'] == 'test' else 'train.X1'
        if (path.is_absolute() or '..' in path.parts or len(path.parts) != 4
                or path.parts[:3] != ('archive', source_split, row['class_id'])
                or row['source_split'] != source_split):
            raise ValueError('Invalid source path/split')
        source = (root / path).resolve()
        if not source.is_relative_to((root / 'archive').resolve()) or not source.is_file():
            raise ValueError(f'Missing or unsafe source: {path}')
        if row['path'] in paths:
            raise ValueError('Duplicate source path')
        paths.add(row['path'])
        digest = row['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid source digest')
        if digest in digest_splits and digest_splits[digest] != row['split']:
            raise ValueError('Byte-identical source leakage across splits')
        digest_splits[digest] = row['split']
        counts[(row['class_id'], row['split'])] += 1
    expected = {(class_id, split): n for class_id, values in split_config['per_class_counts'].items()
                for split, n in values.items()}
    if dict(counts) != expected or set(expected) != {(c, s) for c in IDS for s in SPLITS}:
        raise ValueError('Source class/split counts differ from frozen configuration')
    rows.sort(key=lambda row: (SPLITS.index(row['split']), int(row['class_index']), row['path']))
    return rows, metadata, split_config


def make_config(root, ratio, seed):
    ratio = Fraction(str(ratio))
    if not Fraction(1, 2) < ratio < 1:
        raise ValueError('Correlation must be greater than 0.5 and less than 1')
    return {
        'schema_version': 1, 'template': 'color', 'correlation': str(ratio),
        'seed': seed, 'image_size': [224, 224], 'original_weight': 0.7,
        'tint_weight': 0.3, 'resize': 'Pillow BICUBIC direct resize; no crop',
        'pillow_version': pillow_version, 'numpy_version': np.__version__,
        'split_manifest_sha256': sha((root / 'project_metadata/source_splits.csv').read_bytes()),
        'classes_metadata_sha256': sha((root / 'project_metadata/selected_classes.json').read_bytes()),
        'modes': list(MODES), 'storage_variants': list(VARIANTS),
        'assignment': 'Per-class per-split SHA256 ordering; floor(n * ratio); reversed swaps each correlated assignment',
        'mode_definitions': {
            'original': 'Untinted RGB, standardized exactly like tinted images',
            'correlated': 'Group A majority red; Group B majority blue',
            'randomized': 'Randomized assignment with exact half aligned within each class/split when n is even',
            'reversed': 'Swap red and blue for each correlated source; 10% aligned at ratio 90%',
            'counterfactual': 'Every image receives the group-opposite tint; 0% aligned',
            'grayscale': 'Grayscale from untinted original; removes natural color too; RGB channels retained',
            'all_red': 'Every source red, paired with the identical all_blue source',
            'all_blue': 'Every source blue, paired with the identical all_red source',
        },
    }


def output_path(row, variant):
    filename = Path(row['path']).stem + '__' + sha(row['path'].encode())[:16] + '.png'
    return Path('images') / row['split'] / row['class_id'] / variant / filename


def plan_conditions(rows, config):
    assignments = {}
    ratio = Fraction(config['correlation'])
    for split in SPLITS:
        for class_id in IDS:
            selected = [r for r in rows if r['split'] == split and r['class_id'] == class_id]
            def ordered(tag):
                return sorted(selected, key=lambda r: sha(f"{config['seed']}|{tag}|{r['path']}".encode()))
            correlated = {r['path'] for r in ordered('correlated')[:int(len(selected) * ratio)]}
            balanced = {r['path'] for r in ordered('randomized')[:len(selected) // 2]}
            for row in selected:
                majority = 'red' if row['group'] == 'A' else 'blue'
                opposite = 'blue' if majority == 'red' else 'red'
                chosen = majority if row['path'] in correlated else opposite
                assignments[row['path']] = {
                    'original': 'original', 'correlated': chosen,
                    'randomized': majority if row['path'] in balanced else opposite,
                    'reversed': 'blue' if chosen == 'red' else 'red',
                    'counterfactual': opposite, 'grayscale': 'grayscale',
                    'all_red': 'red', 'all_blue': 'blue',
                }
    return assignments


def manifest_rows(rows, assignments, hashes, split, condition):
    records = []
    for row in rows:
        if row['split'] != split:
            continue
        variant = assignments[row['path']][condition]
        output = output_path(row, variant).as_posix()
        majority = 'red' if row['group'] == 'A' else 'blue'
        records.append({
            'source': row['path'], 'source_sha256': row['sha256'],
            'output': output, 'output_sha256': hashes[output],
            'class_id': row['class_id'], 'class_index': row['class_index'],
            'group': row['group'], 'split': split, 'condition': condition,
            'variant': variant, 'aligned': '' if variant in ('original', 'grayscale') else str(variant == majority),
        })
    return records


def render_variants(data):
    with Image.open(io.BytesIO(data)) as image:
        original = image.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(original).astype(np.float32)
    result = {'original': original, 'grayscale': ImageOps.grayscale(original).convert('RGB')}
    for name, tint in [('red', (255, 0, 0)), ('blue', (0, 0, 255))]:
        pixels = (np.float32(.7) * array + np.float32(.3) * np.array(tint, dtype=np.float32))
        result[name] = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return result


def ensure_output_location(root, output):
    output = output.resolve()
    archive = (root / 'archive').resolve()
    if output == root.resolve() or output.is_relative_to(archive) or archive.is_relative_to(output):
        raise ValueError('Output must be separate from the project root and original archive')
    return output


def generate(root=ROOT, output=None, ratio='0.9', seed=42, workers=4):
    root = root.resolve()
    output = ensure_output_location(root, output or root / 'color_bias_dataset')
    if not 1 <= workers <= 16:
        raise ValueError('workers must be between 1 and 16')
    rows, metadata, split_config = load_sources(root)
    config = make_config(root, ratio, seed)
    config_file = output / 'config.json'
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise ValueError('Existing dataset configuration differs. Use a new output directory.')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Nonempty output without pipeline config; refusing to mix datasets')
    save_json(config_file, config)
    save_json(output / 'status.json', {'status': 'generating', 'audit_passed': False})
    assignments = plan_conditions(rows, config)

    def process(row):
        data = (root / row['path']).read_bytes()
        if sha(data) != row['sha256']:
            raise ValueError(f"Original source hash changed: {row['path']}")
        rendered = render_variants(data)
        hashes, written, reused = {}, 0, 0
        for variant in VARIANTS:
            relative = output_path(row, variant)
            destination = output / relative
            expected = np.asarray(rendered[variant])
            valid = False
            if destination.exists():
                try:
                    with Image.open(destination) as existing:
                        valid = (existing.format == 'PNG' and existing.mode == 'RGB'
                                 and existing.size == (224, 224)
                                 and np.array_equal(np.asarray(existing), expected))
                except (OSError, ValueError):
                    valid = False
            if valid:
                reused += 1
            else:
                buffer = io.BytesIO()
                rendered[variant].save(buffer, format='PNG', compress_level=3)
                atomic_bytes(destination, buffer.getvalue())
                written += 1
            partial = destination.with_suffix('.png.partial')
            if partial.exists():
                partial.unlink()  # Only this pipeline's interrupted temporary output.
            hashes[relative.as_posix()] = sha(destination.read_bytes())
        return hashes, written, reused

    hashes, written, reused = {}, 0, 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (file_hashes, n_written, n_reused) in enumerate(pool.map(process, rows), 1):
                hashes.update(file_hashes)
                written += n_written
                reused += n_reused
                if index % 500 == 0 or index == len(rows):
                    print(f'Generated/validated {index}/{len(rows)} sources; written={written}, reused={reused}', flush=True)
        manifest_hashes = {}
        distributions = {}
        for split in SPLITS:
            for condition in MODES:
                records = manifest_rows(rows, assignments, hashes, split, condition)
                content = csv_bytes(records)
                name = f'manifests/{split}/{condition}.csv'
                atomic_bytes(output / name, content)
                manifest_hashes[name] = sha(content)
                distributions[f'{split}/{condition}'] = {
                    class_id: dict(Counter(r['variant'] for r in records if r['class_id'] == class_id))
                    for class_id in IDS
                }
        generation = {
            'status': 'generated_pending_audit', 'source_images': len(rows),
            'unique_pngs': len(hashes), 'written_this_run': written, 'reused_this_run': reused,
            'manifest_rows': len(rows) * len(MODES), 'manifest_hashes': manifest_hashes,
            'source_split_counts': split_config['counts'], 'distributions': distributions,
            'config_sha256': sha(config_file.read_bytes()),
        }
        save_json(output / 'generation.json', generation)
        save_json(output / 'status.json', {'status': 'generated_pending_audit', 'audit_passed': False})
        return generation
    except Exception as exc:
        save_json(output / 'status.json', {'status': 'failed', 'audit_passed': False, 'error': str(exc)})
        raise
