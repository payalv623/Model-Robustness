"""Audit saved pilot provenance/composites; never treat this as semantic approval."""
import io
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from background_grounded_pilot import run_paths
from background_scenes import make_background
from color_dataset import ROOT, load_sources, sha, save_json


def main(run_name=None):
    REPORT, OUTPUT = run_paths(run_name)
    records = json.loads((REPORT / 'results.json').read_text())
    config = json.loads((REPORT / 'config.json').read_text())
    rows, _, _ = load_sources(ROOT)
    sources = {r['path']: r for r in rows}
    if config['source_splits_sha256'] != sha((ROOT / 'project_metadata/source_splits.csv').read_bytes()):
        raise ValueError('Frozen source split manifest changed')
    if len({r['source'] for r in records}) != len(records):
        raise ValueError('Duplicate pilot sources')
    checks = []
    queue = []
    review_path = REPORT / 'visual_review.json'
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    decisions = ({r['source']: r for r in review.get('images', [])}
                 if review.get('results_sha256') == sha((REPORT / 'results.json').read_bytes()) else {})
    for record in records:
        if config['sam_device'] == 'mps':
            if (record.get('sam_parameter_device') != 'mps:0' or
                    record.get('sam_embedding_device') != 'mps:0' or
                    record.get('mps_allocated_bytes', 0) <= 0):
                raise ValueError('Missing evidence of SAM execution on the Mac GPU')
        row = sources[record['source']]
        if row['split'] != 'train' or row['class_id'] != record['class_id']:
            raise ValueError('Pilot class/split mismatch')
        data = (ROOT / row['path']).read_bytes()
        if sha(data) != row['sha256'] or row['sha256'] != record['source_sha256']:
            raise ValueError('Source hash mismatch')
        with Image.open(io.BytesIO(data)) as opened:
            original = opened.convert('RGB')
        directory = OUTPUT / row['class_id'] / Path(row['path']).stem
        if record['detections']:
            raw = (directory / 'masks.npz').read_bytes()
            if sha(raw) != record['masks_sha256']:
                raise ValueError('Candidate masks hash mismatch')
            with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
                masks = saved['masks']
            if masks.dtype != bool or masks.shape != (3, original.height, original.width):
                raise ValueError('Candidate dimensions/type mismatch')
            index = record['selected_candidate']
            if index != int(np.argmax(record['sam_predicted_ious'])):
                raise ValueError('Selected candidate violates recorded rule')
            mask = masks[index]
            with Image.open(directory / 'mask.png') as saved:
                if not np.array_equal(np.asarray(saved), mask.astype(np.uint8) * 255):
                    raise ValueError('Saved mask differs from selected candidate')
            foreground = np.asarray(original.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.int32)
            alpha = Image.fromarray(mask.astype(np.uint8) * 255).resize((224, 224), Image.Resampling.NEAREST).filter(ImageFilter.GaussianBlur(1.2))
            weights = np.asarray(alpha, dtype=np.int32)[:, :, None]
            for name in ('nature', 'urban_indoor', 'neutral'):
                bg = Image.new('RGB', (224, 224), (127, 127, 127)) if name == 'neutral' else make_background(name, '42|' + row['path'])
                expected = ((foreground * weights + np.asarray(bg, dtype=np.int32) * (255 - weights) + 127) // 255).astype(np.uint8)
                path = directory / f'{name}.png'
                with Image.open(path) as saved:
                    saved.load()
                    if saved.mode != 'RGB' or saved.size != (224, 224) or not np.array_equal(np.asarray(saved), expected):
                        raise ValueError('Saved composite failed independent pixel verification')
                checks.append({'file': str(path.relative_to(ROOT)), 'sha256': sha(path.read_bytes())})
        queue.append({'source': row['path'], 'source_sha256': row['sha256'],
                      'flags': record['flags'], 'required_action': 'visual_review',
                      'candidate_preview': str((directory / 'candidates.png').relative_to(ROOT))})
        if row['path'] in decisions:
            decision = decisions[row['path']]
            queue[-1].update({'visual_decision': decision['decision'],
                              'visual_reason': decision['reason'],
                              'required_action': ('refine_or_review_numeric_flags'
                                  if record['flags'] or decision['decision'] != 'visually_plausible'
                                  else 'broader_validation_pending')})
    result = {'status': 'passed', 'scope': 'numeric and provenance audit only',
              'sam_device': config['sam_device'],
              'mps_operation_cpu_fallback': config.get('mps_operation_cpu_fallback'),
              'images': len(records), 'class_counts': dict(Counter(r['class_id'] for r in records)),
              'verified_composites': len(checks), 'source_hashes_unchanged': True,
              'source_splits_unchanged': True, 'semantic_quality_approved': False,
              'flag_counts': dict(Counter(flag for r in records for flag in r['flags'])),
              'composite_files': checks}
    save_json(REPORT / 'numeric_audit.json', result)
    save_json(REPORT / 'review_queue.json', queue)
    print(json.dumps({k: v for k, v in result.items() if k != 'composite_files'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name')
    main(parser.parse_args().run_name)
