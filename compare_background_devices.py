"""Compare the saved CPU and MPS pilots without claiming segmentation accuracy."""
import json
import statistics
from pathlib import Path

import numpy as np

from background_grounded_pilot import run_paths
from color_dataset import save_json, sha


def main():
    cpu_report, cpu_output = run_paths()
    gpu_report, gpu_output = run_paths('mps_reference')
    cpu_rows = json.loads((cpu_report / 'results.json').read_text())
    gpu_rows = json.loads((gpu_report / 'results.json').read_text())
    cpu = {r['source']: r for r in cpu_rows}
    gpu = {r['source']: r for r in gpu_rows}
    if set(cpu) != set(gpu):
        raise ValueError('CPU/MPS comparison must cover the same sources')
    comparisons = []
    for source, left in cpu.items():
        right = gpu[source]
        if left['source_sha256'] != right['source_sha256']:
            raise ValueError('Source bytes differ between runs')
        masks = []
        for directory, record in ((cpu_output, left), (gpu_output, right)):
            path = directory / record['class_id'] / Path(source).stem / 'masks.npz'
            if sha(path.read_bytes()) != record['masks_sha256']:
                raise ValueError('Saved masks hash mismatch')
            with np.load(path, allow_pickle=False) as saved:
                masks.append(saved['masks'][record['selected_candidate']])
        a, b = masks
        union = int((a | b).sum())
        comparisons.append({'source': source, 'exact_mask_match': bool(np.array_equal(a, b)),
                            'mask_iou': float((a & b).sum() / union) if union else 1.,
                            'changed_pixel_fraction': float((a != b).mean()),
                            'same_selected_candidate': left['selected_candidate'] == right['selected_candidate'],
                            'same_detector_boxes': left['detections'] == right['detections']})
    result = {'images': len(comparisons),
              'exact_mask_matches': sum(r['exact_mask_match'] for r in comparisons),
              'minimum_mask_iou': min(r['mask_iou'] for r in comparisons),
              'median_cpu_sam_seconds': statistics.median(r['segmentation_seconds'] for r in cpu_rows),
              'median_mps_sam_seconds': statistics.median(r['segmentation_seconds'] for r in gpu_rows),
              'timing_scope': 'Observed pilot segmentation blocks include mask serialization; not a controlled benchmark or end-to-end speedup.',
              'meaning': 'Device agreement only, not measured ground-truth segmentation quality.',
              'per_image': comparisons}
    save_json(gpu_report / 'cpu_comparison.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_image'}, indent=2))


if __name__ == '__main__':
    main()
