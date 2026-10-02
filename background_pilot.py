"""SAM quality pilot for Template 2. Does not mark the full dataset complete.

Uses original-resolution RGB inputs and all frozen SAM quality parameters.
Saves candidate masks, actual composites, timings, and numerical checks.
"""

import argparse
import gc
import io
import json
import os
import re
import time
from pathlib import Path

# Unsupported MPS operations may fall back to CPU; keep full float32 precision.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from color_dataset import ROOT, IDS, load_sources, make_config, plan_conditions, sha, save_json, atomic_bytes
from background_scenes import make_background

SAM_SETTINGS = dict(points_per_side=32, points_per_batch=8, pred_iou_thresh=.88,
                    stability_score_thresh=.95, crop_n_layers=0, min_mask_region_area=100)
SAM_COMMIT = 'dca509fe793f601edb92606367a655c15ac00fdf'


def rank_candidates(annotations):
    ranked = []
    for annotation in annotations:
        mask = annotation['segmentation'].astype(bool)
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        height, width = mask.shape
        area = float(mask.mean())
        center = float(1 - np.hypot(xs.mean() / width - .5, ys.mean() / height - .5) / np.sqrt(.5))
        border = np.zeros_like(mask)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        border_fraction = float((mask & border).sum() / mask.sum())
        area_score = 1. if .05 <= area <= .8 else (area / .05 if area < .05 else max(0., (1 - area) / .2))
        iou = float(annotation['predicted_iou'])
        stability = float(annotation['stability_score'])
        score = 2 * iou + 2 * stability + 1.5 * center + area_score - 1.5 * border_fraction
        ranked.append((mask, {'heuristic_score': score, 'area_fraction': area, 'predicted_iou': iou,
                              'stability_score': stability, 'center_score': center,
                              'border_fraction': border_fraction, 'bbox': [float(x) for x in annotation['bbox']]}))
    ranked.sort(key=lambda item: item[1]['heuristic_score'], reverse=True)
    if not ranked:
        raise ValueError('SAM returned no usable candidate masks')
    # This score is not a calibrated confidence or proof of target identity.
    return ranked


def composite(original, mask, background):
    resized = original.resize((224, 224), Image.Resampling.BICUBIC)
    alpha = Image.fromarray(mask.astype(np.uint8) * 255).resize((224, 224), Image.Resampling.NEAREST)
    alpha = alpha.filter(ImageFilter.GaussianBlur(1.2))
    return Image.composite(resized, background, alpha), alpha


def audit_composite(original, output, background, alpha):
    foreground = np.asarray(original.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.int32)
    backdrop = np.asarray(background, dtype=np.int32)
    weights = np.asarray(alpha, dtype=np.int32)[:, :, None]
    expected = ((foreground * weights + backdrop * (255 - weights) + 127) // 255).astype(np.uint8)
    pixels = np.asarray(output)
    if output.mode != 'RGB' or output.size != (224, 224) or not np.array_equal(pixels, expected):
        raise ValueError('Composite pixels differ from independently computed alpha blend')
    return {'pixel_blend_verified': True,
            'unchanged_foreground_pixels': int((weights[:, :, 0] == 255).sum()),
            'replaced_background_pixels': int((weights[:, :, 0] == 0).sum()),
            'soft_boundary_pixels': int(((weights[:, :, 0] > 0) & (weights[:, :, 0] < 255)).sum())}


def save_png(path, image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    atomic_bytes(path, buffer.getvalue())
    with Image.open(path) as saved:
        saved.load()
        if not np.array_equal(np.asarray(saved), np.asarray(image)):
            raise ValueError('Saved PNG verification failed')


def select_device(requested):
    import torch
    if requested != 'auto':
        if requested == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS is unavailable')
        if requested == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable')
        return requested
    return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


def run(pilot_per_class=1, device='auto', class_index=None, run_name='automatic'):
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    rows, metadata, split_config = load_sources(ROOT)
    if pilot_per_class < 1 or pilot_per_class > 10:
        raise ValueError('Pilot size must be 1–10 per class')
    if class_index is not None and class_index not in range(10):
        raise ValueError('Class index must be 0–9')
    if not re.fullmatch(r'[a-z0-9_-]+', run_name):
        raise ValueError('Run name must be a simple lowercase identifier')
    output = ROOT / 'background_bias_dataset' / ('pilot' if run_name == 'automatic' else f'pilot_{run_name}')
    report_dir = ROOT / 'reports/background_bias_current'
    if run_name != 'automatic':
        report_dir = report_dir / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / 'checkpoints/sam_vit_b_01ec64.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    selected_device = select_device(device)
    config = {'schema': 1, 'sam_model': 'vit_b', 'sam_settings': SAM_SETTINGS,
              'sam_commit': SAM_COMMIT, 'checkpoint_sha256': sha(checkpoint.read_bytes()),
              'source_splits_sha256': sha((ROOT / 'project_metadata/source_splits.csv').read_bytes()),
              'device': selected_device, 'torch_version': torch.__version__,
              'mps_coordinate_conversion': 'float32 before tensor transfer' if selected_device == 'mps' else None,
              'selection': 'legacy centrality/area heuristic; semantic target review required',
              'source_resolution': 'original; no downsampling before SAM',
              'mask_boundary_blur_radius_at_224': 1.2}
    identity = sha(json.dumps(config, sort_keys=True).encode())
    save_json(report_dir / 'pilot_config.json', config)
    selected_rows = []
    for item in metadata['classes']:
        if class_index is not None and item['index'] != class_index:
            continue
        selected_rows.extend([r for r in rows if r['split'] == 'train' and r['class_id'] == item['id']][:pilot_per_class])
    status = {'status': 'pilot_running', 'full_dataset_complete': False,
              'requested_images': len(selected_rows), 'completed_images': 0, 'device': selected_device}
    save_json(report_dir / 'status.json', status)
    print(f'Loading SAM ViT-B on {selected_device}; settings: {SAM_SETTINGS}', flush=True)
    # Use restricted state-dictionary loading from the downloaded official checkpoint.
    sam = sam_model_registry['vit_b']()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    sam.load_state_dict(state)
    del state
    sam.to(selected_device).eval()
    generator = SamAutomaticMaskGenerator(sam, **SAM_SETTINGS)
    if selected_device == 'mps':
        # Upstream apply_coords creates float64 NumPy arrays, unsupported on
        # MPS. SAM's positional encoder converts coordinates to float32 anyway.
        # Convert before device transfer without altering model/sampling settings.
        from segment_anything.utils.transforms import ResizeLongestSide
        class Float32Coordinates(ResizeLongestSide):
            def apply_coords(self, coords, original_size):
                return super().apply_coords(coords, original_size).astype(np.float32)
        generator.predictor.transform = Float32Coordinates(sam.image_encoder.img_size)
    records = []
    overview = Image.new('RGB', (1120, len(selected_rows) * 260 + 28), 'white')
    overview_draw = ImageDraw.Draw(overview)
    for column, label in enumerate(('Original', 'Automatic selected mask', 'Nature composite', 'Urban composite', 'Neutral control')):
        overview_draw.text((224 * column + 4, 6), label, fill='black')
    for number, row in enumerate(selected_rows, 1):
        start = time.monotonic()
        source = ROOT / row['path']
        data = source.read_bytes()
        if sha(data) != row['sha256']:
            raise ValueError(f'Source changed: {source}')
        original = Image.open(io.BytesIO(data)).convert('RGB')
        key = sha((identity + row['sha256']).encode())[:24]
        cache = output / '_sam_cache' / key
        mask_file = cache.with_suffix('.npz')
        info_file = cache.with_suffix('.json')
        cached = False
        if mask_file.is_file() and info_file.is_file():
            try:
                info = json.loads(info_file.read_text())
                if info['identity'] != identity or info['source_sha256'] != row['sha256'] or info['masks_sha256'] != sha(mask_file.read_bytes()):
                    raise ValueError('Cache identity mismatch')
                with np.load(mask_file, allow_pickle=False) as saved:
                    masks = saved['masks'].astype(bool)
                if masks.ndim != 3 or masks.shape[1:] != (original.height, original.width):
                    raise ValueError('Cache shape mismatch')
                candidates = info['candidates']
                if len(candidates) != len(masks) or len(masks) == 0:
                    raise ValueError('Cache count mismatch')
                cached = True
            except (OSError, ValueError, KeyError):
                cached = False
        if not cached:
            print(f'[{number}/{len(selected_rows)}] SAM: {row["class_id"]} {source.name} ({original.width}x{original.height})', flush=True)
            with torch.inference_mode():
                annotations = generator.generate(np.asarray(original))
            ranked = rank_candidates(annotations)
            candidates = [item[1] for item in ranked[:12]]
            masks = np.stack([item[0] for item in ranked[:12]])
            buffer = io.BytesIO()
            np.savez_compressed(buffer, masks=masks)
            atomic_bytes(mask_file, buffer.getvalue())
            info = {'identity': identity, 'source_sha256': row['sha256'],
                    'source': row['path'], 'masks_sha256': sha(buffer.getvalue()),
                    'num_sam_candidates': len(annotations), 'candidates': candidates}
            save_json(info_file, info)
            del annotations, ranked
        chosen = masks[0]
        original_224 = original.resize((224, 224), Image.Resampling.BICUBIC)
        seed = f'42|{row["path"]}'  # Same per-image background across all future conditions.
        backgrounds = {'nature': make_background('nature', seed),
                       'urban_indoor': make_background('urban_indoor', seed),
                       'neutral': Image.new('RGB', (224, 224), (127, 127, 127))}
        base = output / row['class_id'] / source.stem
        save_png(base / 'original.png', original_224)
        audits, composites = {}, {}
        for name, background in backgrounds.items():
            image, alpha = composite(original, chosen, background)
            audits[name] = audit_composite(original, image, background, alpha)
            save_png(base / f'{name}.png', image)
            composites[name] = image
        mask_preview = Image.fromarray(chosen.astype(np.uint8) * 255).resize((224, 224), Image.Resampling.NEAREST).convert('RGB')
        save_png(base / 'mask.png', mask_preview)
        y = 28 + (number - 1) * 260
        overview_draw.text((4, y + 3), f'{row["class_index"]}: {row["class_id"]} | {metadata["classes"][int(row["class_index"])]["name"].split(",")[0]} | NOT SEMANTICALLY APPROVED', fill='black')
        for column, image in enumerate((original_224, mask_preview, composites['nature'], composites['urban_indoor'], composites['neutral'])):
            overview.paste(image, (column * 224, y + 25))
        candidate_grid = Image.new('RGB', (896, ((len(masks) + 3) // 4) * 258), 'white')
        candidate_draw = ImageDraw.Draw(candidate_grid)
        for index, mask in enumerate(masks):
            preview, _ = composite(original, mask, Image.new('RGB', (224, 224), (127, 127, 127)))
            x, cy = (index % 4) * 224, (index // 4) * 258
            candidate_draw.text((x + 4, cy + 3), f'Candidate {index}: area {candidates[index]["area_fraction"]:.2f}', fill='black')
            candidate_draw.text((x + 4, cy + 16), f'heuristic score {candidates[index]["heuristic_score"]:.3f}', fill='black')
            candidate_grid.paste(preview, (x, cy + 32))
        save_png(report_dir / f'candidates_{row["class_id"]}_{source.stem}.png', candidate_grid)
        elapsed = time.monotonic() - start
        record = {'source': row['path'], 'source_sha256': row['sha256'], 'class_id': row['class_id'],
                  'source_split': row['split'], 'cache_key': key, 'cache_reused': cached,
                  'elapsed_seconds': elapsed, 'automatic_selected_candidate': 0,
                  'selected_metrics': candidates[0], 'candidate_score_margin': candidates[0]['heuristic_score'] - candidates[1]['heuristic_score'] if len(candidates) > 1 else None,
                  'semantic_review': 'pending', 'composite_audits': audits,
                  'output_directory': str(base.relative_to(ROOT)),
                  'candidate_cache': str(mask_file.relative_to(ROOT))}
        records.append(record)
        save_json(report_dir / 'pilot_results.json', records)
        save_png(report_dir / 'pilot_overview.png', overview)
        status['completed_images'] = len(records)
        save_json(report_dir / 'status.json', status)
        print(f'[{number}/{len(selected_rows)}] finished in {elapsed:.1f}s; candidates retained={len(masks)}; composite pixels verified', flush=True)
        del masks, original, chosen
        generator.predictor.reset_image()
        gc.collect()
        if selected_device == 'mps':
            torch.mps.empty_cache()
    status['status'] = 'pilot_generated_semantic_review_required'
    save_json(report_dir / 'status.json', status)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-per-class', type=int, default=1)
    parser.add_argument('--class-index', type=int)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--run-name', default='automatic')
    args = parser.parse_args()
    run(args.pilot_per_class, args.device, args.class_index, args.run_name)
