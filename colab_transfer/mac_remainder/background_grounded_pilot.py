"""Local class-guided Grounding DINO + SAM pilot, never production approval.

Detection runs on original images with the detector's documented resize. SAM
receives original-resolution RGB and box prompts (not automatic grid sampling).
All candidates and provenance are retained; ambiguous images require review.
"""
import argparse
import gc
import io
import json
import os
import re
import time

import numpy as np
from PIL import Image, ImageDraw

# Fail visibly on unsupported GPU operations unless explicitly overridden.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
from background_pilot import composite, audit_composite, save_png, select_device
from background_scenes import make_background
from color_dataset import ROOT, load_sources, sha, save_json, atomic_bytes

PROMPTS = {
    'n01440764': 'a tench fish. a fish.',
    'n01484850': 'a great white shark. a shark.',
    'n01494475': 'a hammerhead shark. a shark.',
    'n01531178': 'a goldfinch bird. a bird.',
    'n01632777': 'an axolotl. a salamander.',
    'n01665541': 'a leatherback sea turtle. a turtle.',
    'n01687978': 'an agama lizard. a lizard.',
    'n01695060': 'a komodo dragon. a lizard.',
    'n01749939': 'a green mamba snake. a snake.',
    'n01775062': 'a wolf spider. a spider.',
}
REPORT = ROOT / 'reports/background_bias_current/grounded_pilot'
OUTPUT = ROOT / 'background_bias_dataset/pilot_grounded'


def run_paths(run_name=None):
    if run_name is None:
        return REPORT, OUTPUT
    if not re.fullmatch(r'[a-z0-9_-]+', run_name):
        raise ValueError('Run name must be a simple lowercase identifier')
    return (REPORT.parent / f'grounded_{run_name}',
            OUTPUT.parent / f'pilot_grounded_{run_name}')


def select_boxes(boxes, scores, width, height, threshold=.25):
    """Clamp valid boxes and suppress duplicate aliases; retain distinct animals."""
    from torchvision.ops import nms
    import torch
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4).clone()
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError('Box and score counts differ')
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    valid = (torch.isfinite(boxes).all(1) & torch.isfinite(scores) &
             (scores >= threshold) & (boxes[:, 2] > boxes[:, 0]) &
             (boxes[:, 3] > boxes[:, 1]))
    original_indices = torch.where(valid)[0]
    keep = nms(boxes[valid], scores[valid], .5)
    return [{'box_xyxy': boxes[valid][i].tolist(), 'score': float(scores[valid][i]),
             'raw_index': int(original_indices[i])} for i in keep]


def read_source(row):
    data = (ROOT / row['path']).read_bytes()
    if sha(data) != row['sha256']:
        raise ValueError(f'Source hash changed: {row["path"]}')
    with Image.open(io.BytesIO(data)) as image:
        return image.convert('RGB')


def main(per_class=1, device='mps', run_name='mps_reference'):
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from segment_anything import SamPredictor, sam_model_registry
    from segment_anything.utils.amg import calculate_stability_score
    if not 1 <= per_class <= 10:
        raise ValueError('Pilot must contain 1–10 training examples per class')
    REPORT, OUTPUT = run_paths(run_name)
    device = select_device(device)
    print(f'SAM requested device: {device}; detector device: cpu', flush=True)
    torch.set_num_threads(4)
    rows, metadata, _ = load_sources(ROOT)
    selected = [r for item in metadata['classes'] for r in
                [r for r in rows if r['class_id'] == item['id'] and r['split'] == 'train'][:per_class]]
    REPORT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((ROOT / 'project_metadata/grounding_dino_checkpoint.json').read_text())
    detector_path = ROOT / 'checkpoints/grounding-dino-tiny'
    for record in provenance['files']:
        if sha((detector_path / record['file']).read_bytes()) != record['sha256']:
            raise ValueError('Detector file hash mismatch')
    checkpoint = ROOT / 'checkpoints/sam_vit_b_01ec64.pth'
    config = {'schema': 1, 'detector': provenance, 'detector_device': 'cpu',
              'sam_device': device, 'sam_checkpoint_sha256': sha(checkpoint.read_bytes()),
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'prompts': PROMPTS, 'box_threshold': .25, 'text_threshold': .20,
              'duplicate_box_nms_iou': .5, 'sam_mode': 'box prompt, multimask_output=True',
              'selection': 'highest detector score box; highest SAM predicted IoU candidate',
              'sam_min_predicted_iou': .88, 'sam_min_stability': .95,
              'semantic_approval': 'required separately; numeric scores never approve target identity',
              'source_splits_sha256': sha((ROOT / 'project_metadata/source_splits.csv').read_bytes()),
              'mask_blur_radius_at_224': 1.2, 'precision': 'float32',
              'pilot_sampling': 'first N frozen training rows per class; no held-out tuning'}
    if device == 'mps':
        config['mps_operation_cpu_fallback'] = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'
    config_file = REPORT / 'config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError('Pilot config changed; use a new versioned output directory')
    save_json(config_file, config)
    save_json(REPORT / 'status.json', {'status': 'detecting', 'requested': len(selected),
                                      'full_generation_started': False})
    processor = AutoProcessor.from_pretrained(detector_path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        detector_path, local_files_only=True, use_safetensors=True).eval()
    records = []
    for number, row in enumerate(selected, 1):
        image = read_source(row)
        directory = OUTPUT / row['class_id'] / (ROOT / row['path']).stem
        directory.mkdir(parents=True, exist_ok=True)
        cache = directory / 'detection.json'
        identity = sha(json.dumps(config, sort_keys=True).encode() + row['sha256'].encode())
        if cache.exists():
            record = json.loads(cache.read_text())
            if record['identity'] != identity:
                raise ValueError('Detection cache identity mismatch')
        else:
            start = time.monotonic()
            inputs = processor(images=image, text=PROMPTS[row['class_id']], return_tensors='pt')
            with torch.inference_mode():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=.25, text_threshold=.20,
                target_sizes=[image.size[::-1]])[0]
            boxes = select_boxes(result['boxes'], result['scores'], image.width, image.height)
            record = {'source': row['path'], 'source_sha256': row['sha256'],
                      'class_id': row['class_id'], 'split': row['split'], 'identity': identity,
                      'original_size': list(image.size), 'prompt': PROMPTS[row['class_id']],
                      'detections': boxes, 'raw_labels': result['labels'],
                      'detection_seconds': time.monotonic() - start}
            save_json(cache, record)
        records.append(record)
        print(f'Detection [{number}/{len(selected)}] {row["class_id"]}: {len(record["detections"])} boxes', flush=True)
    del model, processor
    gc.collect()
    sam = sam_model_registry['vit_b']()
    sam.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    predictor = SamPredictor(sam.to(device).eval())
    actual_device = str(next(sam.parameters()).device)
    if next(sam.parameters()).device.type != device.split(':')[0]:
        raise RuntimeError(f'SAM model is on {actual_device}, expected {device}')
    print(f'SAM model parameters confirmed on {actual_device}', flush=True)
    for number, (row, record) in enumerate(zip(selected, records), 1):
        image = read_source(row)
        directory = OUTPUT / row['class_id'] / (ROOT / row['path']).stem
        record['flags'] = []
        record['semantic_review'] = 'pending'
        save_png(directory / 'original.png', image.resize((224, 224), Image.Resampling.BICUBIC))
        if not record['detections']:
            record['flags'].append('no_target_detection')
            save_json(directory / 'result.json', record)
            continue
        if len(record['detections']) > 1:
            record['flags'].append('multiple_distinct_detections_review_target_coverage')
        start = time.monotonic()
        box = np.array(record['detections'][0]['box_xyxy'], dtype=np.float32)
        with torch.inference_mode():
            predictor.set_image(np.asarray(image))
            if predictor.features.device.type != device.split(':')[0]:
                raise RuntimeError('SAM image embedding is on the wrong device')
            logits, scores, _ = predictor.predict(box=box, multimask_output=True, return_logits=True)
        if device == 'mps':
            torch.mps.synchronize()
        masks = logits > sam.mask_threshold
        stability = calculate_stability_score(torch.from_numpy(logits), sam.mask_threshold, 1.).tolist()
        choice = int(np.argmax(scores))
        mask = masks[choice]
        buffer = io.BytesIO()
        np.savez_compressed(buffer, masks=masks)
        atomic_bytes(directory / 'masks.npz', buffer.getvalue())
        record.update({'selected_candidate': choice, 'sam_predicted_ious': scores.tolist(),
                       'sam_stability_scores': stability, 'area_fraction': float(mask.mean()),
                       'masks_sha256': sha(buffer.getvalue()),
                       'segmentation_seconds': time.monotonic() - start})
        record['sam_parameter_device'] = actual_device
        record['sam_embedding_device'] = str(predictor.features.device)
        if device == 'mps':
            record['mps_allocated_bytes'] = torch.mps.current_allocated_memory()
        if scores[choice] < .88:
            record['flags'].append('low_sam_predicted_iou')
        if stability[choice] < .95:
            record['flags'].append('low_sam_stability')
        if not .005 < mask.mean() < .95:
            record['flags'].append('extreme_mask_area')
        record['composite_audits'] = {}
        save_png(directory / 'mask.png', Image.fromarray(mask.astype(np.uint8) * 255))
        for name in ('nature', 'urban_indoor', 'neutral'):
            bg = (Image.new('RGB', (224, 224), (127, 127, 127)) if name == 'neutral'
                  else make_background(name, '42|' + row['path']))
            result, alpha = composite(image, mask, bg)
            save_png(directory / f'{name}.png', result)
            record['composite_audits'][name] = audit_composite(image, result, bg, alpha)
        sheet = Image.new('RGB', (896, 3 * 260), 'white')
        draw = ImageDraw.Draw(sheet)
        annotated = image.copy()
        ImageDraw.Draw(annotated).rectangle(tuple(box.tolist()), outline='red', width=max(2, image.width // 150))
        for i, candidate in enumerate(masks):
            neutral, _ = composite(image, candidate, Image.new('RGB', (224, 224), (127, 127, 127)))
            nature, _ = composite(image, candidate, make_background('nature', '42|' + row['path']))
            tiles = (annotated.resize((224, 224)),
                     Image.fromarray(candidate.astype(np.uint8)*255).convert('RGB').resize((224, 224)), nature, neutral)
            draw.text((4, i * 260 + 4), f'{row["class_id"]} candidate {i} | predicted IoU {scores[i]:.3f} | stability {stability[i]:.3f} | selected={i == choice}', fill='black')
            for col, tile in enumerate(tiles):
                sheet.paste(tile, (col * 224, i * 260 + 28))
        save_png(directory / 'candidates.png', sheet)
        save_json(directory / 'result.json', record)
        print(f'SAM [{number}/{len(selected)}] {row["class_id"]}: candidate {choice}, flags={record["flags"]}', flush=True)
    # One sheet per sample position gives ten rows, one for each frozen class.
    for sample in range(per_class):
        overview = Image.new('RGB', (1120, 10 * 256 + 30), 'white')
        draw = ImageDraw.Draw(overview)
        for col, name in enumerate(('Original', 'Selected mask', 'Nature', 'Urban', 'Neutral')):
            draw.text((col * 224 + 4, 6), name, fill='black')
        for c in range(10):
            row, record = selected[c * per_class + sample], records[c * per_class + sample]
            directory = OUTPUT / row['class_id'] / (ROOT / row['path']).stem
            draw.text((4, c * 256 + 33), f'{row["class_id"]} {(ROOT / row["path"]).name} flags={len(record["flags"])}', fill='black')
            for col, name in enumerate(('original', 'mask', 'nature', 'urban_indoor', 'neutral')):
                path = directory / f'{name}.png'
                if path.exists():
                    with Image.open(path) as tile:
                        overview.paste(tile.convert('RGB').resize((224, 224)), (col * 224, c * 256 + 55))
        save_png(REPORT / f'overview_{sample}.png', overview)
    save_json(REPORT / 'results.json', records)
    save_json(REPORT / 'status.json', {'status': 'pilot_complete_pending_visual_review',
              'images': len(records), 'images_with_flags': sum(bool(r['flags']) for r in records),
              'full_generation_started': False, 'quality_gate_passed': False})
    print(f'Pilot saved to {REPORT}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-class', type=int, default=1)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='mps')
    parser.add_argument('--run-name', default='mps_reference', help='Separate versioned pilot outputs')
    args = parser.parse_args()
    try:
        main(args.per_class, args.device, args.run_name)
    except Exception as exc:
        report_dir, _ = run_paths(args.run_name)
        save_json(report_dir / 'status.json', {'status': 'pilot_failed', 'error': str(exc),
                                          'full_generation_started': False})
        raise
