"""One manually guided diagnostic; not a scalable annotation policy or dataset.

The approximate goldfinch prompt was marked from visual inspection of this
training image. It is recorded explicitly and is not ground-truth annotation.
"""

import io
import json
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from segment_anything import SamPredictor, sam_model_registry

from color_dataset import ROOT, load_sources, sha, save_json
from background_pilot import composite, audit_composite, save_png
from background_scenes import make_background


def main():
    torch.set_num_threads(4)
    rows, _, _ = load_sources(ROOT)
    row = next(r for r in rows if r['class_id'] == 'n01531178' and r['split'] == 'train')
    data = (ROOT / row['path']).read_bytes()
    assert sha(data) == row['sha256']
    original = Image.open(io.BytesIO(data)).convert('RGB')
    width, height = original.size
    # Bird box, one point on the bird and one negative point on the cage bar.
    box_normalized = [0.0, .25, .87, .80]
    points_normalized = [[.62, .56], [.41, .32]]
    labels = np.array([1, 0], dtype=np.int32)
    box = np.array(box_normalized, dtype=np.float32) * np.array([width, height, width, height])
    points = np.array(points_normalized, dtype=np.float32) * np.array([width, height])
    checkpoint = ROOT / 'checkpoints/sam_vit_b_01ec64.pth'
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = sam_model_registry['vit_b']()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.to(device).eval()
    predictor = SamPredictor(model)
    start = time.monotonic()
    with torch.inference_mode():
        predictor.set_image(np.asarray(original))
        masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels,
                                             box=box, multimask_output=True)
    directory = ROOT / 'reports/background_bias_current/prompt_probe'
    directory.mkdir(parents=True, exist_ok=True)
    sheet = Image.new('RGB', (896, len(masks) * 252 + 32), 'white')
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(('Original', 'Prompted mask', 'Nature', 'Neutral')):
        draw.text((c * 224 + 4, 8), name, fill='black')
    audits = []
    for index, mask in enumerate(masks):
        background = make_background('nature', '42|' + row['path'])
        result, alpha = composite(original, mask, background)
        neutral, _ = composite(original, mask, Image.new('RGB', (224, 224), (127, 127, 127)))
        audits.append(audit_composite(original, result, background, alpha))
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255).resize((224, 224), Image.Resampling.NEAREST).convert('RGB')
        for col, image in enumerate((original.resize((224, 224)), mask_image, result, neutral)):
            sheet.paste(image, (col * 224, index * 252 + 55))
        draw.text((4, index * 252 + 34), f'Prompted candidate {index}, SAM predicted IoU {scores[index]:.3f} (not measured ground-truth IoU)', fill='black')
        save_png(directory / f'mask_{index}.png', mask_image)
        save_png(directory / f'nature_{index}.png', result)
    save_png(directory / 'comparison.png', sheet)
    report = {'source':row['path'],'source_sha256':row['sha256'],'device':device,
              'sam_model':'vit_b','checkpoint_sha256':sha(checkpoint.read_bytes()),
              'prompt_provenance':'Assistant visual annotation of one training example; approximate diagnostic, not ground truth',
              'box_normalized_xyxy':box_normalized,'points_normalized_xy':points_normalized,
              'point_labels':labels.tolist(),'predicted_iou_scores':scores.tolist(),
              'elapsed_seconds':time.monotonic()-start,'composite_audits':audits,
              'semantic_review':'pending','full_dataset_generated':False,
              'method_difference':'This supplementary probe uses explicit SAM point/box prompting; automatic pilot remains at the frozen 32-point grid settings.'}
    save_json(directory / 'probe.json', report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
