"""Versioned target-guided SAM foreground engine with explicit quality flags."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')

import json
import time
import numpy as np

from background_grounded_pilot import PROMPTS, select_boxes
from background_pilot import select_device
from color_dataset import ROOT, sha

POLICY = {
    'version': 3, 'detector_device': 'cuda', 'sam_device': 'cuda',
    'box_threshold': .25, 'text_threshold': .20,
    'overlap_merge_intersection_over_smaller_box': .5,
    'box_expansions': [0., .15],
    'target_consistency': 'candidate must retain at least 80% of anchor foreground with mask IoU >= 0.5',
    'anchor': 'highest predicted IoU mask from highest-score original detector box',
    'annotation_point_validation': 'minimize violations first; flag any selected mask that violates a recorded positive or negative point',
    'sam_predicted_iou_min': .88, 'sam_stability_min': .95,
    'candidate_selection': 'prefer candidates passing both gates, then maximize 0.75*predicted_iou+0.25*stability',
    'multiple_instances': 'union strong nonduplicate instances; withhold weak secondary boxes pending explicit review; flag all multi-instance images',
    'weak_detection_score': .35,
    'semantic_quality': 'scores never establish correct target identity; flagged masks require review',
    'source_input': 'original RGB resolution', 'precision': 'float32',
    'mask_blur_at_224': 1.2,
}


def merge_boxes(boxes):
    """Merge overlapping alias boxes by their union to avoid truncating limbs."""
    groups = []
    for record in sorted(boxes, key=lambda r: -r['score']):
        group = {'box_xyxy': list(record['box_xyxy']), 'score': record['score'], 'members': [record]}
        changed = True
        while changed:
            changed = False
            for existing in list(groups):
                a, b = group['box_xyxy'], existing['box_xyxy']
                overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
                area = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
                if area > 0 and overlap / area >= .5:
                    group['box_xyxy'] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    group['members'].extend(existing['members'])
                    group['score'] = max(group['score'], existing['score'])
                    groups.remove(existing)
                    changed = True
        groups.append(group)
    return sorted(groups, key=lambda r: -r['score'])


def expand_box(box, fraction, width, height):
    x0, y0, x1, y1 = box
    dx, dy = (x1-x0)*fraction, (y1-y0)*fraction
    return np.array([max(0,x0-dx), max(0,y0-dy), min(width,x1+dx), min(height,y1+dy)], dtype=np.float32)


def choose_candidate(candidates):
    if not candidates:
        raise ValueError('No segmentation candidates')
    def rank(c):
        eligible = c['predicted_iou'] >= .88 and c['stability'] >= .95
        return -c.get('point_errors',0), c.get('target_consistent',True), eligible, .75*c['predicted_iou'] + .25*c['stability']
    return max(range(len(candidates)), key=lambda i: rank(candidates[i]))


class ForegroundEngine:
    def __init__(self):
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from segment_anything import SamPredictor, sam_model_registry
        select_device('cuda')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(4)
        self.torch = torch
        provenance = json.loads((ROOT/'project_metadata/grounding_dino_checkpoint.json').read_text())
        detector_dir = ROOT/'checkpoints/grounding-dino-tiny'
        for item in provenance['files']:
            if sha((detector_dir/item['file']).read_bytes()) != item['sha256']:
                raise ValueError('Detector checkpoint hash mismatch')
        sam_file = ROOT/'checkpoints/sam_vit_b_01ec64.pth'
        sam_hash = sha(sam_file.read_bytes())
        if sam_hash != 'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912':
            raise ValueError('SAM checkpoint hash mismatch')
        self.config = {'policy': POLICY, 'prompts': PROMPTS, 'detector': provenance,
                       'sam_checkpoint_sha256': sam_hash, 'torch': torch.__version__,
                       'transformers': transformers.__version__, 'cpu_fallback': False, 'gpu': torch.cuda.get_device_name(0)}
        self.processor = AutoProcessor.from_pretrained(detector_dir, local_files_only=True)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector_dir, local_files_only=True, use_safetensors=True).to('cuda').eval()
        sam = sam_model_registry['vit_b']()
        sam.load_state_dict(torch.load(sam_file, map_location='cpu', weights_only=True))
        self.predictor = SamPredictor(sam.to('cuda').eval())
        if str(next(sam.parameters()).device) != 'cuda:0':
            raise RuntimeError('SAM is not on CUDA')

    def segment(self, image, class_id, annotation=None):
        from segment_anything.utils.amg import calculate_stability_score
        start = time.monotonic()
        torch = self.torch
        inputs = self.processor(images=image, text=PROMPTS[class_id], return_tensors='pt').to('cuda')
        with torch.inference_mode():
            output = self.detector(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=.25, text_threshold=.20,
            target_sizes=[image.size[::-1]])[0]
        # Keep raw alias boxes before NMS so a larger full-body box is not lost.
        raw = []
        for box, score in zip(result['boxes'].tolist(), result['scores'].tolist()):
            valid = select_boxes([box], [score], image.width, image.height)
            if valid:
                raw.append(valid[0])
        groups = merge_boxes(raw)
        torch.cuda.synchronize()
        del inputs, output, result
        detection_seconds = time.monotonic()-start
        flags = []
        if annotation:
            # Explicit source-bound annotations may replace detector boxes.
            if 'boxes_normalized' in annotation:
                groups = [{'box_xyxy': (np.array(b)*[image.width,image.height,image.width,image.height]).tolist(),
                           'score': None, 'members': []} for b in annotation['boxes_normalized']]
        if not groups:
            return None, {'flags':['no_target_detection'], 'detections':raw, 'instances':[],
                          'detection_seconds':detection_seconds, 'sam_seconds':0}, []
        if len(groups)>1:
            flags.append('multiple_instances_require_review')
            if any(g['score'] is not None and g['score']<.35 for g in groups[1:]):
                flags.append('weak_secondary_detection_withheld')
                groups=[groups[0]]+[g for g in groups[1:] if g['score'] is None or g['score']>=.35]
        if len(groups)>10:
            return None, {'flags':['too_many_detections_require_annotation'], 'detections':raw,
                          'instances':[], 'detection_seconds':detection_seconds, 'sam_seconds':0}, []
        start_sam = time.monotonic()
        all_masks, instances = [], []
        with torch.inference_mode():
            self.predictor.set_image(np.asarray(image))
            if str(self.predictor.features.device) != 'cuda:0':
                raise RuntimeError('SAM image embedding is not on CUDA')
            for instance, group in enumerate(groups):
                if group['score'] is not None and group['score'] < .35:
                    flags.append('weak_target_detection')
                candidates = []
                instance_masks = []
                anchor_box = max(group['members'],key=lambda m:m['score'])['box_xyxy'] if group['members'] else group['box_xyxy']
                prompts=[('anchor',0.,anchor_box),('merged',0.,group['box_xyxy']),('expanded',.15,group['box_xyxy'])]
                for prompt_kind,expansion,base_box in prompts:
                    box = expand_box(base_box, expansion, image.width, image.height)
                    kwargs = {}
                    if annotation and annotation.get('points_normalized'):
                        kwargs = {'point_coords': np.array(annotation['points_normalized'],dtype=np.float32)*[image.width,image.height],
                                  'point_labels': np.array(annotation['point_labels'],dtype=np.int32)}
                    logits, scores, _ = self.predictor.predict(box=box, multimask_output=True, return_logits=True, **kwargs)
                    stabilities = calculate_stability_score(torch.from_numpy(logits), 0., 1.).tolist()
                    for mask, score, stability in zip(logits>0, scores, stabilities):
                        candidates.append({'predicted_iou':float(score),'stability':stability,
                                           'prompt_kind':prompt_kind,
                                           'box_expansion':expansion,'box_xyxy':box.tolist(),
                                           'area_fraction':float(mask.mean())})
                        instance_masks.append(mask)
                for candidate,candidate_mask in zip(candidates,instance_masks):
                    candidate['point_errors']=0
                    if annotation:
                        for point,label in zip(annotation.get('points_normalized',[]),annotation.get('point_labels',[])):
                            x=min(image.width-1,max(0,int(point[0]*image.width)))
                            y=min(image.height-1,max(0,int(point[1]*image.height)))
                            candidate['point_errors']+=int(bool(candidate_mask[y,x])!=bool(label))
                anchor_index=max(range(3),key=lambda i:(-candidates[i]['point_errors'],candidates[i]['predicted_iou']))
                anchor=instance_masks[anchor_index]
                for candidate,candidate_mask in zip(candidates,instance_masks):
                    intersection=int((anchor & candidate_mask).sum())
                    union=int((anchor | candidate_mask).sum())
                    candidate['anchor_coverage']=intersection/max(1,int(anchor.sum()))
                    candidate['anchor_mask_iou']=intersection/max(1,union)
                    candidate['target_consistent']=(candidate['anchor_coverage']>=.8 and candidate['anchor_mask_iou']>=.5)
                choice = choose_candidate(candidates)
                chosen = candidates[choice]
                if chosen['point_errors']:
                    flags.append('annotation_point_constraints_failed')
                if chosen['predicted_iou'] < .88:
                    flags.append('low_sam_predicted_iou')
                if chosen['stability'] < .95:
                    flags.append('low_sam_stability')
                instances.append({'detector_group':group, 'selected_candidate':choice,
                                  'candidates':candidates,'mask_offset':len(all_masks)})
                all_masks.extend(instance_masks)
        selected = [all_masks[r['mask_offset']+r['selected_candidate']] for r in instances]
        mask = np.logical_or.reduce(selected)
        if not .005 < mask.mean() < .95:
            flags.append('extreme_mask_area')
        torch.cuda.synchronize()
        return mask, {'flags':sorted(set(flags)), 'detections':raw, 'instances':instances,
                      'detection_seconds':detection_seconds, 'sam_seconds':time.monotonic()-start_sam,
                      'sam_parameter_device':str(next(self.predictor.model.parameters()).device),
                      'sam_embedding_device':str(self.predictor.features.device),
                      'cuda_allocated_bytes':torch.cuda.memory_allocated(),
                      'detector_parameter_device':str(next(self.detector.parameters()).device),
                      'area_fraction':float(mask.mean()),
                      'annotation':annotation}, all_masks
