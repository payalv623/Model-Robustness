"""Verify retained original CUDA/MPS records, policies, annotations, and device provenance."""
import json
from pathlib import Path
from color_dataset import sha

def safe(root,relative):
    p=root/relative
    if not p.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe provenance path')
    return p

def verify_backend(record,row,config,annotation,output):
    from background_dataset import fingerprint
    proof=record['merge_provenance'];provider=proof['provider']
    if provider not in ('cuda','mps'):raise ValueError('Unexpected provider')
    evidence=config['merge_inputs'][provider]
    original=safe(output,proof['record']).read_bytes()
    if sha(original)!=proof['record_sha256']:raise ValueError('Original record changed')
    old=json.loads(original)
    cfg_bytes=safe(output,evidence['config']).read_bytes()
    ann_bytes=safe(output,evidence['annotations']).read_bytes()
    if sha(cfg_bytes)!=evidence['config_sha256'] or sha(ann_bytes)!=evidence['annotations_sha256']:
        raise ValueError('Original configuration or annotations changed')
    for relative,h in evidence['support_files'].items():
        if sha(safe(output,relative).read_bytes())!=h:raise ValueError('Original supporting code/engine metadata changed')
    old_cfg=json.loads(cfg_bytes);old_ann=json.loads(ann_bytes).get('images',{}).get(row['path'])
    if old_ann!=annotation:raise ValueError('Merged annotation differs from chosen original')
    if old['identity']!=fingerprint(old_cfg,row,old_ann):raise ValueError('Original identity is invalid')
    for k in ('source','source_sha256','directory','generation_status','segmentation','class_id','class_index','group','split'):
        if old[k]!=record[k]:raise ValueError('Original record field altered: '+k)
    if old['source']!=row['path'] or old['source_sha256']!=row['sha256'] or old['generation_status']!='generated':
        raise ValueError('Original record/source mismatch')
    for relative,h in old['file_hashes'].items():
        if record['file_hashes'].get(relative)!=h:raise ValueError('Original output hash changed')
    info=old['segmentation']
    expected='cuda:0' if provider=='cuda' else 'mps:0'
    if info.get('sam_parameter_device')!=expected or info.get('sam_embedding_device')!=expected:
        raise ValueError('SAM backend mismatch')
    if provider=='cuda' and info.get('detector_parameter_device')!='cuda:0':raise ValueError('CUDA detector mismatch')
    if provider=='mps' and old_cfg['policy']['detector_device']!='cpu':raise ValueError('MPS detector policy mismatch')
