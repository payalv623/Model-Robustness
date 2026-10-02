"""Import verified CUDA records without recomputing images; preserve their original records."""
import json
import os
import shutil
from pathlib import Path
from color_dataset import ROOT, sha, save_json


def import_cuda(output, config, rows, annotations):
    from background_dataset import key_for, fingerprint, cached_record
    parent=ROOT.parent/'template2_project/background_bias_dataset/production_v3_cuda'
    parent_config=json.loads((parent/'config.json').read_text())
    if sha((parent/'config.json').read_bytes())!=config['policy']['imported_cuda_config_sha256']:
        raise ValueError('Parent CUDA configuration changed')
    save_json(output/'imported_cuda_config.json',parent_config)
    imported=0
    for row in rows:
        key=key_for(row); annotation=annotations.get(row['path'])
        target_identity=fingerprint(config,row,annotation)
        if cached_record(output,row,target_identity):
            continue
        old=cached_record(parent,row,fingerprint(parent_config,row,annotation))
        if not old:
            continue
        if old['source_sha256']!=row['sha256'] or old['source']!=row['path']:
            raise ValueError('Parent source identity mismatch')
        info=old['segmentation']
        if any(info.get(k)!='cuda:0' for k in ('detector_parameter_device','sam_parameter_device','sam_embedding_device')):
            raise ValueError('Imported record was not verified CUDA')
        for relative in old['file_hashes']:
            source=parent/relative;dest=output/relative
            if not source.resolve().is_relative_to(parent.resolve()) or not dest.resolve().is_relative_to(output.resolve()):
                raise ValueError('Unsafe import path')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                try:os.link(source,dest)
                except OSError:shutil.copyfile(source,dest)
        old_bytes=(parent/'records'/f'{key}.json').read_bytes()
        provenance_path=f'imported_cuda_records/{key}.json'
        target=output/provenance_path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(old_bytes)
        updated=dict(old);updated['file_hashes']=dict(old['file_hashes'])
        updated['file_hashes'][provenance_path]=sha(old_bytes)
        updated['identity']=target_identity
        updated['imported_cuda_provenance']={'record':provenance_path,'record_sha256':sha(old_bytes)}
        save_json(output/'records'/f'{key}.json',updated)
        imported+=1
    print(f'Imported {imported} verified CUDA records; their image bytes are unchanged.',flush=True)


def verify_backend(record,row,config,annotation,output):
    from background_dataset import fingerprint
    info=record['segmentation']
    devices={info.get(k) for k in ('detector_parameter_device','sam_parameter_device','sam_embedding_device')}
    if devices=={'cpu'}:
        if 'imported_cuda_provenance' in record:raise ValueError('CPU record wrongly claims CUDA import')
        return
    if devices!={'cuda:0'}:raise ValueError('Unexpected or mixed model devices')
    proof=record['imported_cuda_provenance'];p=output/proof['record']
    if not p.resolve().is_relative_to(output.resolve()):raise ValueError('Unsafe provenance path')
    data=p.read_bytes()
    if sha(data)!=proof['record_sha256']:raise ValueError('CUDA provenance record changed')
    old=json.loads(data);old_config=json.loads((output/'imported_cuda_config.json').read_text())
    if sha((output/'imported_cuda_config.json').read_bytes())!=config['policy']['imported_cuda_config_sha256']:
        raise ValueError('CUDA provenance configuration changed')
    if old['identity']!=fingerprint(old_config,row,annotation):raise ValueError('Invalid original CUDA identity')
    if old['segmentation']!=info or old['source_sha256']!=row['sha256'] or old['directory']!=record['directory']:
        raise ValueError('Imported CUDA data changed')
    for relative,h in old['file_hashes'].items():
        if record['file_hashes'].get(relative)!=h:raise ValueError('CUDA output hash was altered')
