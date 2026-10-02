"""Resumable Template 2 generation. Generated candidates are not quality approval.

Every source is retained. Uncertain masks go into a review queue; only the final
audit can mark a dataset verified. No model-training manifest is silently filtered.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import numpy as np
from PIL import Image, ImageDraw

from background_foreground import ForegroundEngine, POLICY
from background_pilot import composite, save_png
from background_scenes import make_background
from background_progress_report import render as render_progress
from color_dataset import ROOT, IDS, SPLITS, load_sources, plan_conditions, sha, save_json, atomic_bytes, csv_bytes

CONDITIONS = {'correlated':'correlated', 'balanced':'randomized', 'reversed':'reversed',
              'counterfactual':'counterfactual', 'all_nature':'all_red', 'all_urban':'all_blue'}
FIELDS = ('source','source_sha256','output','output_sha256','class_id','class_index',
          'group','split','condition','variant','aligned','quality_status')
VARIANTS = ('original','nature','urban_indoor','neutral')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def key_for(row):
    return sha(row['path'].encode())[:20]


@contextmanager
def job_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'worker.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another worker already owns this output directory') from exc
        yield


def paths(full):
    suffix = ('production_v' if full else 'refined_pilot_v') + str(POLICY['version']) + '_cuda'
    return ROOT/'background_bias_dataset'/suffix, ROOT/'reports/background_bias_current'/suffix


def load_annotations():
    path = ROOT/'project_metadata/background_annotations.json'
    return json.loads(path.read_text()).get('images', {}) if path.exists() else {}


def fingerprint(config, row, annotation):
    return sha(json.dumps({'config':config, 'source':row, 'annotation':annotation}, sort_keys=True).encode())


def cached_record(output, row, identity):
    path = output/'records'/f'{key_for(row)}.json'
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        if record['identity'] != identity or record['generation_status'] != 'generated':
            return None
        for relative, digest in record['file_hashes'].items():
            if sha((output/relative).read_bytes()) != digest:
                return None
        return record
    except (OSError, KeyError, ValueError):
        return None


def render_preview(output, row, image, masks, info, directory):
    cols = 3
    sheet = Image.new('RGB', (672, max(1, len(masks))*252), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, mask in enumerate(masks):
        selected = any(x['mask_offset']+x['selected_candidate']==i for x in info['instances'])
        candidate = next(x['candidates'][i-x['mask_offset']] for x in info['instances']
                         if x['mask_offset'] <= i < x['mask_offset']+len(x['candidates']))
        draw.text((4,i*252+4),f'{i} chosen={selected} IoU-est={candidate["predicted_iou"]:.3f} stability={candidate["stability"]:.3f}',fill='black')
        neutral,_ = composite(image, mask, Image.new('RGB',(224,224),(127,127,127)))
        for col,tile in enumerate((image.resize((224,224)),Image.fromarray(mask.astype(np.uint8)*255).convert('RGB').resize((224,224)),neutral)):
            sheet.paste(tile,(col*224,i*252+26))
    save_png(directory/'candidates.png',sheet)


def write_overviews(output, report, rows):
    # Interleaved by sample position: ten classes per page.
    by_class = {c:[r for r in rows if r['class_id']==c] for c in IDS}
    for sample in range(min(len(v) for v in by_class.values())):
        sheet=Image.new('RGB',(1120,2590),'white'); draw=ImageDraw.Draw(sheet)
        for col,name in enumerate(('Original','Selected mask','Nature','Urban','Neutral')):
            draw.text((col*224+4,6),name,fill='black')
        for index,c in enumerate(IDS):
            row=by_class[c][sample]; path=output/'records'/f'{key_for(row)}.json'
            if not path.exists(): continue
            r=json.loads(path.read_text()); directory=output/r['directory']
            draw.text((4,index*256+33),f'{Path(row["path"]).name} flags={len(r["segmentation"]["flags"])}',fill='black')
            for col,name in enumerate(('original','mask','nature','urban_indoor','neutral')):
                f=directory/f'{name}.png'
                if f.exists():
                    with Image.open(f) as im: sheet.paste(im.convert('RGB').resize((224,224)),(col*224,index*256+55))
        save_png(report/f'overview_{sample}.png',sheet)


def export_manifests(output, report, rows):
    assignments=plan_conditions(rows,{'correlation':'9/10','seed':42})
    manifests={(s,c):[] for s in SPLITS for c in (*CONDITIONS,'original','neutral')}
    queue=[]; counts=Counter(); errors=[]
    for row in rows:
        record_path=output/'records'/f'{key_for(row)}.json'
        if not record_path.exists():
            errors.append(row['path']);continue
        record=json.loads(record_path.read_text())
        if record['generation_status']!='generated':
            errors.append(row['path']);continue
        flags=record['segmentation']['flags']
        quality='review_required' if flags else 'automatic_checks_passed'
        counts[quality]+=1
        if flags:
            queue.append({'source':row['path'],'source_sha256':row['sha256'],
                          'mask_sha256':record['file_hashes'][record['directory']+'/mask.png'],
                          'flags':flags,'record':str(record_path.relative_to(output)),
                          'decision':'pending'})
        variants={name:{'red':'nature','blue':'urban_indoor'}[assignments[row['path']][mode]] for name,mode in CONDITIONS.items()}
        variants.update(original='original',neutral='neutral')
        for condition,variant in variants.items():
            relative=f'{record["directory"]}/{variant}.png'
            aligned='' if variant in ('original','neutral') else str(variant==('nature' if row['group']=='A' else 'urban_indoor'))
            manifests[row['split'],condition].append({
                'source':row['path'],'source_sha256':row['sha256'],'output':relative,
                'output_sha256':record['file_hashes'][relative],
                **{k:row[k] for k in ('class_id','class_index','group','split')},
                'condition':condition,'variant':variant,'aligned':aligned,'quality_status':quality})
    if not errors:
        for (split,condition),records in manifests.items():
            atomic_bytes(output/'candidate_manifests'/split/f'{condition}.csv',csv_bytes(records,FIELDS))
    save_json(report/'review_queue.json',queue)
    result={'status':'generated_pending_quality_review' if not errors else 'generation_incomplete',
            'total_sources':len(rows),'generated_sources':sum(counts.values()),
            'quality_counts':dict(counts),'failed_or_missing_sources':errors,
            'candidate_manifest_rows':sum(len(v) for v in manifests.values()) if not errors else 0,
            'training_ready':False,'full_dataset_verified':False,
            'note':'Candidate manifests retain all sources; flags are never silently filtered.'}
    save_json(report/'generation_summary.json',result)
    return result


def run(full=False, pilot_per_class=3):
    if not 1<=pilot_per_class<=10: raise ValueError('Pilot size must be 1–10')
    output,report=paths(full);report.mkdir(parents=True,exist_ok=True)
    rows,_,_=load_sources(ROOT)
    if not full:
        rows=[r for c in IDS for r in [x for x in rows if x['class_id']==c and x['split']=='train'][:pilot_per_class]]
    annotations=load_annotations()
    engine=None
    config={'schema':2,'policy':POLICY,'source_split_sha256':sha((ROOT/'project_metadata/source_splits.csv').read_bytes()),
            'source_selection_sha256':sha(json.dumps(rows,sort_keys=True).encode()),'full':full,
            'foreground_code_sha256':sha((ROOT/'background_foreground.py').read_bytes()),
            'generator_code_sha256':sha(Path(__file__).read_bytes()),
            'sam_checkpoint_sha256':sha((ROOT/'checkpoints/sam_vit_b_01ec64.pth').read_bytes()),
            'detector_provenance_sha256':sha((ROOT/'project_metadata/grounding_dino_checkpoint.json').read_bytes()),
            'correlation':'9/10','seed':42,'variants':VARIANTS}
    from colab_checkpoint import Backup
    backup = Backup(ROOT, output)
    with job_lock(output):
        cfg=output/'config.json'
        if cfg.exists() and json.loads(cfg.read_text())!=json.loads(json.dumps(config)):
            raise ValueError('Output configuration differs; use a new versioned dataset directory')
        save_json(cfg,config)
        started=time.monotonic(); completed=0;failed=0;resumed=0;flagged=0
        for row in rows:
            if shutil.disk_usage(output).free<2*1024**3:
                raise OSError('Less than 2 GiB free; stopped safely before filling the disk')
            annotation=annotations.get(row['path'])
            if annotation and annotation['source_sha256']!=row['sha256']:
                raise ValueError('Annotation source hash mismatch')
            data=(ROOT/row['path']).read_bytes()
            if sha(data)!=row['sha256']:
                raise ValueError(f'Source changed: {row["path"]}')
            identity=fingerprint(config,row,annotation)
            record=cached_record(output,row,identity)
            if record:
                completed+=1;resumed+=1;flagged+=bool(record['segmentation']['flags'])
                continue
            if engine is None:
                print('Loading class detector and SAM on CUDA (no CPU fallback)',flush=True)
                engine=ForegroundEngine();save_json(output/'engine_config.json',engine.config)
            directory=output/'images'/row['split']/row['class_id']/key_for(row)
            directory.mkdir(parents=True,exist_ok=True)
            record={'source':row['path'],'source_sha256':row['sha256'],'identity':identity,
                    'directory':str(directory.relative_to(output)),
                    **{k:row[k] for k in ('class_id','class_index','group','split')}}
            try:
                with Image.open(io.BytesIO(data)) as opened: image=opened.convert('RGB')
                mask,info,candidates=engine.segment(image,row['class_id'],annotation)
                record['segmentation']=info
                save_png(directory/'original.png',image.resize((224,224),Image.Resampling.BICUBIC))
                if mask is None:
                    record['generation_status']='needs_annotation';failed+=1
                else:
                    save_png(directory/'mask.png',Image.fromarray(mask.astype(np.uint8)*255))
                    for variant in ('nature','urban_indoor','neutral'):
                        bg=(Image.new('RGB',(224,224),(127,127,127)) if variant=='neutral'
                            else make_background(variant,'42|'+row['path']))
                        result,_=composite(image,mask,bg);save_png(directory/f'{variant}.png',result)
                    # Retain candidates even for unflagged images, so a later visual
                    # audit can inspect them. Always replace on annotation repair.
                    buffer=io.BytesIO();np.savez_compressed(buffer,masks=np.stack(candidates))
                    atomic_bytes(directory/'candidates.npz',buffer.getvalue())
                    if not full: render_preview(output,row,image,candidates,info,directory)
                    record['file_hashes']={str((directory/f'{name}.png').relative_to(output)):sha((directory/f'{name}.png').read_bytes()) for name in (*VARIANTS,'mask')}
                    if (directory/'candidates.npz').exists():
                        record['file_hashes'][str((directory/'candidates.npz').relative_to(output))]=sha((directory/'candidates.npz').read_bytes())
                    record['generation_status']='generated';completed+=1;flagged+=bool(info['flags'])
            except (RuntimeError,ValueError,OSError) as exc:
                record.update(generation_status='error',error=str(exc));failed+=1
                # Backend/device errors must stop instead of producing thousands of failures.
                if isinstance(exc,RuntimeError):
                    save_json(output/'records'/f'{key_for(row)}.json',record);raise
            save_json(output/'records'/f'{key_for(row)}.json',record)
            backup.save()
            elapsed=time.monotonic()-started
            status={'status':'running','full_run':full,'total_sources':len(rows),'completed':completed,
                    'failed':failed,'resumed':resumed,'flagged':flagged,'elapsed_seconds':elapsed,
                    'sam_device':'cuda','last_source':row['path'],'updated_utc':utc_now(),
                    'training_ready':False,'full_dataset_verified':False,'pid':os.getpid()}
            save_json(report/'status.json',status)
            if (completed+failed)%10==0 or failed:
                render_progress(report,status)
            print(f'[{completed+failed}/{len(rows)}] {Path(row["path"]).name}: {record["generation_status"]}; flagged={flagged}; elapsed={elapsed:.0f}s',flush=True)
        if not full: write_overviews(output,report,rows)
        # Exact 90/10 ratios only exist on the full frozen splits, not tiny pilots.
        summary=export_manifests(output,report,rows) if full else {
            'status':'pilot_generated_pending_review','generated_sources':completed,'failed_sources':failed,
            'flagged_sources':flagged,'total_sources':len(rows),'training_ready':False,'full_dataset_verified':False}
        summary.update(sam_device='cuda',elapsed_seconds=time.monotonic()-started,updated_utc=utc_now())
        save_json(report/'status.json',summary)
        if full: render_progress(report,summary)
        backup.save(force=True)
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--pilot-per-class',type=int,default=3)
    args=parser.parse_args()
    try: run(args.full,args.pilot_per_class)
    except Exception as exc:
        _,report=paths(args.full)
        save_json(report/'status.json',{'status':'stopped_with_error','error':str(exc),'training_ready':False,'full_dataset_verified':False,'updated_utc':utc_now()})
        raise
