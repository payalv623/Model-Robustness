"""Independent saved-pixel, manifest, split, device and quality-release audit."""
import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from background_dataset import paths, key_for, VARIANTS, CONDITIONS, fingerprint, load_annotations
from background_scenes import make_background
from color_dataset import ROOT, IDS, SPLITS, load_sources, read_csv, save_json, sha, atomic_bytes


def audit(output, report, root=ROOT):
    config=json.loads((output/'config.json').read_text())
    rows,_,_=load_sources(root)
    if not config['full']:
        record_sources={json.loads(p.read_text())['source'] for p in (output/'records').glob('*.json')}
        rows=[r for r in rows if r['path'] in record_sources]
    issues=[]; unresolved=[]; checked=0; source_count=0; records={}
    if config['source_split_sha256']!=sha((root/'project_metadata/source_splits.csv').read_bytes()):
        issues.append('Frozen split hash mismatch')
    if config['source_selection_sha256']!=sha(json.dumps(rows,sort_keys=True).encode()):
        issues.append('Source selection differs from frozen configuration')
    review_path=report/'quality_review.json'
    review=json.loads(review_path.read_text()) if review_path.exists() else {}
    reviews=review.get('images',{}) if review.get('config_sha256')==sha((output/'config.json').read_bytes()) else {}
    if root==ROOT:
        annotations=load_annotations()
    else:
        annotations={}
    for row in rows:
        try:
            record=json.loads((output/'records'/f'{key_for(row)}.json').read_text())
            if record['source']!=row['path'] or record['source_sha256']!=row['sha256']:
                raise ValueError('Source record identity mismatch')
            if record['identity']!=fingerprint(config,row,annotations.get(row['path'])):
                raise ValueError('Stale source configuration or annotation')
            if record['generation_status']!='generated': raise ValueError('Source not generated')
            info=record['segmentation']
            if info['sam_parameter_device']!='mps:0' or info['sam_embedding_device']!='mps:0':
                raise ValueError('SAM device is not MPS')
            directory=output/record['directory']
            if not directory.resolve().is_relative_to(output.resolve()): raise ValueError('Unsafe image path')
            for relative,digest in record['file_hashes'].items():
                if not (output/relative).resolve().is_relative_to(output.resolve()): raise ValueError('Unsafe hash path')
                if sha((output/relative).read_bytes())!=digest: raise ValueError('Output hash mismatch')
            data=(root/row['path']).read_bytes()
            if sha(data)!=row['sha256']: raise ValueError('Source bytes changed')
            with Image.open(io.BytesIO(data)) as opened: original=opened.convert('RGB')
            with Image.open(directory/'mask.png') as opened:
                mask=np.asarray(opened)
                if opened.mode!='L' or opened.size!=original.size or not set(np.unique(mask)).issubset({0,255}):
                    raise ValueError('Mask dimensions/mode/binary values invalid')
                alpha=opened.resize((224,224),Image.Resampling.NEAREST).filter(ImageFilter.GaussianBlur(1.2))
            resized=original.resize((224,224),Image.Resampling.BICUBIC)
            foreground=np.asarray(resized,dtype=np.int32)
            weights=np.asarray(alpha,dtype=np.int32)[:,:,None]
            for variant in VARIANTS:
                if variant=='original': expected=np.asarray(resized)
                else:
                    bg=Image.new('RGB',(224,224),(127,127,127)) if variant=='neutral' else make_background(variant,'42|'+row['path'])
                    expected=((foreground*weights+np.asarray(bg,dtype=np.int32)*(255-weights)+127)//255).astype(np.uint8)
                with Image.open(directory/f'{variant}.png') as saved:
                    saved.load()
                    if saved.mode!='RGB' or saved.size!=(224,224) or not np.array_equal(np.asarray(saved),expected):
                        raise ValueError(f'Independent pixel check failed: {variant}')
                checked+=1
            source_count+=1;records[row['path']]=record
            decision=reviews.get(row['path'],{})
            valid_review=(decision.get('mask_sha256')==record['file_hashes'][record['directory']+'/mask.png'] and
                          decision.get('source_sha256')==row['sha256'])
            if info['flags'] and not (valid_review and decision.get('decision')=='approve'):
                unresolved.append(row['path'])
            if valid_review and decision.get('decision')=='reject' and row['path'] not in unresolved:
                unresolved.append(row['path'])
        except (OSError,ValueError,KeyError) as exc:
            issues.append(f'{row["path"]}: {exc}')
    manifest_rows=0
    if config['full']:
        row_map={r['path']:r for r in rows}; reference={}
        for split in SPLITS:
            expected_sources={r['path'] for r in rows if r['split']==split}
            for condition in (*CONDITIONS,'original','neutral'):
                file=output/'candidate_manifests'/split/f'{condition}.csv'
                try:
                    manifest=read_csv(file);manifest_rows+=len(manifest)
                    if len(manifest)!=len(expected_sources) or {r['source'] for r in manifest}!=expected_sources:
                        raise ValueError('Manifest missing, extra or repeated sources')
                    counts=Counter()
                    for item in manifest:
                        row=row_map[item['source']];r=records[item['source']]
                        if any(item[k]!=str(row[k]) for k in ('class_id','class_index','group','split')):
                            raise ValueError('Manifest class/split mismatch')
                        if item['source_sha256']!=row['sha256'] or item['condition']!=condition:
                            raise ValueError('Manifest condition/source mismatch')
                        v=item['variant'];relative=r['directory']+'/'+v+'.png'
                        if item['output']!=relative or item['output_sha256']!=r['file_hashes'][relative]:
                            raise ValueError('Manifest output mismatch')
                        expected_quality='review_required' if r['segmentation']['flags'] else 'automatic_checks_passed'
                        if item['quality_status']!=expected_quality: raise ValueError('Manifest hides quality flag')
                        if condition in ('original','neutral'):
                            if v!=condition or item['aligned']!='': raise ValueError('Invalid control variant')
                        else:
                            if v not in ('nature','urban_indoor'): raise ValueError('Invalid biased variant')
                            aligned=v==('nature' if row['group']=='A' else 'urban_indoor')
                            if item['aligned']!=str(aligned): raise ValueError('Wrong alignment label')
                            counts[row['class_id']]+=aligned
                            reference[split,condition,row['path']]=v
                            if condition=='all_nature' and v!='nature': raise ValueError('Wrong paired nature variant')
                            if condition=='all_urban' and v!='urban_indoor': raise ValueError('Wrong paired urban variant')
                    for c in IDS:
                        n=sum(r['class_id']==c for r in manifest)
                        expected={'correlated':n*9//10,'balanced':n//2,'reversed':n//10,'counterfactual':0}
                        if condition in expected and counts[c]!=expected[condition]: raise ValueError('Class correlation count mismatch')
                except (OSError,ValueError,KeyError) as exc: issues.append(f'{file}: {exc}')
            for source in expected_sources:
                if reference.get((split,'correlated',source))==reference.get((split,'reversed',source)):
                    issues.append(f'Reversed assignment not paired: {source}')
            # Independently compare aligned/minority membership with saved Color manifests.
            color=root/'color_bias_dataset/manifests'/split/'correlated.csv'
            if color.exists():
                for row in read_csv(color):
                    expected={'red':'nature','blue':'urban_indoor'}[row['variant']]
                    if reference.get((split,'correlated',row['source']))!=expected:
                        issues.append(f'Background/Color source membership differs: {row["source"]}')
    # Freeze a review sample independently of model scores: five per class/split.
    sample=[]
    for split in SPLITS:
        for c in IDS:
            candidates=[r for r in rows if r['class_id']==c and r['split']==split]
            sample.extend(sorted(candidates,key=lambda r:sha(('background-audit-42|'+r['path']).encode()))[:5])
    sample_missing=[]
    for row in sample:
        r=records.get(row['path']);decision=reviews.get(row['path'],{})
        if (not r or decision.get('decision')!='approve' or decision.get('source_sha256')!=row['sha256'] or
                decision.get('mask_sha256')!=r['file_hashes'][r['directory']+'/mask.png']):
            sample_missing.append(row['path'])
    save_json(report/'required_visual_sample.json',{'config_sha256':sha((output/'config.json').read_bytes()),'sources':[r['path'] for r in sample],'pending':sample_missing})
    release=(config['full'] and not issues and not unresolved and not sample_missing and source_count==len(rows))
    result={'status':'failed' if issues else 'numeric_audit_passed', 'source_images_verified':source_count,
            'pixel_verified_pngs':checked,'manifest_rows':manifest_rows,'issues':issues,
            'quality_review_unresolved':len(unresolved),'unresolved_sources':unresolved,
            'required_visual_sample_pending':len(sample_missing),'full_dataset_verified':release,
            'training_ready':release,'sam_device':'mps'}
    save_json(report/'audit.json',result)
    if release:
        for p in (output/'candidate_manifests').glob('*/*.csv'):
            atomic_bytes(output/'manifests'/p.relative_to(output/'candidate_manifests'),p.read_bytes())
        save_json(output/'status.json',{'status':'complete_verified','training_ready':True})
    print(json.dumps({k:v for k,v in result.items() if k not in ('unresolved_sources','issues')},indent=2))
    if issues: print('\n'.join(issues[:10]))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--full',action='store_true')
    args=parser.parse_args();out,report=paths(args.full);result=audit(out,report)
    raise SystemExit(1 if result['issues'] else 0)
