"""Independent saved-pixel, source, manifest and quality-release checks for Shape."""
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from color_dataset import ROOT, IDS, SPLITS, load_sources, read_csv, save_json, sha, atomic_bytes
from shape_dataset import CONDITIONS, VARIANTS


def audit(root=ROOT,output=None,report=None):
    output=output or root/'shape_bias_dataset/production_v1';report=report or root/'reports/shape_bias_current'
    rows,_,_=load_sources(root);mapping={r['path']:r for r in rows}
    config=json.loads((output/'config.json').read_text());hashes=json.loads((output/'hashes.json').read_text())
    issues=[];checked=0;manifest_count=0;variants={};glyphs={};geometry={}
    if config['source_split_sha256']!=sha((root/'project_metadata/source_splits.csv').read_bytes()):issues.append('Split hash mismatch')
    if config['classes_metadata_sha256']!=sha((root/'project_metadata/selected_classes.json').read_bytes()):issues.append('Class metadata hash mismatch')
    for row in rows:
        try:
            source=root/row['path'];assert sha(source.read_bytes())==row['sha256'],'Source bytes changed'
            with Image.open(source) as im:
                rgb=im.convert('RGB');original=np.asarray(rgb.resize((224,224),Image.Resampling.BICUBIC));photo=np.asarray(rgb.resize((192,192),Image.Resampling.BICUBIC))
            framed=np.full((224,224,3),230,dtype=np.uint8);framed[:192,16:208]=photo
            arrays={}
            for v in VARIANTS:
                p=f'images/{row["split"]}/{row["class_id"]}/{sha(row["path"].encode())[:20]}/{v}.png'
                assert sha((output/p).read_bytes())==hashes[p],'Output hash mismatch'
                with Image.open(output/p) as im:
                    assert im.mode=='RGB' and im.size==(224,224),'Image format differs'
                    arrays[v]=np.array(im)
                variants[row['path'],v]=(p,hashes[p]);checked+=1
            assert np.array_equal(arrays['original'],original),'Original mismatch'
            assert np.array_equal(arrays['cue_removed'],framed),'Framing control mismatch'
            for v in ('circle','triangle'):
                a=arrays[v];different=np.any(a!=framed,axis=2)
                assert not different[:192].any(),'Cue occludes photograph'
                assert different.sum()==256,'Cue area differs'
                assert np.all(a[different]==55),'Cue color differs'
                assert np.array_equal(a[~different],framed[~different]),'Pixels outside cue changed'
                if v in glyphs:assert np.array_equal(different,glyphs[v]),'Cue geometry varies by source'
                else:
                    glyphs[v]=different;y,x=np.where(different)
                    geometry[v]={'area':int(len(x)),'centroid_xy':[float(x.mean()),float(y.mean())],
                                 'bbox_xyxy':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]}
                    assert abs(x.mean()-111.5)<1 and abs(y.mean()-207.5)<1,'Cue centroid offset'
            assert not np.array_equal(glyphs['circle'],glyphs['triangle']),'Cue shapes identical'
        except (AssertionError,OSError,KeyError,ValueError) as exc:issues.append(row['path']+': '+str(exc))
    # Independent row-width checks distinguish a symmetric disk from a widening triangle.
    if len(glyphs)==2:
        circle=glyphs['circle'].sum(axis=1);circle=circle[circle>0]
        triangle=glyphs['triangle'].sum(axis=1);triangle=triangle[triangle>0]
        if not np.array_equal(circle,circle[::-1]):issues.append('Circle lacks vertical symmetry')
        if not np.all(np.diff(triangle)>=0):issues.append('Triangle does not widen towards its base')
    selected={}
    for split in SPLITS:
        expected={r['path'] for r in rows if r['split']==split}
        for condition in CONDITIONS:
            try:
                items=read_csv(output/'candidate_manifests'/split/(condition+'.csv'));manifest_count+=len(items)
                assert len(items)==len(expected) and {r['source'] for r in items}==expected,'Missing/extra/duplicate sources'
                counts=Counter()
                for item in items:
                    r=mapping[item['source']];v=item['variant'];p,digest=variants[r['path'],v]
                    assert all(item[k]==str(r[k]) for k in ('class_id','class_index','group','split')),'Label/split differs'
                    assert item['source_sha256']==r['sha256'] and item['condition']==condition,'Source/condition differs'
                    assert item['output']==p and item['output_sha256']==digest,'Output differs'
                    if condition in ('original','cue_removed'):
                        assert v==condition and item['aligned']=='','Control differs'
                    else:
                        assert v in ('circle','triangle'),'Wrong cue variant'
                        aligned=v==('circle' if r['group']=='A' else 'triangle')
                        assert item['aligned']==str(aligned),'Alignment differs'
                        counts[r['class_id']]+=aligned
                        selected[split,condition,r['path']]=v
                        if condition=='all_circle':assert v=='circle'
                        if condition=='all_triangle':assert v=='triangle'
                for c in IDS:
                    n=sum(r['class_id']==c for r in items)
                    if condition in ('correlated','balanced','reversed','counterfactual'):
                        exp={'correlated':n*9//10,'balanced':n//2,'reversed':n//10,'counterfactual':0}[condition]
                        assert counts[c]==exp,'Per-class correlation differs'
            except (AssertionError,OSError,KeyError,ValueError) as exc:issues.append(f'{split}/{condition}: {exc}')
        for s in expected:
            if selected.get((split,'correlated',s))==selected.get((split,'reversed',s)):issues.append('Reversal not paired: '+s)
        color=root/'color_bias_dataset/manifests'/split/'correlated.csv'
        if not color.exists():issues.append('Missing Color comparison manifest: '+str(color))
        else:
            for item in read_csv(color):
                if selected.get((split,'correlated',item['source']))!={'red':'circle','blue':'triangle'}[item['variant']]:issues.append('Color/Shape membership differs: '+item['source'])
    # Thirty observations: one per frozen class/split; explicit config-bound review.
    config_sha=sha((output/'config.json').read_bytes());sample_file=report/'visual_sample.json';review_file=report/'visual_review.json'
    sample=json.loads(sample_file.read_text()) if sample_file.exists() else {};review=json.loads(review_file.read_text()) if review_file.exists() else {}
    expected_sample=[min((r for r in rows if r['class_id']==c and r['split']==s),key=lambda r:sha(('shape-review-42|'+r['path']).encode()))['path'] for s in SPLITS for c in IDS]
    valid_sample=(sample.get('config_sha256')==config_sha and [r['source'] for r in sample.get('sources',[])]==expected_sample)
    pending=[]
    for source in expected_sample:
        d=review.get('images',{}).get(source,{}) if review.get('config_sha256')==config_sha else {}
        valid=(valid_sample and d.get('decision')=='approve' and d.get('source_sha256')==mapping[source]['sha256']
               and d.get('variant_hashes')=={v:variants.get((source,v),(None,None))[1] for v in VARIANTS})
        if not valid:pending.append(source)
    ready=not issues and not pending
    result={'status':'complete_verified' if ready else ('failed' if issues else 'numeric_audit_passed'),
            'source_images_verified':len(rows)-sum(': ' in i and i.startswith('archive/') for i in issues),
            'pixel_verified_pngs':checked,'manifest_rows':manifest_count,'issues':issues,
            'cue_geometry':geometry,'visual_sample_pending':len(pending),'training_ready':ready}
    save_json(report/'audit.json',result);save_json(output/'status.json',result)
    if ready:
        for p in (output/'candidate_manifests').glob('*/*.csv'):atomic_bytes(output/'manifests'/p.relative_to(output/'candidate_manifests'),p.read_bytes())
    elif (output/'manifests').exists():
        # Keep stale releases out of training paths; preserve them for diagnosis.
        old=output/'previous_manifests_after_failed_audit'
        if old.exists():raise RuntimeError('An older invalid release is already preserved; inspect before another release')
        (output/'manifests').rename(old)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    raise SystemExit(1 if audit()['issues'] else 0)
