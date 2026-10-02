"""Template 3: a synthetic geometric shortcut, with color and cue area controlled."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from color_dataset import (ROOT, IDS, SPLITS, atomic_bytes, csv_bytes, load_sources,
                           plan_conditions, read_csv, save_json, sha)

VARIANTS=('original','cue_removed','circle','triangle')
CONDITIONS=('original','cue_removed','correlated','balanced','reversed',
            'counterfactual','all_circle','all_triangle')
FIELDS=('source','source_sha256','output','output_sha256','class_id','class_index',
        'group','split','condition','variant','aligned')
BACKGROUND=230
INK=55
CUE_AREA=256


def glyph(kind):
    """Select equal numbers of pixels from concentric circle/triangle level sets."""
    y,x=np.mgrid[192:224,0:224]
    dx=x-111.5;dy=y-207.5
    score=dx*dx+dy*dy if kind=='circle' else np.maximum(dy,(np.sqrt(3)*np.abs(dx)-dy)/2)
    assert kind in ('circle','triangle')
    order=np.argsort(score.ravel(),kind='stable')
    result=np.zeros((224,224),dtype=bool)
    footer=np.zeros(score.size,dtype=bool);footer[order[:CUE_AREA]]=True
    result[192:,:]=footer.reshape(score.shape)
    return result


def render(data):
    with Image.open(io.BytesIO(data)) as source:
        rgb=source.convert('RGB')
        original=rgb.resize((224,224),Image.Resampling.BICUBIC)
        photo=rgb.resize((192,192),Image.Resampling.BICUBIC)
    framed=Image.new('RGB',(224,224),(BACKGROUND,)*3)
    framed.paste(photo,(16,0))
    result={'original':original,'cue_removed':framed}
    for kind in ('circle','triangle'):
        a=np.array(framed);a[glyph(kind)]=INK;result[kind]=Image.fromarray(a)
    return result


def relative(row,variant):
    return f'images/{row["split"]}/{row["class_id"]}/{sha(row["path"].encode())[:20]}/{variant}.png'


def make_config(root):
    return {'schema':1,'template':'shape','version':1,'correlation':'9/10','seed':42,
            'image_size':[224,224],'source_split_sha256':sha((root/'project_metadata/source_splits.csv').read_bytes()),
            'classes_metadata_sha256':sha((root/'project_metadata/selected_classes.json').read_bytes()),
            'generator_code_sha256':sha(Path(__file__).read_bytes()),
            'pillow_version':pillow_version,'numpy_version':np.__version__,
            'variants':list(VARIANTS),'conditions':list(CONDITIONS),
            'photo_region_xyxy':[16,0,208,192],
            'photo_resize':'Pillow BICUBIC direct resize to 192x192; full source, no crop',
            'cue_region_xyxy':[0,192,224,224],'background_rgb':[BACKGROUND]*3,'cue_rgb':[INK]*3,
            'cue_pixels':CUE_AREA,'cue_masks_sha256':{k:sha(glyph(k).tobytes()) for k in ('circle','triangle')},
            'groups':{'A':'circle','B':'triangle'},
            'assignment':'Identical per-source majority/minority membership to Color; exact 90/10 per class/split',
            'scope':'Sensitivity to an added geometric marker; does not measure intrinsic object shape-vs-texture bias.',
            'control':'cue_removed retains identical photo framing and footer; original is an additional unframed baseline.',
            'known_geometry_difference':'Equal cue color and pixel area; outline/perimeter and bounding box differ by shape. Raster centroids are reported.'}


def assignments(rows):
    old=plan_conditions(rows,{'correlation':'9/10','seed':42})
    modes={'correlated':'correlated','balanced':'randomized','reversed':'reversed',
           'counterfactual':'counterfactual','all_circle':'all_red','all_triangle':'all_blue'}
    return {r['path']:{'original':'original','cue_removed':'cue_removed',
            **{c:{'red':'circle','blue':'triangle'}[old[r['path']][m]] for c,m in modes.items()}}
            for r in rows}


def generate(root=ROOT, output=None, workers=4):
    root=root.resolve();output=(output or root/'shape_bias_dataset/production_v1').resolve()
    if output==root or output.is_relative_to(root/'archive') or (root/'archive').is_relative_to(output):
        raise ValueError('Output must be separate from original sources')
    output.mkdir(parents=True,exist_ok=True)
    rows,_,_=load_sources(root);config=make_config(root);config_file=output/'config.json'
    if config_file.exists() and json.loads(config_file.read_text())!=config:
        raise ValueError('Existing output has different configuration; choose a new output directory')
    save_json(config_file,config)
    status={'status':'generating','training_ready':False,'total_sources':len(rows)}
    save_json(output/'status.json',status)
    cache_path=output/'hashes.json'
    hashes=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    def one(row):
        data=(root/row['path']).read_bytes()
        if sha(data)!=row['sha256']:raise ValueError('Source changed: '+row['path'])
        names={v:relative(row,v) for v in VARIANTS}
        if all(p in hashes and (output/p).is_file() and sha((output/p).read_bytes())==hashes[p] for p in names.values()):
            return {p:hashes[p] for p in names.values()}
        values={}
        for v,im in render(data).items():
            b=io.BytesIO();im.save(b,format='PNG');value=b.getvalue()
            atomic_bytes(output/names[v],value);values[names[v]]=sha(value)
        return values
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i,result in enumerate(pool.map(one,rows),1):
                hashes.update(result)
                if i%500==0 or i==len(rows):
                    save_json(cache_path,hashes)
                    status.update(generated_sources=i,updated_utc=datetime.now(timezone.utc).isoformat())
                    save_json(output/'status.json',status)
                    print(f'Shape {i}/{len(rows)} sources',flush=True)
        planned=assignments(rows)
        for split in SPLITS:
            for condition in CONDITIONS:
                items=[]
                for r in rows:
                    if r['split']!=split:continue
                    v=planned[r['path']][condition];p=relative(r,v)
                    items.append({'source':r['path'],'source_sha256':r['sha256'],'output':p,'output_sha256':hashes[p],
                                  **{k:r[k] for k in ('class_id','class_index','group','split')},
                                  'condition':condition,'variant':v,
                                  'aligned':'' if v in ('original','cue_removed') else str(v==('circle' if r['group']=='A' else 'triangle'))})
                atomic_bytes(output/'candidate_manifests'/split/f'{condition}.csv',csv_bytes(items,FIELDS))
        status.update(status='generated_pending_audit',generated_sources=len(rows))
        save_json(output/'status.json',status)
        return output
    except BaseException as exc:
        save_json(cache_path,hashes);status.update(status='stopped_with_error',error=str(exc));save_json(output/'status.json',status)
        raise


def visual_sample(root,output,report):
    rows,_,_=load_sources(root)
    chosen=[min((r for r in rows if r['class_id']==c and r['split']==s),
                key=lambda r:sha(('shape-review-42|'+r['path']).encode())) for s in SPLITS for c in IDS]
    items=[]
    for page in range(3):
        sheet=Image.new('RGB',(896,2580),'white');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(chosen[page*10:(page+1)*10]):
            draw.text((4,i*258+3),f'{page*10+i:02d} {r["class_id"]} {r["split"]} | original, cue removed, circle, triangle',fill='black')
            for j,v in enumerate(VARIANTS):
                with Image.open(output/relative(r,v)) as im:sheet.paste(im,(224*j,i*258+24))
            items.append({'source':r['path'],'source_sha256':r['sha256'],'page':page})
        report.mkdir(parents=True,exist_ok=True);sheet.save(report/f'sample_{page}.jpg',quality=95,subsampling=0)
    save_json(report/'visual_sample.json',{'config_sha256':sha((output/'config.json').read_bytes()),'sources':items})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('workers must be 1..8')
    out=generate(workers=args.workers)
    visual_sample(ROOT,out,ROOT/'reports/shape_bias_current')
