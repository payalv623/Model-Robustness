"""Hash-bound contact sheets and explicit decisions for saved Template 2 masks.

This tool never infers approval from SAM scores and never releases manifests.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from color_dataset import ROOT, load_sources, save_json, sha


def records_for(bundle):
    return {r['source']: r for p in sorted((bundle/'records').glob('*.json'))
            for r in [json.loads(p.read_text())]}


def make_sheets(bundle, destination, sources, page_size=10):
    records = records_for(bundle)
    frozen = {r['path']: r for r in load_sources()[0]}
    destination.mkdir(parents=True, exist_ok=True)
    index = []
    for start in range(0, len(sources), page_size):
        batch = sources[start:start+page_size]
        sheet = Image.new('RGB', (1344, ((len(batch)+1)//2)*260), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, source in enumerate(batch):
            r = records[source]
            raw = (ROOT/source).read_bytes()
            mask_path = bundle/r['directory']/'mask.png'
            assert sha(raw) == r['source_sha256'] == frozen[source]['sha256']
            assert sha(mask_path.read_bytes()) == r['file_hashes'][r['directory']+'/mask.png']
            with Image.open(ROOT/source) as opened:
                original = opened.convert('RGB').resize((224,224), Image.Resampling.BICUBIC)
            with Image.open(mask_path) as opened:
                assert opened.mode == 'L'
                mask = opened.resize((224,224), Image.Resampling.NEAREST)
            alpha = mask.filter(ImageFilter.GaussianBlur(1.2))
            a=np.asarray(alpha,dtype=np.int32)[:,:,None]
            neutral=Image.fromarray(((np.asarray(original,dtype=np.int32)*a+127*(255-a)+127)//255).astype(np.uint8))
            # If an exported neutral exists, verify both its bytes and independent reconstruction.
            neutral_path=mask_path.with_name('neutral.png')
            if neutral_path.exists():
                assert sha(neutral_path.read_bytes()) == r['file_hashes'][r['directory']+'/neutral.png']
                assert np.array_equal(np.asarray(Image.open(neutral_path)), np.asarray(neutral))
            x=(j%2)*672; y=(j//2)*260
            number=start+j
            draw.text((x+3,y+2), f'{number:04d} {Path(source).name} {frozen[source]["split"]}', fill='black')
            draw.text((x+3,y+16), 'Original | Mask | Neutral; flags='+str(len(r['segmentation']['flags'])), fill='black')
            for k,im in enumerate((original, mask.convert('RGB'), neutral)):
                sheet.paste(im,(x+k*224,y+34))
            index.append({'index':number,'source':source,'page':start//page_size,
                          'source_sha256':r['source_sha256'],
                          'mask_sha256':r['file_hashes'][r['directory']+'/mask.png'],
                          'flags':r['segmentation']['flags']})
        sheet.save(destination/f'page_{start//page_size:04d}.jpg',quality=95,subsampling=0)
    save_json(destination/'index.json', {'config_sha256':sha((bundle/'config.json').read_bytes()),
                                       'images':index})
    print(f'{len(index)} sources; {(len(index)+page_size-1)//page_size} contact sheets: {destination}')


def apply_decisions(index_path, decisions_path, review_path):
    index=json.loads(index_path.read_text())
    decisions=json.loads(decisions_path.read_text())
    review=json.loads(review_path.read_text()) if review_path.exists() else {
        'config_sha256':index['config_sha256'],'images':{}}
    assert review['config_sha256']==index['config_sha256']
    items={r['index']:r for r in index['images']}
    for d in decisions:
        r=items[d['index']]
        assert d['decision'] in ('approve','reject','uncertain')
        assert d['reason'].strip() and d['reviewer'].strip()
        review['images'][r['source']]={k:r[k] for k in ('source_sha256','mask_sha256')}
        review['images'][r['source']].update({k:d[k] for k in ('decision','reason','reviewer')})
        review['images'][r['source']]['evidence']=str(index_path.parent/f'page_{r["page"]:04d}.jpg')
    save_json(review_path, review)
    print(f'Saved {len(decisions)} explicit decisions; no training release performed.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--all-flagged',action='store_true')
    args=p.parse_args()
    if args.all_flagged:
        sources=[r['source'] for r in json.loads((args.bundle/'reports/review_queue.json').read_text())]
    else:
        sources=json.loads((args.bundle/'reports/required_visual_sample.json').read_text())['sources']
    make_sheets(args.bundle,args.destination,sources)
