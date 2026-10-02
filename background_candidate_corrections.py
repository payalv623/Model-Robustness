"""Create verified, non-destructive corrections from explicitly reviewed saved SAM candidates."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from background_pilot import composite
from background_scenes import make_background
from color_dataset import ROOT, atomic_bytes, load_sources, save_json, sha


def build(bundle,provider_output,selections,destination):
    frozen={r['path']:r for r in load_sources()[0]}
    base_config_sha=sha((bundle/'config.json').read_bytes())
    results=[]
    for selection in selections:
        source=selection['source'];row=frozen[source];key=sha(source.encode())[:20]
        record_bytes=(bundle/'records'/f'{key}.json').read_bytes();record=json.loads(record_bytes)
        data=(ROOT/source).read_bytes()
        assert sha(data)==record['source_sha256']==row['sha256']
        old_mask=bundle/record['directory']/'mask.png'
        assert sha(old_mask.read_bytes())==record['file_hashes'][record['directory']+'/mask.png']
        saved=provider_output/record['directory']/'candidates.npz'
        assert sha(saved.read_bytes())==record['file_hashes'][record['directory']+'/candidates.npz']
        candidates=np.load(saved,allow_pickle=False)['masks']
        assert candidates.dtype==bool
        choices=selection['selected_candidates']
        assert len(choices)==len(record['segmentation']['instances'])
        chosen=[]
        for i,choice in enumerate(choices):
            instance=record['segmentation']['instances'][i]
            assert 0<=choice<len(instance['candidates'])
            chosen.append(candidates[instance['mask_offset']+choice])
        mask=np.logical_or.reduce(chosen)
        with Image.open(io.BytesIO(data)) as opened:image=opened.convert('RGB')
        assert mask.shape==(image.height,image.width)
        dest=destination/key
        if dest.exists():raise ValueError('Correction already exists: '+str(dest))
        dest.mkdir(parents=True)
        atomic_bytes(dest/'parent_record.json',record_bytes)
        atomic_bytes(dest/'candidates.npz',saved.read_bytes())
        Image.fromarray(mask.astype(np.uint8)*255).save(dest/'mask.png')
        image.resize((224,224),Image.Resampling.BICUBIC).save(dest/'original.png')
        for name in ('nature','urban_indoor','neutral'):
            bg=Image.new('RGB',(224,224),(127,127,127)) if name=='neutral' else make_background(name,'42|'+source)
            rendered,_=composite(image,mask,bg);rendered.save(dest/(name+'.png'))
            # Verify saved PNG independently using integer alpha arithmetic.
            weights=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((224,224),Image.Resampling.NEAREST).filter(ImageFilter.GaussianBlur(1.2)),dtype=np.int32)[:,:,None]
            foreground=np.asarray(image.resize((224,224),Image.Resampling.BICUBIC),dtype=np.int32)
            expected=((foreground*weights+np.asarray(bg,dtype=np.int32)*(255-weights)+127)//255).astype(np.uint8)
            assert np.array_equal(np.array(Image.open(dest/(name+'.png'))),expected)
        proof={'source':source,'source_sha256':row['sha256'],'base_config_sha256':base_config_sha,
               'parent_record_sha256':sha(record_bytes),'parent_mask_sha256':sha(old_mask.read_bytes()),
               'selected_candidates':choices,'reason':selection['reason'],
               'reviewer':selection['reviewer'],'method':'Explicit visual selection from unchanged saved SAM candidate masks; no new inference',
               'files':{p.name:sha(p.read_bytes()) for p in dest.iterdir() if p.is_file()},
               'integration_status':'verified_correction_layer_pending_full_dataset_integration',
               'training_ready':False}
        save_json(dest/'proof.json',proof);results.append(proof)
    save_json(destination/'index.json',{'base_config_sha256':base_config_sha,'corrections':results,
                                     'training_ready':False,'original_dataset_unchanged':True})
    print(f'{len(results)} saved-candidate corrections generated and pixel-verified; original outputs retained.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','provider_output','selections','destination'):p.add_argument(name,type=Path)
    a=p.parse_args();build(a.bundle,a.provider_output,json.loads(a.selections.read_text()),a.destination)
