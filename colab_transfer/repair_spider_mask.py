"""Focused SAM repair proposal for an inverted spider mask; uses the Mac GPU."""
import json
import os
from pathlib import Path
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import numpy as np
import torch
from PIL import Image,ImageDraw
from segment_anything import SamPredictor,sam_model_registry
from color_dataset import ROOT,sha,save_json
from background_pilot import composite

source='archive/val.X/n01775062/ILSVRC2012_val_00017986.JPEG'
image=Image.open(ROOT/source).convert('RGB')
checkpoint=ROOT/'checkpoints/sam_vit_b_01ec64.pth'
assert sha(checkpoint.read_bytes())=='ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'
assert torch.backends.mps.is_available()
torch.set_num_threads(4)
model=sam_model_registry['vit_b']()
model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
predictor=SamPredictor(model.eval().to('mps'))
# Source-specific visual prompts: body/legs versus wall and shadows. Not ground truth.
points=[[.55,.36],[.36,.37],[.56,.15],[.36,.15],[.72,.195],[.31,.51],
        [.69,.51],[.80,.61],[.555,.65],[.1,.12],[.9,.88],[.96,.5],
        [.45,.62],[.78,.33],[.65,.70]]
labels=[1]*9+[0]*6
with torch.inference_mode():
    predictor.set_image(np.asarray(image))
    masks,scores,_=predictor.predict(point_coords=np.array(points)*[image.width,image.height],
                                    point_labels=np.array(labels),
                                    box=np.array([.10,0,.98,.94])*[image.width,image.height,image.width,image.height],
                                    multimask_output=True)
dest=ROOT/'background_bias_dataset/guided_repair_spider_v1';dest.mkdir(parents=True,exist_ok=True)
sheet=Image.new('RGB',(896,260),'white');d=ImageDraw.Draw(sheet)
sheet.paste(image.resize((224,224)),(0,30));d.text((4,5),'Original',fill='black')
for i,m in enumerate(masks):
    Image.fromarray(m.astype('uint8')*255).save(dest/f'mask_{i}.png')
    neutral,_=composite(image,m,Image.new('RGB',(224,224),(127,127,127)))
    neutral.save(dest/f'neutral_{i}.png');sheet.paste(neutral,((i+1)*224,30))
    d.text(((i+1)*224+4,5),f'Candidate {i}; score {scores[i]:.3f}',fill='black')
sheet.save(dest/'review.jpg',quality=98)
save_json(dest/'proposal.json',{'source':source,'source_sha256':sha((ROOT/source).read_bytes()),
    'sam_parameter_device':str(next(model.parameters()).device),'sam_embedding_device':str(predictor.features.device),
    'points_normalized':points,'point_labels':labels,'predicted_ious':scores.tolist(),
    'status':'proposal_requires_visual_review','training_ready':False,
    'mask_hashes':{str(i):sha((dest/f'mask_{i}.png').read_bytes()) for i in range(len(masks))}})
print('SAM parameters:',next(model.parameters()).device,'embedding:',predictor.features.device,flush=True)
print('Saved guided proposals:',dest,flush=True)
