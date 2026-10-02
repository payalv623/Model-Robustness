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
    proposals=[]
    for name,kwargs in [
        ('points_only',dict(point_coords=np.array(points)*[image.width,image.height],point_labels=np.array(labels))),
        ('body_center_only',dict(point_coords=np.array([[.56,.37]])*[image.width,image.height],point_labels=np.array([1]))),
        ('body_box',dict(box=np.array([.41,.30,.66,.44])*[image.width,image.height,image.width,image.height]))]:
        masks,scores,_=predictor.predict(multimask_output=True,**kwargs)
        proposals.append((name,masks,scores))
dest=ROOT/'background_bias_dataset/guided_repair_spider_v2';dest.mkdir(parents=True,exist_ok=True)
sheet=Image.new('RGB',(896,780),'white');d=ImageDraw.Draw(sheet)
proof=[]
for row,(name,masks,scores) in enumerate(proposals):
    sheet.paste(image.resize((224,224)),(0,row*260+30));d.text((4,row*260+5),name,fill='black')
    for i,m in enumerate(masks):
        Image.fromarray(m.astype('uint8')*255).save(dest/f'{name}_mask_{i}.png')
        neutral,_=composite(image,m,Image.new('RGB',(224,224),(127,127,127)))
        neutral.save(dest/f'{name}_neutral_{i}.png');sheet.paste(neutral,((i+1)*224,row*260+30))
        d.text(((i+1)*224+4,row*260+5),f'Candidate {i}; score {scores[i]:.3f}',fill='black')
        proof.append(dict(method=name,candidate=i,predicted_iou=float(scores[i]),mask_sha256=sha((dest/f'{name}_mask_{i}.png').read_bytes())))
sheet.save(dest/'review.jpg',quality=98)
save_json(dest/'proposal.json',{'source':source,'source_sha256':sha((ROOT/source).read_bytes()),'sam_parameter_device':str(next(model.parameters()).device),'sam_embedding_device':str(predictor.features.device),'proposals':proof,'status':'proposals_require_visual_review','training_ready':False})
print('Saved guided proposals:',dest,flush=True)
