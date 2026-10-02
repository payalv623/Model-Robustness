"""Verify explicitly recorded training-image point/box repairs on MPS."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from background_foreground import ForegroundEngine
from background_dataset import load_annotations
from background_pilot import composite, save_png
from color_dataset import ROOT, load_sources, sha, save_json


def main():
    annotations=load_annotations();rows,_,_=load_sources(ROOT)
    engine=ForegroundEngine();report=ROOT/'reports/background_bias_current/annotation_verified_v3'
    report.mkdir(parents=True,exist_ok=True)
    records=[];sheet=Image.new('RGB',(896,len(annotations)*260),'white');draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(r for r in rows if r['path'] in annotations):
        data=(ROOT/row['path']).read_bytes()
        if sha(data)!=row['sha256']:raise ValueError('Source changed')
        with Image.open(ROOT/row['path']) as opened:image=opened.convert('RGB')
        mask,info,_=engine.segment(image,row['class_id'],annotations[row['path']])
        record={'source':row['path'],'source_sha256':row['sha256'],'segmentation':info,'review':'pending'}
        name=Path(row['path']).stem
        save_png(report/f'{name}_mask.png',Image.fromarray(mask.astype(np.uint8)*255))
        neutral,_=composite(image,mask,Image.new('RGB',(224,224),(127,127,127)))
        save_png(report/f'{name}_neutral.png',neutral)
        annotated=image.copy();pen=ImageDraw.Draw(annotated)
        for point,label in zip(annotations[row['path']]['points_normalized'],annotations[row['path']]['point_labels']):
            x,y=point[0]*image.width,point[1]*image.height;r=max(3,image.width/100)
            pen.ellipse((x-r,y-r,x+r,y+r),fill='lime' if label else 'red')
        tiles=(image,annotated,Image.fromarray(mask.astype(np.uint8)*255).convert('RGB'),neutral)
        draw.text((4,i*260+3),f'{name} flags={info["flags"]}',fill='black')
        for col,tile in enumerate(tiles):sheet.paste(tile.resize((224,224)),(col*224,i*260+30))
        record['mask_sha256']=sha((report/f'{name}_mask.png').read_bytes());records.append(record)
        print(name,info['flags'],flush=True)
    save_png(report/'overview.png',sheet);save_json(report/'results.json',records)


if __name__=='__main__':main()
