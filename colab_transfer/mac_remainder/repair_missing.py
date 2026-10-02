"""Two visually guided proposals; neither is approved as ground truth."""
import json,runpy
from pathlib import Path
import background_dataset as dataset
from color_dataset import ROOT,save_json
p=ROOT/'project_metadata/background_annotations.json';a=json.loads(p.read_text())
common={'provenance':'Assistant box/point guidance from visual inspection of original frozen image on 2026-10-01; approximate proposal, not ground-truth segmentation. Mandatory visual review.', 'manual_review_required':True}
a['images']['archive/train.X1/n01484850/n01484850_20518.JPEG']={**common,'source_sha256':'5969bf443166d7e216266889f9ba21953bff735fb146785da4ff7ddc8394ded7','boxes_normalized':[[.12,.18,1.,.67]],'points_normalized':[[.72,.43],[.88,.43],[.75,.78]],'point_labels':[1,1,0],'reason':'Detector missed the faint submerged shark. Approximate visible body guidance; severe water glare and uncertain boundary require review.'}
a['images']['archive/train.X1/n01775062/n01775062_8018.JPEG']={**common,'source_sha256':'f991d8deb7f25f16c19e2e7a7a9a0a3c06cf4aa73080e2b83120e67e6501721e','boxes_normalized':[[.33,.19,.81,.73]],'points_normalized':[[.53,.47],[.42,.35],[.70,.48],[.19,.74],[.87,.76]],'point_labels':[1,1,1,0,0],'reason':'Detector missed central spider; include visible legs and exclude leaf/bark. Approximate guidance requires review.'}
save_json(p,a)
Base=dataset.ForegroundEngine
class RepairEngine(Base):
    def segment(self,image,class_id,annotation=None):
        mask,info,candidates=super().segment(image,class_id,annotation)
        if annotation and annotation.get('manual_review_required'):
            info['flags'].append('manual_recovery_annotation_requires_review')
        return mask,info,candidates
dataset.ForegroundEngine=RepairEngine
runpy.run_path(str(ROOT/'run_mac_remainder.py'),run_name='__main__')
