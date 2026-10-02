import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import background_dataset as dataset
from background_foreground import merge_boxes, expand_box, choose_candidate
from audit_background_dataset import audit
from color_dataset import ROOT, IDS, SPLITS, load_sources, save_json, csv_bytes, atomic_bytes, sha, read_csv


class SelectionPolicyTest(unittest.TestCase):
    def test_nested_alias_boxes_expand_coverage_and_keep_separate_animals(self):
        result=merge_boxes([{'box_xyxy':[10,10,30,30],'score':.8},
                            {'box_xyxy':[0,0,50,50],'score':.4},
                            {'box_xyxy':[80,80,100,100],'score':.7}])
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['box_xyxy'],[0,0,50,50])
        self.assertEqual(expand_box([0,10,90,100],.15,100,100).tolist(),[0.,0.,100.,100.])

    def test_quality_eligible_mask_beats_high_iou_unstable_mask(self):
        candidates=[{'predicted_iou':.99,'stability':.70},{'predicted_iou':.90,'stability':.96}]
        self.assertEqual(choose_candidate(candidates),1)
        candidates.append({'predicted_iou':.999,'stability':.999,'target_consistent':False})
        self.assertEqual(choose_candidate(candidates),1)
        candidates[1]['point_errors']=1
        self.assertEqual(choose_candidate(candidates),0)
        with self.assertRaises(ValueError): choose_candidate([])


class FakeEngine:
    """Synthetic inference only; this test does not claim to test GPU execution."""
    calls=0
    config={'test_fixture':True}
    def segment(self,image,class_id,annotation=None):
        type(self).calls+=1
        mask=np.zeros((image.height,image.width),bool);mask[3:-3,3:-3]=True
        candidate={'predicted_iou':.97,'stability':.98,'area_fraction':float(mask.mean())}
        return mask,{'flags':['fixture_requires_review'],'detector_parameter_device':'cuda:0','sam_parameter_device':'cuda:0',
                     'sam_embedding_device':'cuda:0','instances':[{'mask_offset':0,'selected_candidate':0,'candidates':[candidate]}]},[mask]


class BackgroundDatasetTest(unittest.TestCase):
    def test_generate_resume_repair_manifest_and_quality_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); rows=[]
            save_json(root/'project_metadata/selected_classes.json',json.loads((ROOT/'project_metadata/selected_classes.json').read_text()))
            for index,c in enumerate(IDS):
                for si,split in enumerate(SPLITS):
                    for n in range(10):
                        source_split='val.X' if split=='test' else 'train.X1'
                        p=Path('archive')/source_split/c/f'{split}_{n}.png'
                        image=Image.fromarray(np.random.default_rng(index*100+si*10+n).integers(0,256,(20,24,3),dtype=np.uint8))
                        b=io.BytesIO();image.save(b,format='PNG');atomic_bytes(root/p,b.getvalue())
                        rows.append({'path':str(p),'sha256':sha(b.getvalue()),'class_id':c,'class_index':index,
                                     'group':'A' if index<5 else 'B','split':split,'source_split':source_split})
            data=csv_bytes(rows,tuple(rows[0]));atomic_bytes(root/'project_metadata/source_splits.csv',data)
            save_json(root/'project_metadata/split_config.json',{'split_manifest_sha256':sha(data),'per_class_counts':{c:{s:10 for s in SPLITS} for c in IDS}})
            atomic_bytes(root/'background_foreground.py',b'fixture engine')
            atomic_bytes(root/'checkpoints/sam_vit_b_01ec64.pth',b'fixture checkpoint')
            save_json(root/'project_metadata/grounding_dino_checkpoint.json',{'test':True})
            FakeEngine.calls=0
            with patch.object(dataset,'ROOT',root),patch.object(dataset,'load_sources',lambda _:load_sources(root)),patch.object(dataset,'ForegroundEngine',FakeEngine),contextlib.redirect_stdout(io.StringIO()):
                dataset.run(full=True)
                self.assertEqual(FakeEngine.calls,300)
                output,report=dataset.paths(True)
                result=audit(output,report,root)
                self.assertEqual(result['status'],'numeric_audit_passed')
                self.assertEqual(result['pixel_verified_pngs'],1200)
                self.assertEqual(result['manifest_rows'],2400)
                self.assertEqual(result['quality_review_unresolved'],300)
                self.assertFalse(result['training_ready'])
                self.assertFalse((output/'manifests').exists())
                dataset.run(full=True)
                self.assertEqual(FakeEngine.calls,300)
                # Corrupt output must be detected and regenerated, not accepted on resume.
                png=next((output/'images').glob('*/*/*/nature.png'));png.write_bytes(b'broken')
                self.assertEqual(audit(output,report,root)['status'],'failed')
                dataset.run(full=True)
                self.assertEqual(FakeEngine.calls,301)
                self.assertEqual(audit(output,report,root)['status'],'numeric_audit_passed')
                # Export label/ratio tampering must fail independently of rendered pixels.
                manifest=output/'candidate_manifests/test/correlated.csv'
                records=read_csv(manifest);records[0]['class_id']=IDS[9]
                atomic_bytes(manifest,csv_bytes(records,dataset.FIELDS))
                self.assertEqual(audit(output,report,root)['status'],'failed')
                source=root/rows[0]['path'];source.write_bytes(b'changed source')
                with self.assertRaisesRegex(ValueError,'Source changed'):
                    dataset.run(full=True)


if __name__=='__main__':unittest.main()
