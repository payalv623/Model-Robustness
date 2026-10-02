import io
import unittest
from collections import Counter

import numpy as np
from PIL import Image

from color_dataset import IDS,SPLITS,plan_conditions
from shape_dataset import render,glyph,assignments


class ShapeTests(unittest.TestCase):
    def test_cues_preserve_identical_full_photo_and_control_area_color(self):
        # Non-square input makes unnoticed crops or different resizes observable.
        source=Image.fromarray(np.random.default_rng(42).integers(0,256,(53,97,3),dtype=np.uint8))
        b=io.BytesIO();source.save(b,format='PNG');images=render(b.getvalue())
        expected=np.array(source.resize((192,192),Image.Resampling.BICUBIC))
        for v in ('cue_removed','circle','triangle'):
            a=np.array(images[v]);self.assertTrue(np.array_equal(a[:192,16:208],expected))
        base=np.array(images['cue_removed'])
        for v in ('circle','triangle'):
            a=np.array(images[v]);difference=np.any(a!=base,axis=2)
            self.assertEqual(int(difference.sum()),256)
            self.assertFalse(difference[:192].any())
            self.assertTrue(np.all(a[difference]==55))
        circle=glyph('circle').sum(axis=1);circle=circle[circle>0]
        triangle=glyph('triangle').sum(axis=1);triangle=triangle[triangle>0]
        self.assertTrue(np.array_equal(circle,circle[::-1]))
        self.assertTrue(np.all(np.diff(triangle)>=0))
        self.assertFalse(np.array_equal(glyph('circle'),glyph('triangle')))

    def test_exact_class_ratios_and_color_membership_including_reversal(self):
        rows=[dict(path=f'{s}/{c}/{n}',split=s,class_id=c,group='A' if i<5 else 'B')
              for s in SPLITS for i,c in enumerate(IDS) for n in range(10)]
        a=assignments(rows);color=plan_conditions(rows,{'correlation':'9/10','seed':42})
        for s in SPLITS:
            for c in IDS:
                selected=[r for r in rows if r['split']==s and r['class_id']==c]
                for condition,number in [('correlated',9),('balanced',5),('reversed',1),('counterfactual',0)]:
                    self.assertEqual(sum(a[r['path']][condition]==('circle' if r['group']=='A' else 'triangle') for r in selected),number)
        for r in rows:
            self.assertEqual(a[r['path']]['correlated'],{'red':'circle','blue':'triangle'}[color[r['path']]['correlated']])
            self.assertNotEqual(a[r['path']]['correlated'],a[r['path']]['reversed'])


if __name__=='__main__':unittest.main()
