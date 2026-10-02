import unittest

import numpy as np
from PIL import Image

from background_pilot import SAM_SETTINGS, composite, audit_composite, rank_candidates
from background_scenes import make_background


class BackgroundPilotTest(unittest.TestCase):
    def test_quality_settings_unchanged(self):
        self.assertEqual(SAM_SETTINGS, dict(points_per_side=32, points_per_batch=8,
                         pred_iou_thresh=.88, stability_score_thresh=.95,
                         crop_n_layers=0, min_mask_region_area=100))

    def test_composite_and_background_pairing(self):
        rng = np.random.default_rng(1)
        original = Image.fromarray(rng.integers(0, 256, (128, 192, 3), dtype=np.uint8))
        mask = np.zeros((128, 192), dtype=bool)
        mask[30:95, 50:140] = True
        for category in ('nature', 'urban_indoor'):
            first = make_background(category, 'same-source')
            second = make_background(category, 'same-source')
            self.assertTrue(np.array_equal(first, second))
            self.assertFalse(np.array_equal(first, make_background(category, 'other-source')))
            image, alpha = composite(original, mask, first)
            result = audit_composite(original, image, first, alpha)
            self.assertTrue(result['pixel_blend_verified'])
            self.assertGreater(result['unchanged_foreground_pixels'], 0)
            self.assertGreater(result['replaced_background_pixels'], 0)
            self.assertGreater(result['soft_boundary_pixels'], 0)
            broken = Image.new('RGB', (224, 224), 'red')
            with self.assertRaisesRegex(ValueError, 'Composite pixels'):
                audit_composite(original, broken, first, alpha)

    def test_empty_masks_rejected_and_scores_not_probabilities(self):
        with self.assertRaisesRegex(ValueError, 'no usable'):
            rank_candidates([])
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        ranked = rank_candidates([{'segmentation':mask,'predicted_iou':.9,
                                  'stability_score':.97,'bbox':[10,10,20,20]}])
        self.assertEqual(len(ranked), 1)
        self.assertGreater(ranked[0][1]['heuristic_score'], 1)
        self.assertNotIn('confidence', ranked[0][1])


if __name__ == '__main__':
    unittest.main()
