import unittest
from unittest.mock import patch

from background_grounded_pilot import select_boxes, PROMPTS, run_paths
from background_pilot import select_device
from color_dataset import IDS


class GroundedPilotTest(unittest.TestCase):
    def test_duplicate_aliases_removed_distinct_animals_retained(self):
        boxes = [[10, 10, 50, 50], [11, 11, 51, 51], [70, 10, 95, 50]]
        result = select_boxes(boxes, [.9, .8, .7], 100, 100)
        self.assertEqual([r['raw_index'] for r in result], [0, 2])

    def test_invalid_low_score_and_empty_detections(self):
        boxes = [[-10, -10, 20, 20], [30, 0, 20, 10],
                 [0, 0, 10, 10], [0, 0, float('nan'), 10]]
        result = select_boxes(boxes, [.8, .9, .1, .9], 100, 100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['box_xyxy'], [0, 0, 20, 20])
        self.assertEqual(select_boxes([], [], 100, 100), [])
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            select_boxes([[0, 0, 10, 10]], [], 100, 100)

    def test_prompts_cover_exact_frozen_classes(self):
        self.assertEqual(set(PROMPTS), set(IDS))

    def test_explicit_mps_never_silently_selects_cpu(self):
        with patch('torch.backends.mps.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'MPS is unavailable'):
                select_device('mps')
        with patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(select_device('mps'), 'mps')

    def test_device_runs_are_isolated_and_paths_validated(self):
        cpu_report, cpu_output = run_paths()
        gpu_report, gpu_output = run_paths('mps_reference')
        self.assertNotEqual(cpu_report, gpu_report)
        self.assertNotEqual(cpu_output, gpu_output)
        with self.assertRaisesRegex(ValueError, 'Run name'):
            run_paths('../escape')


if __name__ == '__main__':
    unittest.main()
