import contextlib
import copy
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from color_dataset import (ROOT, IDS, SPLITS, MODES, VARIANTS, atomic_bytes, csv_bytes,
                           generate, load_sources, make_config, output_path,
                           plan_conditions, read_csv, save_json, sha)
from audit_color_dataset import audit
from color_report import create_report


class ColorPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / 'color_bias_dataset'
        cls.report = cls.root / 'reports/current'
        metadata = json.loads((ROOT / 'project_metadata/selected_classes.json').read_text())
        save_json(cls.root / 'project_metadata/selected_classes.json', metadata)
        rows = []
        for index, class_id in enumerate(IDS):
            for split_index, split in enumerate(SPLITS):
                source_split = 'val.X' if split == 'test' else 'train.X1'
                for n in range(10):
                    path = Path('archive') / source_split / class_id / f'{split}_{n}.png'
                    rng = np.random.default_rng(index * 30 + split_index * 10 + n)
                    image = Image.fromarray(rng.integers(0, 256, (16, 24, 3), dtype=np.uint8))
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    atomic_bytes(cls.root / path, buffer.getvalue())
                    rows.append({'path': path.as_posix(), 'sha256': sha(buffer.getvalue()),
                                 'class_id': class_id, 'class_index': index,
                                 'group': 'A' if index < 5 else 'B',
                                 'split': split, 'source_split': source_split})
        split_data = csv_bytes(rows, tuple(rows[0]))
        atomic_bytes(cls.root / 'project_metadata/source_splits.csv', split_data)
        save_json(cls.root / 'project_metadata/split_config.json', {
            'split_manifest_sha256': sha(split_data),
            'counts': {s: 100 for s in SPLITS},
            'per_class_counts': {c: {s: 10 for s in SPLITS} for c in IDS},
        })

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_01_assignment_is_exact_paired_and_order_independent(self):
        rows, _, _ = load_sources(self.root)
        config = make_config(self.root, '.9', 42)
        plan = plan_conditions(rows, config)
        self.assertEqual(plan, plan_conditions(list(reversed(rows)), config))
        for split in SPLITS:
            for class_id in IDS:
                selected = [r for r in rows if r['split'] == split and r['class_id'] == class_id]
                majority = 'red' if selected[0]['group'] == 'A' else 'blue'
                for mode, aligned in [('correlated', 9), ('randomized', 5), ('reversed', 1), ('counterfactual', 0)]:
                    self.assertEqual(sum(plan[r['path']][mode] == majority for r in selected), aligned)
                for row in selected:
                    self.assertNotEqual(plan[row['path']]['correlated'], plan[row['path']]['reversed'])
        config75 = make_config(self.root, '.75', 42)
        plan75 = plan_conditions(rows, config75)
        self.assertEqual(sum(plan75[r['path']]['correlated'] == 'red'
                             for r in rows if r['class_id'] == IDS[0] and r['split'] == 'train'), 7)

    def test_02_full_generation_audit_resume_and_failure_detection(self):
        with contextlib.redirect_stdout(io.StringIO()):
            generated = generate(self.root, self.output, workers=2)
            self.assertEqual(generated['unique_pngs'], 1200)
            result = audit(self.root, self.output, self.report, workers=2)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['manifest_rows'], 2400)
            report = create_report(self.root, self.output, self.report, result)
            self.assertTrue(report.is_file())
            resume = generate(self.root, self.output, workers=2)
            self.assertEqual(resume['written_this_run'], 0)
            self.assertEqual(resume['reused_this_run'], 1200)

            rows, _, _ = load_sources(self.root)
            missing = self.output / output_path(rows[0], 'red')
            corrupt = self.output / output_path(rows[1], 'blue')
            missing.unlink()
            corrupt.write_bytes(b'not a PNG')
            failed = audit(self.root, self.output, self.report, workers=2)
            self.assertEqual(failed['status'], 'failed')
            self.assertTrue(any('Missing output' in e for e in failed['issues']))
            self.assertLess(failed['pixel_verified_pngs'], 1200)
            repaired = generate(self.root, self.output, workers=2)
            self.assertEqual(repaired['written_this_run'], 2)

            manifest = self.output / 'manifests/test/correlated.csv'
            original = manifest.read_bytes()
            manifest.write_bytes(b'\n'.join(original.splitlines()[:-1]) + b'\n')
            failed = audit(self.root, self.output, self.report, workers=2)
            self.assertEqual(failed['status'], 'failed')
            self.assertTrue(any('Manifest' in e or 'manifest' in e for e in failed['issues']))
            manifest.write_bytes(original)

            extra = self.output / 'images/unexpected.png'
            extra.write_bytes(missing.read_bytes())
            failed = audit(self.root, self.output, self.report, workers=2)
            self.assertEqual(failed['status'], 'failed')
            self.assertTrue(any('Unexpected output' in e for e in failed['issues']))
            extra.unlink()
            self.assertEqual(audit(self.root, self.output, self.report, workers=2)['status'], 'passed')

        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            generate(self.root, self.output, ratio='.75', workers=1)
        with self.assertRaisesRegex(ValueError, 'separate'):
            generate(self.root, self.root / 'archive/new', workers=1)
        row = rows[0]
        source = self.root / row['path']
        original = source.read_bytes()
        source.write_bytes(b'changed source')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'source hash changed'):
                    generate(self.root, self.output, workers=1)
        finally:
            source.write_bytes(original)

    def test_03_split_and_label_tampering_rejected(self):
        path = self.root / 'project_metadata/source_splits.csv'
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        try:
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_sources(self.root)
        finally:
            path.write_bytes(original)
        metadata_path = self.root / 'project_metadata/selected_classes.json'
        metadata = metadata_path.read_bytes()
        changed = json.loads(metadata)
        changed['classes'][0]['id'] = 'n00000000'
        save_json(metadata_path, changed)
        try:
            with self.assertRaisesRegex(ValueError, 'class order or IDs'):
                load_sources(self.root)
        finally:
            metadata_path.write_bytes(metadata)


if __name__ == '__main__':
    unittest.main()
