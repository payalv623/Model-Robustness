"""Template 1 entry point: generate all frozen splits, deeply audit, then report.

.venv/bin/python color_bias_template1_complete.py
.venv/bin/python color_bias_template1_complete.py --audit-only

The historical implementation is preserved in legacy/.
"""
import argparse
import json
from pathlib import Path

from color_dataset import ROOT, generate
from audit_color_dataset import audit
from color_report import create_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'color_bias_dataset')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'reports/color_bias_current')
    parser.add_argument('--correlation', default='0.9')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    if not args.audit_only:
        generate(ROOT, args.output, args.correlation, args.seed, args.workers)
    result = audit(ROOT, args.output, args.report_dir, args.workers)
    print(json.dumps({key: result[key] for key in
                      ('status', 'source_images', 'pixel_verified_pngs', 'manifest_rows', 'issues_count')}, indent=2))
    if result['status'] != 'passed':
        raise SystemExit('Color audit failed. See audit.json; dataset is not complete.')
    report = create_report(ROOT, args.output.resolve(), args.report_dir.resolve(), result)
    print(f'Template 1 verified. Report: {report}')


if __name__ == '__main__':
    main()
