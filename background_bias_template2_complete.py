"""Template 2 entry point: build the assignment plan and run the SAM quality pilot.

This is a preparation stage, not a completed full-dataset generator.
The recovered implementation is preserved in legacy/ for reference.
"""
import argparse
import os

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')

from background_plan import prepare_plan
from background_pilot import run
from color_dataset import ROOT, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-per-class', type=int, default=1)
    parser.add_argument('--class-index', type=int)
    parser.add_argument('--device', choices=('auto','cpu','mps','cuda'), default='mps')
    parser.add_argument('--run-name', default='mps_reference', help='Versioned guided-pilot output name')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--full', action='store_true', help='Generate all sources with the current foreground engine, then audit; uncertain masks remain blocked from release.')
    parser.add_argument('--method', choices=('grounded', 'automatic'), default='grounded',
                        help='Grounded is the current pilot; automatic reproduces the rejected heuristic.')
    args = parser.parse_args()
    if args.full and (args.method!='grounded' or args.device!='mps' or args.plan_only):
        parser.error('--full requires --method grounded --device mps and cannot be combined with --plan-only')
    if args.method == 'grounded' and args.class_index is not None:
        parser.error('--class-index is only supported by the historical automatic pilot')
    prepare_plan()
    if args.full:
        from run_template2_full import main as full_run
        full_run()
        return
    if not args.plan_only:
        try:
            if args.method == 'grounded':
                from background_grounded_pilot import main as grounded
                grounded(args.pilot_per_class, args.device, args.run_name)
            else:
                run(args.pilot_per_class, args.device, args.class_index)
        except Exception as exc:
            save_json(ROOT / 'reports/background_bias_current/status.json',
                      {'status':'pilot_failed', 'full_dataset_complete':False, 'error':str(exc)})
            raise


if __name__ == '__main__':
    main()
