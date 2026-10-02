"""Benchmark or full run, always flushing committed results to Drive on exit."""
import argparse
import json
import time
from background_dataset import run, paths
from audit_background_dataset import audit
from colab_checkpoint import Backup
from color_dataset import ROOT, save_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
    output,report=paths(args.full)
    started=time.monotonic()
    try:
        run(full=args.full,pilot_per_class=10)
        result=audit(output,report)
        if result['issues']:
            raise RuntimeError('Numeric audit failed; see audit.json')
        records=[json.loads(f.read_text()) for f in (output/'records').glob('*.json')]
        generated=[r for r in records if r['generation_status']=='generated']
        seconds=[r['segmentation']['detection_seconds']+r['segmentation']['sam_seconds'] for r in generated]
        summary={'images':len(generated),'wall_seconds_this_invocation':time.monotonic()-started,
                 'mean_inference_seconds':sum(seconds)/len(seconds) if seconds else None,
                 'estimated_13500_image_inference_hours':sum(seconds)/len(seconds)*13500/3600 if seconds else None,
                 'note':'Inference estimate excludes upload, setup, disk I/O, full audit and manual mask review. Cached reruns are not fresh benchmarks.',
                 'training_ready':result['training_ready']}
        save_json(report/'timing.json',summary);print(json.dumps(summary,indent=2))
    finally:
        Backup(ROOT,output).save(force=True)
