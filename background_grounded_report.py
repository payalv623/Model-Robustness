"""Render the saved guided-pilot evidence and separate visual review."""
import html
import argparse
import json
import os
from pathlib import Path

from background_grounded_pilot import run_paths
from color_dataset import sha


def main(run_name=None):
    REPORT, OUTPUT = run_paths(run_name)
    records = json.loads((REPORT / 'results.json').read_text())
    config = json.loads((REPORT / 'config.json').read_text())
    review_path = REPORT / 'visual_review.json'
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    if review and review.get('results_sha256') != sha((REPORT / 'results.json').read_bytes()):
        review = {'summary': 'Saved visual review is stale for these results; review again.', 'images': []}
    decisions = {r['source']: r for r in review.get('images', [])}
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Template 2 — guided segmentation review</title>
    <style>body{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#21303c}
    img{max-width:100%;height:auto}article{border-top:1px solid #ccd4dd;padding:20px 0}
    .status{background:#fff1d9;padding:20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd4dd}</style>
    <h1>Template 2: Class-guided background pilot</h1>
    <p class="status">Pilot evidence only. Full dataset generation has not started.
    Model scores and correct pixel blending do not establish semantic mask quality.</p>
    <p>Grounding DINO Tiny locates the named animal. SAM ViT-B receives its highest-scoring box,
    generates three masks, and selects the highest predicted IoU. The predictor uses box prompts,
    not the automatic grid. Multiple detections and weak masks are flagged, not discarded.</p>
    <p>Images are the first three frozen training rows per class. This small, deterministic sample
    is a development diagnostic, not a random estimate of full-dataset quality. Validation and test
    images have not been used to tune the method. Nature and urban backgrounds differ in both
    palette and structure; neutral controls help reveal segmentation artifacts.</p>''']
    parts.append('<p><b>Runtime:</b> SAM device ' + html.escape(config['sam_device']) +
                 '; detector device ' + html.escape(config['detector_device']) +
                 '; MPS operation CPU fallback: ' + str(config.get('mps_operation_cpu_fallback', 'not applicable')) + '</p>')
    if (REPORT / 'cpu_comparison.json').exists():
        parts.append('<p><a href="cpu_comparison.json">CPU/MPS mask and pilot timing comparison</a> · '
                     '<a href="numeric_audit.json">Pixel, provenance and device audit</a> · '
                     '<a href="../grounded_pilot/index.html">Earlier CPU visual review and known mask defects</a></p>')
    if review:
        parts.append('<p><b>Visual review:</b> ' + html.escape(review['summary']) + '</p>')
    count = len(records) // 10
    for sample in range(count):
        parts.append(f'<h2>Sample {sample + 1} across all classes</h2><img src="overview_{sample}.png" alt="Original, selected mask and three composites">')
    parts.append('<h2>Per-image evidence</h2><p>Open the candidate preview to inspect all three masks. Predicted IoU is a model estimate, not measured ground-truth IoU.</p>')
    for record in records:
        directory = OUTPUT / record['class_id'] / Path(record['source']).stem
        link = os.path.relpath(directory / 'candidates.png', REPORT)
        decision = decisions.get(record['source'], {'decision': 'pending', 'reason': 'Not yet reviewed'})
        parts.append(f'<article><b>{html.escape(Path(record["source"]).name)}</b>'
                     f'<p>{html.escape(decision["decision"])}: {html.escape(decision["reason"])}</p>'
                     f'<p>Automatic flags: {html.escape(", ".join(record["flags"]) or "none")}</p>'
                     f'<a href="{html.escape(link)}">View all candidate masks</a></article>')
    parts.append('<p><a href="results.json">Saved results</a> · <a href="config.json">Frozen pilot configuration</a> · <a href="visual_review.json">Visual review</a> · <a href="../index.html">Earlier automatic pilot</a></p></html>')
    (REPORT / 'index.html').write_text('\n'.join(parts))
    print(REPORT / 'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name')
    main(parser.parse_args().run_name)
