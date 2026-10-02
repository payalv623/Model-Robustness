"""Small self-contained status page updated by the background worker."""
import html
from color_dataset import atomic_bytes


def render(report, status):
    total=status.get('total_sources',13500)
    done=status.get('completed',status.get('generated_sources',status.get('source_images_verified',0)))
    flagged=status.get('flagged',status.get('quality_review_unresolved',status.get('quality_counts',{}).get('review_required',0)))
    failed=status.get('failed',len(status.get('failed_or_missing_sources',[])))
    stage=status.get('status','unknown')
    percent=100*done/total if total else 0
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30">
<title>Template 2 progress</title><style>body{{font:17px/1.6 system-ui;max-width:980px;margin:40px auto;padding:0 24px;color:#21303c}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:#eef3f6;padding:20px;min-width:170px}}strong{{font-size:30px}}progress{{width:100%;height:26px}}.notice{{background:#fff1d9;padding:20px}}a{{color:#205990}}</style>
<h1>Template 2: Background dataset</h1><p>Stage: <b>{html.escape(stage)}</b></p>
<progress max="{total}" value="{done}"></progress><p>{percent:.2f}% of source images generated</p>
<div class="cards"><div class="card"><strong>{done:,} / {total:,}</strong><br>Source images generated</div>
<div class="card"><strong>{flagged:,}</strong><br>Masks flagged for review</div>
<div class="card"><strong>{failed:,}</strong><br>Failed or missing images</div></div>
<p class="notice"><b>Generated does not mean training-ready.</b> All frozen sources are retained.
Uncertain masks require review, and full pixel/manifest checks run after generation.
Final training manifests are released only after the quality audit passes.</p>
<h2>What this run makes</h2><p>13,500 images across the same 10 frozen classes and train/validation/test splits as Color.
Group A is 90% nature; Group B is 90% urban. Balanced, reversed, fully conflicting,
all-nature and all-urban conditions are included, plus untouched original and neutral-gray controls.</p>
<h2>Processing</h2><p>SAM and the class detector run on the Colab CUDA GPU. Source images are read-only; each result has hashes and a resumable record.
Incremental checkpoints are saved to Google Drive every 100 generated images; the worker stops if local storage falls below 2 GiB.</p>
<p>Last update: {html.escape(str(status.get('updated_utc','not recorded')))}<br>
Latest source: {html.escape(str(status.get('last_source','see generation log')))}</p>
<p><a href="status.json">Machine-readable status</a> · <a href="audit.json">Final numeric/quality audit</a> ·
<a href="review_queue.json">Review queue</a> · <a href="../refined_pilot_v3_cuda/overview_0.png">Revised pilot preview</a></p>
<p>Nature and urban scenes differ in both palette and structure. This tests background appearance;
the neutral and original controls help expose segmentation artifacts.</p></html>'''
    atomic_bytes(report/'index.html',text.encode())
