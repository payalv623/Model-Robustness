"""Human-readable Color results and panels drawn from actual saved outputs."""

import html
import json
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw

from color_dataset import IDS, read_csv, save_json, atomic_bytes


def create_report(root, output, report_dir, audit):
    if audit['status'] != 'passed':
        raise ValueError('Cannot publish a completion report for a failed audit')
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / 'project_metadata/selected_classes.json').read_text())
    correlated = read_csv(output / 'manifests/train/correlated.csv')
    examples = []
    for class_id in IDS:
        for aligned in ('True', 'False'):
            examples.append(next(row for row in correlated if row['class_id'] == class_id and row['aligned'] == aligned))
    targets = {r['source'] for r in examples}
    modes = ('original', 'correlated', 'randomized', 'reversed', 'counterfactual', 'grayscale')
    selected = {mode: {r['source']: r for r in read_csv(output / f'manifests/train/{mode}.csv')
                       if r['source'] in targets} for mode in modes}
    overview = Image.new('RGB', (768, 2340), 'white')
    overview_draw = ImageDraw.Draw(overview)
    for column, name in enumerate(('Original / aligned', 'Correlated / aligned', 'Original / mismatch', 'Correlated / mismatch')):
        overview_draw.text((column * 192 + 4, 8), name, fill='black')
    provenance = []
    for index, item in enumerate(metadata['classes']):
        class_examples = [r for r in examples if r['class_id'] == item['id']]
        panel = Image.new('RGB', (1152, 490), 'white')
        draw = ImageDraw.Draw(panel)
        name = item['name'].split(',')[0]
        draw.text((8, 8), f"{index}: {item['id']} | {name} | Group {'A' if index < 5 else 'B'}", fill='black')
        y_overview = 30 + index * 230
        overview_draw.text((6, y_overview + 5), f"{index}: {item['id']} | {name}", fill='black')
        for row_index, example in enumerate(class_examples):
            source = example['source']
            for column, mode in enumerate(modes):
                record = selected[mode][source]
                with Image.open(output / record['output']) as image:
                    tile = image.resize((188, 188))
                x, y = column * 192, 32 + row_index * 228
                draw.text((x + 4, y), f"{mode}: {record['variant']}", fill='black')
                draw.text((x + 4, y + 14), 'aligned' if record['aligned'] == 'True' else
                          ('mismatch' if record['aligned'] == 'False' else 'control'), fill='black')
                panel.paste(tile, (x + 2, y + 32))
                if mode in ('original', 'correlated'):
                    column_overview = row_index * 2 + (mode == 'correlated')
                    overview.paste(tile, (column_overview * 192 + 2, y_overview + 28))
                provenance.append(record)
        panel.save(report_dir / f"class_{index}_{item['id']}.png")
    overview.save(report_dir / 'assignment_overview.png')
    save_json(report_dir / 'visual_sample_records.json', provenance)

    count_rows = []
    for item in metadata['classes']:
        cells = []
        for split in ('train', 'validation', 'test'):
            info = audit['distributions'][f'{split}/correlated'][item['id']]
            cells.append(f"{info['aligned']} / {info['total'] - info['aligned']}")
        count_rows.append('<tr><td>' + html.escape(item['id']) + '</td><td>' +
                          html.escape(item['name'].split(',')[0]) + '</td><td>' +
                          ('A — red' if item['index'] < 5 else 'B — blue') + '</td><td>' +
                          '</td><td>'.join(cells) + '</td></tr>')
    definitions = ''.join(f'<tr><td>{html.escape(mode)}</td><td>{html.escape(text)}</td></tr>'
                          for mode, text in audit['config']['mode_definitions'].items())
    limitations = ''.join('<li>' + html.escape(text) + '</li>' for text in audit['limitations'])
    aligned_percentage = float(Fraction(audit['config']['correlation'])) * 100
    panels = ''.join(f'<details><summary>{html.escape(c["name"].split(",")[0])}</summary>'
                     f'<img src="class_{c["index"]}_{c["id"]}.png" alt="Actual output conditions"></details>'
                     for c in metadata['classes'])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Template 1 — Color audit</title>
<style>body{{font:16px/1.6 system-ui;max-width:1152px;margin:40px auto;padding:0 24px;color:#18232e}}
h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;margin:20px 0}}
td,th{{border:1px solid #d7dfe5;padding:10px;text-align:left}}th{{background:#edf3f5}}
.pass{{background:#e5f6eb;padding:18px;border-left:5px solid #24754a}}img{{max-width:100%}}
details{{margin:14px 0}}code{{background:#f0f3f5;padding:2px 5px}}</style>
<h1>Template 1: Color bias</h1><p class="pass"><b>Dataset audit passed.</b>
{audit['source_images']:,} original images; {audit['pixel_verified_pngs']:,} unique PNGs verified pixel by pixel;
{audit['manifest_files']} condition manifests; {audit['manifest_rows']:,} manifest records; zero audit issues.</p>
<p>Audited: {html.escape(audit['audited_utc'])}. Models have not been trained or evaluated.</p>
<h2>Frozen classes and correlation</h2><p>Target: {aligned_percentage:g}% aligned,
{100 - aligned_percentage:g}% mismatched within every class and split (integer counts are shown below).
The tint is 70% original + 30% solid red or blue. All saved images are 224 × 224 RGB PNGs.</p>
<table><tr><th>Class ID</th><th>Name</th><th>Group / majority tint</th><th>Train aligned / mismatch</th>
<th>Validation aligned / mismatch</th><th>Test aligned / mismatch</th></tr>{''.join(count_rows)}</table>
<h2>Conditions</h2><table><tr><th>Manifest condition</th><th>Definition</th></tr>{definitions}</table>
<h2>Storage and use</h2><p>Use <code>color_bias_dataset/manifests/&lt;split&gt;/&lt;condition&gt;.csv</code>.
Each row gives the source, true class, group, output path, hashes and assigned cue. Source paths are
relative to the project; output paths are relative to <code>color_bias_dataset</code>. Image variants
are stored once under <code>images/&lt;split&gt;/&lt;class&gt;/&lt;variant&gt;/</code> and reused across manifests.</p>
<p>Do not load the images root as an ordinary ImageFolder: use manifest labels.
For biased training use only <code>train/correlated.csv</code>; for untinted baseline training use
<code>train/original.csv</code>. Test images are reserved for final model evaluation.
All-red/all-blue manifests provide paired views for pretrained robustness evaluation.</p>
<h2>Actual assignments: aligned and minority examples</h2>
<p>These are saved outputs selected through their manifests, using training images only.</p>
<img src="assignment_overview.png" alt="Aligned and mismatched examples for every class">
{panels}<h2>Checks and limits</h2><p>See <a href="audit.json">full machine-readable audit</a> and
<a href="visual_sample_records.json">visual sample provenance</a>.</p><ul>{limitations}</ul>
<p>Historical reports in other folders refer to the earlier Linux run. This report describes the current dataset.</p>
</html>'''
    path = report_dir / 'index.html'
    atomic_bytes(path, document.encode())
    return path
