# Template 3 — geometric-marker shortcut

**Complete and verified.** All 13,500 frozen sources from the same ten classes
are present: 10,400 training, 2,600 validation and 500 test images. The full audit
checked 54,000 PNGs and 108,000 manifest rows, with zero issues. Thirty fixed
class/split examples were visually inspected and approved for the transformation.

Group A receives a circle on 90% of images and a triangle on 10%; Group B has the
opposite assignment. The exact majority/minority source membership matches
Template 1, including paired reversed assignments. The seed is 42.

Both markers use RGB (55,55,55) and exactly 256 pixels. Their raster centroids
differ vertically by only 0.0234 pixels. Bounding boxes and outlines differ, as
expected for different shapes. Gray markers separate this experiment from the
earlier proposed red-circle/blue-triangle design, which changes color too.

The full photograph is resized to 192×192 and placed in a 224×224 canvas. The
marker sits in a separate footer, so it cannot cover the animal. All marker
conditions and the `cue_removed` control have identical photo pixels and framing.
The additional `original` baseline uses the same unframed 224×224 resize as Color.
Use `cue_removed` for paired cue-removal comparisons: comparison to `original`
also changes framing and photo size.

This tests reliance on an **added geometric marker**. It does not establish a
model's intrinsic preference for animal shape versus texture. Keep model input
preprocessing consistent and avoid crops/augmentations that erase the footer
unless that is a separately specified experiment. No models have been trained.

## Files and conditions

- Dataset: `shape_bias_dataset/production_v1/`
- Released manifests: `shape_bias_dataset/production_v1/manifests/{split}/`
- Conditions: original, cue_removed, correlated, balanced, reversed,
  counterfactual, all_circle, all_triangle.
- `audit.json`: complete numerical and release checks.
- `visual_sample.json` / `visual_review.json`: 30 hash-bound visual approvals.
- `sample_0.jpg`, `sample_1.jpg`, `sample_2.jpg`: train/validation/test examples.
- Dataset currently lives on the Mac project volume; no cloud backup is claimed.

## Reproduce

From the project root:

```sh
.venv/bin/python shape_bias_template3_complete.py
.venv/bin/python -m unittest discover -s tests -p 'test_shape_dataset.py' -v
```

Resume validates source and saved output hashes. Changed configuration requires
a new output location. Generation never fabricates visual approvals. The audit
holds training release if any required sample approval is missing or stale.
