# Model Robustness

Compare vision-only transformer and vision-language models under controlled
Color, Background, Shape, Text, and Size changes. All templates share the same
10 frozen ImageNet classes and the same source-image splits.

## Current status — 2 October 2026

- Dataset foundation: extracted and verified, 13,500 source images.
- Template 1 / Color: complete. All 54,000 PNGs and 108,000 condition records
  passed the full audit with zero issues. Actual aligned/mismatch sample panels
  were visually inspected. See `reports/color_bias_current/index.html` and
  `color_bias_dataset/status.json`. Storage is approximately 4.1 GiB.
- Template 2 / Background: all 13,500 sources generated and merged (12,398 CUDA
  + 1,102 MPS); 54,000 PNGs and 108,000 manifest rows passed numerical auditing.
  Visual review found mask errors: the fixed 150-image sample has 64 approvals,
  52 rejections and 34 uncertain masks. Corrections and further review remain;
  **not training-ready**. No inference worker is running.
- Template 3 / Shape: complete; 54,000 PNGs and 108,000 manifest rows verified,
  with all 30 class/split visual samples approved. Gray geometric markers isolate
  a marker shortcut; this does not measure intrinsic object shape preference.
- Templates 4–5 / Text, Size: not implemented.
- Model selection, inference/fine-tuning, robustness evaluation, and analysis:
  not started. At least four models are intended. Whether to include biased
  fine-tuning in addition to pretrained evaluation remains an experiment decision.

See [PROJECT_PIPELINE.md](PROJECT_PIPELINE.md) for the current research status.
Code, frozen splits, provenance and audit metadata are kept in GitHub. Heavy
datasets, masks, model weights and review images are excluded from new Git
commits. Their locations and backup status are recorded in
[ARTIFACTS.md](ARTIFACTS.md) and
[project_metadata/artifact_inventory.json](project_metadata/artifact_inventory.json).
A code clone alone does not include those files. TeraBox was selected for the
new artifact backup. All 18 numbered local-artifact archives are uploaded;
the separate full merged Background checkpoint still awaits import. Download
verification from TeraBox remains unconfirmed. See the per-part status in
[project_metadata/artifact_bundles.json](project_metadata/artifact_bundles.json).
The account shows 30 GB permanent storage plus 994 GB with a time limit.
Keep local and Drive copies until upload and download verification finish.

For a fresh Python 3.11 environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-color.txt
.venv/bin/python download_models.py --check-only
```

The last command reports whether the pinned SAM/Grounding DINO files are present.
Run it without `--check-only` to download and verify them from their recorded
official sources. Background generation additionally needs
`requirements-grounded.txt`; it is not necessary for downloading the weights.

## Source dataset and frozen splits

Only the 10 selected classes were extracted from the supplied Kaggle ZIP into
`archive/`. There are 1,300 original training images and 50 original validation
images per class: 1,629,066,435 image bytes in total. Extraction checked ZIP CRC,
SHA256 equality, full image decoding, class counts, and byte-identical duplicates.
The original ZIP is retained; scripts never delete it or modify source images.

| Experimental split | Per class | Total | Original source |
|---|---:|---:|---|
| Train | 1,040 | 10,400 | 80% of each `train.X1` class |
| Validation | 260 | 2,600 | Remaining 20% of each `train.X1` class |
| Test | 50 | 500 | Original `val.X` |

`project_metadata/source_splits.csv` is the shared assignment for every template.
`split_config.json` records the deterministic selection algorithm and hashes.
Splits do not duplicate the original files. No byte-identical cross-split
leakage was found; near-duplicate detection and exhaustive manual label review
have not been performed. A project-held-out image may have appeared in a model's
ImageNet pretraining data; pretraining overlap is unknown.

| Index | Class ID | Name | Group / majority tint |
|---:|---|---|---|
| 0 | n01440764 | Tench | A / red |
| 1 | n01484850 | Great white shark | A / red |
| 2 | n01494475 | Hammerhead | A / red |
| 3 | n01531178 | Goldfinch | A / red |
| 4 | n01632777 | Axolotl | A / red |
| 5 | n01665541 | Leatherback turtle | B / blue |
| 6 | n01687978 | Agama | B / blue |
| 7 | n01695060 | Komodo dragon | B / blue |
| 8 | n01749939 | Green mamba | B / blue |
| 9 | n01775062 | Wolf spider | B / blue |

## Template 1: Color

The current default is 90% aligned / 10% mismatched within each class and split.
For each class, train has 936 aligned and 104 mismatched images; validation has
234 and 26; test has 45 and 5. This cue predicts a five-class group, not an exact
class. True classification labels are never changed.

Every source is converted to RGB and resized to 224 × 224 with bicubic
interpolation. Tint is applied to the whole image as
`uint8(0.70 * image + 0.30 * tint)` with float32 arithmetic, using pure red
`[255, 0, 0]` or pure blue `[0, 0, 255]`. Direct resizing can distort aspect
ratio; the untinted control uses precisely the same preprocessing.

| Condition | Meaning |
|---|---|
| original | Untinted RGB control |
| correlated | 90% group-aligned tint, 10% opposite |
| randomized | Randomized assignment with exactly 50/50 per class and split |
| reversed | Swap each correlated assignment: 10% aligned, 90% opposite |
| counterfactual | 100% group-opposite tint |
| grayscale | Grayscale from the untinted image, saved with three RGB channels |
| all_red | Red version of every source |
| all_blue | Blue version of every source, paired with all_red |

Grayscale removes natural color too; it is an additional control, not a pure
removal of the synthetic tint. For pretrained evaluation, all_red/all_blue
allow paired cue changes without assuming the model learned our group mapping.
Evaluation conditions are also available in train and validation for controlled
experiments; they are not an instruction to train on every condition.

### Storage and loading

```text
color_bias_dataset/
  config.json
  generation.json
  status.json
  images/<split>/<class_id>/<original|red|blue|grayscale>/<image>.png
  manifests/<train|validation|test>/<condition>.csv
reports/color_bias_current/
  audit.json
  index.html
  assignment_overview.png
  class_<index>_<class_id>.png
  visual_sample_records.json
  pipeline.log
```

Four unique variants are stored once for each source: 54,000 PNGs total.
The eight conditions reference those variants through 24 manifests, with
108,000 records total. Reusing a variant across conditions is intentional;
source images never cross experimental splits.

**Use a manifest-based loader, not ImageFolder on the images root.** Each CSV
row supplies the true `class_index`, source path/hash, output path/hash, group,
split, condition, variant, and alignment flag. Output paths are relative to
`color_bias_dataset`; source paths are relative to the project root.
For biased training use `manifests/train/correlated.csv`; for an untinted
training baseline use `manifests/train/original.csv`. Reserve test manifests
for final evaluation, not hyperparameter or prompt tuning.

### Run and verify

Python 3.11 is configured in `.venv`. Template 1 needs only NumPy and Pillow:

```bash
.venv/bin/python -m pip install -r requirements-color.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python color_bias_template1_complete.py --workers 4
.venv/bin/python color_bias_template1_complete.py --audit-only --workers 4
```

The entry point runs generation, independent full auditing, and report creation.
Use `--output` and `--report-dir` for a separate experiment. `--correlation`
and `--seed` are recorded in the configuration. A different configuration
cannot overwrite an existing dataset. Integer assignments use floor(n × ratio)
and report actual counts; at our default 90/10 all split counts are exact.

Rerunning validates existing images against the expected pixels, repairs
missing/corrupted outputs, and rebuilds deterministic manifests. PNGs and
metadata use temporary files followed by replacement. Source hashes are
checked; altered originals stop generation. Do not edit generated images or
run concurrent generators into the same destination.

The separate audit checks every output's existence, dimensions, RGB channels,
PNG encoding, full decoding, hash, and all pixels against an independently
computed transformation. It verifies every manifest row and all per-class
ratios. Missing/extra files, source/split changes, and incorrect manifests fail
completion. Visual panels use actual manifest assignments, including minority
examples, from training images only. Dataset completion does not establish
model robustness; that requires the model experiments.

## Recovery history

Code and historical Color reports came from GitHub commit
`1efcba41993118d9968eaafb4ae9786e9c263dc3`. Those reports describe 12,973 source
images on Linux; our new training source has 13,000. Without the old manifest,
the 27-image discrepancy cannot be explained.

`legacy/color_bias_template1_legacy.py` preserves the recovered implementation
for reference, not production execution. `reports/color_readiness/` records
its audit and sample tests; `audit_color_readiness.py` reproduces that legacy
audit (also needs Matplotlib). Historical `reports/color_bias/` and other old
reports remain untouched. **Current results belong in `reports/color_bias_current/`.**

## Template 2: Background preparation

### Completed generation; semantic review pending

The full merged checkpoint is in Google Drive at
`Template2-Colab/merged_checkpoints/production_v3_merged_cuda_mps`.
It contains all 13,500 sources' original, nature, urban and neutral variants,
full-resolution masks, cached candidate masks and provenance records. The
partial local `background_bias_dataset/production_v3` is an earlier Mac run,
not the complete merged result. Do not resume it to recover the merged dataset.

Use `colab_transfer/Template2_Merge.ipynb` and its support scripts to reproduce
the merge. Reports are under
`reports/background_bias_current/production_v3_merged_cuda_mps/`. The latest
semantic decisions are separate from the original merged checkpoint; retain
them when restoring. Two approved correction layers in
`background_bias_dataset/review_corrections_v2` have not been integrated yet.

Six bias conditions plus two controls yield 108,000 candidate manifest rows.
Correlation remains 90/10 and aligned/minority membership matches Color.

`background_foreground.py` version 3 compares original, merged-alias and expanded
boxes. Candidate selection is constrained to retain the original target mask,
preventing the background inversion observed in the rejected version 2 pilot.
Multiple instances, weak detections, low stability, low predicted IoU, extreme
mask area and violated annotation points are flagged. The 0.88/0.95 thresholds
have not been relaxed. Two source-hash-bound training-image annotations are
active in `project_metadata/background_annotations.json`; unsuccessful probes
are archived separately and not applied.

Generation does not establish semantic mask quality. Candidate manifests retain
all sources and flags, with no silent filtering. The final `manifests/` directory
is created only after independent pixels, hashes, dimensions, labels, splits,
ratios and paired assignments pass, all flagged masks have valid recorded reviews,
and the fixed 150-image class/split-stratified visual sample is approved. Until
then `training_ready` remains false. Fourteen tests cover assignment invariants,
GPU-device requirements, resume, corruption repair, changed-source rejection,
manifest tampering and release blocking.

The completed base version 3 pilot checked 30 images and 120 original/composite
PNGs numerically; 19 source masks retained quality flags. This is evidence for
the review workflow, not proof of full-dataset segmentation accuracy. Original
resolution checking corrected an earlier spider review: its prominent brown
appendages are legs, not plant stems. Full-run source snapshots and validation
metadata are saved alongside the generation log for reproducibility.

### Earlier pilots (retained as evidence)

Install with `.venv/bin/python -m pip install -r requirements-grounded.txt`.
The official ViT-B checkpoint is in `checkpoints/`; its download source and
SHA256 are recorded in `project_metadata/sam_checkpoint.json`. PyTorch 2.5.1,
TorchVision 0.20.1, OpenCV 4.13.0.92 and pinned SAM code are installed. Color's
NumPy/Pillow versions were preserved; all eleven current tests pass.

The entry point now prepares split-aware assignments and runs a quality pilot:

```bash
.venv/bin/python background_bias_template2_complete.py --plan-only
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python background_bias_template2_complete.py --method grounded --pilot-per-class 3 --device mps --run-name mps_reference
.venv/bin/python audit_background_grounded.py --run-name mps_reference
.venv/bin/python background_grounded_report.py --run-name mps_reference
```

SAM ViT-B uses 32 points per side, 8 points per batch, predicted IoU threshold
0.88, stability threshold 0.95, zero crop layers, and minimum region area 100.
Original-resolution images are supplied to SAM. The Mac GPU was tested after
converting upstream point coordinates to float32 before MPS transfer. SAM's
positional encoder already uses float32; model and sampling settings remain
unchanged. Unsupported MPS operations may fall back to CPU.

The pilot caches source/checkpoint/configuration-identified candidate masks and
records hashes. Ten class samples produced nature, urban and neutral composites;
all 30 passed independent pixel blend checks. However, eight automatic masks
were visually rejected and two require further review. Failures include image
borders, cage bars, background, and object fragments. The heuristic score is
not a semantic confidence score. A CPU check selected the same cage-bar mask
as MPS on the goldfinch example (mask IoU 1.0).

`background_prompt_probe.py` tests an explicitly recorded visual box and
positive/negative points on that goldfinch training image. It substantially
improves target retention but is only a diagnostic, not a validated method for
all images. The automatic heuristic can be reproduced with `--method automatic`;
the entry point now defaults to the guided pilot described below. It does not
offer a full-generation mode. Pilot outputs are not training-ready data.

### Current guided pilot

The guided entry points now require **MPS by default** for SAM. The Codex sandbox
can report `mps_available=False` even though the Mac supports it; GPU runs must
execute outside that sandbox. Explicit `--device mps` fails if MPS is unavailable
instead of silently using CPU. `PYTORCH_ENABLE_MPS_FALLBACK=0` disables fallback
of unsupported GPU operations; the code records and checks model-parameter and
image-embedding devices plus allocated MPS memory. Detection and image/file
processing still run on CPU. The MPS pilot is isolated under
`background_bias_dataset/pilot_grounded_mps_reference` with its report under
`reports/background_bias_current/grounded_mps_reference`; the prior CPU pilot
and visual review remain separate.

The completed 30-image MPS reference run verified both SAM parameters and image
embeddings on `mps:0`, with CPU fallback disabled. All 90 composites passed audit.
Twenty-three selected masks exactly match CPU; minimum mask IoU across all 30
is 0.99968. Median observed SAM blocks were 2.05 seconds on CPU and 0.92 seconds
on MPS, including mask serialization; this is a small pilot comparison, not an
end-to-end speedup guarantee. Device agreement does not fix the known semantic
mask defects. See the MPS report and `cpu_comparison.json` for recorded evidence.

`background_grounded_pilot.py` uses local Grounding DINO Tiny with a class-specific
animal prompt to provide a box to the same SAM ViT-B model. This is an explicit
change from automatic grid sampling to box-prompted segmentation, in float32
with original-resolution RGB supplied to SAM. The detector uses its standard
image preprocessing. The detector checkpoint revision and file hashes are in
`project_metadata/grounding_dino_checkpoint.json`; local safetensors loading
requires no network or remote model code during inference.

The completed CPU pilot covers three frozen training images per class (30 total).
It saves all detections, all three SAM masks, the selected mask, nature/urban/neutral
composites, source hashes, model/configuration provenance, timings, and candidate
previews under `background_bias_dataset/pilot_grounded`. Detection caches are
configuration-identified. Changing configuration requires a new versioned output
directory; rerunning currently recomputes SAM masks.

`reports/background_bias_current/grounded_pilot/index.html` is the current review.
All 90 saved composites pass independent pixel and provenance checks. Visual
inspection marked 14 previews plausible, 11 needing refinement, and 5 rejected.
These are qualitative development judgments, not measured segmentation accuracy
or production approval. Twenty masks fail the unchanged 0.95 stability gate;
three images have competing detections. No threshold was relaxed. Numerical
flags remain binding even when a preview looks plausible.

Remaining failures include water/railing attached to a shark, substrate around
an axolotl, grass around a Komodo dragon, and missing spider legs or retained
stems. `visual_review.json` records every decision and `review_queue.json` lists
repairs. Next: recorded point-prompt refinement and a consistent policy for all
visible target instances, then broader training-only validation. Do not silently
drop difficult sources or change the shared split to improve apparent quality.
The earlier pilot itself is not a production dataset. The active full candidate
run is described above; Shape/Text/Size and model comparisons remain pending.

The verified assignment plan uses the same aligned/mismatched sources as Color:
Group A is 90% nature, Group B is 90% urban; balanced, reversed, fully conflicting
and paired all-nature/all-urban conditions are also planned. These synthetic
scene families differ in color and structure, so the intervention is compound
background appearance, not isolated semantic context. The pilot retains neutral
and untouched-original controls for separating segmentation effects.

The recovered Background implementation is preserved in
`legacy/background_bias_template2_legacy.py` for reference only.

Original data, generated datasets, checkpoints, source ZIPs, and `.venv` are
ignored by Git. Source code, metadata, and reports are retained for reproducibility.
