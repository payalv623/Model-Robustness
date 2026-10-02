# Template 2 visual review — 1 October 2026

All 150 fixed class/split sample images were inspected as original/mask/neutral
triplets at the model's 224-pixel resolution. Decisions are conservative visual
judgments, not pixelwise ground truth or species-label adjudication.

| Decision | Images |
|---|---:|
| Approved | 64 |
| Clear failure | 52 |
| Needs closer inspection/annotation | 34 |

Observed errors include background water/rocks/bark retained, animal parts or
additional animals omitted, and foreground/background inversion. Five clear
failures had no automatic flags. Therefore simply lowering the SAM stability
threshold would not establish dataset quality.

The inspection decisions refer to the original merged masks. They stay rejected
until a corrected mask is integrated and reviewed under its new hash. No images
were dropped or relabeled. The frozen splits and 90/10 assignments remain intact.

## Corrected examples

- `n01775062_7053.JPEG`: saved SAM candidate 1 retains the spider instead of the
  surrounding hand/background. No new inference needed.
- `ILSVRC2012_val_00017986.JPEG`: the old mask was inverted. A new SAM MPS run with
  one source-specific positive body point, candidate 0, removes the background
  while retaining the visible spider. Earlier incomplete proposals are retained
  separately for provenance and must not be used as the final correction.

Both current corrections are in `background_bias_dataset/review_corrections_v2/`.
Their source, parent-record and mask hashes, derivation and saved variant hashes
are recorded. The eight RGB variants passed independent pixel checks. These are
approved correction layers, **not a completed full dataset release**.

The faint underwater shark `n01484850_20518.JPEG` remains rejected. Glare obscures
its outline; the current proposal retains water. A trustworthy annotation is
still needed before its cutout can be approved.

## Resume without repeating generation

1. Open `../visual_sample/page_0000.jpg` through `page_0014.jpg`; use the numbered
   index and `fixed150_decisions.json` for exact failure reasons.
2. Use saved candidate masks first. Otherwise record source-specific guidance,
   run only that source, and inspect the corrected original/mask/neutral triplet.
3. Record each decision with source and mask SHA-256. `background_review.py`
   provides the contact-sheet and explicit-decision utilities; it never
   auto-approves unseen masks or releases training manifests.
4. Inspect the remaining flagged sheets, integrate accepted corrections with
   provenance, then rerun the full numerical and quality-release audit.

The current `audit.json` is the earlier full numerical audit. It is preserved,
not rewritten to imply that a new full release audit has happened. Latest review
counts are in `summary.json`. Training-ready remains false.
