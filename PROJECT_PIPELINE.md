# Project pipeline

All templates use the same 10 frozen classes and 13,500 source images. The split
is 10,400 training, 2,600 validation and 500 test images. Source files stay intact.

| Phase | Work | Current status |
|---|---|---|
| 1. Build biased datasets | Color, Background, Shape, Text, Size | Color and Shape verified; Background generated and numerically verified, but visual review found mask errors; Text and Size pending |
| 2. Prepare model experiments | At least four vision/VLM models, consistent inputs and class labels | Pending; final model list and pretrained-only versus fine-tuning design still need selection |
| 3. Measure robustness | Compare aligned, balanced, reversed and counterfactual conditions with original/neutral controls | Pending verified datasets and model setup |
| 4. Compare and report | Per-class results, robustness drops, cue sensitivity, uncertainty and examples of failures | Pending experiments |

## Background: generation and merge

- All 13,500 frozen sources have generated candidates: 12,398 from CUDA and 1,102 from the Mac GPU remainder.
- Same ten classes, 10,400/2,600/500 split membership, seed 42 and 90/10 assignments as Color.
- The Mac remainder passed independent numerical checks for 1,102 sources and 4,408 PNGs.
- Original CUDA/MPS records, source hashes, annotations, backend labels and code are retained in the merged dataset.
- The full merged audit passed: 13,500 sources, 54,000 PNGs and 108,000 manifest rows; labels, splits, 90/10 ratios and Color membership checked.
- Merged results are backed up to Drive; the full reports are in `reports/background_bias_current/production_v3_merged_cuda_mps/`.
- Recovery/merge instructions: `colab_transfer/merge_support/README.md` and `colab_transfer/Template2_Merge.ipynb`.

## Background: still required

1. Correct the rejected faint-shark mask and resolve the flagged-mask review queue. The original queue contains 8,575 images; 8,482 of those have not yet been visually inspected.
2. The fixed 150-image sample has now been inspected: 64 approved, 52 rejected, 34 uncertain. Five rejected masks had passed the automatic confidence checks. The 86 unapproved sample masks still need corrections or closer review.
3. Two corrected spider masks are saved with source hashes, derivation evidence and independently checked composites in `background_bias_dataset/review_corrections_v2/`. They are approved as correction layers but have not yet been integrated into a newly audited full dataset. Original records remain intact.
4. Latest semantic status: `reports/background_bias_current/production_v3_merged_cuda_mps/semantic_review/summary.json`. The original merged dataset has 8,555 unresolved flagged/rejected sources; including additional required sample cases gives 8,568 unique sources awaiting approval. These counts overlap and must not be summed.
5. Integrate approved corrections, complete remaining quality review and rerun the quality-release audit. Release training manifests only when all required checks pass.

Two detector failures received source-bound manual guidance to generate reviewable proposals. The shark proposal visibly retains excessive water/glare and is rejected for training. The spider proposal remains flagged. No source was silently removed to make the audit pass.

Template 2 is not training-ready yet. More SAM inference alone does not resolve ambiguous source boundaries; unclear cases need trustworthy source-specific annotations. No confidence threshold was reduced and no source was removed.

## Shape: complete

- Template 3 is generated and independently verified for all 13,500 sources: 54,000 PNGs and 108,000 manifest rows, with zero audit issues.
- Same ten classes, frozen splits, seed 42, exact per-class 90/10 assignments and Color majority/minority membership.
- Gray circle versus gray triangle, equal cue color and 256-pixel area, placed below the photograph without occluding it.
- Eight conditions, including an identically framed cue-removed control. All 30 class/split sample transformations were visually approved.
- Training manifests: `shape_bias_dataset/production_v1/manifests/`.
- Method and verification: `reports/shape_bias_current/README.md` and `audit.json`.
- This measures a geometric-marker shortcut, not intrinsic object shape-versus-texture preference. The earlier colored-shape proposal would mix shape and color, so this implementation controls marker color.

Text, Size and model experiments remain pending. Dataset approval does not mean model training has happened.

Nature and urban backgrounds differ in both palette and structure. Report this
as a combined background-appearance shift; use neutral and original controls
to help distinguish segmentation artifacts from background sensitivity.
