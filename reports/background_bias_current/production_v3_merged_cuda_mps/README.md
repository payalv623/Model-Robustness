# Template 2: generation complete; visual review found mask errors

Latest update: all 150 fixed sample images have been inspected. **64 approved,
52 rejected, 34 uncertain.** Five clear failures were unflagged by SAM's automatic
checks. This is evidence that confidence scores cannot replace visual review.
Two spider corrections are prepared and verified in
`background_bias_dataset/review_corrections_v2/`; integration is still pending.

The original table below records the earlier full numerical audit. Current
semantic counts and per-class decisions are in `semantic_review/summary.json`;
image-specific decisions are in `quality_review.json`. The current base dataset
has 8,555 unresolved flagged/rejected sources, with 86 required sample cases not
approved. Their union is 8,568 unique sources. The 8,482 unseen flagged sources
must not be treated as approved because some sampled masks passed.

The same **10 frozen classes and 13,500 sources** are retained. The split remains **10,400 training / 2,600 validation / 500 test**. Generation, merge, provenance checks, independent pixel checks and candidate-manifest checks have finished.

| Verified item | Result |
|---|---:|
| CUDA source results | 12,398 |
| Mac MPS source results | 1,102 |
| Total source images | 13,500 |
| Rendered PNG variants checked | 54,000 |
| Candidate manifest rows checked | 108,000 |
| Numerical audit issues | 0 |
| Sources flagged for quality review | 8,575 |
| Fixed visual sample pending at the original audit | 150 |

**Training-ready: no.** Numerical verification establishes file integrity, rendering arithmetic, labels, split membership, condition ratios and alignment with the Color template. It does not establish accurate foreground segmentation.

The 150-image sample may overlap the flagged queue. Flags below overlap each other and must not be summed to count affected images. A flag is a review trigger, not a confirmed segmentation failure.

## Review triggers

| Flag | Sources |
|---|---:|
| low_sam_stability | 8,032 |
| weak_target_detection | 811 |
| multiple_instances_require_review | 759 |
| low_sam_predicted_iou | 416 |
| weak_secondary_detection_withheld | 399 |
| extreme_mask_area | 140 |
| manual_recovery_annotation_requires_review | 2 |

## Next steps

1. Resolve the 52 rejected and 34 uncertain sample masks. The sample inspection itself is complete; passing the sample gate still requires acceptable corrected masks.
2. Integrate the two verified corrections, correct the explicitly rejected faint-shark mask, and review/repair the remaining flagged masks, retaining source membership and recording source/mask hashes with each decision.
3. Rerun the numerical and quality-release audit after repairs. Release training manifests only after the review gates pass.
4. Shape is now complete and verified independently of SAM. Text and Size, model experiments and comparisons remain pending.

## Saved evidence

- `audit.json`: full numerical audit, with zero issues.
- `review_queue.json`: 8,575 flagged sources and reasons.
- `required_visual_sample.json`: fixed 150-source class/split sample.
- `quality_review.json`: source/mask-bound rejection of the faint-shark proposal.
- `merged_config.json`: merged configuration and provenance hashes.
- `merge.log`: completed merge, audit and Drive checkpoint confirmation.

Cloud results: **My Drive/Template2-Colab/merged_checkpoints/production_v3_merged_cuda_mps**, checkpoint **chunk-000001.zip**, containing 13,500 records. Original CUDA checkpoints and the Mac archive remain preserved. Local reports were downloaded and verified with SHA-256 `2f8fb94901326878dc9e564f93dad2b79ba4f51330a6091ee43a9b17a09bc522`.
