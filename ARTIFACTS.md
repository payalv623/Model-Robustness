# Datasets, models and recovery

GitHub contains source code, tests, frozen class/split definitions, provenance,
and audit/review metadata. Generated images, masks, model weights, transfer ZIPs
and large review sheets stay outside new Git commits. Existing historical report
images already tracked in the repository are retained.

**Storage migration is not complete.** On 2 October 2026 the requested Supabase
project (`poxcywpjiivokukekvok`) showed the Free plan, with 1 GB of file storage.
No artifacts were uploaded there. TeraBox was selected instead; sign-in and
uploads are pending.
The paths below record the current locations; they are not new cloud download
links. Do not delete the local files based on this inventory.

## Current inventory

Exact local byte counts and migration status are recorded in
`project_metadata/artifact_inventory.json`. Paths are relative to the clone.

| Artifact | Restore location | Current copy |
|---|---|---|
| 13,500 frozen source images | `archive/` | Local; verified source ZIP also in Drive `Template2-Colab` |
| SAM and Grounding DINO weights | `checkpoints/` | Local; pinned official download URLs and hashes in `project_metadata/` |
| Template 1 Color | `color_bias_dataset/` | Local, complete and verified |
| Template 3 Shape | `shape_bias_dataset/production_v1/` | Local, complete and verified |
| Full Template 2 merged candidates | `background_bias_dataset/production_v3_merged_cuda_mps/` | Google Drive merged checkpoint below; **not training-ready** |
| Template 2 masks/records and visual sample | `colab_transfer/mask_review/` | Local review bundle, not the entire merged dataset |
| Template 2 Mac remainder | `colab_transfer/mac_remainder/` | Local; transfer ZIP also in Drive |
| Template 2 approved correction layers | `background_bias_dataset/review_corrections_v2/` | Local; two corrections, not integrated |
| Audits and review media | `reports/` | Local; lightweight metadata also in GitHub |
| Transfer archives | `colab_transfer/*.zip` | Local; some also in Drive as recorded in transfer metadata |
| Original ImageNet-100 download | `archive (1).zip` | Local, optional; not required for the frozen ten-class experiment |

The local project artifacts occupy about 14.5 GB, excluding the 17.3 GB original
ImageNet-100 ZIP and the full merged checkpoint on Drive. Some transfer copies
duplicate other artifacts. The Python environment is reproducible from the
requirements files and is not an artifact to upload.

## Existing Google Drive backup

Folder: <https://drive.google.com/drive/folders/1_net-2l_JI-PXmEwz21WXy9Y__QUJFT1>

- `Template2-Colab/merged_checkpoints/production_v3_merged_cuda_mps/chunk-000001.zip`
  contains the original 13,500-source merged result. Verify its SHA-256 against
  the accompanying `chunk-000001.zip.sha256` before extraction.
- `Template2-Colab/semantic_review_v1/` holds the newer sample decisions and
  summary. The original merged ZIP predates these decisions.
- The two current corrected masks remain local. They must be preserved alongside
  the checkpoint; they must not directly overwrite its original records.
- Do not treat a restored checkpoint as quality-approved. Template 2 remains
  blocked from training release pending semantic corrections and review.

## Model restoration available now

Python 3.11 or newer is sufficient; this downloader needs no ML dependencies:

```bash
python3 download_models.py
python3 download_models.py --check-only
```

This downloads the exact SAM ViT-B and Grounding DINO Tiny files recorded in
`project_metadata/sam_checkpoint.json` and
`project_metadata/grounding_dino_checkpoint.json`. Every file is checked against
its recorded SHA-256 before replacing the destination. Existing correct files
are reused. These are official-source downloads, not a completed cloud migration.

Once artifact storage is selected, its repository/bucket identifiers, immutable
file revisions and hashes will be recorded here, with a verified restore command.
Credentials belong in local environment variables or the provider's login store,
never in GitHub or downloadable manifests.
