# Datasets, models and recovery

GitHub contains source code, tests, frozen class/split definitions, provenance,
and audit/review metadata. Generated images, masks, model weights, transfer ZIPs
and large review sheets stay outside new Git commits. Existing historical report
images already tracked in the repository are retained.

**Storage migration is not complete.** On 2 October 2026 the requested Supabase
project (`poxcywpjiivokukekvok`) showed the Free plan, with 1 GB of file storage.
No artifacts were uploaded there. TeraBox was selected instead. Its signed-in
account shows **30 GB permanent + 994 GB time-limited**, not 1 TB of permanent
free storage. The expiry of the temporary allowance has not been verified.
The project backup is in progress; do not delete local or Drive copies.

TeraBox artifact folder (requires the owner's login):
<https://dm.1024terabox.com/main?category=all&path=%2FModel-Robustness-artifacts-2026-10-02>

Imported Drive files:
<https://dm.1024terabox.com/main?category=all&path=%2FImport%2FFrom%20Google%20Drive%2FTemplate2-Colab>

The import initially copied seven top-level files (reported as 1.85 GB). This
does not establish that nested checkpoint directories were copied. In particular,
the full merged checkpoint still requires separate confirmation. Its Drive
folder is [production_v3_merged_cuda_mps](https://drive.google.com/drive/folders/1Eu7mht4br4TplaSl3rTSFmcJiwjsd11r);
the ZIP is displayed as 2.82 GB.

As of 3 October 2026, **all 18 numbered local-artifact archives are present**
in the artifact folder (12,683,164,950 bytes in total). TeraBox reported the
16-file final batch complete; both earlier archives are also present.
Remote downloads have not yet been checked against the local SHA-256 hashes.
The initial `artifact_inventory.json` on TeraBox is an older snapshot; use
GitHub's inventory. A separate 8.9 MB recovery metadata export has been
prepared with the restore tools and latest semantic-review transfer ZIP; its
upload is not yet confirmed. See `project_metadata/recovery_metadata_export.json`.

The full merged Background checkpoint remains a separate transfer. Three
private transfer copies were prepared at the top level of My Drive:
`Template2-production_v3_merged_cuda_mps.zip`, its `.zip.sha256` sidecar, and
`Template2-production_v3_merged_cuda_mps-index.json`. The original checkpoint
folder remains unchanged. Only these project items should be selected in the
TeraBox importer; never use its default selection of all Drive files.

Direct browser uploads need the extension file-access setting. The 16-file
batch succeeded using keyboard selection in the native file picker, without
changing that setting. Reconnecting the Drive importer encountered automatic
approval review because its consent covers all Drive files and account
information; explicit approval is pending for the remaining connection step.

## Current inventory

Exact local byte counts and migration status are recorded in
`project_metadata/artifact_inventory.json`. Paths are relative to the clone.

| Artifact | Restore location | Current copy |
|---|---|---|
| 13,500 frozen source images | `archive/` | Local; verified source ZIP also in Drive `Template2-Colab` |
| SAM and Grounding DINO weights | `checkpoints/` | TeraBox numbered archives uploaded; official download URLs and hashes also in `project_metadata/` |
| Template 1 Color | `color_bias_dataset/` | TeraBox numbered archives uploaded; local generation verified |
| Template 3 Shape | `shape_bias_dataset/production_v1/` | TeraBox numbered archives uploaded; local generation verified |
| Full Template 2 merged candidates | `background_bias_dataset/production_v3_merged_cuda_mps/` | Google Drive merged checkpoint below; **not training-ready** |
| Template 2 masks/records and visual sample | `colab_transfer/mask_review/` | Local review bundle, not the entire merged dataset |
| Template 2 Mac remainder | `colab_transfer/mac_remainder/` | Local; transfer ZIP also in Drive |
| Template 2 approved correction layers | `background_bias_dataset/review_corrections_v2/` | Included in uploaded Background archive; two corrections, not integrated |
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

Prepared archive names, byte counts, SHA-256 hashes and per-part upload status
are recorded in `project_metadata/artifact_bundles.json`.
Credentials belong in the provider's login store, never in GitHub or manifests.

## Restore numbered TeraBox archives

TeraBox downloads currently use its signed-in web interface. A supported
automatic download API has not been verified; the script does not use unofficial
downloaders or browser cookies. Download the numbered ZIPs for the groups you
need into one local directory. Keep their filenames unchanged, then run:

```bash
# Restore model files, frozen sources, and the two completed templates:
python3 artifact_bundle.py restore --downloads /path/to/downloaded-parts \
  --group models --group sources --group color --group shape

# Restore the local Background runs/corrections and their review assets:
python3 artifact_bundle.py restore --downloads /path/to/downloaded-parts \
  --group background_local --group background_mask_review --group mac_remainder --group reports
```

Python 3.11+ is sufficient; the archive utility uses the standard library.
It checks complete part sets, archive SHA-256, safe member paths and restored
file SHA-256. It reuses identical files and refuses to replace changed local
files. No source dataset, mask or model is deleted. Restoring Background groups
above does **not** restore the full Drive-only merged checkpoint and does not
approve masks for training.

To create an archive group locally:

```bash
python3 artifact_bundle.py pack models
python3 artifact_bundle.py pack color
```

Archives are approximately 1 GB each, with a small ZIP/index overhead. A large
single-file group can be split into `.part` files and reassembled by the same
restore command. `--part N` packages only one numbered part when disk space is
limited. Source snapshots and generated part hashes protect resumed packaging.
Packaged is not uploaded: each manifest entry records its upload status separately.
