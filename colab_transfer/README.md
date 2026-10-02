# Template 2 on Google Colab

Prepared for the existing ten-class project. The original Mac workers and sources are unchanged.

## Files

- `Template2_Colab.ipynb`: upload/open in Colab and run cells in order.
- `template2_imagenet10_sources.zip`: exactly 13,500 frozen images, 1.63 GB, verified individually against the frozen split manifest.
- `template2_colab_code.zip`: CUDA worker, audits, class/split metadata, two source-bound annotations, and Template 1 correlation references. No model weights or Color image outputs.
- `transfer_manifest.json`: archive sizes and SHA-256 digests; these are also embedded in the notebook.

## Cloud flow

1. Put both ZIP files in `My Drive/Template2-Colab`.
2. Open the notebook and choose Runtime > Change runtime type > T4 GPU. Authorize the notebook's Google Drive connection when prompted.
3. The notebook verifies and extracts the bundles on VM-local disk, installs an isolated Python 3.11/PyTorch 2.5.1 CUDA environment, downloads the same pinned model weights, and verifies all source hashes.
4. Run the 100-image benchmark (ten training images per class), independent pixel audit and preview. Actual cloud throughput is unknown until this runs.
5. Run full generation. Both Grounding DINO and SAM use CUDA float32; TF32 is disabled. GPU device evidence is recorded per image and checked by the auditor. CPU image preparation and file writing remain normal.
6. Incremental archives are saved to `My Drive/Template2-Colab/checkpoints` every 100 records and at normal completion/errors. After a runtime reset, rerun the notebook with the same bundles; it restores checkpoints and rechecks identities/hashes. A hard interruption may lose up to 99 uncheckpointed records. Corrupt/incomplete checkpoints stop restoration for investigation rather than silently dropping data.
7. Review flagged masks and the independent 150-image sample. Only after passing numerical AND visual quality gates can training manifests be released. This notebook never trains evaluation models or treats mask confidence as semantic approval.

## Output isolation

The CUDA dataset is `background_bias_dataset/production_v3_cuda`; reports are under `reports/background_bias_current/production_v3_cuda`. Mac MPS candidates are retained separately. CUDA and MPS can produce slightly different masks, so their records are not silently mixed. This migration starts a reproducible CUDA run; it does not reuse incomplete MPS results.

## Local verification

The adapted synthetic generation/resume/repair/manifest/quality-gate test and the two selection-policy tests pass. Backup tests verify save/restore, repaired-record replacement, corruption detection and rejection of path traversal. Notebook Python cells compile. All 13,500 archived source images match the frozen SHA-256 records. These checks do not establish actual CUDA execution; that requires the cloud benchmark.

The local scripts `build_bundle.py` and `build_notebook.py` regenerate archives/notebook after a deliberate code change. Never rebuild a deployed bundle casually: configuration hashes protect resumability.

## Current cloud session

Notebook: https://colab.research.google.com/drive/1Syv0DzN3GdgcT9khYgAGgV1lxTMhusy3

Transfer folder: https://drive.google.com/drive/folders/1_net-2l_JI-PXmEwz21WXy9Y__QUJFT1

The notebook is connected to a T4 GPU with Drive mounted. Environment setup and weight downloads completed successfully. The code archive is uploaded. The source upload failed once and was retried; last observed progress was 33%. Run All is queued and the notebook currently prints that it is waiting for the source upload. It will wait at most one hour, then fail clearly if the file remains missing. Once the upload appears, verification, the benchmark and full generation proceed in order if prior checks pass. Actual CUDA inference and timing remain unverified until the benchmark finishes. Keep the Drive upload tab open and the Mac awake while transferring.
