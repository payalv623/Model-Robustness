# Free CPU continuation after Colab GPU exhaustion

This recovery preserves the original CUDA checkpoint folder and creates a separate `production_v3_cpu_recovery` dataset. It keeps the same ten classes, 13,500 sources, frozen splits, annotations, 90/10 correlation, seed, model weights, float32 precision, segmentation rules, and review gates.

The Colab notebook has a dedicated **FREE CPU RECOVERY** code cell immediately after the source-bundle extraction cell. With a CPU runtime, run the Drive mount cell, bundle extraction cell, and this recovery cell. Do not run the older CUDA installation/benchmark/full-run cells on the CPU runtime.

The recovery cell contains a compressed copy of these three readable scripts:

- `start_cpu_recovery.py`: installs the pinned CPU environment, checks model weights, restores original CUDA checkpoints, verifies every frozen source, and starts recovery.
- `prepare_cpu_recovery.py`: creates separate CPU code/output directories without changing the original CUDA code or results.
- `recovery_provenance.py`: imports only complete CUDA records whose identities and output hashes match. It preserves their original records and backend labels; incomplete or corrupted outputs are generated on CPU.

## Persistent files

- Original results: `My Drive/Template2-Colab/checkpoints/production_v3_cuda`.
- Recovery results: `My Drive/Template2-Colab/cpu_recovery_checkpoints/production_v3_cpu_recovery`.
- Recovery source: `My Drive/Template2-Colab/cpu_recovery_code` (copied before inference starts).

Recovery backups are attempted every ten processed sources, plus final/error backup. Up to nine newly processed sources may require reprocessing after an abrupt runtime loss; an in-progress checkpoint is not yet durable. The first recovery backup includes all imported results and their provenance. Original CUDA checkpoints remain untouched.

## Validation completed locally

`test_recovery.py` uses synthetic inference; it does not claim hardware inference testing. It verified 298 unchanged CUDA imports, two CPU repairs, restart without recomputation, rejection of corrupted provenance, 1,200 independently reconstructed PNGs, 2,400 manifest rows, and checkpoint restoration. Training remains blocked by unresolved review requirements.

Full real-data generation, independent numerical audit, flagged-mask review, and required visual sample review must still finish before Template 2 can be declared training-ready. CPU and CUDA may select different masks due to numerical differences; backend provenance is retained per image.
