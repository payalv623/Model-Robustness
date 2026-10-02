# Template 2 full candidate merge

`merge_results.py CUDA_ROOT MAC_ROOT MERGED_ROOT` creates a separate merged candidate dataset without running inference. It verifies the original generator/foreground code hashes, source manifest, model identifiers, policy equivalence, scene renderer, and condition-assignment code before merging.

For each frozen source it selects the Mac record if one exists in the audited remainder, otherwise the original CUDA record. It rejects missing, stale, corrupted, or incompatible inputs. Original record JSON, model-device labels, annotations, code, and available engine metadata are retained and hashed. The merged record's identity refers to the merged configuration while its original identity is independently verified against the retained original configuration.

All sources are retained. Candidate manifests preserve the original 90/10, balanced, reversed, counterfactual, all-nature, all-urban, original, and neutral conditions, with exact class counts and Color-template minority membership checked by the independent audit. No training manifests are released while quality review is unresolved.

The faint shark recovery proposal (`n01484850_20518.JPEG`) is explicitly rejected for training because visual inspection shows excess water/glare in the retained foreground. The spider recovery is also flagged and not approved. Their numerical generation success is not a semantic-quality claim.

## Validation

`test_merge.py` verifies a 300-source synthetic merge (298 CUDA and two MPS), 1,200 independently checked PNGs, 2,400 manifest rows, retained quality gates, and rejection of modified provenance. It does not claim to test ML inference.

The real merge is launched by the final **FINAL TEMPLATE 2 MERGE** cell in the existing Colab notebook. Its persistent destination is `My Drive/Template2-Colab/merged_checkpoints/production_v3_merged_cuda_mps`. The original CUDA and Mac archives remain intact.
