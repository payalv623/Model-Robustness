# Model Robustness

Research project for benchmarking spurious feature reliance
(shortcut learning) in computer vision.

## Dataset

ImageNet-100 is used as the base dataset.

Ten fixed classes are selected and reused across all five
bias templates.

## Bias Templates

1. Color Bias
2. Background Bias
3. Shape Bias
4. Text Bias
5. Size Bias

## Current Progress

### Template 1 — Color Bias
- Correlated
- Randomized
- Reversed
- Grayscale

### Template 2 — Background Bias
- Correlated
- Balanced
- Counterfactual

## Current Stage

Dataset generation, numerical auditing, and visual verification.

Model training has not yet been performed.

## Important

The original ImageNet-100 dataset, generated datasets, and model
checkpoints are excluded from this repository through `.gitignore`.
