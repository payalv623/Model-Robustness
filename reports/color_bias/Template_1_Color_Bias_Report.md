# Template 1 — COLOR Bias

## 1. Objective
Introduce a label-correlated color factor while retaining the original image content.

## 2. Image Standardization
- Resolution: 224 × 224 pixels
- Output channels: RGB
- Original dataset directory: `archive/`
- Original files are not modified.

## 3. Color Transformation
The generated color-biased image uses:

**Output = 0.70 × Original + 0.30 × Color Overlay**

Two color overlays are used:
- Warm / Red
- Cool / Blue

## 4. Correlation Rule
### Group A — Classes 0–4
90% correlated with red and 10% mismatched with blue.

### Group B — Classes 5–9
90% correlated with blue and 10% mismatched with red.

## 5. Frozen Classes
- Class 0: n01440764 (Group A)
- Class 1: n01484850 (Group A)
- Class 2: n01494475 (Group A)
- Class 3: n01531178 (Group A)
- Class 4: n01632777 (Group A)
- Class 5: n01665541 (Group B)
- Class 6: n01687978 (Group B)
- Class 7: n01695060 (Group B)
- Class 8: n01749939 (Group B)
- Class 9: n01775062 (Group B)

## 6. Evaluation Modes
Four modes were generated:
1. Correlated
2. Randomized
3. Reversed Tint
4. Grayscale

### Correlated
- Total images: 12973
- Correlated images: 11675
- Mismatch images: 1298
- Actual correlated percentage: 89.99%
- Actual mismatch percentage: 10.01%

### Randomized
The color assignments are shuffled across all selected images while preserving the total red/blue assignment counts from the correlated construction.

### Reversed Tint
Classes 0–4 receive blue and classes 5–9 receive red.

### Grayscale
Images are converted to grayscale and retained as three-channel RGB output.

## 7. Integrity Verification
### correlated
- Files checked: 12973
- Bad/unreadable files: 0
- Wrong size: 0
- Wrong channel mode: 0
### randomized
- Files checked: 12973
- Bad/unreadable files: 0
- Wrong size: 0
- Wrong channel mode: 0
### reversed
- Files checked: 12973
- Bad/unreadable files: 0
- Wrong size: 0
- Wrong channel mode: 0
### grayscale
- Files checked: 12973
- Bad/unreadable files: 0
- Wrong size: 0
- Wrong channel mode: 0

## 8. Visual Verification
Visual comparison panels are stored in:
`reports/color_bias/visual_samples/`

Each panel compares:
- Original
- Red
- Blue
- Class-correlated color
- Reversed color
- Grayscale

## 9. Numerical and Visual Artifacts
- `color_bias_dataset/correlated/`
- `color_bias_dataset/randomized/`
- `color_bias_dataset/reversed/`
- `color_bias_dataset/grayscale/`
- `color_bias_dataset/correlated_audit.json`
- `color_bias_dataset/image_integrity_audit.json`
- `reports/color_bias/correlation_distribution.png`
- `reports/color_bias/correlated_vs_randomized.png`
- `reports/color_bias/visual_samples/`

## 10. Model Training
No model training is performed in this stage.
The generated datasets are prepared for the later model-training and evaluation stage.