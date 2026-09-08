# Template 1 — COLOR Bias Benchmark

## 1. Objective
The purpose of this template is to introduce a label-correlated color factor while retaining the original image content.

## 2. Image Standardization
- Output resolution: 224 × 224 pixels
- Output color format: RGB
- Original images were read without modifying the source files.

## 3. Color Transformation
Each generated image uses the following pixel-wise blend:

**Output = 0.70 × Original + 0.30 × Color Overlay**

The two color conditions are:
- Warm / Red
- Cool / Blue

## 4. Class Groups

### Group A — Classes 0–4
90% receive the warm/red tint.
10% receive the cool/blue tint as deliberate mismatch.

### Group B — Classes 5–9
90% receive the cool/blue tint.
10% receive the warm/red tint as deliberate mismatch.

## 5. Frozen Classes
- Class 0: n01440764 — Group A
- Class 1: n01484850 — Group A
- Class 2: n01494475 — Group A
- Class 3: n01531178 — Group A
- Class 4: n01632777 — Group A
- Class 5: n01665541 — Group B
- Class 6: n01687978 — Group B
- Class 7: n01695060 — Group B
- Class 8: n01749939 — Group B
- Class 9: n01775062 — Group B

## 6. Numerical Audit

- Total generated images: 12973
- Correlated images: 11675
- Mismatch images: 1298
- Overall correlated percentage: 89.99%
- Overall mismatch percentage: 10.01%

## 7. Evaluation Modes

The project specification defines four evaluation conditions for the Color template:
1. Correlated
2. Randomized
3. Reversed Tint
4. Grayscale

This execution generates the correlated dataset. The visual and numerical audit is produced before model training.

## 8. Visual Verification
Visual comparison images are stored in:
`reports/color_bias_samples/`

The comparison contains:
- Original image
- Warm/red transformation
- Cool/blue transformation
- Grayscale transformation

## 9. Generated Artifacts
`color_bias_dataset/` — generated color-biased images
`project_metadata/selected_classes.json` — frozen classes
`color_bias_dataset/numerical_audit.json` — numerical audit
`reports/color_bias_distribution.png` — correlation graph
`reports/color_bias_samples/` — visual comparisons
`reports/color_pixel_statistics.json` — pixel statistics

## 10. Model Training
No model training is performed by this template-generation script.