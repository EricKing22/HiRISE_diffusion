# Evaluation Metrics for HiRISE Diffusion

## Why Multiple Metrics?

A single number cannot characterise image generation quality. Pixels can be accurate on average (low MSE) yet look structurally wrong; a model can reproduce fine textures (high SSIM) yet shift the mean brightness (low Pearson r); or it can interpolate plausibly per image but produce a generated distribution that diverges from real (high FID). The metrics below each probe a distinct failure mode.

---

## Dual-Space Metric Framework

All metrics are reported in **two complementary spaces**:

**[Physical Space]** — reverse-normalized to original DN values
```
x_phys = (clamp(x_norm, -10, 10) + dc) × scale + center
```
- MSE/MAE: absolute error in DN units. **Not directly comparable across scenes** — physical MSE scales with `scale²`, so a low-contrast scene (scale = 0.05) always has smaller physical MSE than a high-contrast scene (scale = 0.20) even with identical model performance.
- PSNR: fixed MAX = 1.0, computed from pooled MSE (see §3).
- Use for: reporting absolute error in physical units.

**[Normalized Space]** — model's operating range `clamp(-10, 10)`
```
x_norm = clamp(x, -10, 10)
```
- MSE: `MSE_norm = MSE_phys / scale²`. Two scenes with identical relative error produce the same `MSE_norm` — **comparable across all scenes**.
- PSNR: fixed MAX = 20, computed from pooled MSE (see §3).
- Use for: **primary model quality comparison across runs, λ sweeps, or different architectures**.

**[Structural Metrics]**
- **SSIM:** computed in physical space with per-image GT data_range via custom `_ssim_single` (see §4).
- **Pearson r:** computed in normalized space (invariant to affine transforms by definition).

---

## 1  MSE — Mean Squared Error

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\hat{x}_i - x_i)^2$$

**Physical space:** MSE in original DN units. A scene with scale=0.15 has 225× larger MSE_phys than a scene with scale=0.01 for the same relative error.

**Normalized space:** `MSE_norm = MSE_phys / scale²`. Two scenes with identical relative error produce the same `MSE_norm` regardless of their absolute brightness.

---

## 2  MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\hat{x}_i - x_i|$$

L1 pixel accuracy. More robust to outlier pixels than MSE (linear vs. quadratic penalty).

If `MSE >> MAE²`, error is concentrated in sparse spikes. If `MSE ≈ MAE²`, errors are spatially uniform — useful diagnostic for HiRISE with occasional bright surface features.

---

## 3  PSNR — Peak Signal-to-Noise Ratio

$$\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{\text{MAX}^2}{\overline{\text{MSE}}}\right) \quad \text{[dB]}$$

where $\overline{\text{MSE}}$ is the **pooled (averaged) MSE across all images**.

**Physical space PSNR:** MAX = 1.0 (fixed).
**Normalized space PSNR:** MAX = 20 (fixed, from clamp range [-10, 10]).

### Why pooled MSE, not per-image PSNR averaging

The naive approach — compute PSNR per image, then average — is biased upward by Jensen's inequality. Since `log` is concave:

$$\text{mean}\bigl(\text{PSNR}_i\bigr) = \text{mean}\!\left(10\log_{10}\frac{\text{MAX}^2}{\text{MSE}_i}\right) \;\geq\; 10\log_{10}\frac{\text{MAX}^2}{\text{mean}(\text{MSE}_i)}$$

A few "easy" images with tiny MSE produce disproportionately high PSNR values (e.g. 50+ dB), pulling up the arithmetic mean. In our HiRISE evaluation, this inflated per-image averaged PSNR by ~6 dB (33 dB vs 27 dB from pooled MSE).

The pooled approach — `10·log10(MAX² / mean(MSE))` — gives a single global PSNR that faithfully reflects average pixel-level error. This is the more conservative and statistically robust measure.

### Fixed MAX vs. per-image range

Per-image range as MAX (`max - min` of each GT image) is problematic for HiRISE: scenes have narrow brightness ranges (e.g. range = 0.03), and `MSE / range²` can exceed 1, producing negative PSNR values. A fixed MAX avoids this while keeping PSNR interpretable and comparable.

---

## 4  SSIM — Structural Similarity Index

$$\text{SSIM} = \frac{(2\mu_x\mu_{\hat{x}}+C_1)(2\sigma_{x\hat{x}}+C_2)}{(\mu_x^2+\mu_{\hat{x}}^2+C_1)(\sigma_x^2+\sigma_{\hat{x}}^2+C_2)}$$

where $C_1 = (K_1 \cdot L)^2$, $C_2 = (K_2 \cdot L)^2$, $K_1=0.01$, $K_2=0.03$, and $L$ = `data_range`.

Computed in **physical space** using custom `_ssim_single`（uniform $11 \times 11$ sliding window），with **per-image GT `data_range`**:

```python
data_range = (gt_image.max() - gt_image.min()).clamp(min=0.01)
```

Numerical safeguards in `_ssim_single`:
- Variance `clamp(min=0)` — prevents tiny negatives from `E[x²] - E[x]²` float rounding
- `+1e-8` in denominator — prevents division-by-zero in flat patches

### Why per-image GT data_range

The stabilisation constants $C_1$ and $C_2$ scale as $L^2$. The choice of $L$ must match the actual signal amplitude to avoid two failure modes:

**Failure mode 1 — Constant masking ($L$ too large):** When $L$ is much larger than the image's dynamic range, $C_2$ dominates local patch variances and SSIM → 1.0 regardless of prediction quality.

Example: HiRISE reflectance range ≈ 0.03, but `data_range = 1.0`:
- $C_2 = (0.03 \times 1.0)^2 = 9 \times 10^{-4}$
- Local patch variance ≈ $10^{-5}$
- $C_2$ is 90× larger → contrast-structure term degenerates to $C_2 / C_2 = 1.0$

**Failure mode 2 — Numerical divergence ($L$ too small):** When $L$ approaches 0, $C_1 \approx C_2 \approx 0$, the SSIM formula becomes an unstable bare quotient. In earlier code with `clamp(min=1e-6)`, degenerate patches produced SSIM < −1 (e.g. −5.25).

**Fix:** Per-image GT `data_range` with floor `clamp(min=0.01)`:
- $L$ tracks each image's actual contrast → $C_2$ scales correctly → no masking
- Floor at 0.01 → $C_2 \geq 9 \times 10^{-8}$ → numerically stable even for flat patches
- Custom function adds variance clamping and denominator epsilon as extra safeguards

SSIM ∈ [-1, 1]; 1 = perfect.

---

## 5  Pearson r — Correlation Coefficient

$$r = \frac{\sum_i(x_i - \bar{x})(\hat{x}_i - \bar{\hat{x}})}{\sqrt{\sum_i(x_i - \bar{x})^2}\;\sqrt{\sum_i(\hat{x}_i - \bar{\hat{x}})^2}}$$

Linear correlation, invariant to any affine transform (global scale or offset). Always computed in normalized space.

- r = 1: perfect linear agreement regardless of scale/brightness errors
- r < 0.8: model struggles to capture spatial structure
- Complementary to SSIM: Pearson r measures global linear trend; SSIM measures local structure

---

## 6  FID — Fréchet Inception Distance

$$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \operatorname{Tr}\!\left(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g}\right)$$

Computed in **normalized space** ([-10, 10]) for maximum contrast in InceptionV3 feature extraction:
1. Tile: `(B, 1, H, W) → (B, 3, H, W)`
2. Resize to $299 × 299$
3. Rescale `[-10, 10] → [0, 1]`: `(x + 10) / 20`

### Why normalized space, not physical space

**Bug found in earlier implementation:** the rescaling assumed input in `[-1, 1]` (`(x + 1) / 2`), but was given physical-space data concentrated in `[0.05, 0.25]`. This mapped all pixel values to a narrow band around 0.55, destroying contrast. InceptionV3 extracted near-identical features for all images, making FID uninformative.

Normalized space preserves the per-scene contrast amplification from the MAD-based z-scoring. The `[-10, 10] → [0, 1]` mapping places data with std ≈ 0.2–0.3 across the `[0.35, 0.65]` range — enough dynamic range for InceptionV3 to extract meaningful texture and structure features.

Lower FID = generated distribution closer to real. Requires N >> 2048 for reliable estimates (N ≥ 2000 for publication-quality numbers).

---

## Summary Table

| Metric | Space | MAX / data_range | ↑/↓ | Use for |
|--------|-------|------------------|-----|---------|
| MSE | Physical | — | ↓ | Absolute error in DN units (scale-dependent) |
| MSE | Normalized | — | ↓ | **Model quality comparison** (scene-invariant) |
| MAE | Physical | — | ↓ | Sparse bright/dark artefacts |
| PSNR | Physical | 1.0 (fixed), pooled MSE | ↑ | Physical-space SNR |
| PSNR | Normalized | 20 (fixed), pooled MSE | ↑ | **Model quality comparison** (scene-invariant) |
| SSIM | Physical | per-image GT range | ↑ | Local structural distortion |
| Pearson r | Normalized | — | ↑ | Global brightness/scale mismatch |
| FID | Normalized | — | ↓ | Distributional shift, mode collapse |

---

## Bugs Found and Fixed

### 1. SSIM constant masking (CRITICAL — two rounds of fixes)

**Round 1 — Symptom:** SSIM ≈ 0.99 across all samples — unrealistically perfect.

**Root cause:** SSIM was computed with fixed `data_range = 1.0` (first in normalized [0,1] space, then in physical space). HiRISE local patch variances ($\sim10^{-5}$) were dwarfed by $C_2 = (0.03)^2 = 9 \times 10^{-4}$, collapsing the contrast-structure term to 1.0.

**Round 2 — Symptom:** SSIM < −1 (e.g. −5.25) with per-image GT range and `clamp(min=1e-6)`.

**Root cause:** For near-flat patches, `data_range ≈ 1e-6` made $C_1 \approx C_2 \approx 0$. The SSIM formula became a numerically unstable bare quotient, producing values outside [-1, 1].

**Final fix:** Custom `_ssim_single` with per-image GT `data_range`, floor `clamp(min=0.01)`, variance `clamp(min=0)`, and denominator `+1e-8`.

### 2. PSNR inflated by per-image averaging (SIGNIFICANT)

**Symptom:** Reported PSNR 33 dB, but PSNR from pooled MSE gives 27 dB — a 6 dB gap.

**Root cause:** Per-image PSNR averaging `mean(10·log10(MAX²/MSE_i))` is biased upward by Jensen's inequality (log is concave). A few low-MSE images produce disproportionately high PSNR values (50+ dB), pulling up the arithmetic mean.

**Fix:** Compute single global PSNR from pooled MSE: `10·log10(MAX² / mean(MSE))`.

### 3. Physical/Normalized PSNR identity revealed scale clamping

**Symptom:** Physical PSNR and Normalized PSNR were **identical** to 2 decimal places (33.02 = 33.02, 35.86 = 35.86).

**Root cause:** All validation patches had `scale = 0.05` exactly (the `clamp_min` floor in the dataset). When `scale = 0.05`:

$$\text{PSNR}_\text{phys} - \text{PSNR}_\text{norm} = -20\log_{10}(\text{scale}) - 10\log_{10}(400) = 26.02 - 26.02 = 0$$

Since scale is clamped at a minimum of 0.05 (cannot go lower), and the difference averaged to exactly 0.0 across all 100 images, **every single** image must have `scale = 0.05`. This means all eval patches had `1.4826 × MAD < 0.05` — extremely low contrast.

**Implication:** Not a code bug per se, but a data characteristic that inflated all metrics. Low-contrast patches have little signal to reconstruct, making metrics artificially easy.

### 4. FID preprocessing space mismatch

**Symptom:** FID values potentially uninformative.

**Root cause:** `_inception_features` used `(x + 1) / 2` (designed for `[-1, 1]` input) but received physical-space data in `[0, ~0.3]`. All pixels mapped to `[0.5, 0.65]` — InceptionV3 extracted near-identical features regardless of content.

**Fix:** Pass normalized tensors (`[-10, 10]`) and use `(x + 10) / 20` mapping to `[0, 1]`.

---

## SCI Lambda Sweep Analysis

### Observation: all metrics improve monotonically with lambda

Sweeping `lambda_scl = lambda_ccl` from 300 to 1000 (bidirectional model, 100k steps, DexiNed edges, n=100):

| λ | MSE (norm) | PSNR (dB) | SSIM (norm) | SSIM (phys) | **Pearson r** |
|---|---|---|---|---|---|
| 300 | 0.5583 | 28.56 | 0.4909 | 0.6455 | **0.82** |
| 400 | 0.4512 | 29.48 | 0.5414 | 0.6643 | **0.83** |
| 500 | 0.3678 | 30.36 | 0.5767 | 0.6785 | **0.83** |
| 600 | 0.2984 | 31.27 | 0.6185 | 0.6902 | **0.83** |
| 700 | 0.2424 | 32.18 | 0.6549 | 0.6991 | **0.83** |
| 800 | 0.1993 | 33.03 | 0.6800 | 0.7058 | **0.84** |
| 900 | 0.1659 | 33.82 | 0.7065 | 0.7101 | **0.83** |

MSE, PSNR, SSIM all improve with no sign of plateau at λ=1000. This is **not a bug**.

### Root cause: SCI corrects global statistics, not spatial structure

The two SCI losses (`compute_prior.py:56-84`) operate on **global image statistics**:

- **L_scl** = `|mean(x̂₀) - μ_prior| + |std(x̂₀) - σ_prior|` — corrects DC offset and variance
- **L_ccl** = chi-squared(soft_histogram(x̂₀), h_prior) — corrects value distribution shape

Neither loss has any spatial component. The gradient from L_scl is nearly uniform across all pixels — it shifts and rescales the entire image to match the prior's moments.

### Why MSE keeps dropping

MSE decomposes as:

```
MSE = Var(pred - gt) + [E(pred) - E(gt)]²
                         ↑ bias² — reduced by SCI
```

Higher lambda forces `E(pred) → μ_prior` and `std(pred) → σ_prior`. Since the ground truth comes from the same distribution, this reduces the **bias component** of MSE. There is no natural saturation because L_scl uses L1 (|·|), whose gradient is a constant ±1/N — it never reaches zero.

### Proof: Pearson r is flat

**Pearson r is invariant to mean and scale shifts** (it only measures spatial correlation). It barely moves from 0.82 to 0.83 across the entire λ range. This proves:

1. The **spatial/structural quality is constant** regardless of lambda
2. All MSE/PSNR/SSIM improvement comes from better global moment matching
3. The U-Net's structural predictions are **not improving** — SCI just wraps them in better statistics

### Why SSIM also improves

SSIM = luminance × contrast × structure. The luminance and contrast terms depend on per-patch means and variances, which benefit from global moment correction. The structure term (local Pearson-like) does not change.

### Physical PSNR = Normalized PSNR (identity, not a bug)

Confirmed across all λ values. Since all images have `scale = 0.05` (MAD floor clamp):

```
mse_phys = scale² × mse_norm = mse_norm / 400
PSNR_phys = 10·log10(1.0 / (mse_norm / 400)) = 10·log10(400 / mse_norm) = PSNR_norm
```

These are mathematically identical. Physical and normalized PSNR are redundant for this dataset.

### Practical guidance for lambda tuning

- **MSE / PSNR / SSIM will not show an optimal lambda** — they reward moment matching indefinitely
- **Pearson r** (or structure-only SSIM) is the correct metric for finding where SCI begins to distort spatial content
- For reporting: note that MSE/PSNR improvements above ~λ=100 are dominated by global moment matching, not structural improvement
- Consider computing metrics after **per-image mean/variance normalization** to factor out the SCI statistical correction and reveal true structural quality

---

## Implementation

- `evaluate_images()` in `src/eval.py` computes all metrics batch-by-batch
- DDPM outputs clamped to `[-10, 10]` before any metric computation (suppresses outlier explosion from 1000-step numerical drift)
- Denormalization: `x_phys = (clamp(x, -10, 10) + dc) × scale + center` using per-sample `norm_stats` (3,) = [center, scale, dc]
- Physical PSNR: `10·log10(1.0² / mean(MSE_phys))` | Normalized PSNR: `10·log10(20² / mean(MSE_norm))`
- SSIM: custom `_ssim_single` in physical space, uniform 11×11 window, per-image GT `data_range` with floor 0.01
- Pearson r: normalized space, per-image then averaged
- FID features: extracted from normalized tensors with `[-10, 10] → [0, 1]` rescaling for InceptionV3
