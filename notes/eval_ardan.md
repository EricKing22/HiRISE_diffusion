# Ardan-Variant Evaluation (`eval_ardan.py`)

Reproduces the evaluation framework from `HiRISE_img_reconstruction` for the
CM-Diff diffusion model, enabling direct metric comparison between the CNN
baseline and the diffusion approach.

---

## Why a Separate Evaluation Script?

`eval.py` is not modified. All new code lives in `src/eval_ardan.py` and
`src/metrics_ardan.py`. This avoids disrupting the existing training/eval
pipeline while adding the HiRISE benchmark metrics.

---

## Metrics Added

The following metrics from `HiRISE_img_reconstruction/src/post/run_evals.py`
(`compute_per_img_metrics()`) are added on top of the existing MSE/MAE/PSNR/SSIM/
Pearson/FID set:

### NMAE% — Normalised Mean Absolute Error
```
NMAE% = 100 × mean(|pred − GT|) / mean(|GT|)
```
- Denominator is mean signal magnitude (`use_magnitude=True` in img_reconstruction).
- Normalises out absolute brightness — a low-contrast scene (scale=0.05) and a
  high-contrast scene (scale=0.20) give comparable NMAE for the same relative error.
- Computed in both **physical DN space** and **normalized space**.

### PWT@1% — Percentage Within Tolerance
```
PWT@1% = 100 × fraction of pixels where |pred − GT| ≤ 0.01 × (GT_max − GT_min)
```
- Threshold = 1% of per-image GT dynamic range (matches `PWT_THRESH=0.01`).
- Intuitive: "what fraction of pixels are within 1% of the image's contrast range".
- Computed in both physical and normalized space.
- Higher = better.

### Bias% — Mean Signed Error
```
Bias% = 100 × mean(pred − GT) / mean(|GT|)
```
- Signed: positive → systematic over-prediction, negative → under-prediction.
- Same denominator as NMAE.
- Computed in both physical and normalized space.

All three metrics are computed as batch tensor operations in `metrics_ardan.py`,
returning `[B]` per-image values. These are then averaged across the full eval set.

---

## Existing Metrics (Unchanged)

Carried over from `eval.py` with identical formulas:

| Metric | Space | Notes |
|--------|-------|-------|
| MSE | Physical + Normalized | Scene-dependent (phys) / comparable (norm) |
| MAE | Physical | Robustness diagnostic |
| PSNR | Physical (MAX=1.0) + Normalized (MAX=20) | Pooled MSE — avoids Jensen's inflation |
| SSIM | Physical | GT-normalized to [0,1] per image via `ssim_safe()` |
| Pearson r | Normalized | Invariant to affine transforms |
| FID | Normalized | Optional; disabled by default (`--no_fid`) |

---

## Evaluation Validated Against img_reconstruction

**Matched:**
- NMAE formula: confirmed against `run_evals.py` line 90
- PWT formula: confirmed against `run_evals.py` line 89 (threshold 0.01, dynamic range denominator)
- Bias formula: confirmed against `run_evals.py` line 91
- SSIM: identical `ssim_safe()` function — normalise GT to [0,1], apply same
  transform to pred, call `sk_ssim(..., data_range=1.0)`

**Sanity checks passed (see test run above):**
- Perfect prediction (pred=GT): MSE=0, NMAE=0, PWT=100%, Bias=0, SSIM=1.0
- Noisy prediction (σ=0.01): NMAE~5.7%, PWT~2%, Bias~0%, PSNR≈40 dB

---

## What is Ignored / Not Reproduced

The following aspects of `HiRISE_img_reconstruction` evaluation are **not
reproduced** because they are either not applicable to a diffusion model or
require infrastructure that doesn't exist in this project:

### 1. Baseline / SYN4 comparison
The img_reconstruction project compares the CNN against a synthetic linear
baseline (SYN4). **No equivalent baseline exists** for the diffusion model.
All metrics are reported as absolute values only. Cross-model comparison is
done offline (e.g. in `interpolate.ipynb` or by importing both CSVs).

### 2. Win rate
Win rate (`% images where CNN_NMAE < SYN4_NMAE`) requires a baseline — not applicable.

### 3. LF/HF error decomposition
The img_reconstruction project decomposes residuals into low-frequency (DC/cast bias)
and high-frequency (texture/edges) components using Gaussian blur kernels.
This decomposition is specific to the CNN's `BrightnessHeadLocal` architecture.
The diffusion model does not have an analogous component, so LF/HF analysis would
be ambiguous. **Skipped.**

### 4. DC diagnostics
Layer-by-layer DC decomposition (`out_centered + m_hat + lf_bias + global_bias`)
is specific to the CNN architecture. **Skipped.**

### 5. Loss function decomposition
The img_reconstruction loss has multiple weighted components (MAE, GDL, local_mae,
mean_mae, lf_l1). The CM-Diff model uses only noise-prediction MSE. Reporting
component-wise loss breakdown is not meaningful. **Skipped.**

### 6. Theme categories
The img_reconstruction `evals.yaml` contains 18 scientific theme categories
(volcanic, eolian, fluvial, etc.) built from manually curated observation lists
that require Mars science theme annotations. These annotations are not in
`data_record_bin12.csv`. **Deferred** — can be added manually if needed.

### 7. Spatial loss maps
Hexagonal binning of per-observation errors on a Mars map (using lat/lon).
Useful visualisation but requires Cartopy/matplotlib infrastructure.
**Skipped** in this eval script; can be added to `interpolate.ipynb` separately.

### 8. ECDF plots
Empirical CDF of error distributions across the dataset.
**Skipped** in the script; trivial to add in a notebook from the exported CSV.

### 9. Gradient diagnostics during evaluation
Not applicable to diffusion models. **Skipped.**

---

## Per-Category Evaluation

14 categories are auto-generated from `data_record_bin12.csv` metadata by
`src/configs/gen_eval_categories_ardan.py` → `src/configs/evals_ardan.yaml`.

| Group | Categories |
|-------|-----------|
| Geometry | nadir, off_nadir |
| Illumination | low_sun, high_sun |
| Location | polar (|lat|>60°), equatorial (|lat|<5°) |
| Seasonal | summer_north/south, winter_north/south |
| Contrast | high_contrast, low_contrast |
| Brightness | high_brightness, low_brightness |

20 observations per category, from the validation split only.

**Cost warning**: Each category requires ~20 × 1000-step DDPM × 2 directions =
~40 full sampling runs. Across 14 categories this is ~560 passes. Reserve
`--time=08:00:00` or more in the SLURM job when using `--category_eval`.

---

## CSV Output

Two CSVs are saved to `--save_dir` (default: `src/output/evals/`):

`eval_per_image_lscl{X}_lccl{Y}.csv` — one row per image per direction:
```
obs_id, set_name, date, direction, mse_phys, mae_phys, nmae_phys, pwt_phys,
bias_phys, ssim, mse_norm, nmae_norm, pwt_norm, bias_norm, pearson
```

`eval_categories_lscl{X}_lccl{Y}.csv` — same columns plus `category` tag
(only written when `--category_eval` is used).

---

## Files

| File | Purpose |
|------|---------|
| `src/eval_ardan.py` | Main evaluation script — load model, run sampling, compute metrics, report, export CSV |
| `src/metrics_ardan.py` | Metric helpers: NMAE, PWT, Bias, SSIM, Pearson, PSNR |
| `src/configs/gen_eval_categories_ardan.py` | One-time script: generate `evals_ardan.yaml` from bin12 metadata |
| `src/configs/evals_ardan.yaml` | Generated evaluation categories (20 obs each) |
| `scripts/eval_ardan.sh` | SLURM submission: λ sweep + optional category eval |

No existing files are modified.

---

## Usage

```bash
# 1. Generate categories (once)
cd /scratch_root/ed425/HiRISE_diffusion
python src/configs/gen_eval_categories_ardan.py \
    --csv_path data/files/data_record_bin12.csv \
    --output   src/configs/evals_ardan.yaml

# 2. Global evaluation
python src/eval_ardan.py \
    --checkpoint src/output/latest.pt \
    --lambda_scl 20.0 --lambda_ccl 20.0 \
    --max_samples 100 --no_fid

# 3. With per-category evaluation
python src/eval_ardan.py \
    --checkpoint src/output/latest.pt \
    --lambda_scl 20.0 --lambda_ccl 20.0 \
    --category_eval \
    --evals_config src/configs/evals_ardan.yaml

# 4. Submit full sweep to SLURM
sbatch scripts/eval_ardan.sh
```
