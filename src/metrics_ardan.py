"""
Per-image evaluation metrics for HiRISE diffusion — Ardan evaluation variant.

All batch functions take [B, 1, H, W] tensors and return [B] tensors (one
value per image). Aggregation (mean / median) is left to the caller so that
per-image CSVs can be produced before summarising.

Matches the reference formulas in:
  HiRISE_img_reconstruction/src/post/run_evals.py  compute_per_img_metrics()
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim


# =============================================================================
# New metrics matching HiRISE_img_reconstruction
# =============================================================================

def nmae_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    NMAE% = 100 × mean(|pred − target|) / mean(|target|)  per image.

    Denominator is the mean signal magnitude (same as img_reconstruction
    use_magnitude=True default). This normalises out absolute brightness
    differences so that low-contrast and high-contrast scenes are comparable.

    Args:
        pred   : [B, 1, H, W]
        target : [B, 1, H, W]

    Returns:
        [B] tensor of NMAE values in percent (higher = worse)
    """
    B = pred.shape[0]
    abs_err = (pred - target).abs().view(B, -1)
    mag = target.abs().view(B, -1).mean(dim=1).clamp(min=1e-8)
    return 100.0 * abs_err.mean(dim=1) / mag


def pwt_batch(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.01) -> torch.Tensor:
    """
    PWT% = 100 × fraction of pixels within `threshold` of GT dynamic range.

    Default threshold=0.01 → "within 1% of GT range" (matches img_reconstruction
    PWT_THRESH=0.01). Dynamic range = max(target) − min(target) per image.

    Args:
        pred      : [B, 1, H, W]
        target    : [B, 1, H, W]
        threshold : fraction of per-image dynamic range (0.01 = 1%)

    Returns:
        [B] tensor of PWT values in percent (higher = better)
    """
    B = pred.shape[0]
    flat_t = target.view(B, -1)
    dy_range = (flat_t.max(dim=1).values - flat_t.min(dim=1).values).clamp(min=1e-8)
    abs_err = (pred - target).abs().view(B, -1)
    within = (abs_err <= threshold * dy_range.unsqueeze(1)).float()
    return 100.0 * within.mean(dim=1)


def bias_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Bias% = 100 × mean(pred − target) / mean(|target|)  per image (signed).

    Positive → systematic over-prediction.
    Negative → systematic under-prediction.
    Same denominator as NMAE (mean signal magnitude).

    Args:
        pred   : [B, 1, H, W]
        target : [B, 1, H, W]

    Returns:
        [B] tensor of signed bias values in percent
    """
    B = pred.shape[0]
    err = (pred - target).view(B, -1)
    mag = target.abs().view(B, -1).mean(dim=1).clamp(min=1e-8)
    return 100.0 * err.mean(dim=1) / mag


# =============================================================================
# Structural metrics (already in eval.py; mirrored here for standalone use)
# =============================================================================

def ssim_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    SSIM for 2-D numpy arrays.

    Normalises GT to [0, 1] using per-image min/max, applies the same linear
    transform to pred, then calls skimage SSIM with data_range=1.0.

    This puts the data at SSIM's design working point where K1=0.01, K2=0.03
    with data_range=1.0 are correctly calibrated relative to local variance.
    Avoids two failure modes:
      - constant masking (data_range >> actual dynamic range)
      - numerical divergence (data_range ≈ 0 → C1/C2 ≈ 0)

    Args:
        y_true, y_pred : 2-D float arrays (H, W)

    Returns:
        SSIM scalar in [-1, 1]; 1 = perfect.
    """
    tmin, tmax = np.min(y_true), np.max(y_true)
    rng = max(float(tmax - tmin), 1e-8)
    yt = (y_true - tmin) / rng
    yp = (y_pred - tmin) / rng
    return float(sk_ssim(yt, yp, data_range=1.0))


def pearson_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-image Pearson correlation for [B, 1, H, W] tensors.

    r = Σ(x−x̄)(y−ȳ) / [‖x−x̄‖ · ‖y−ȳ‖]

    Invariant to per-image affine transforms (global scale/offset).

    Returns:
        [B] tensor in [-1, 1]
    """
    B = pred.shape[0]
    p = pred.view(B, -1)
    t = target.view(B, -1)
    p_c = p - p.mean(dim=1, keepdim=True)
    t_c = t - t.mean(dim=1, keepdim=True)
    r = (p_c * t_c).sum(dim=1) / (p_c.norm(dim=1) * t_c.norm(dim=1) + 1e-8)
    return r


def psnr_pooled(mse_list: list, max_val: float) -> float:
    """
    Single global PSNR from the pooled (averaged) MSE.

    PSNR = 10·log10(MAX² / mean(MSE))

    Using the arithmetic mean of per-image PSNRs is biased upward by Jensen's
    inequality (log is concave).  Pooled MSE is the correct, conservative
    estimator.

    Args:
        mse_list : list of per-image MSE values
        max_val  : signal peak for PSNR formula
                   (1.0 for physical space, 20.0 for normalized space)

    Returns:
        PSNR in dB (float)
    """
    avg_mse = float(np.mean(mse_list))
    if avg_mse <= 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / avg_mse)
