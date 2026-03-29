"""
Evaluate a CM-Diff checkpoint on the validation set.

Two evaluation modes:
  - Noise-prediction MSE: fast, used by train.py during training (evaluate())
  - Image-level full metrics: runs full 1000-step sampling, computes
    MSE / MAE / PSNR / SSIM / Pearson-r and FID (evaluate_images())

Usage (standalone — image-level):
    cd <project_root>
    python src/eval.py \
        --checkpoint src/output/latest.pt \
        --data_root  /path/to/data \
        --csv_path   /path/to/data_record_bin12.csv
"""

import os
import sys
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy import linalg as sp_linalg

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ModelConfig, TrainConfig, DataConfig, InferenceConfig
from models.cm_diff_unet import UNet
from diffusion.scheduler import DDPMScheduler
from diffusion.process import q_sample, sobel_edge
from compute_prior import load_prior_stats
from inference import sample
from data.dataset import DiffusionDataset, diffusion_collate_fn, get_loader


# =============================================================================
# Noise-prediction evaluation (used by train.py)
# =============================================================================

def eval_batch(
    batch:     dict,
    model:     torch.nn.Module,
    scheduler: DDPMScheduler,
    device:    torch.device,
) -> tuple:
    """
    Compute noise-prediction MSE for one batch, split by direction.

    Returns (loss_ir2red, loss_red2ir) as Python floats.
    """
    ir  = batch["ir"].to(device)
    red = batch["red"].to(device)
    B   = ir.shape[0]

    half      = B // 2
    direction = torch.cat([
        torch.zeros(half,     dtype=torch.long, device=device),
        torch.ones(B - half,  dtype=torch.long, device=device),
    ])

    mask     = (direction == 0)[:, None, None, None]
    x_target = torch.where(mask, red, ir)
    x_source = torch.where(mask, ir,  red)

    edge = sobel_edge(x_source)
    t    = torch.randint(0, scheduler.timesteps, (B,), device=device)

    noise    = torch.randn_like(x_target)
    x_t      = q_sample(scheduler, x_target, t, noise)
    eps_pred = model(x_t, x_source, edge, direction, t)

    loss_per_sample = F.mse_loss(eps_pred, noise, reduction="none").mean(dim=[1, 2, 3])

    mask_1d     = (direction == 0)
    loss_ir2red = loss_per_sample[mask_1d].mean().item()  if mask_1d.any()    else 0.0
    loss_red2ir = loss_per_sample[~mask_1d].mean().item() if (~mask_1d).any() else 0.0

    return loss_ir2red, loss_red2ir


def evaluate(
    model:      torch.nn.Module,
    scheduler:  DDPMScheduler,
    val_loader: torch.utils.data.DataLoader,
    device:     torch.device,
) -> dict:
    """
    Noise-prediction evaluation over the full validation loader.
    Fast — used by train.py during training.

    Returns dict with keys: loss, loss_ir2red, loss_red2ir.
    """
    model.eval()
    ir2red_accum = 0.0
    red2ir_accum = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            v_ir2red, v_red2ir = eval_batch(batch, model, scheduler, device)
            ir2red_accum += v_ir2red
            red2ir_accum += v_red2ir
            n += 1

    avg_ir2red = ir2red_accum / max(n, 1)
    avg_red2ir = red2ir_accum / max(n, 1)
    avg_loss   = avg_ir2red + avg_red2ir

    return dict(
        loss=avg_loss,
        loss_ir2red=avg_ir2red,
        loss_red2ir=avg_red2ir,
    )


# =============================================================================
# Per-image metric helpers
# =============================================================================

def _ssim_batch(pred: torch.Tensor, target: torch.Tensor,
                data_range: float = 2.0, window_size: int = 11) -> torch.Tensor:
    """
    Per-image SSIM over a batch [B, 1, H, W] in [-1, 1].
    Approximates Gaussian-windowed SSIM with a uniform sliding window.
    Returns [B] tensor.

    SSIM(x, y) = [l(x,y)]·[c(x,y)]·[s(x,y)]
      l = (2μ_x μ_y + C1) / (μ_x² + μ_y² + C1)       luminance
      c = (2σ_x σ_y + C2) / (σ_x² + σ_y² + C2)       contrast
      s = (σ_xy + C2/2)   / (σ_x σ_y + C2/2)          structure
    Combined: SSIM = [(2μ_xμ_y+C1)(2σ_xy+C2)] / [(μ_x²+μ_y²+C1)(σ_x²+σ_y²+C2)]
    """
    C1  = (0.01 * data_range) ** 2
    C2  = (0.03 * data_range) ** 2
    pad = window_size // 2

    mu1 = F.avg_pool2d(pred,   window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=pad)

    sigma1_sq = F.avg_pool2d(pred    ** 2,    window_size, stride=1, padding=pad) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(target  ** 2,    window_size, stride=1, padding=pad) - mu2 ** 2
    sigma12   = F.avg_pool2d(pred * target,   window_size, stride=1, padding=pad) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    return ssim_map.mean(dim=[1, 2, 3])   # [B]


def _pearson_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-image Pearson correlation coefficient over a batch [B, 1, H, W].
    Returns [B] tensor in [-1, 1].

    r = Σ(x_i - x̄)(y_i - ȳ) / [||x - x̄|| · ||y - ȳ||]
    Invariant to per-image affine transforms (scale/offset).
    """
    B   = pred.shape[0]
    p   = pred.view(B, -1)
    t   = target.view(B, -1)
    p_c = p - p.mean(dim=1, keepdim=True)
    t_c = t - t.mean(dim=1, keepdim=True)
    r   = (p_c * t_c).sum(dim=1) / (p_c.norm(dim=1) * t_c.norm(dim=1) + 1e-8)
    return r   # [B]


def _psnr_from_mse(mse: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """
    PSNR in dB from per-image MSE tensor.
    PSNR = 10 · log10(MAX² / MSE),  where MAX = data_range (=2 for [-1,1]).
    Returns [B] tensor.
    """
    return 10.0 * torch.log10(torch.tensor(data_range ** 2) / (mse + 1e-10))


# =============================================================================
# FID helpers
# =============================================================================

def _build_inception(device: torch.device) -> torch.nn.Module:
    """
    Load ImageNet-pretrained InceptionV3 as a 2048-dim feature extractor.
    The FC layer is replaced with Identity so forward() returns pool3 features.
    """
    from torchvision import models
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception.fc = torch.nn.Identity()   # avgpool (2048-dim) → output
    inception.eval()
    return inception.to(device)


@torch.no_grad()
def _inception_features(images: torch.Tensor,
                         inception: torch.nn.Module) -> np.ndarray:
    """
    Extract InceptionV3 pool3 features from single-channel images.
    images : [B, 1, H, W] in [-1, 1]
    Returns numpy [B, 2048].

    Preprocessing:
      1. Tile 1→3 channels (grayscale repeated across RGB)
      2. Resize to 299×299 (InceptionV3 input resolution)
      3. Rescale [-1,1] → [0,1] (InceptionV3 transform_input handles normalization)
    """
    imgs = images.repeat(1, 3, 1, 1)
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    imgs = ((imgs + 1.0) / 2.0).clamp(0.0, 1.0)
    return inception(imgs).cpu().numpy()   # [B, 2048]


def _fid_from_features(real_feats: np.ndarray, fake_feats: np.ndarray,
                        eps: float = 1e-6) -> float:
    """
    Fréchet Inception Distance from two feature arrays [N, D].

    FID = ||μ_r - μ_g||₂² + Tr(Σ_r + Σ_g - 2·sqrt(Σ_r · Σ_g))

    where (μ_r, Σ_r) and (μ_g, Σ_g) are the mean and covariance of
    real and generated feature distributions.
    """
    mu_r    = real_feats.mean(axis=0)
    mu_g    = fake_feats.mean(axis=0)
    sigma_r = np.cov(real_feats, rowvar=False) + np.eye(real_feats.shape[1]) * eps
    sigma_g = np.cov(fake_feats, rowvar=False) + np.eye(fake_feats.shape[1]) * eps

    diff_sq  = np.sum((mu_r - mu_g) ** 2)
    sqrt_cov, _ = sp_linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real

    fid = diff_sq + np.trace(sigma_r + sigma_g - 2.0 * sqrt_cov)
    return float(np.real(fid))


# =============================================================================
# Image-level evaluation (full sampling)
# =============================================================================

def evaluate_images(
    model:          torch.nn.Module,
    scheduler:      DDPMScheduler,
    val_dataset:    DiffusionDataset,
    prior_red:      dict,
    prior_ir:       dict,
    cfg_inf:        InferenceConfig,
    device:         torch.device,
    max_samples:    int = 0,
    batch_size:     int = 4,
    seed:           int = 42,
    compute_fid:    bool = True,
) -> dict:
    """
    Image-level evaluation: full T-step reverse diffusion in batches.

    Metrics computed per direction (IR→RED and RED→IR):
      MSE      — mean squared error (pixel accuracy)
      MAE      — mean absolute error (robust pixel accuracy)
      PSNR     — peak signal-to-noise ratio in dB
      SSIM     — structural similarity index (luminance + contrast + structure)
      Pearson r — linear correlation (scale/offset invariant)
      FID      — Fréchet Inception Distance (distributional quality, optional)

    Data range is 2.0 (images normalised to [-1, 1]).
    """
    DATA_RANGE = 2.0

    model.eval()
    n = len(val_dataset)
    if max_samples > 0:
        n = min(n, max_samples)

    dataset = (torch.utils.data.Subset(val_dataset, list(range(n)))
               if n < len(val_dataset) else val_dataset)
    loader  = get_loader(dataset, batch_size=batch_size, collate_fn=diffusion_collate_fn,
                         num_workers=0, shuffle=False)

    # Per-image metric accumulators
    mse_0, mae_0, psnr_0, ssim_0, pearson_0 = [], [], [], [], []
    mse_1, mae_1, psnr_1, ssim_1, pearson_1 = [], [], [], [], []

    # FID feature accumulators
    if compute_fid:
        inception = _build_inception(device)
        real_f0, fake_f0 = [], []   # IR→RED: real=RED GT, fake=pred RED
        real_f1, fake_f1 = [], []   # RED→IR: real=IR  GT, fake=pred IR

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    n_done = 0
    for batch in loader:
        ir_batch  = batch["ir"].to(device)    # [B, 1, H, W]
        red_batch = batch["red"].to(device)   # [B, 1, H, W]

        with torch.no_grad():
            pred_red = sample(model, scheduler, ir_batch,  direction=0,
                              prior_stats=prior_red, cfg_inf=cfg_inf, device=device,
                              verbose=False)
            pred_ir  = sample(model, scheduler, red_batch, direction=1,
                              prior_stats=prior_ir,  cfg_inf=cfg_inf, device=device,
                              verbose=False)

        # ── pixel-level metrics ──────────────────────────────────────────────
        mse0_b = F.mse_loss(pred_red, red_batch, reduction="none").mean(dim=[1,2,3])
        mse1_b = F.mse_loss(pred_ir,  ir_batch,  reduction="none").mean(dim=[1,2,3])

        mae0_b = F.l1_loss(pred_red, red_batch, reduction="none").mean(dim=[1,2,3])
        mae1_b = F.l1_loss(pred_ir,  ir_batch,  reduction="none").mean(dim=[1,2,3])

        psnr0_b = _psnr_from_mse(mse0_b, DATA_RANGE)
        psnr1_b = _psnr_from_mse(mse1_b, DATA_RANGE)

        # ── structural metrics ───────────────────────────────────────────────
        with torch.no_grad():
            ssim0_b    = _ssim_batch(pred_red, red_batch, DATA_RANGE)
            ssim1_b    = _ssim_batch(pred_ir,  ir_batch,  DATA_RANGE)
            pearson0_b = _pearson_batch(pred_red, red_batch)
            pearson1_b = _pearson_batch(pred_ir,  ir_batch)

        mse_0.extend(mse0_b.cpu().tolist());    mse_1.extend(mse1_b.cpu().tolist())
        mae_0.extend(mae0_b.cpu().tolist());    mae_1.extend(mae1_b.cpu().tolist())
        psnr_0.extend(psnr0_b.cpu().tolist());  psnr_1.extend(psnr1_b.cpu().tolist())
        ssim_0.extend(ssim0_b.cpu().tolist());  ssim_1.extend(ssim1_b.cpu().tolist())
        pearson_0.extend(pearson0_b.cpu().tolist()); pearson_1.extend(pearson1_b.cpu().tolist())

        # ── FID features ────────────────────────────────────────────────────
        if compute_fid:
            real_f0.append(_inception_features(red_batch, inception))
            fake_f0.append(_inception_features(pred_red,  inception))
            real_f1.append(_inception_features(ir_batch,  inception))
            fake_f1.append(_inception_features(pred_ir,   inception))

        n_done += ir_batch.shape[0]
        print(f"  [{n_done}/{n}]  "
              f"MSE(IR→RED)={mse0_b.mean():.4f}  SSIM={ssim0_b.mean():.4f}  | "
              f"MSE(RED→IR)={mse1_b.mean():.4f}  SSIM={ssim1_b.mean():.4f}")

    def _avg(lst): return float(np.mean(lst))

    results = dict(
        n_samples   = n_done,
        # IR → RED
        mse_ir2red  = _avg(mse_0),
        mae_ir2red  = _avg(mae_0),
        psnr_ir2red = _avg(psnr_0),
        ssim_ir2red = _avg(ssim_0),
        pearson_ir2red = _avg(pearson_0),
        # RED → IR
        mse_red2ir  = _avg(mse_1),
        mae_red2ir  = _avg(mae_1),
        psnr_red2ir = _avg(psnr_1),
        ssim_red2ir = _avg(ssim_1),
        pearson_red2ir = _avg(pearson_1),
        # combined (average of both directions)
        mse_combined  = (_avg(mse_0)  + _avg(mse_1))  / 2,
        psnr_combined = (_avg(psnr_0) + _avg(psnr_1)) / 2,
        ssim_combined = (_avg(ssim_0) + _avg(ssim_1)) / 2,
    )

    if compute_fid:
        real_0 = np.concatenate(real_f0, axis=0)
        fake_0 = np.concatenate(fake_f0, axis=0)
        real_1 = np.concatenate(real_f1, axis=0)
        fake_1 = np.concatenate(fake_f1, axis=0)
        results["fid_ir2red"] = _fid_from_features(real_0, fake_0)
        results["fid_red2ir"] = _fid_from_features(real_1, fake_1)
        print(f"  FID(IR→RED)={results['fid_ir2red']:.2f}  FID(RED→IR)={results['fid_red2ir']:.2f}")

    return results


# =============================================================================
# Dataset split (same logic as train.py to guarantee identical val set)
# =============================================================================

def get_val_split(dr: pd.DataFrame):
    """
    80/20 train/val split at Observation level with fixed seed=42.
    Returns (train_sets, val_sets) as sets of Set IDs.
    """
    all_obs = sorted(dr["Observation"].unique())
    rng     = torch.Generator().manual_seed(42)
    perm    = torch.randperm(len(all_obs), generator=rng).tolist()
    n_train = int(len(all_obs) * 0.8)
    train_obs = set(all_obs[i] for i in perm[:n_train])
    val_obs   = set(all_obs[i] for i in perm[n_train:])

    train_sets = set(dr[dr["Observation"].isin(train_obs)]["Set"].unique())
    val_sets   = set(dr[dr["Observation"].isin(val_obs)]["Set"].unique())
    return train_sets, val_sets


# =============================================================================
# CLI (image-level evaluation)
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CM-Diff — full image metrics")
    parser.add_argument("--checkpoint",  default="src/output/latest.pt")
    parser.add_argument("--prior_dir",   default="src/output",
                        help="Directory containing prior_ir.pt and prior_red.pt")
    parser.add_argument("--data_root",   default="")
    parser.add_argument("--csv_path",    default="")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Max samples (0 = all)")
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--lambda_scl",  type=float, default=0.0)
    parser.add_argument("--lambda_ccl",  type=float, default=0.0)
    parser.add_argument("--no_fid",      action="store_true",
                        help="Skip FID computation (faster, no torchvision needed)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    cfg_model = ModelConfig()
    cfg_data  = DataConfig()
    cfg_inf   = InferenceConfig(lambda_scl=args.lambda_scl, lambda_ccl=args.lambda_ccl)
    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = args.data_root or cfg_data.data_root or os.path.join(project_root, "data")
    csv_path  = args.csv_path  or cfg_data.csv_path  or os.path.join(project_root, "data", "files", "data_record_bin12.csv")

    print(f"Device       : {device}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"SCI          : lambda_scl={args.lambda_scl}  lambda_ccl={args.lambda_ccl}")
    print(f"Max samples  : {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"FID          : {'disabled' if args.no_fid else 'enabled'}")
    print()

    model = UNet(
        in_channels=cfg_model.in_channels,
        out_channels=cfg_model.out_channels,
        base_channels=cfg_model.base_channels,
        num_res_blocks=cfg_model.num_res_blocks,
        dropout=cfg_model.dropout,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: step={ckpt.get('step','?')}  loss={ckpt.get('loss','?'):.4f}")

    scheduler = DDPMScheduler(
        timesteps=cfg_model.timesteps,
        beta_start=cfg_model.beta_start,
        beta_end=cfg_model.beta_end,
    ).to(device)

    prior_red = load_prior_stats(os.path.join(args.prior_dir, "prior_red.pt"), device)
    prior_ir  = load_prior_stats(os.path.join(args.prior_dir, "prior_ir.pt"),  device)
    print(f"Prior RED: mu={prior_red['mu'].item():.4f}  sigma={prior_red['sigma'].item():.4f}")
    print(f"Prior IR:  mu={prior_ir['mu'].item():.4f}  sigma={prior_ir['sigma'].item():.4f}")

    dr = pd.read_csv(csv_path)
    _, val_sets = get_val_split(dr)
    val_dataset = DiffusionDataset(
        data_record=dr, data_root=data_root, sweep=True, allowed_sets=val_sets,
    )
    print(f"Val set: {len(val_dataset)} sets\n")

    results = evaluate_images(
        model, scheduler, val_dataset,
        prior_red=prior_red, prior_ir=prior_ir,
        cfg_inf=cfg_inf, device=device,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        compute_fid=not args.no_fid,
    )

    W = 14
    print(f"\n{'='*52}")
    print(f"{'Metric':<{W}}  {'IR→RED':>10}  {'RED→IR':>10}  {'Combined':>10}")
    print(f"{'-'*52}")
    print(f"{'MSE':<{W}}  {results['mse_ir2red']:>10.4f}  {results['mse_red2ir']:>10.4f}  {results['mse_combined']:>10.4f}")
    print(f"{'MAE':<{W}}  {results['mae_ir2red']:>10.4f}  {results['mae_red2ir']:>10.4f}  {'—':>10}")
    print(f"{'PSNR (dB)':<{W}}  {results['psnr_ir2red']:>10.2f}  {results['psnr_red2ir']:>10.2f}  {results['psnr_combined']:>10.2f}")
    print(f"{'SSIM':<{W}}  {results['ssim_ir2red']:>10.4f}  {results['ssim_red2ir']:>10.4f}  {results['ssim_combined']:>10.4f}")
    print(f"{'Pearson r':<{W}}  {results['pearson_ir2red']:>10.4f}  {results['pearson_red2ir']:>10.4f}  {'—':>10}")
    if "fid_ir2red" in results:
        print(f"{'FID':<{W}}  {results['fid_ir2red']:>10.2f}  {results['fid_red2ir']:>10.2f}  {'—':>10}")
    print(f"{'='*52}")
    print(f"(n={results['n_samples']}  λ_scl={args.lambda_scl}  λ_ccl={args.lambda_ccl})")


if __name__ == "__main__":
    main()
