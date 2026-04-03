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

from config import ModelConfig, DataConfig, InferenceConfig
from models.cm_diff_unet import UNet
from models.ir2red_ddpm import UNet as IR2REDUNet
from models.red2ir_ddpm import UNet as RED2IRUNet
from diffusion.scheduler import DDPMScheduler
from diffusion.process import q_sample
from diffusion.edge import compute_edge, load_dexined
from compute_prior import load_prior_stats
from inference import sample
from data.dataset import DiffusionDataset, diffusion_collate_fn, get_loader
from skimage.metrics import structural_similarity as sk_ssim

# =============================================================================
# Noise-prediction evaluation (used by train.py)
# =============================================================================

def eval_batch(
    batch:     dict,
    model:     torch.nn.Module,
    scheduler: DDPMScheduler,
    device:    torch.device,
    edge_mode: str = "sobel",
    dexined_model: torch.nn.Module = None,
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

    edge = compute_edge(x_source, edge_mode, dexined_model)
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
    edge_mode:  str = "sobel",
    dexined_model: torch.nn.Module = None,
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
            v_ir2red, v_red2ir = eval_batch(batch, model, scheduler, device,
                                            edge_mode, dexined_model)
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


def evaluate_unidirectional(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    train_mode: str,
    edge_mode: str = "sobel",
    dexined_model: torch.nn.Module = None,
) -> dict:
    """Noise-prediction evaluation for single-direction models (ir2red/red2ir)."""
    model.eval()
    loss_accum = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            ir = batch["ir"].to(device)
            red = batch["red"].to(device)
            B = ir.shape[0]

            if train_mode == "ir2red":
                x_source, x_target = ir, red
            elif train_mode == "red2ir":
                x_source, x_target = red, ir
            else:
                raise ValueError(f"Unsupported mode for evaluate_unidirectional: {train_mode}")

            edge = compute_edge(x_source, edge_mode, dexined_model)
            t = torch.randint(0, scheduler.timesteps, (B,), device=device)
            noise = torch.randn_like(x_target)
            x_t = q_sample(scheduler, x_target, t, noise)
            eps_pred = model(x_t, x_source, edge, t)
            loss = F.mse_loss(eps_pred, noise)

            loss_accum += loss.item()
            n += 1

    avg = loss_accum / max(n, 1)
    if train_mode == "ir2red":
        avg_ir2red, avg_red2ir = avg, 0.0
    else:
        avg_ir2red, avg_red2ir = 0.0, avg

    avg_loss = avg_ir2red + avg_red2ir
    return dict(
        loss=avg_loss,
        loss_ir2red=avg_ir2red,
        loss_red2ir=avg_red2ir,
    )


# =============================================================================
# Per-image metric helpers
# =============================================================================

def _ssim_single(pred: torch.Tensor, target: torch.Tensor,
                  data_range: float, window_size: int = 11) -> float:
    """
    SSIM for a single image pair [1, 1, H, W] with explicit data_range.

    Uses per-image GT dynamic range so that the stabilisation constants
    C1/C2 scale with the actual signal amplitude.  Numerical safeguards:
      - variance clamp(min=0) against E[x²]-E[x]² float rounding
      - +1e-8 in denominator against zero-variance flat patches
    """
    C1  = (0.01 * data_range) ** 2
    C2  = (0.03 * data_range) ** 2
    pad = window_size // 2

    mu1 = F.avg_pool2d(pred,   window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=pad)

    sigma1_sq = (F.avg_pool2d(pred   ** 2, window_size, stride=1, padding=pad) - mu1 ** 2).clamp(min=0)
    sigma2_sq = (F.avg_pool2d(target ** 2, window_size, stride=1, padding=pad) - mu2 ** 2).clamp(min=0)
    sigma12   =  F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu1 * mu2

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
    return (num / den).mean().item()


def _ssim_safe(y_true, y_pred, data_range: float):
    """Compute SSIM for 2D arrays with explicit data_range."""
    dr = max(float(data_range), 1e-8)
    return float(sk_ssim(y_true, y_pred, data_range=dr))

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


def _psnr_from_mse(mse: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
    """
    PSNR in dB from per-image MSE tensor.
    PSNR = 10 · log10(MAX² / MSE),  where MAX = data_range (=2 for [-1,1]).
    Returns [B] tensor.
    """
    return 10.0 * torch.log10(data_range ** 2 / (mse + 1e-10))


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
    images : [B, 1, H, W] in [-10, 10] (normalized space)
    Returns numpy [B, 2048].

    Preprocessing:
      1. Tile 1→3 channels (grayscale repeated across RGB)
      2. Resize to 299×299 (InceptionV3 input resolution)
      3. Rescale [-10,10] → [0,1] (InceptionV3 expects [0,1] input)
    """
    imgs = images.repeat(1, 3, 1, 1)
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    imgs = ((imgs + 10.0) / 20.0).clamp(0.0, 1.0)
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

def _eval_one_direction(
    model, scheduler, src, tgt, direction, prior,
    center, scale, dc,
    cfg_inf, device, edge_mode, dexined_model,
    compute_fid, inception, label,
):
    """
    Full reverse diffusion for one (src→tgt) direction.

    Args:
        direction: int (0 or 1) for bidirectional, None for unidirectional.
    Returns per-image metric lists and a progress string.
    """
    B = src.shape[0]

    with torch.no_grad():
        pred = sample(model, scheduler, src, direction=direction,
                      prior_stats=prior, cfg_inf=cfg_inf, device=device,
                      verbose=False, edge_mode=edge_mode,
                      dexined_model=dexined_model)

    pred_norm = pred.clamp(-10.0, 10.0)
    pred_phys = (pred_norm + dc) * scale + center
    tgt_phys  = (tgt       + dc) * scale + center

    mse_phys_b = F.mse_loss(pred_phys, tgt_phys, reduction="none").mean(dim=[1,2,3])
    mae_phys_b = F.l1_loss( pred_phys, tgt_phys, reduction="none").mean(dim=[1,2,3])
    mse_norm_b = F.mse_loss(pred_norm, tgt,       reduction="none").mean(dim=[1,2,3])

    ssim_phy_vals, ssim_norm_vals = [], []
    for i in range(B):
        tgt_phy_np  = tgt_phys[i].squeeze().cpu().numpy()
        pred_phy_np = pred_phys[i].squeeze().cpu().numpy()
        dr_phy = max(float(np.max(tgt_phy_np) - np.min(tgt_phy_np)), 1e-2)
        ssim_phy_vals.append(_ssim_safe(tgt_phy_np, pred_phy_np, data_range=dr_phy))
        ssim_norm_vals.append(_ssim_safe(
            tgt[i].squeeze().cpu().numpy(),
            pred_norm[i].squeeze().cpu().numpy(),
            data_range=20.0))

    with torch.no_grad():
        pearson_b = _pearson_batch(pred_norm, tgt)

    return dict(
        mse_phys  = mse_phys_b.cpu().tolist(),
        mae_phys  = mae_phys_b.cpu().tolist(),
        mse_norm  = mse_norm_b.cpu().tolist(),
        ssim_phys = ssim_phy_vals,
        ssim_norm = ssim_norm_vals,
        pearson   = pearson_b.cpu().tolist(),
        fid_real  = _inception_features(tgt,       inception) if compute_fid else None,
        fid_fake  = _inception_features(pred_norm, inception) if compute_fid else None,
        progress  = (f"MSE({label})={mse_phys_b.mean():.4f}/{mse_norm_b.mean():.4f}  "
                     f"SSIM phy={np.mean(ssim_phy_vals):.4f} norm={np.mean(ssim_norm_vals):.4f}"),
    )


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
    edge_mode:      str = "sobel",
    dexined_model:  torch.nn.Module = None,
    train_mode:     str = "bidirectional",
) -> dict:
    """
    Image-level evaluation: full T-step reverse diffusion in batches.

    train_mode controls which directions are evaluated:
      "bidirectional" — both IR→RED and RED→IR
      "ir2red"        — IR→RED only  (source=IR, target=RED)
      "red2ir"        — RED→IR only  (source=RED, target=IR)

    Inactive directions return float('nan') in the results dict.
    """
    model.eval()
    n = len(val_dataset)
    if max_samples > 0:
        n = min(n, max_samples)
    dataset = (torch.utils.data.Subset(val_dataset, list(range(n)))
               if n < len(val_dataset) else val_dataset)
    loader  = get_loader(dataset, batch_size=batch_size, collate_fn=diffusion_collate_fn,
                         num_workers=4, shuffle=False)

    # Per-direction accumulators — index 0 = IR→RED, index 1 = RED→IR
    accs = [{k: [] for k in ("mse_phys", "mae_phys", "mse_norm",
                              "ssim_norm", "ssim_phys", "pearson",
                              "fid_real",  "fid_fake")}
            for _ in range(2)]

    inception = _build_inception(device) if compute_fid else None

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    n_done = 0

    for batch in loader:
        ir  = batch["ir"].to(device)
        red = batch["red"].to(device)
        B   = ir.shape[0]

        norm_stats = batch["norm_stats"]
        center = norm_stats[:, :1, None, None].to(device)
        scale  = norm_stats[:, 1:2, None, None].to(device)
        dc     = norm_stats[:, 2:3, None, None].to(device)

        shared = dict(center=center, scale=scale, dc=dc,
                      cfg_inf=cfg_inf, device=device, edge_mode=edge_mode,
                      dexined_model=dexined_model,
                      compute_fid=compute_fid, inception=inception)

        progress_parts = [f"  [{n_done + B}/{n}]"]

        if train_mode in ("bidirectional", "ir2red"):
            d = 0 if train_mode == "bidirectional" else None
            m = _eval_one_direction(model, scheduler, src=ir, tgt=red,
                                    direction=d, prior=prior_red,
                                    label="IR→RED", **shared)
            for k in ("mse_phys", "mae_phys", "mse_norm", "ssim_phys", "ssim_norm", "pearson"):
                accs[0][k].extend(m[k])
            if compute_fid:
                accs[0]["fid_real"].append(m["fid_real"])
                accs[0]["fid_fake"].append(m["fid_fake"])
            progress_parts.append(m["progress"])

        if train_mode in ("bidirectional", "red2ir"):
            d = 1 if train_mode == "bidirectional" else None
            m = _eval_one_direction(model, scheduler, src=red, tgt=ir,
                                    direction=d, prior=prior_ir,
                                    label="RED→IR", **shared)
            for k in ("mse_phys", "mae_phys", "mse_norm", "ssim_phys", "ssim_norm", "pearson"):
                accs[1][k].extend(m[k])
            if compute_fid:
                accs[1]["fid_real"].append(m["fid_real"])
                accs[1]["fid_fake"].append(m["fid_fake"])
            progress_parts.append(m["progress"])

        n_done += B
        print("  |  ".join(progress_parts))

    def _avg(lst):
        return float(np.mean(lst)) if lst else float("nan")
    def _psnr_pooled(lst, max_val):
        if not lst:
            return float("nan")
        avg_mse = float(np.mean(lst))
        return 10.0 * math.log10(max_val ** 2 / avg_mse) if avg_mse > 0 else 0.0

    a0, a1 = accs[0], accs[1]
    results = dict(
        n_samples        = n_done,
        mse_phys_ir2red  = _avg(a0["mse_phys"]),
        mae_phys_ir2red  = _avg(a0["mae_phys"]),
        psnr_phys_ir2red = _psnr_pooled(a0["mse_phys"], 1.0),
        mse_phys_red2ir  = _avg(a1["mse_phys"]),
        mae_phys_red2ir  = _avg(a1["mae_phys"]),
        psnr_phys_red2ir = _psnr_pooled(a1["mse_phys"], 1.0),
        mse_norm_ir2red  = _avg(a0["mse_norm"]),
        psnr_norm_ir2red = _psnr_pooled(a0["mse_norm"], 20.0),
        mse_norm_red2ir  = _avg(a1["mse_norm"]),
        psnr_norm_red2ir = _psnr_pooled(a1["mse_norm"], 20.0),
        ssim_norm_ir2red = _avg(a0["ssim_norm"]),
        ssim_norm_red2ir = _avg(a1["ssim_norm"]),
        ssim_phys_ir2red = _avg(a0["ssim_phys"]),
        ssim_phys_red2ir = _avg(a1["ssim_phys"]),
        pearson_ir2red   = _avg(a0["pearson"]),
        pearson_red2ir   = _avg(a1["pearson"]),
    )

    if compute_fid:
        for key, a in (("fid_ir2red", a0), ("fid_red2ir", a1)):
            if a["fid_real"]:
                results[key] = _fid_from_features(
                    np.concatenate(a["fid_real"]), np.concatenate(a["fid_fake"]))
        fid_parts = [f"FID({k})={results[k]:.2f}"
                     for k in ("fid_ir2red", "fid_red2ir") if k in results]
        if fid_parts:
            print(f"  {'  '.join(fid_parts)}")

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
    parser.add_argument("--checkpoint",  default="src/output/latest_bidirectional.pt")
    parser.add_argument("--prior_dir",   default="src/output",
                        help="Directory containing prior_ir.pt and prior_red.pt")
    parser.add_argument("--data_root",   default=r"D:\Imperial\IRP\HiRISE_diffusion\data")
    parser.add_argument("--csv_path",    default=r"D:\Imperial\IRP\HiRISE_diffusion\data\files\data_record_bin12.csv")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples (0 = all)")
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--lambda_scl",  type=float, default=0.0)
    parser.add_argument("--lambda_ccl",  type=float, default=0.0)
    parser.add_argument("--no_fid",      default=True, action="store_true",
                        help="Skip FID computation (faster, no torchvision needed)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument(
        "--train_mode",
        default="bidirectional",
        choices=["bidirectional", "ir2red", "red2ir"],
        help="Model/eval mode: bidirectional (full image metrics) or single-direction validation",
    )
    parser.add_argument("--edge_mode",  default="dexined", choices=["sobel", "dexined"],
                        help="Edge detector: sobel (default) or dexined")
    parser.add_argument("--dexined_weights", default="checkpoints/dexined_biped.pth",
                        help="Path to DexiNed pretrained weights (.pth)")
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
    print(f"Mode         : {args.train_mode}")
    print(f"SCI          : lambda_scl={args.lambda_scl}  lambda_ccl={args.lambda_ccl}")
    print(f"Max samples  : {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"FID          : {'disabled' if args.no_fid else 'enabled'}")
    print()

    if args.train_mode == "bidirectional":
        model = UNet(
            in_channels=cfg_model.in_channels,
            out_channels=cfg_model.out_channels,
            base_channels=cfg_model.base_channels,
            num_res_blocks=cfg_model.num_res_blocks,
            dropout=cfg_model.dropout,
        ).to(device)
    elif args.train_mode == "ir2red":
        model = IR2REDUNet(
            in_channels=cfg_model.in_channels,
            out_channels=cfg_model.out_channels,
            base_channels=cfg_model.base_channels,
            num_res_blocks=cfg_model.num_res_blocks,
            dropout=cfg_model.dropout,
        ).to(device)
    else:
        model = RED2IRUNet(
            in_channels=cfg_model.in_channels,
            out_channels=cfg_model.out_channels,
            base_channels=cfg_model.base_channels,
            num_res_blocks=cfg_model.num_res_blocks,
            dropout=cfg_model.dropout,
        ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: step={ckpt.get('step','?')}  loss={ckpt.get('loss','?'):.4f}")

    # ── Edge detector ─────────────────────────────────────────────────────────
    dexined_model = None
    if args.edge_mode == "dexined":
        dexined_model = load_dexined(args.dexined_weights, device)

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
    val_sets = [19645, 7292, 7293, 14774]
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
        edge_mode=args.edge_mode,
        dexined_model=dexined_model,
        train_mode=args.train_mode,
    )

    def _f4(v):  return f"{v:>10.4f}" if not math.isnan(v) else f"{'—':>10}"
    def _f2(v):  return f"{v:>10.2f}" if not math.isnan(v) else f"{'—':>10}"
    def _avg2(a, b): return f"{(a+b)/2:>10.4f}" if not (math.isnan(a) or math.isnan(b)) else f"{'—':>10}"
    def _avg2f2(a, b): return f"{(a+b)/2:>10.2f}" if not (math.isnan(a) or math.isnan(b)) else f"{'—':>10}"

    W = 14
    print(f"\n{'='*68}")
    print(f"{'Metric':<{W}}  {'IR→RED':>10}  {'RED→IR':>10}  {'Combined':>10}")
    print(f"{'─'*68}")
    print(f"{'[Physical]':<{W}}")
    print(f"{'  MSE':<{W}}  {_f4(results['mse_phys_ir2red'])}  {_f4(results['mse_phys_red2ir'])}  "
          f"{_avg2(results['mse_phys_ir2red'], results['mse_phys_red2ir'])}")
    print(f"{'  MAE':<{W}}  {_f4(results['mae_phys_ir2red'])}  {_f4(results['mae_phys_red2ir'])}  {'—':>10}")
    print(f"{'  PSNR (dB)':<{W}}  {_f2(results['psnr_phys_ir2red'])}  {_f2(results['psnr_phys_red2ir'])}  "
          f"{_avg2f2(results['psnr_phys_ir2red'], results['psnr_phys_red2ir'])}")
    print(f"{'[Normalized]':<{W}}")
    print(f"{'  MSE':<{W}}  {_f4(results['mse_norm_ir2red'])}  {_f4(results['mse_norm_red2ir'])}  "
          f"{_avg2(results['mse_norm_ir2red'], results['mse_norm_red2ir'])}")
    print(f"{'  PSNR (dB)':<{W}}  {_f2(results['psnr_norm_ir2red'])}  {_f2(results['psnr_norm_red2ir'])}  "
          f"{_avg2f2(results['psnr_norm_ir2red'], results['psnr_norm_red2ir'])}")
    print(f"{'[Structural]':<{W}}")
    print(f"{'SSIM (norm)':<{W}}  {_f4(results['ssim_norm_ir2red'])}  {_f4(results['ssim_norm_red2ir'])}  "
          f"{_avg2(results['ssim_norm_ir2red'], results['ssim_norm_red2ir'])}")
    print(f"{'SSIM (phys)':<{W}}  {_f4(results['ssim_phys_ir2red'])}  {_f4(results['ssim_phys_red2ir'])}  "
          f"{_avg2(results['ssim_phys_ir2red'], results['ssim_phys_red2ir'])}")
    print(f"{'Pearson r':<{W}}  {_f4(results['pearson_ir2red'])}  {_f4(results['pearson_red2ir'])}  {'—':>10}")
    if "fid_ir2red" in results or "fid_red2ir" in results:
        fid0 = results.get("fid_ir2red", float("nan"))
        fid1 = results.get("fid_red2ir", float("nan"))
        print(f"{'FID':<{W}}  {_f2(fid0)}  {_f2(fid1)}  {'—':>10}")
    print(f"{'='*68}")
    print(f"(n={results['n_samples']}  mode={args.train_mode}  λ_scl={args.lambda_scl}  λ_ccl={args.lambda_ccl})")
    print(f"  Physical PSNR: MAX=1.0  |  Normalized PSNR: MAX=20  |  SSIM physical: per-image GT range, floor=0.01")


if __name__ == "__main__":
    main()
