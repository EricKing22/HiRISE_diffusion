"""
Evaluate a Flow Matching checkpoint on the validation set.

Two evaluation modes:
  - Velocity-prediction MSE: fast, used by train_fm.py during training
  - Image-level full metrics: runs full Euler ODE sampling, computes
    MSE / MAE / PSNR / SSIM / Pearson-r (and optionally FID)

Usage (standalone — image-level):
    cd <project_root>
    python src/eval_fm.py \
        --checkpoint src/output/latest_fm_ir2red.pt \
        --train_mode ir2red
"""

import os
import sys
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FMModelConfig, FMInferenceConfig, DataConfig
from models.ir2red_fm import FMUNet as IR2REDFMUNet
from models.red2ir_fm import FMUNet as RED2IRFMUNet
from diffusion.fm_utils import fm_interpolate, fm_velocity_target
from inference_fm import sample_fm
from data.dataset import DiffusionDataset, diffusion_collate_fn, get_loader
from eval import (
    _ssim_safe, _pearson_batch,
    _build_inception, _inception_features, _fid_from_features,
    get_val_split,
)


# =============================================================================
# Velocity-prediction evaluation (used by train_fm.py)
# =============================================================================

def eval_fm_batch(
    batch:     dict,
    model:     torch.nn.Module,
    device:    torch.device,
) -> tuple:
    """
    Compute FM velocity-prediction MSE for one batch, split by direction.

    Returns (loss_ir2red, loss_red2ir) as Python floats.
    """
    ir  = batch["ir"].to(device)
    red = batch["red"].to(device)
    B   = ir.shape[0]

    half      = B // 2
    direction = torch.cat([
        torch.zeros(half,       dtype=torch.long, device=device),
        torch.ones(B - half,    dtype=torch.long, device=device),
    ])

    mask     = (direction == 0)[:, None, None, None]
    x_target = torch.where(mask, red, ir)
    x_source = torch.where(mask, ir, red)

    t        = torch.rand(B, device=device)
    noise    = torch.randn_like(x_target)
    x_t      = fm_interpolate(x_target, noise, t)
    v_target = fm_velocity_target(x_target, noise)

    v_pred = model(x_t, x_source, t)
    loss_per_sample = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=[1, 2, 3])

    mask_1d     = (direction == 0)
    loss_ir2red = loss_per_sample[mask_1d].mean().item() if mask_1d.any() else 0.0
    loss_red2ir = loss_per_sample[~mask_1d].mean().item() if (~mask_1d).any() else 0.0
    return loss_ir2red, loss_red2ir


def evaluate_fm(
    model:      torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device:     torch.device,
) -> dict:
    """
    Bidirectional FM velocity-prediction evaluation over the validation loader.

    Returns dict with keys: loss, loss_ir2red, loss_red2ir.
    """
    model.eval()
    ir2red_accum = 0.0
    red2ir_accum = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            v_ir2red, v_red2ir = eval_fm_batch(batch, model, device)
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


def evaluate_fm_unidirectional(
    model:      torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device:     torch.device,
    train_mode: str,
) -> dict:
    """
    Velocity-prediction MSE evaluation over the full validation loader.

    Args:
        train_mode: "ir2red" or "red2ir".

    Returns dict with keys: loss, loss_ir2red, loss_red2ir.
    """
    model.eval()
    loss_accum = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            ir  = batch["ir"].to(device)
            red = batch["red"].to(device)
            B   = ir.shape[0]

            if train_mode == "ir2red":
                x_source, x_target = ir, red
            elif train_mode == "red2ir":
                x_source, x_target = red, ir
            else:
                raise ValueError(f"Unsupported FM eval mode: {train_mode}")

            t     = torch.rand(B, device=device)
            noise = torch.randn_like(x_target)
            x_t      = fm_interpolate(x_target, noise, t)
            v_target = fm_velocity_target(x_target, noise)

            v_pred = model(x_t, x_source, t)
            loss = F.mse_loss(v_pred, v_target)

            loss_accum += loss.item()
            n += 1

    avg = loss_accum / max(n, 1)
    if train_mode == "ir2red":
        return dict(loss=avg, loss_ir2red=avg, loss_red2ir=0.0)
    return dict(loss=avg, loss_ir2red=0.0, loss_red2ir=avg)


# =============================================================================
# Image-level evaluation (full Euler ODE sampling)
# =============================================================================

def _eval_one_direction_fm(
    model, cfg_inf,
    src, tgt,
    center, scale, dc,
    device,
    compute_fid, inception,
    label,
):
    """
    Full FM Euler ODE sampling for one (src→tgt) direction.

    Mirrors eval._eval_one_direction but uses sample_fm instead of DDPM sample.
    Returns per-image metric lists and a progress string.
    """
    B = src.shape[0]

    with torch.no_grad():
        pred = sample_fm(model, src, cfg_inf, device, verbose=False)

    pred_norm = pred.clamp(-10.0, 10.0)
    pred_phys = (pred_norm + dc) * scale + center
    tgt_phys  = (tgt       + dc) * scale + center

    mse_phys_b = F.mse_loss(pred_phys, tgt_phys, reduction="none").mean(dim=[1, 2, 3])
    mae_phys_b = F.l1_loss( pred_phys, tgt_phys, reduction="none").mean(dim=[1, 2, 3])
    mse_norm_b = F.mse_loss(pred_norm, tgt,       reduction="none").mean(dim=[1, 2, 3])
    mae_norm_b = F.l1_loss( pred_norm, tgt,        reduction="none").mean(dim=[1, 2, 3])

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
        mae_norm  = mae_norm_b.cpu().tolist(),
        ssim_phys = ssim_phy_vals,
        ssim_norm = ssim_norm_vals,
        pearson   = pearson_b.cpu().tolist(),
        fid_real  = _inception_features(tgt,       inception) if compute_fid else None,
        fid_fake  = _inception_features(pred_norm, inception) if compute_fid else None,
        progress  = (f"MSE({label})={mse_phys_b.mean():.4f}/{mse_norm_b.mean():.4f}  "
                     f"SSIM phy={np.mean(ssim_phy_vals):.4f} norm={np.mean(ssim_norm_vals):.4f}"),
    )


def evaluate_images_fm(
    model:          torch.nn.Module,
    val_dataset:    DiffusionDataset,
    cfg_inf:        FMInferenceConfig,
    device:         torch.device,
    max_samples:    int  = 0,
    batch_size:     int  = 4,
    seed:           int  = 42,
    compute_fid:    bool = True,
    train_mode:     str  = "ir2red",
) -> dict:
    """
    Image-level evaluation: full Euler ODE sampling in batches.

    train_mode controls which direction is evaluated:
      "ir2red" — IR→RED  (source=IR, target=RED)
      "red2ir" — RED→IR  (source=RED, target=IR)

    The inactive direction returns float('nan') in the results dict,
    matching the layout of eval.evaluate_images for easy side-by-side
    comparison.
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
    accs = [{k: [] for k in ("mse_phys", "mae_phys", "mse_norm", "mae_norm",
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

        shared = dict(cfg_inf=cfg_inf, center=center, scale=scale, dc=dc,
                      device=device, compute_fid=compute_fid, inception=inception)

        progress_parts = [f"  [{n_done + B}/{n}]"]

        dir_idx        = 0 if train_mode == "ir2red" else 1
        src, tgt       = (ir, red) if train_mode == "ir2red" else (red, ir)
        label          = "IR→RED" if train_mode == "ir2red" else "RED→IR"

        m = _eval_one_direction_fm(model, src=src, tgt=tgt, label=label, **shared)
        for k in ("mse_phys", "mae_phys", "mse_norm", "mae_norm", "ssim_phys", "ssim_norm", "pearson"):
            accs[dir_idx][k].extend(m[k])
        if compute_fid:
            accs[dir_idx]["fid_real"].append(m["fid_real"])
            accs[dir_idx]["fid_fake"].append(m["fid_fake"])
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
        # Physical space
        mse_phys_ir2red  = _avg(a0["mse_phys"]),
        mae_phys_ir2red  = _avg(a0["mae_phys"]),
        ssim_phys_ir2red = _avg(a0["ssim_phys"]),
        mse_phys_red2ir  = _avg(a1["mse_phys"]),
        mae_phys_red2ir  = _avg(a1["mae_phys"]),
        ssim_phys_red2ir = _avg(a1["ssim_phys"]),
        # Normalized space
        mse_norm_ir2red  = _avg(a0["mse_norm"]),
        mae_norm_ir2red  = _avg(a0["mae_norm"]),
        psnr_norm_ir2red = _psnr_pooled(a0["mse_norm"], 20.0),
        ssim_norm_ir2red = _avg(a0["ssim_norm"]),
        mse_norm_red2ir  = _avg(a1["mse_norm"]),
        mae_norm_red2ir  = _avg(a1["mae_norm"]),
        psnr_norm_red2ir = _psnr_pooled(a1["mse_norm"], 20.0),
        ssim_norm_red2ir = _avg(a1["ssim_norm"]),
        # Statistical
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
# Dataset split (identical to eval.py / train.py)
# =============================================================================

# get_val_split is imported from eval.py


# =============================================================================
# CLI (image-level evaluation)
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FM — full image metrics")
    parser.add_argument("--checkpoint",  default="src/output/latest_fm_ir2red.pt")
    parser.add_argument("--data_root",   default=r"D:\Imperial\IRP\HiRISE_diffusion\data",
                        help="Data root directory (default: DataConfig.data_root)")
    parser.add_argument("--csv_path",    default=r"D:\Imperial\IRP\HiRISE_diffusion\data\files\data_record_bin12.csv",
                        help="CSV path (default: DataConfig.csv_path)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to evaluate (0 = all)")
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_steps",   type=int, default=50,
                        help="Euler ODE steps (0 = FMInferenceConfig default, currently 50)")
    parser.add_argument("--no_fid",      default=True, action="store_true",
                        help="Skip FID computation (faster)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument(
        "--train_mode",
        default="ir2red",
        choices=["ir2red", "red2ir"],
        help="Evaluation direction: ir2red or red2ir",
    )
    args = parser.parse_args()

    cfg_model = FMModelConfig()
    cfg_data  = DataConfig()
    cfg_inf   = FMInferenceConfig()
    if args.num_steps > 0:
        cfg_inf = FMInferenceConfig(num_steps=args.num_steps)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = args.data_root or cfg_data.data_root or os.path.join(project_root, "data")
    csv_path  = args.csv_path  or cfg_data.csv_path  or os.path.join(
        project_root, "data", "files", "data_record_bin12.csv")

    print(f"Device       : {device}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Mode         : {args.train_mode}")
    print(f"ODE steps    : {cfg_inf.num_steps}")
    print(f"Max samples  : {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"FID          : {'disabled' if args.no_fid else 'enabled'}")
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    ModelCls = IR2REDFMUNet if args.train_mode == "ir2red" else RED2IRFMUNet
    model = ModelCls(
        in_channels    = cfg_model.in_channels,
        out_channels   = cfg_model.out_channels,
        base_channels  = cfg_model.base_channels,
        num_res_blocks = cfg_model.num_res_blocks,
        dropout        = cfg_model.dropout,
        t_scale        = cfg_model.t_scale,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: step={ckpt.get('step','?')}  loss={ckpt.get('loss','?'):.4f}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dr = pd.read_csv(csv_path)
    _, val_sets = get_val_split(dr)
    val_sets = [19645, 7292, 7293, 14774]
    val_dataset = DiffusionDataset(
        data_record=dr, data_root=data_root, sweep=True,
        allowed_sets=val_sets,
    )
    print(f"Val set: {len(val_dataset)} sets\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = evaluate_images_fm(
        model, val_dataset,
        cfg_inf=cfg_inf, device=device,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        compute_fid=not args.no_fid,
        train_mode=args.train_mode,
    )

    # ── Print results table (same format as eval.py) ──────────────────────────
    def _f4(v):      return f"{v:>10.4f}" if not math.isnan(v) else f"{'—':>10}"
    def _f2(v):      return f"{v:>10.2f}" if not math.isnan(v) else f"{'—':>10}"
    def _avg2(a, b):
        return f"{(a+b)/2:>10.4f}" if not (math.isnan(a) or math.isnan(b)) else f"{'—':>10}"
    def _avg2f2(a, b):
        return f"{(a+b)/2:>10.2f}" if not (math.isnan(a) or math.isnan(b)) else f"{'—':>10}"

    W = 14
    print(f"\n{'='*68}")
    print(f"{'Metric':<{W}}  {'IR→RED':>10}  {'RED→IR':>10}  {'Combined':>10}")
    print(f"{'─'*68}")
    print(f"{'[Physical]':<{W}}")
    print(f"{'  MSE':<{W}}  {_f4(results['mse_phys_ir2red'])}  {_f4(results['mse_phys_red2ir'])}  "
          f"{_avg2(results['mse_phys_ir2red'], results['mse_phys_red2ir'])}")
    print(f"{'  MAE':<{W}}  {_f4(results['mae_phys_ir2red'])}  {_f4(results['mae_phys_red2ir'])}  "
          f"{_avg2(results['mae_phys_ir2red'], results['mae_phys_red2ir'])}")
    print(f"{'  SSIM':<{W}}  {_f4(results['ssim_phys_ir2red'])}  {_f4(results['ssim_phys_red2ir'])}  "
          f"{_avg2(results['ssim_phys_ir2red'], results['ssim_phys_red2ir'])}")
    print(f"{'[Normalized]':<{W}}")
    print(f"{'  MSE':<{W}}  {_f4(results['mse_norm_ir2red'])}  {_f4(results['mse_norm_red2ir'])}  "
          f"{_avg2(results['mse_norm_ir2red'], results['mse_norm_red2ir'])}")
    print(f"{'  MAE':<{W}}  {_f4(results['mae_norm_ir2red'])}  {_f4(results['mae_norm_red2ir'])}  "
          f"{_avg2(results['mae_norm_ir2red'], results['mae_norm_red2ir'])}")
    print(f"{'  PSNR (dB)':<{W}}  {_f2(results['psnr_norm_ir2red'])}  {_f2(results['psnr_norm_red2ir'])}  "
          f"{_avg2f2(results['psnr_norm_ir2red'], results['psnr_norm_red2ir'])}")
    print(f"{'  SSIM':<{W}}  {_f4(results['ssim_norm_ir2red'])}  {_f4(results['ssim_norm_red2ir'])}  "
          f"{_avg2(results['ssim_norm_ir2red'], results['ssim_norm_red2ir'])}")
    print(f"{'─'*68}")
    print(f"{'Pearson r':<{W}}  {_f4(results['pearson_ir2red'])}  {_f4(results['pearson_red2ir'])}  {'—':>10}")
    if "fid_ir2red" in results or "fid_red2ir" in results:
        fid0 = results.get("fid_ir2red", float("nan"))
        fid1 = results.get("fid_red2ir", float("nan"))
        print(f"{'FID':<{W}}  {_f2(fid0)}  {_f2(fid1)}  {'—':>10}")
    print(f"{'='*68}")
    print(f"(n={results['n_samples']}  mode={args.train_mode}  steps={cfg_inf.num_steps})")
    print(f"  PSNR (normalized, MAX=20) ≡ PSNR (physical, MAX=1.0) under scale=0.05 normalisation")
    print(f"  SSIM physical: per-image GT dynamic range, floor=0.01")


if __name__ == "__main__":
    main()
