"""
Evaluate a CM-Diff checkpoint on the validation set.

Two evaluation modes:
  - Noise-prediction MSE: fast, used by train.py during training (evaluate())
  - Image-level MSE: runs full 1000-step sampling, compares pred vs GT (evaluate_images())

Usage (standalone — image-level):
    cd <project_root>
    python src/eval.py \
        --checkpoint src/output/latest.pt \
        --data_root  /path/to/data \
        --csv_path   /path/to/data_record_bin12.csv
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
import pandas as pd

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
) -> dict:
    """
    Image-level evaluation: run full T-step reverse diffusion in batches,
    compare predicted image vs ground truth.

    Evaluates both directions for each sample:
      direction 0: IR10 → RED4  (pred vs GT RED4)
      direction 1: RED4 → IR10  (pred vs GT IR10)

    Returns dict with per-direction and combined MSE.
    """
    model.eval()
    n = len(val_dataset)
    if max_samples > 0:
        n = min(n, max_samples)

    dataset = torch.utils.data.Subset(val_dataset, list(range(n))) if n < len(val_dataset) else val_dataset
    loader  = get_loader(dataset, batch_size=batch_size, collate_fn=diffusion_collate_fn,
                         num_workers=0, shuffle=False)

    mse_ir2red_list = []
    mse_red2ir_list = []

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    n_done = 0
    for batch in loader:
        ir_batch  = batch["ir"].to(device)    # [B, 1, H, W]
        red_batch = batch["red"].to(device)   # [B, 1, H, W]

        # Direction 0: IR10 → RED4
        with torch.no_grad():
            pred_red = sample(model, scheduler, ir_batch, direction=0,
                              prior_stats=prior_red, cfg_inf=cfg_inf, device=device,
                              verbose=False)

        # Direction 1: RED4 → IR10
        with torch.no_grad():
            pred_ir = sample(model, scheduler, red_batch, direction=1,
                             prior_stats=prior_ir, cfg_inf=cfg_inf, device=device,
                             verbose=False)

        # Vectorised per-sample MSE: direction 0 predicts RED, direction 1 predicts IR
        mse_0 = F.mse_loss(pred_red, red_batch, reduction="none").mean(dim=[1, 2, 3])  # [B]
        mse_1 = F.mse_loss(pred_ir,  ir_batch,  reduction="none").mean(dim=[1, 2, 3])  # [B]

        mse_ir2red_list.extend(mse_0.cpu().tolist())
        mse_red2ir_list.extend(mse_1.cpu().tolist())

        n_done += ir_batch.shape[0]
        print(f"  [{n_done}/{n}]  MSE(IR→RED)={mse_0.mean().item():.4f}  MSE(RED→IR)={mse_1.mean().item():.4f}")

    avg_ir2red = sum(mse_ir2red_list) / len(mse_ir2red_list)
    avg_red2ir = sum(mse_red2ir_list) / len(mse_red2ir_list)

    return dict(
        mse_ir2red=avg_ir2red,
        mse_red2ir=avg_red2ir,
        mse_combined=(avg_ir2red + avg_red2ir) / 2,
        n_samples=n_done,
        per_sample_ir2red=mse_ir2red_list,
        per_sample_red2ir=mse_red2ir_list,
    )


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
    parser = argparse.ArgumentParser(description="Evaluate CM-Diff — image-level MSE")
    parser.add_argument("--checkpoint",  default="src/output/latest.pt",
                        help="Model checkpoint path")
    parser.add_argument("--prior_dir",   default="src/output",
                        help="Directory containing prior_ir.pt and prior_red.pt")
    parser.add_argument("--data_root",   default="",
                        help="Root directory for .npy files (overrides config)")
    parser.add_argument("--csv_path",    default="",
                        help="Path to data_record CSV (overrides config)")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Max samples to evaluate (0 = all, default 50)")
    parser.add_argument("--batch_size",  type=int, default=4,
                        help="Batch size for image-level evaluation (default 4)")
    parser.add_argument("--lambda_scl",  type=float, default=0.0,
                        help="SCI lambda_scl weight")
    parser.add_argument("--lambda_ccl",  type=float, default=0.0,
                        help="SCI lambda_ccl weight")
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
    print(f"Prior dir    : {args.prior_dir}")
    print(f"SCI          : lambda_scl={args.lambda_scl}  lambda_ccl={args.lambda_ccl}")
    print(f"Max samples  : {args.max_samples if args.max_samples > 0 else 'all'}")
    print()

    # ── Model ─────────────────────────────────────────────────────────────
    model = UNet(
        in_channels=cfg_model.in_channels,
        out_channels=cfg_model.out_channels,
        base_channels=cfg_model.base_channels,
        num_res_blocks=cfg_model.num_res_blocks,
        dropout=cfg_model.dropout,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: step={ckpt.get('step', '?')}  loss={ckpt.get('loss', '?'):.4f}")

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = DDPMScheduler(
        timesteps=cfg_model.timesteps,
        beta_start=cfg_model.beta_start,
        beta_end=cfg_model.beta_end,
    ).to(device)

    # ── Prior stats ───────────────────────────────────────────────────────
    prior_red = load_prior_stats(os.path.join(args.prior_dir, "prior_red.pt"), device)
    prior_ir  = load_prior_stats(os.path.join(args.prior_dir, "prior_ir.pt"),  device)
    print(f"Prior RED: mu={prior_red['mu'].item():.4f}  sigma={prior_red['sigma'].item():.4f}")
    print(f"Prior IR:  mu={prior_ir['mu'].item():.4f}  sigma={prior_ir['sigma'].item():.4f}")

    # ── Val dataset (identical split to train.py) ─────────────────────────
    dr = pd.read_csv(csv_path)
    _, val_sets = get_val_split(dr)

    val_dataset = DiffusionDataset(
        data_record=dr, data_root=data_root, sweep=True,
        allowed_sets=val_sets,
    )
    print(f"Val set: {len(val_dataset)} sets")
    print()

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = evaluate_images(
        model, scheduler, val_dataset,
        prior_red=prior_red,
        prior_ir=prior_ir,
        cfg_inf=cfg_inf,
        device=device,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"\n{'='*50}")
    print(f"Image-level MSE  ({results['n_samples']} samples)")
    print(f"  IR→RED : {results['mse_ir2red']:.4f}")
    print(f"  RED→IR : {results['mse_red2ir']:.4f}")
    print(f"  Combined: {results['mse_combined']:.4f}")


if __name__ == "__main__":
    main()
