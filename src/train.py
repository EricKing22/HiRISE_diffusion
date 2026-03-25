"""
CM-Diff training script for HiRISE IR10 ↔ RED4 bidirectional diffusion.

Training loop (BDT — Bidirectional Diffusion Training):
  Each batch is split into two halves:
    Direction 0 (IR10 → RED4): source=ir,  target=red
    Direction 1 (RED4 → IR10): source=red, target=ir
  Both directions share the same UNet; L_joint = L_A + L_B.

Usage:
    cd src
    python train.py
"""

import os
import sys
import math
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import wandb

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from config import ModelConfig, TrainConfig, DataConfig
from models.cm_diff_unet import UNet
from diffusion.scheduler import DDPMScheduler
from diffusion.process import q_sample, sobel_edge

# Add project root for data imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.dataset import DiffusionDataset, diffusion_collate_fn, get_loader


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(
    step:      int,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss:      float,
    path:      str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step":       step,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "loss":       loss,
        },
        path,
    )
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(
    path:      str,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load checkpoint; returns the step to resume from."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  [ckpt] resumed from step {ckpt['step']} (loss={ckpt['loss']:.4f})")
    return ckpt["step"]


# =============================================================================
# Single training step
# =============================================================================

def train_step(
    batch:     dict,
    model:     torch.nn.Module,
    scheduler: DDPMScheduler,
    device:    torch.device,
    cfg_train: TrainConfig,
) -> torch.Tensor:
    """
    Compute L_joint for one batch.

    The batch is split in half by direction:
      first  B//2 samples → direction 0 (IR10 → RED4)
      second B//2 samples → direction 1 (RED4 → IR10)

    If batch size is odd, the last sample is assigned direction 0.

    Returns scalar loss tensor (already normalised by batch size).
    """
    ir  = batch["ir"].to(device)    # [B, 1, H, W]
    red = batch["red"].to(device)   # [B, 1, H, W]
    B   = ir.shape[0]

    # Assign directions: first half = 0, second half = 1
    half      = B // 2
    direction = torch.cat([
        torch.zeros(half,       dtype=torch.long, device=device),
        torch.ones( B - half,   dtype=torch.long, device=device),
    ])                                  # [B]

    # Build source / target per sample based on direction
    # direction=0: target=red, source=ir
    # direction=1: target=ir,  source=red
    mask       = (direction == 0)[:, None, None, None]   # [B,1,1,1] bool
    x_target   = torch.where(mask, red, ir)              # [B,1,H,W]
    x_source   = torch.where(mask, ir,  red)             # [B,1,H,W]

    # Sobel edge map from source (fixed structural prior)
    with torch.no_grad():
        edge = sobel_edge(x_source)                      # [B,1,H,W]

    # Sample random timesteps t ~ Uniform(0, T-1)
    t     = torch.randint(0, scheduler.timesteps, (B,), device=device)

    # Forward diffusion: add noise to target
    noise = torch.randn_like(x_target)
    x_t   = q_sample(scheduler, x_target, t, noise)      # [B,1,H,W]

    # Predict noise
    eps_pred = model(x_t, x_source, edge, direction, t)  # [B,1,H,W]

    # MSE loss between predicted and actual noise
    loss_per_sample = F.mse_loss(eps_pred, noise, reduction="none").mean(dim=[1, 2, 3])

    # Separate losses by direction for weighted L_joint
    mask_1d   = (direction == 0)                          # [B]
    loss_ir2red = loss_per_sample[mask_1d].mean()    if mask_1d.any()    else torch.tensor(0.0, device=device)
    loss_red2ir = loss_per_sample[~mask_1d].mean()  if (~mask_1d).any() else torch.tensor(0.0, device=device)

    l_joint = cfg_train.lambda_ir_to_red * loss_ir2red + cfg_train.lambda_red_to_ir * loss_red2ir
    return l_joint, loss_ir2red.item(), loss_red2ir.item()


# =============================================================================
# Main training loop
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CM-Diff BDT training")
    parser.add_argument("--data_root",      default="",          help="Root directory for .npy files (overrides config)")
    parser.add_argument("--csv_path",       default="",          help="Absolute path to data_record CSV (overrides config)")
    parser.add_argument("--ckpt_dir",       default="",          help="Checkpoint output directory (overrides default <project_root>/checkpoints)")
    parser.add_argument("--wandb_project",  default="HiRISE_diffusion", help="W&B project name")
    parser.add_argument("--run_name",       default=None,        help="W&B run name (auto-generated if omitted)")
    parser.add_argument("--no_wandb",       action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    cfg_model = ModelConfig()
    cfg_train = TrainConfig()
    cfg_data  = DataConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Diffusion schedule ────────────────────────────────────────────────
    scheduler = DDPMScheduler(
        timesteps=cfg_model.timesteps,
        beta_start=cfg_model.beta_start,
        beta_end=cfg_model.beta_end,
    ).to(device)

    # ── Model ─────────────────────────────────────────────────────────────
    model = UNet(
        in_channels=cfg_model.in_channels,
        out_channels=cfg_model.out_channels,
        base_channels=cfg_model.base_channels,
        num_res_blocks=cfg_model.num_res_blocks,
        dropout=cfg_model.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet parameters: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train.lr)
    scheduler_lr = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg_train.lr_decay_every,
        gamma=cfg_train.lr_decay,
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root    = args.data_root or cfg_data.data_root or os.path.join(project_root, "data")
    csv_path     = args.csv_path  or cfg_data.csv_path  or os.path.join(project_root, "data", "files", "data_record_bin12.csv")
    dr = pd.read_csv(csv_path)

    dataset = DiffusionDataset(
        data_record=dr,
        data_root=data_root,
        sweep=True,            # suppress per-set prints during training
        allowed_sets=None,     # use all available data
    )
    loader = get_loader(
        dataset,
        batch_size=cfg_train.batch_size,
        collate_fn=diffusion_collate_fn,
        num_workers=4,
        shuffle=True,
    )
    print(f"Dataset: {len(dataset)} sets  |  batch_size={cfg_train.batch_size}")

    # ── W&B ───────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                **cfg_model.model_dump(),
                **cfg_train.model_dump(),
                "data_root": data_root,
                "csv_path":  csv_path,
                "n_params":  n_params,
                "device":    str(device),
            },
            resume="allow",
        )

    # ── Resume from checkpoint if available ───────────────────────────────
    ckpt_dir  = args.ckpt_dir or os.path.join(project_root, "checkpoints")
    start_step = 0
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest_ckpt):
        start_step = load_checkpoint(latest_ckpt, model, optimizer)

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    step         = start_step
    loss_accum       = 0.0
    loss_ir2red_accum = 0.0
    loss_red2ir_accum = 0.0
    data_iter    = iter(loader)

    print(f"Training from step {step} to {cfg_train.total_steps} ...")

    while step < cfg_train.total_steps:
        # Reload dataloader when exhausted
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch     = next(data_iter)

        optimizer.zero_grad()
        loss, loss_ir2red_val, loss_red2ir_val = train_step(batch, model, scheduler, device, cfg_train)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler_lr.step()

        loss_accum        += loss.item()
        loss_ir2red_accum += loss_ir2red_val
        loss_red2ir_accum += loss_red2ir_val
        step              += 1

        # ── Logging ───────────────────────────────────────────────────────
        if step % cfg_train.log_every == 0:
            avg_loss        = loss_accum        / cfg_train.log_every
            avg_loss_ir2red = loss_ir2red_accum / cfg_train.log_every
            avg_loss_red2ir = loss_red2ir_accum / cfg_train.log_every
            lr_now          = optimizer.param_groups[0]["lr"]
            print(f"step {step:>6}/{cfg_train.total_steps}  "
                  f"loss={avg_loss:.4f}  ir2red={avg_loss_ir2red:.4f}  red2ir={avg_loss_red2ir:.4f}  "
                  f"lr={lr_now:.2e}")
            if use_wandb:
                wandb.log({
                    "train/loss":        avg_loss,
                    "train/loss_ir2red": avg_loss_ir2red,
                    "train/loss_red2ir": avg_loss_red2ir,
                    "train/lr":          lr_now,
                    "train/grad_norm":   grad_norm.item(),
                }, step=step)
            loss_accum        = 0.0
            loss_ir2red_accum = 0.0
            loss_red2ir_accum = 0.0

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % cfg_train.save_every == 0 or step == cfg_train.total_steps:
            save_checkpoint(
                step, model, optimizer, loss.item(),
                os.path.join(ckpt_dir, f"step_{step:07d}.pt"),
            )
            save_checkpoint(
                step, model, optimizer, loss.item(),
                latest_ckpt,
            )
            if use_wandb:
                wandb.log({"checkpoint/step": step}, step=step)

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
