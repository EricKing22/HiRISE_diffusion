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
import datetime
import torch
import torch.nn.functional as F
import pandas as pd
import wandb

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from config import ModelConfig, TrainConfig, DataConfig
from models.cm_diff_unet import UNet as BidirectionalUNet
from models.ir2red_ddpm import UNet as IR2REDUNet
from models.red2ir_ddpm import UNet as RED2IRUNet
from diffusion.scheduler import DDPMScheduler
from diffusion.process import q_sample, sobel_edge

# Add project root for data imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.dataset import DiffusionDataset, diffusion_collate_fn, get_loader
from eval import evaluate, evaluate_unidirectional, get_val_split


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
    train_mode: str,
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

    if train_mode == "bidirectional":
        half      = B // 2
        direction = torch.cat([
            torch.zeros(half,       dtype=torch.long, device=device),
            torch.ones(B - half,    dtype=torch.long, device=device),
        ])

        mask     = (direction == 0)[:, None, None, None]
        x_target = torch.where(mask, red, ir)
        x_source = torch.where(mask, ir, red)

        with torch.no_grad():
            edge = sobel_edge(x_source)

        t     = torch.randint(0, scheduler.timesteps, (B,), device=device)
        noise = torch.randn_like(x_target)
        x_t   = q_sample(scheduler, x_target, t, noise)
        eps_pred = model(x_t, x_source, edge, direction, t)

        loss_per_sample = F.mse_loss(eps_pred, noise, reduction="none").mean(dim=[1, 2, 3])
        mask_1d = (direction == 0)
        loss_ir2red = loss_per_sample[mask_1d].mean() if mask_1d.any() else torch.tensor(0.0, device=device)
        loss_red2ir = loss_per_sample[~mask_1d].mean() if (~mask_1d).any() else torch.tensor(0.0, device=device)
        l_joint = cfg_train.lambda_ir_to_red * loss_ir2red + cfg_train.lambda_red_to_ir * loss_red2ir
        return l_joint, loss_ir2red.item(), loss_red2ir.item()

    if train_mode == "ir2red":
        x_source, x_target = ir, red
    elif train_mode == "red2ir":
        x_source, x_target = red, ir
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    with torch.no_grad():
        edge = sobel_edge(x_source)

    t     = torch.randint(0, scheduler.timesteps, (B,), device=device)
    noise = torch.randn_like(x_target)
    x_t   = q_sample(scheduler, x_target, t, noise)
    eps_pred = model(x_t, x_source, edge, t)
    loss = F.mse_loss(eps_pred, noise)

    if train_mode == "ir2red":
        return loss, loss.item(), 0.0
    return loss, 0.0, loss.item()


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
    parser.add_argument(
        "--train_mode",
        default="bidirectional",
        choices=["bidirectional", "ir2red", "red2ir"],
        help="Training mode: shared bidirectional model or single-direction DDPM",
    )
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
    if args.train_mode == "bidirectional":
        model = BidirectionalUNet(
            in_channels=cfg_model.in_channels,
            out_channels=cfg_model.out_channels,
            base_channels=cfg_model.base_channels,
            num_res_blocks=cfg_model.num_res_blocks,
            dropout=cfg_model.dropout,
        ).to(device)
        model_tag = "cm_diff_unet"
    elif args.train_mode == "ir2red":
        model = IR2REDUNet(
            in_channels=cfg_model.in_channels,
            out_channels=cfg_model.out_channels,
            base_channels=cfg_model.base_channels,
            num_res_blocks=cfg_model.num_res_blocks,
            dropout=cfg_model.dropout,
        ).to(device)
        model_tag = "ir2red_ddpm"
    else:
        model = RED2IRUNet(
            in_channels=cfg_model.in_channels,
            out_channels=cfg_model.out_channels,
            base_channels=cfg_model.base_channels,
            num_res_blocks=cfg_model.num_res_blocks,
            dropout=cfg_model.dropout,
        ).to(device)
        model_tag = "red2ir_ddpm"

    print(f"Training mode: {args.train_mode}  model={model_tag}")

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

    # ── 8:2 train/val split at Observation level (fixed seed) ─────────────
    train_sets, val_sets = get_val_split(dr)
    train_obs = dr[dr["Set"].isin(train_sets)]["Observation"].unique()
    val_obs   = dr[dr["Set"].isin(val_sets)]["Observation"].unique()

    train_dataset = DiffusionDataset(
        data_record=dr, data_root=data_root, sweep=True,
        allowed_sets=train_sets,
    )
    val_dataset = DiffusionDataset(
        data_record=dr, data_root=data_root, sweep=True,
        allowed_sets=val_sets,
    )

    loader = get_loader(
        train_dataset,
        batch_size=cfg_train.batch_size,
        collate_fn=diffusion_collate_fn,
        num_workers=4,
        shuffle=True,
    )
    val_loader = get_loader(
        val_dataset,
        batch_size=cfg_train.batch_size,
        collate_fn=diffusion_collate_fn,
        num_workers=4,
        shuffle=False,
    )
    print(f"Train: {len(train_dataset)} sets ({len(train_obs)} obs)  "
          f"Val: {len(val_dataset)} sets ({len(val_obs)} obs)  "
          f"batch_size={cfg_train.batch_size}")

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
                "train_mode": args.train_mode,
                "model_tag": model_tag,
            },
            resume="allow",
        )

    # ── Resume from checkpoint if available ───────────────────────────────
    run_ts   = datetime.datetime.now().strftime("%m%d_%H%M")
    base_dir = args.ckpt_dir or os.path.join(os.path.dirname(__file__), "output")
    ckpt_dir = os.path.join(base_dir, f"{args.train_mode}_{run_ts}")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_step  = 0
    latest_ckpt = os.path.join(base_dir, f"latest_{args.train_mode}.pt")
    if cfg_train.resume and os.path.exists(latest_ckpt):
        start_step = load_checkpoint(latest_ckpt, model, optimizer)
    elif not cfg_train.resume:
        print("  [ckpt] resume=False — starting from scratch")

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
        loss, loss_ir2red_val, loss_red2ir_val = train_step(batch, model, scheduler, device, cfg_train, args.train_mode)
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

        # ── Validation ────────────────────────────────────────────────────
        if step % cfg_train.val_every == 0:
            if args.train_mode == "bidirectional":
                val_results = evaluate(model, scheduler, val_loader, device)
            else:
                val_results = evaluate_unidirectional(model, scheduler, val_loader, device, args.train_mode)

            print(f"  [val] step {step:>6}  loss={val_results['loss']:.4f}  "
                  f"ir2red={val_results['loss_ir2red']:.4f}  red2ir={val_results['loss_red2ir']:.4f}")
            if use_wandb:
                wandb.log({
                    "val/loss":        val_results["loss"],
                    "val/loss_ir2red": val_results["loss_ir2red"],
                    "val/loss_red2ir": val_results["loss_red2ir"],
                }, step=step)
            model.train()

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
