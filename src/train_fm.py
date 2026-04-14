"""
Flow Matching training script for HiRISE IR10 → RED4 (and RED4 → IR10).

Mirrors the structure of train.py but replaces the DDPM forward process
with rectified-flow linear interpolation and velocity-prediction loss.

Usage:
    cd src
    python train_fm.py --train_mode ir2red
    python train_fm.py --train_mode ir2red --no_wandb
"""

import os
import sys
import datetime
import argparse

import torch
import torch.nn.functional as F
import pandas as pd
import wandb

sys.path.insert(0, os.path.dirname(__file__))

from config import FMModelConfig, FMTrainConfig, DataConfig
from models.ir2red_fm import FMUNet as IR2REDFMUNet
from models.red2ir_fm import FMUNet as RED2IRFMUNet
from diffusion.fm_utils import fm_interpolate, fm_velocity_target
from eval_fm import evaluate_fm_unidirectional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import DiffusionDataset, diffusion_collate_fn, get_loader
from eval import get_val_split


# =============================================================================
# Checkpoint helpers  (identical to train.py)
# =============================================================================

def save_checkpoint(step, model, optimizer, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"step": step, "model": model.state_dict(),
         "optimizer": optimizer.state_dict(), "loss": loss},
        path,
    )
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  [ckpt] resumed from step {ckpt['step']} (loss={ckpt['loss']:.4f})")
    return ckpt["step"]


# =============================================================================
# Single training step
# =============================================================================

def fm_train_step(
    batch:      dict,
    model:      torch.nn.Module,
    device:     torch.device,
    train_mode: str,
) -> tuple:
    """
    One Flow Matching training step.

    1. Sample t ~ U(0, 1)
    2. x_t = (1-t)·noise + t·x_target
    3. v_target = x_target - noise
    4. loss = MSE(v_pred, v_target)

    Returns (loss_tensor, loss_ir2red_float, loss_red2ir_float).
    """
    ir  = batch["ir"].to(device)
    red = batch["red"].to(device)
    B   = ir.shape[0]

    if train_mode == "ir2red":
        x_source, x_target = ir, red
    elif train_mode == "red2ir":
        x_source, x_target = red, ir
    else:
        raise ValueError(f"Unsupported FM train_mode: {train_mode}")

    t     = torch.rand(B, device=device)
    noise = torch.randn_like(x_target)
    x_t      = fm_interpolate(x_target, noise, t)
    v_target = fm_velocity_target(x_target, noise)

    v_pred = model(x_t, x_source, t)
    loss   = F.mse_loss(v_pred, v_target)

    if train_mode == "ir2red":
        return loss, loss.item(), 0.0
    return loss, 0.0, loss.item()


# =============================================================================
# Main training loop
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Flow Matching training")
    parser.add_argument("--data_root",      default="")
    parser.add_argument("--csv_path",       default="")
    parser.add_argument("--ckpt_dir",       default="")
    parser.add_argument("--wandb_project",  default="HiRISE_diffusion")
    parser.add_argument("--run_name",       default=None)
    parser.add_argument("--no_wandb",       action="store_true")
    parser.add_argument(
        "--train_mode",
        default="ir2red",
        choices=["ir2red", "red2ir"],
        help="Training direction (unidirectional only)",
    )
    args = parser.parse_args()

    cfg_model = FMModelConfig()
    cfg_train = FMTrainConfig()
    cfg_data  = DataConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    if args.train_mode == "red2ir":
        model = RED2IRFMUNet(
            in_channels    = cfg_model.in_channels,
            out_channels   = cfg_model.out_channels,
            base_channels  = cfg_model.base_channels,
            num_res_blocks = cfg_model.num_res_blocks,
            dropout        = cfg_model.dropout,
            t_scale        = cfg_model.t_scale,
        ).to(device)
        model_tag = "red2ir_fm"
    elif args.train_mode == "ir2red":
        model = IR2REDFMUNet(
            in_channels    = cfg_model.in_channels,
            out_channels   = cfg_model.out_channels,
            base_channels  = cfg_model.base_channels,
            num_res_blocks = cfg_model.num_res_blocks,
            dropout        = cfg_model.dropout,
            t_scale        = cfg_model.t_scale,
        ).to(device)
        model_tag = "ir2red_fm"
    else:
        raise ValueError(f"Unsupported train_mode: {args.train_mode}")

    print(f"Training mode: {args.train_mode}  model={model_tag}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FMUNet parameters: {n_params:,}")

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
                "data_root":  data_root,
                "csv_path":   csv_path,
                "n_params":   n_params,
                "device":     str(device),
                "train_mode": args.train_mode,
                "model_tag":  model_tag,
            },
            resume="allow",
        )

    # ── Resume from checkpoint if available ───────────────────────────────
    run_ts   = datetime.datetime.now().strftime("%m%d_%H%M")
    base_dir = args.ckpt_dir or os.path.join(os.path.dirname(__file__), "output")
    ckpt_dir = os.path.join(base_dir, f"fm_{args.train_mode}_{run_ts}")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_step  = 0
    latest_ckpt = os.path.join(base_dir, f"latest_fm_{args.train_mode}.pt")
    if cfg_train.resume and os.path.exists(latest_ckpt):
        start_step = load_checkpoint(latest_ckpt, model, optimizer)
    elif not cfg_train.resume:
        print("  [ckpt] resume=False — starting from scratch")

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    step              = start_step
    loss_accum        = 0.0
    loss_ir2red_accum = 0.0
    loss_red2ir_accum = 0.0
    data_iter         = iter(loader)

    print(f"Training from step {step} to {cfg_train.total_steps} ...")

    while step < cfg_train.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch     = next(data_iter)

        optimizer.zero_grad()
        loss, loss_ir2red_val, loss_red2ir_val = fm_train_step(
            batch, model, device, args.train_mode,
        )
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
            val_results = evaluate_fm_unidirectional(
                model, val_loader, device, args.train_mode,
            )
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
