"""
CM-Diff inference with SCI (Statistical Constraint Inference).

Implements Algorithm 2 from the CM-Diff paper adapted for HiRISE IR10 ↔ RED4.

Algorithm per denoising step:
  1. Predict noise eps via U-Net
  2. Compute x_0_tilde (predicted clean image from x_t)
  3. Compute L_cons = lambda_scl * L_scl(x_0_tilde) + lambda_ccl * L_ccl(x_0_tilde)
  4. Compute gradient g = d(L_cons) / d(x_0_tilde)
  5. Compute posterior mean mu_q from DDPM posterior
  6. Correct:  mu_updated = mu_q - sigma_q * g
  7. Sample x_{t-1} ~ N(mu_updated, sigma_q)    [t=0: no noise, return mu]

Direction labels:
    0 = IR10 → RED4
    1 = RED4 → IR10

Usage:
    cd src
    python inference.py \\
        --source   path/to/source.npy   \\
        --direction 0                   \\
        --checkpoint ../checkpoints/latest.pt \\
        --prior      ../checkpoints/prior_stats_red.pt \\
        --output     ../outputs/result.npy
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import ModelConfig, InferenceConfig
from models.cm_diff_unet import UNet
from diffusion.scheduler import DDPMScheduler
from diffusion.process import sobel_edge
from compute_prior import (
    sci_l_scl, sci_l_ccl,
    load_prior_stats,
    compute_prior_from_dataset,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# Single denoising step with SCI correction
# =============================================================================

def ddpm_step_sci(
    model:       torch.nn.Module,
    scheduler:   DDPMScheduler,
    x_t:         torch.Tensor,       # [B, 1, H, W]  current noisy image
    x_source:    torch.Tensor,       # [B, 1, H, W]  source modality (clean)
    edge:        torch.Tensor,       # [B, 1, H, W]  edge map of source
    direction:   torch.Tensor,       # [B]  long (0 or 1)
    t_batch:     torch.Tensor,       # [B]  long  current timestep indices
    prior_stats: dict,
    cfg_inf:     InferenceConfig,
) -> torch.Tensor:
    """
    One reverse diffusion step t → t-1 with SCI gradient correction.

    Steps:
      1. eps_pred  = U-Net(x_t, x_source, edge, direction, t)
      2. x_0_tilde = (x_t - sqrt(1-ab_t) * eps_pred) / sqrt(ab_t)
      3. mu_q      = DDPM posterior mean  (Eq. 11)
      4. g         = grad_{x_0_tilde}(L_cons)
      5. mu_upd    = mu_q + sigma_q * g
      6. x_{t-1}  ~ N(mu_upd, sigma_q)   [pure mu at t=0]
    """

    # ── 1. Noise prediction ───────────────────────────────────────────────────
    with torch.no_grad():
        eps_pred = model(x_t, x_source, edge, direction, t_batch)   # [B,1,H,W]

    # ── 2. Predicted clean image x_0_tilde ────────────────────────────────────
    sqrt_ab     = scheduler.gather(scheduler.sqrt_alpha_bar,    t_batch)   # [B,1,1,1]
    sqrt_one_ab = scheduler.gather(scheduler.sqrt_one_minus_ab, t_batch)   # [B,1,1,1]
    x_0_tilde   = (x_t.detach() - sqrt_one_ab * eps_pred.detach()) / (sqrt_ab + 1e-8)

    # ── 3. Posterior mean mu_q and variance sigma_q  (paper Eq. 11) ───────────
    beta_t         = scheduler.gather(scheduler.betas,          t_batch)   # [B,1,1,1]
    alpha_bar_t    = scheduler.gather(scheduler.alpha_bar,      t_batch)
    alpha_bar_prev = scheduler.gather(scheduler.alpha_bar_prev, t_batch)
    sigma_q        = scheduler.gather(scheduler.posterior_var,  t_batch)   # β̃_t

    #   mu_q = [ √α_t·(1-ᾱ_{t-1})·x_t  +  √ᾱ_{t-1}·β_t·x̃_0 ] / (1-ᾱ_t)
    sqrt_alpha_t = torch.sqrt(scheduler.gather(scheduler.alphas, t_batch))
    mu_q = (
        sqrt_alpha_t * (1.0 - alpha_bar_prev) * x_t.detach()
        + torch.sqrt(alpha_bar_prev) * beta_t * x_0_tilde.detach()
    ) / (1.0 - alpha_bar_t + 1e-8)

    # ── 4. SCI gradient correction ────────────────────────────────────────────
    x_leaf = x_0_tilde.detach().requires_grad_(True)

    loss_cons = torch.zeros(1, device=x_t.device)
    if cfg_inf.lambda_scl > 0.0:
        loss_cons = loss_cons + cfg_inf.lambda_scl * sci_l_scl(
            x_leaf,
            prior_stats["mu"],
            prior_stats["sigma"],
        )
    if cfg_inf.lambda_ccl > 0.0:
        loss_cons = loss_cons + cfg_inf.lambda_ccl * sci_l_ccl(
            x_leaf,
            prior_stats["histogram"],
            int(prior_stats["bins"].item()),
            hist_min=float(prior_stats["hist_min"].item()),
            hist_max=float(prior_stats["hist_max"].item()),
        )

    if loss_cons.requires_grad:
        grad = torch.autograd.grad(loss_cons, x_leaf)[0]   # [B,1,H,W]
        mu_updated = mu_q - sigma_q * grad.detach()
    else:
        mu_updated = mu_q   # SCI disabled — no correction

    # ── 5. Sample x_{t-1} ─────────────────────────────────────────────────────
    if t_batch[0].item() > 0:
        noise  = torch.randn_like(x_t)
        x_prev = mu_updated + torch.sqrt(sigma_q.clamp(min=1e-10)) * noise
    else:
        x_prev = mu_updated   # final step — deterministic

    return x_prev.detach()


# =============================================================================
# Full reverse diffusion loop
# =============================================================================

def sample(
    model:       torch.nn.Module,
    scheduler:   DDPMScheduler,
    x_source:    torch.Tensor,     # [B, 1, H, W]
    direction:   int,              # 0 = IR→RED,  1 = RED→IR
    prior_stats: dict,
    cfg_inf:     InferenceConfig,
    device:      torch.device,
    verbose:     bool = True,
) -> torch.Tensor:
    """
    Full T-step reverse diffusion with SCI.

    Returns generated image x_0  [B, 1, H, W].
    """
    model.eval()
    B, _, H, W = x_source.shape

    edge        = sobel_edge(x_source)
    direction_t = torch.full((B,), direction, dtype=torch.long, device=device)

    # Initialise x_T to match the training forward-process marginal at t=T:
    #   x_T = sqrt(ᾱ_T) * x_target + sqrt(1-ᾱ_T) * ε
    # The target is unknown at inference time, so we approximate E[x_target]
    # with prior_stats["mu"] (the target-domain mean computed from training data).
    #
    # IMPORTANT: must NOT use x_source here.  x_source belongs to the SOURCE
    # domain (e.g. RED4 for direction 1) whose mean differs from the target.
    # Seeding x_T with x_source injects a DC bias of sqrt(ᾱ_T)*(μ_source−μ_target),
    # which the first x̃₀ computation amplifies by 1/sqrt(ᾱ_T) ≈ 167×.
    t_max         = cfg_inf.timesteps - 1
    sqrt_ab_T     = scheduler.sqrt_alpha_bar[t_max].item()
    sqrt_one_ab_T = scheduler.sqrt_one_minus_ab[t_max].item()
    mu_target     = prior_stats["mu"].item()
    x_t = sqrt_ab_T * mu_target + sqrt_one_ab_T * torch.randn(B, 1, H, W, device=device)
    if verbose:
        print(f"[init] sqrt_ab_T={sqrt_ab_T:.4f}  mu_target={mu_target:.4f}  x_T mean≈{x_t.mean().item():.4f}")

    for t in reversed(range(cfg_inf.timesteps)):
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)

        # enable_grad is required for the SCI autograd step inside ddpm_step_sci
        with torch.enable_grad():
            x_t = ddpm_step_sci(
                model, scheduler, x_t, x_source, edge,
                direction_t, t_batch, prior_stats, cfg_inf,
            )

        if verbose and t % 100 == 0:
            print(f"  t={t:4d}  x range=[{x_t.min().item():.3f}, {x_t.max().item():.3f}]")

    return x_t


# =============================================================================
# CLI entry-point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CM-Diff SCI inference")
    parser.add_argument("--source",     required=True,
                        help="Source image path (.npy), normalised to [-1, 1]")
    parser.add_argument("--direction",  type=int, required=True, choices=[0, 1],
                        help="Translation direction: 0=IR→RED, 1=RED→IR")
    parser.add_argument("--checkpoint", default="src/output/latest.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--prior_dir",  default="src/output",
                        help="Directory containing prior_ir.pt and prior_red.pt")
    parser.add_argument("--data_root",  default="",
                        help="Root directory for .npy files (only needed if prior must be computed)")
    parser.add_argument("--csv_path",   default="",
                        help="Path to data_record CSV (only needed if prior must be computed)")
    parser.add_argument("--output",     default="../outputs/result.npy",
                        help="Output path (.npy)")
    parser.add_argument("--ground_truth", default="",
                        help="Ground truth target image (.npy) for comparison (optional)")
    parser.add_argument("--device",     default="cuda",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg_model = ModelConfig()
    cfg_inf   = InferenceConfig()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet(
        in_channels   = cfg_model.in_channels,
        out_channels  = cfg_model.out_channels,
        base_channels = cfg_model.base_channels,
        num_res_blocks= cfg_model.num_res_blocks,
        dropout       = cfg_model.dropout,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint  step={ckpt.get('step', '?')}  loss={ckpt.get('loss', '?'):.4f}")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = DDPMScheduler(
        timesteps  = cfg_model.timesteps,
        beta_start = cfg_model.beta_start,
        beta_end   = cfg_model.beta_end,
    ).to(device)

    # ── Prior statistics ───────────────────────────────────────────────────────
    # direction=0 (IR→RED): target domain is RED4  → prior_red.pt
    # direction=1 (RED→IR): target domain is IR10  → prior_ir.pt
    prior_name = "prior_red.pt" if args.direction == 0 else "prior_ir.pt"
    prior_path = os.path.join(args.prior_dir, prior_name)

    if not os.path.exists(prior_path):
        print(f"[prior] {prior_path} not found — computing from dataset ...")
        prior_ir, prior_red = compute_prior_from_dataset(
            args.prior_dir, device, data_root=args.data_root, csv_path=args.csv_path
        )
        prior_stats = prior_red if args.direction == 0 else prior_ir
    else:
        prior_stats = load_prior_stats(prior_path, device)

    print(f"[prior] mu={prior_stats['mu'].item():.4f}  "
          f"sigma={prior_stats['sigma'].item():.4f}  "
          f"bins={int(prior_stats['bins'].item())}")

    # ── Source image ───────────────────────────────────────────────────────────
    src_np   = np.load(args.source).astype(np.float32)
    x_source = torch.from_numpy(src_np)

    # Ensure shape is [1, 1, H, W]
    if x_source.ndim == 2:
        x_source = x_source[None, None]          # [H,W] → [1,1,H,W]
    elif x_source.ndim == 3:
        x_source = x_source[None, :1]            # [C,H,W] → [1,1,H,W] (take first channel)
    elif x_source.ndim == 4:
        x_source = x_source[:, :1]               # [B,C,H,W] → [B,1,H,W]

    # Shared per-scene normalisation using IR10 statistics — matches DiffusionDataset.
    # For direction=0 (source=IR10): compute stats directly from source.
    # For direction=1 (source=RED4): load companion IR10 to get the same stats.
    src_path = args.source
    if "_IR10.npy" in src_path:
        ir10_path = src_path
    elif "_RED4.npy" in src_path:
        ir10_path = src_path.replace("_RED4.npy", "_IR10.npy")
    else:
        ir10_path = None

    if ir10_path and os.path.exists(ir10_path):
        ir10_t = torch.from_numpy(np.load(ir10_path).astype(np.float32))
        flat   = ir10_t.reshape(-1)
        print(f"[norm] IR10 stats from {os.path.basename(ir10_path)}")
    else:
        flat = x_source.reshape(-1)
        print(f"[norm] IR10 companion not found — using source-only stats")

    center = flat.median()
    mad    = (flat - center).abs().median()
    scale  = (1.4826 * mad).clamp_min(0.05)
    x_source = ((x_source - center) / scale).clamp(-10, 10)

    # Method B: subtract IR10 normalized mean so scene enters UNet with IR mean≈0
    # flat already holds the raw IR10 pixels (or source pixels as fallback)
    dc = ((flat - center) / scale).clamp(-10, 10).mean()
    x_source = x_source - dc
    print(f"[norm] center={center.item():.4f}  scale={scale.item():.4f}  dc={dc.item():.4f}")

    x_source = x_source.to(device)
    print(f"Source image:  shape={tuple(x_source.shape)}  "
          f"range=[{x_source.min().item():.3f}, {x_source.max().item():.3f}]")

    # ── Inference ──────────────────────────────────────────────────────────────
    direction_name = {0: "IR10 → RED4", 1: "RED4 → IR10"}[args.direction]
    print(f"Sampling ({direction_name}, T={cfg_inf.timesteps}) ...")

    with torch.no_grad():
        result = sample(model, scheduler, x_source, args.direction,
                        prior_stats, cfg_inf, device)

    # ── Restore dc offset ─────────────────────────────────────────────────────
    result = result + dc.to(device)

    # ── Save ───────────────────────────────────────────────────────────────────
    out_np = result.squeeze().cpu().numpy()           # [H, W]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, out_np)
    print(f"Saved → {args.output}  shape={out_np.shape}  range=[{out_np.min():.3f}, {out_np.max():.3f}]")


if __name__ == "__main__":
    main()
