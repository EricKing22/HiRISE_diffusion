"""
CM-Diff inference with SCI (Statistical Constraint Inference).

Implements Algorithm 2 from the CM-Diff paper adapted for HiRISE IR10 ↔ RED4.

Algorithm per denoising step:
  1. Predict noise eps via U-Net
  2. Compute x_0_tilde (predicted clean image from x_t)
  3. Compute L_cons = lambda_scl * L_scl(x_0_tilde) + lambda_ccl * L_ccl(x_0_tilde)
  4. Compute gradient g = d(L_cons) / d(x_0_tilde)
  5. Compute posterior mean mu_q from DDPM posterior
  6. Correct:  mu_updated = mu_q + sigma_q * g
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# SCI loss functions
# =============================================================================

def soft_histogram(
    x:       torch.Tensor,       # any shape — treated as flat
    bins:    int,
    min_val: float = -1.0,
    max_val: float =  1.0,
) -> torch.Tensor:
    """
    Differentiable soft histogram via Gaussian bin assignment.

    Each pixel is softly assigned to nearby bins using a Gaussian kernel
    of width = bin_width, so the output is differentiable w.r.t. x.

    Returns normalised histogram [bins] that sums to ~1.
    """
    x_flat      = x.reshape(-1)
    bin_edges   = torch.linspace(min_val, max_val, bins + 1, device=x.device, dtype=x.dtype)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width   = (max_val - min_val) / bins

    # [N, bins] soft assignment weights
    diffs   = x_flat.unsqueeze(1) - bin_centers.unsqueeze(0)
    weights = torch.exp(-0.5 * (diffs / bin_width) ** 2)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    hist = weights.sum(dim=0)                    # [bins]
    hist = hist / (hist.sum() + 1e-8)            # normalise
    return hist


def sci_l_scl(
    x_0_tilde:   torch.Tensor,   # [B, 1, H, W]
    mu_prior:    torch.Tensor,   # scalar
    sigma_prior: torch.Tensor,   # scalar
) -> torch.Tensor:
    """
    Statistical Constraint Loss  (paper Eq. 14).

    L_scl = |mu_pred - mu_prior| + |sigma_pred - sigma_prior|

    Aligns first- and second-order statistics of the predicted image with
    the empirical statistics of the target domain training set.
    Single-channel version (no RGB loop needed for IR/RED images).
    """
    mu_pred    = x_0_tilde.mean()
    sigma_pred = x_0_tilde.std()
    return (mu_pred - mu_prior).abs() + (sigma_pred - sigma_prior).abs()


def sci_l_ccl(
    x_0_tilde: torch.Tensor,   # [B, 1, H, W]
    h_prior:   torch.Tensor,   # [bins]  normalised histogram from training set
    bins:      int,
) -> torch.Tensor:
    """
    Channel Constraint Loss  (paper Eq. 13).

    L_ccl = sum_i  (h_pred_i - h_prior_i)^2 / (h_pred_i + h_prior_i + eps)

    Chi-squared distance between predicted-image histogram and target-domain
    prior histogram computed from the training set.
    Single-channel version.
    """
    h_pred = soft_histogram(x_0_tilde, bins)
    eps    = 1e-6
    return ((h_pred - h_prior) ** 2 / (h_pred + h_prior + eps)).sum()


# =============================================================================
# Prior statistics  (pre-computed from training set target domain)
# =============================================================================

def compute_prior_stats(
    tensors: list,                          # list of [1,H,W] or [H,W] tensors
    bins:    int            = 256,
    device:  torch.device  = torch.device("cpu"),
) -> dict:
    """
    Compute mu, sigma, and histogram over all pixels of the target domain.

    Call once on the training split of the target modality and save with
    save_prior_stats() so inference can load it without the dataset.
    """
    all_pixels = torch.cat([t.reshape(-1).to(device) for t in tensors])
    mu    = all_pixels.mean()
    sigma = all_pixels.std()
    h     = soft_histogram(all_pixels, bins)
    return {"mu": mu, "sigma": sigma, "histogram": h, "bins": torch.tensor(bins)}


def save_prior_stats(stats: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(stats, path)
    print(f"[prior] saved → {path}")


def load_prior_stats(path: str, device: torch.device) -> dict:
    data = torch.load(path, map_location=device)
    return {k: v.to(device) for k, v in data.items()}


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

    #   mu_q = [ (1-α_t)(1-ᾱ_{t-1})·x_t  +  √ᾱ_{t-1}·β_t·x̃_0 ] / (1-ᾱ_t)
    mu_q = (
        (1.0 - scheduler.gather(scheduler.alphas, t_batch)) * (1.0 - alpha_bar_prev) * x_t.detach()
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
        )

    grad = torch.autograd.grad(loss_cons, x_leaf)[0]   # [B,1,H,W]
    mu_updated = mu_q + sigma_q * grad.detach()

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
) -> torch.Tensor:
    """
    Full T-step reverse diffusion with SCI.

    Returns generated image x_0  [B, 1, H, W].
    """
    model.eval()
    B, _, H, W = x_source.shape

    edge        = sobel_edge(x_source)
    direction_t = torch.full((B,), direction, dtype=torch.long, device=device)

    # Start from pure Gaussian noise
    x_t = torch.randn(B, 1, H, W, device=device)

    for t in reversed(range(cfg_inf.timesteps)):
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)

        # enable_grad is required for the SCI autograd step inside ddpm_step_sci
        with torch.enable_grad():
            x_t = ddpm_step_sci(
                model, scheduler, x_t, x_source, edge,
                direction_t, t_batch, prior_stats, cfg_inf,
            )

        if t % 100 == 0:
            print(f"  t={t:4d}  x range=[{x_t.min().item():.3f}, {x_t.max().item():.3f}]")

    return x_t


# =============================================================================
# CLI entry-point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CM-Diff SCI inference")
    parser.add_argument("--source",     required=True,
                        help="Source image path (.npy), normalised to [-1, 1]")
    parser.add_argument("--direction",  type=int, required=True,
                        help="Translation direction: 0=IR→RED, 1=RED→IR")
    parser.add_argument("--checkpoint", default="../checkpoints/latest.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--prior",      default="../checkpoints/prior_stats.pt",
                        help="Path to prior statistics file (.pt)")
    parser.add_argument("--output",     default="../outputs/result.npy",
                        help="Output path (.npy)")
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
    if not os.path.exists(args.prior):
        raise FileNotFoundError(
            f"Prior stats not found at '{args.prior}'.\n"
            "Compute them from your training set with compute_prior_stats() "
            "and save with save_prior_stats()."
        )
    prior_stats = load_prior_stats(args.prior, device)
    print(f"Prior stats:  mu={prior_stats['mu'].item():.4f}  "
          f"sigma={prior_stats['sigma'].item():.4f}  "
          f"bins={int(prior_stats['bins'].item())}")

    # ── Source image ───────────────────────────────────────────────────────────
    src_np   = np.load(args.source).astype(np.float32)
    x_source = torch.from_numpy(src_np)
    if x_source.ndim == 2:
        x_source = x_source[None, None]          # [1,1,H,W]
    elif x_source.ndim == 3:
        x_source = x_source[None]                # [1,1,H,W]
    x_source = x_source.to(device)
    print(f"Source image:  shape={tuple(x_source.shape)}  "
          f"range=[{x_source.min().item():.3f}, {x_source.max().item():.3f}]")

    # ── Inference ──────────────────────────────────────────────────────────────
    direction_name = {0: "IR10 → RED4", 1: "RED4 → IR10"}[args.direction]
    print(f"Sampling ({direction_name}, T={cfg_inf.timesteps}) ...")

    with torch.no_grad():
        result = sample(model, scheduler, x_source, args.direction,
                        prior_stats, cfg_inf, device)

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_np = result.squeeze().cpu().numpy()
    np.save(args.output, out_np)
    print(f"Saved → {args.output}  "
          f"shape={out_np.shape}  "
          f"range=[{out_np.min():.3f}, {out_np.max():.3f}]")


if __name__ == "__main__":
    main()
