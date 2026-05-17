"""
Flow Matching inference via Euler ODE integration.

Generates a target-domain image from a source image using a trained FM velocity
network. Starts from x_0 ~ N(0, I) and integrates the learned velocity field to
x_1 (clean image). Optional SGI guidance can constrain x_1 estimates with
global target-domain priors.

Usage:
    cd src
    python inference_fm.py \
        --source   path/to/source_IR10.npy \
        --checkpoint output/latest_fm_ir2red.pt \
        --num_steps 50 \
        --output   ../outputs/fm_result.npy
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from config import FMModelConfig, FMInferenceConfig
from models import IR2REDFMUNet, RED2IRFMUNet, BidirectionalFMUNet
from diffusion.fm_utils import fm_euler_step
from compute_prior import load_prior_stats, sci_l_scl, sci_l_ccl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# Full Euler ODE sampling loop
# =============================================================================

def _sgi_loss(
    x1_hat: torch.Tensor,
    prior_stats: dict,
    cfg_inf: FMInferenceConfig,
) -> torch.Tensor:
    loss_sgi = x1_hat.new_tensor(0.0)
    if cfg_inf.lambda_sgi_scl > 0.0:
        loss_sgi = loss_sgi + cfg_inf.lambda_sgi_scl * sci_l_scl(
            x1_hat, prior_stats["mu"], prior_stats["sigma"],
        )
    if cfg_inf.lambda_sgi_ccl > 0.0:
        loss_sgi = loss_sgi + cfg_inf.lambda_sgi_ccl * sci_l_ccl(
            x1_hat,
            prior_stats["histogram"],
            int(prior_stats["bins"].item()),
            hist_min=float(prior_stats["hist_min"].item()),
            hist_max=float(prior_stats["hist_max"].item()),
        )
    return loss_sgi


def _sgi_loss_parts(
    x1_hat: torch.Tensor,
    prior_stats: dict,
    cfg_inf: FMInferenceConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_scl = x1_hat.new_tensor(0.0)
    loss_ccl = x1_hat.new_tensor(0.0)
    if cfg_inf.lambda_sgi_scl > 0.0:
        loss_scl = sci_l_scl(x1_hat, prior_stats["mu"], prior_stats["sigma"])
    if cfg_inf.lambda_sgi_ccl > 0.0:
        loss_ccl = sci_l_ccl(
            x1_hat,
            prior_stats["histogram"],
            int(prior_stats["bins"].item()),
            hist_min=float(prior_stats["hist_min"].item()),
            hist_max=float(prior_stats["hist_max"].item()),
        )
    loss_total = cfg_inf.lambda_sgi_scl * loss_scl + cfg_inf.lambda_sgi_ccl * loss_ccl
    return loss_total, loss_scl, loss_ccl


def _tensor_rms(x: torch.Tensor) -> torch.Tensor:
    return x.square().mean(dim=[1, 2, 3]).sqrt()


def _sgi_guidance_correction(
    grad: torch.Tensor,
    v_pred: torch.Tensor,
    lambda_t: float,
    cfg_inf: FMInferenceConfig,
) -> torch.Tensor:
    """
    Convert the SGI loss gradient into the actual correction applied at this
    Euler step. Ratio scaling is the experimental default because the raw
    global-statistic gradient was orders of magnitude smaller than the FM
    velocity, even with large SGI loss weights.
    """
    if cfg_inf.sgi_scale_mode == "raw":
        return float(lambda_t) * grad

    if cfg_inf.sgi_scale_mode == "ratio":
        grad_norm = _tensor_rms(grad).view(-1, 1, 1, 1).clamp_min(1e-12)
        v_norm = _tensor_rms(v_pred.detach()).view(-1, 1, 1, 1).clamp_min(1e-12)
        target_norm = float(lambda_t) * float(cfg_inf.sgi_guidance_ratio) * v_norm
        return target_norm * grad / grad_norm

    raise ValueError(f"Unsupported FM SGI scale mode: {cfg_inf.sgi_scale_mode}")


def _append_sgi_diagnostics(
    diagnostics: list[dict],
    *,
    step_idx: int,
    t_val: float,
    lambda_t: float,
    v_pred: torch.Tensor,
    grad: torch.Tensor,
    guidance_correction: torch.Tensor,
    loss_scl: torch.Tensor,
    loss_ccl: torch.Tensor,
    x1_hat: torch.Tensor,
    prior_stats: dict,
) -> None:
    with torch.no_grad():
        v_norm = _tensor_rms(v_pred.detach())
        grad_norm = _tensor_rms(grad.detach())
        guided_norm = _tensor_rms(guidance_correction.detach())
        ratio = guided_norm / v_norm.clamp_min(1e-12)
        cosine = F.cosine_similarity(
            v_pred.detach().flatten(1),
            (-grad.detach()).flatten(1),
            dim=1,
            eps=1e-12,
        )
        x1_flat = x1_hat.detach().flatten(1)
        diagnostics.append(dict(
            step=step_idx,
            t=t_val,
            lambda_t=float(lambda_t),
            v_norm=float(v_norm.mean().item()),
            grad_norm=float(grad_norm.mean().item()),
            guided_norm=float(guided_norm.mean().item()),
            guided_to_v=float(ratio.mean().item()),
            cosine=float(cosine.mean().item()),
            loss_scl=float(loss_scl.detach().item()),
            loss_ccl=float(loss_ccl.detach().item()),
            x1_mean=float(x1_flat.mean(dim=1).mean().item()),
            x1_sigma=float(x1_flat.std(dim=1, unbiased=False).mean().item()),
            prior_mean=float(prior_stats["mu"].detach().item()),
            prior_sigma=float(prior_stats["sigma"].detach().item()),
        ))


def sample_fm(
    model:    torch.nn.Module,
    x_source: torch.Tensor,       # [B, 1, H, W]
    cfg_inf:  FMInferenceConfig,
    device:   torch.device,
    direction=None,               # int or [B] tensor for bidirectional models
    prior_stats: dict | None = None,
    verbose:  bool = True,
    return_diagnostics: bool = False,
) -> torch.Tensor:
    """
    FM inference via Euler ODE integration.

    Starts from x_0 ~ N(0, I)  (t = 0, pure noise)
    Integrates to x_1          (t = 1, clean image)

    Returns generated image x_1  [B, 1, H, W].
    """
    model.eval()
    B, _, H, W = x_source.shape

    x_t = torch.randn(B, 1, H, W, device=device)
    dt  = 1.0 / cfg_inf.num_steps
    use_guidance = (
        prior_stats is not None
        and (cfg_inf.lambda_sgi_scl > 0.0 or cfg_inf.lambda_sgi_ccl > 0.0)
    )
    diagnostics: list[dict] = []
    diagnostic_every = max(1, int(getattr(cfg_inf, "sgi_diagnostic_every", 5)))

    if direction is None:
        direction_t = None
    elif torch.is_tensor(direction):
        direction_t = direction.to(device=device, dtype=torch.long)
    else:
        direction_t = torch.full((B,), int(direction), dtype=torch.long, device=device)

    def model_forward(x_state: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
        if direction_t is None:
            return model(x_state, x_source, t_batch)
        return model(x_state, x_source, direction_t, t_batch)

    if verbose:
        guide_msg = (
            f", SGI mode={cfg_inf.sgi_mode} "
            f"λ_scl={cfg_inf.lambda_sgi_scl:g} λ_ccl={cfg_inf.lambda_sgi_ccl:g} "
            f"scale={cfg_inf.sgi_scale_mode}"
            f"{f' ratio={cfg_inf.sgi_guidance_ratio:g}' if cfg_inf.sgi_scale_mode == 'ratio' else ''}"
            if use_guidance else ""
        )
        print(f"[FM] Euler ODE: {cfg_inf.num_steps} steps, dt={dt:.4f}{guide_msg}")

    for step_idx in range(cfg_inf.num_steps):
        t_val   = step_idx * dt
        t_batch = torch.full((B,), t_val, device=device)

        if use_guidance:
            with torch.enable_grad():
                lambda_t = float(t_val ** cfg_inf.sgi_schedule_power)
                if cfg_inf.sgi_mode == "velocity":
                    x_t_req = x_t.detach().requires_grad_(True)
                    v_pred = model_forward(x_t_req, t_batch)
                    x1_hat = x_t_req + (1.0 - t_val) * v_pred
                    loss_sgi, loss_scl, loss_ccl = _sgi_loss_parts(x1_hat, prior_stats, cfg_inf)
                    grad = torch.autograd.grad(loss_sgi, x_t_req, retain_graph=False)[0]
                    guidance_correction = _sgi_guidance_correction(grad, v_pred, lambda_t, cfg_inf)
                    if return_diagnostics and step_idx % diagnostic_every == 0:
                        _append_sgi_diagnostics(
                            diagnostics,
                            step_idx=step_idx,
                            t_val=t_val,
                            lambda_t=lambda_t,
                            v_pred=v_pred,
                            grad=grad,
                            guidance_correction=guidance_correction,
                            loss_scl=loss_scl,
                            loss_ccl=loss_ccl,
                            x1_hat=x1_hat,
                            prior_stats=prior_stats,
                        )
                    v_pred = (v_pred - guidance_correction).detach()
                elif cfg_inf.sgi_mode == "reproject":
                    x_t_req = x_t.detach().requires_grad_(True)
                    v_pred = model_forward(x_t_req, t_batch)
                    x1_hat = x_t_req + (1.0 - t_val) * v_pred
                    noise_hat = x_t_req - t_val * v_pred
                    loss_sgi, loss_scl, loss_ccl = _sgi_loss_parts(x1_hat, prior_stats, cfg_inf)
                    grad_x1 = torch.autograd.grad(loss_sgi, x1_hat, retain_graph=False)[0]
                    guidance_correction = _sgi_guidance_correction(grad_x1, v_pred, lambda_t, cfg_inf)
                    if return_diagnostics and step_idx % diagnostic_every == 0:
                        _append_sgi_diagnostics(
                            diagnostics,
                            step_idx=step_idx,
                            t_val=t_val,
                            lambda_t=lambda_t,
                            v_pred=v_pred,
                            grad=grad_x1,
                            guidance_correction=guidance_correction,
                            loss_scl=loss_scl,
                            loss_ccl=loss_ccl,
                            x1_hat=x1_hat,
                            prior_stats=prior_stats,
                        )
                    x1_guided = x1_hat - guidance_correction
                    x_t = ((1.0 - t_val) * noise_hat + t_val * x1_guided).detach()
                    with torch.no_grad():
                        v_pred = model_forward(x_t, t_batch)
                else:
                    raise ValueError(f"Unsupported FM SGI mode: {cfg_inf.sgi_mode}")
        else:
            with torch.no_grad():
                v_pred = model_forward(x_t, t_batch)

        x_t = fm_euler_step(v_pred, x_t, dt)

        if verbose and step_idx % max(1, cfg_inf.num_steps // 10) == 0:
            print(f"  step {step_idx:>4}/{cfg_inf.num_steps}  t={t_val:.3f}  "
                  f"x range=[{x_t.min().item():.3f}, {x_t.max().item():.3f}]")

    if return_diagnostics:
        return x_t, diagnostics
    return x_t


# =============================================================================
# CLI entry-point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Flow Matching inference")
    parser.add_argument("--source",      required=True,
                        help="Source image path (.npy)")
    parser.add_argument("--checkpoint",  default="src/output/latest_fm_ir2red.pt",
                        help="Path to FM model checkpoint (.pt)")
    parser.add_argument("--train_mode",  default="ir2red",
                        choices=["bidirectional", "ir2red", "red2ir"])
    parser.add_argument("--direction",   type=int, default=0, choices=[0, 1],
                        help="For bidirectional checkpoints: 0=IR10->RED4, 1=RED4->IR10")
    parser.add_argument("--prior_dir",   default="src/output",
                        help="Directory containing prior_ir.pt and prior_red.pt")
    parser.add_argument("--output",      default="../outputs/fm_result.npy",
                        help="Output path (.npy)")
    parser.add_argument("--num_steps",   type=int, default=0,
                        help="Euler ODE steps (0 = use FMInferenceConfig default)")
    parser.add_argument("--lambda_sgi_scl", type=float, default=0.0)
    parser.add_argument("--lambda_sgi_ccl", type=float, default=0.0)
    parser.add_argument("--sgi_schedule_power", type=float, default=2.0)
    parser.add_argument("--sgi_mode", choices=["velocity", "reproject"], default="velocity",
                        help="SGI update mode: velocity correction or PnP-style re-interpolation")
    parser.add_argument("--sgi_scale_mode", choices=["ratio", "raw"], default="ratio",
                        help="SGI scaling: ratio controls correction RMS relative to velocity; raw uses lambda_t * grad")
    parser.add_argument("--sgi_guidance_ratio", type=float, default=0.01,
                        help="Target RMS correction / RMS velocity when --sgi_scale_mode=ratio")
    parser.add_argument("--no_dc", action="store_true",
                        help="Disable source dc subtraction; also selects non-DC priors")
    parser.add_argument("--norm_gain", type=float, default=4.0,
                        help="Fixed gain applied after DC-normalized scene scaling")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_dc = not args.no_dc
    print(f"Device: {device}")

    cfg_model = FMModelConfig()
    cfg_inf   = FMInferenceConfig()
    if args.num_steps > 0:
        cfg_inf = FMInferenceConfig(
            num_steps=args.num_steps,
            lambda_sgi_scl=args.lambda_sgi_scl,
            lambda_sgi_ccl=args.lambda_sgi_ccl,
            sgi_schedule_power=args.sgi_schedule_power,
            sgi_mode=args.sgi_mode,
            sgi_scale_mode=args.sgi_scale_mode,
            sgi_guidance_ratio=args.sgi_guidance_ratio,
        )
    else:
        cfg_inf = FMInferenceConfig(
            lambda_sgi_scl=args.lambda_sgi_scl,
            lambda_sgi_ccl=args.lambda_sgi_ccl,
            sgi_schedule_power=args.sgi_schedule_power,
            sgi_mode=args.sgi_mode,
            sgi_scale_mode=args.sgi_scale_mode,
            sgi_guidance_ratio=args.sgi_guidance_ratio,
        )

    # ── Model ─────────────────────────────────────────────────────────────
    if args.train_mode == "bidirectional":
        ModelCls = BidirectionalFMUNet
    elif args.train_mode == "red2ir":
        ModelCls = RED2IRFMUNet
    else:
        ModelCls = IR2REDFMUNet

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
    model.eval()
    print(f"Loaded checkpoint  step={ckpt.get('step', '?')}  loss={ckpt.get('loss', '?'):.4f}")

    prior_stats = None
    if cfg_inf.lambda_sgi_scl > 0.0 or cfg_inf.lambda_sgi_ccl > 0.0:
        if args.train_mode == "red2ir":
            target_direction = 1
        elif args.train_mode == "ir2red":
            target_direction = 0
        else:
            target_direction = args.direction
        prior_suffix = "_dc" if use_dc else ""
        if args.norm_gain != 1.0:
            prior_suffix = f"{prior_suffix}_g{args.norm_gain:g}"
        prior_modality = "red" if target_direction == 0 else "ir"
        prior_name = f"prior_{prior_modality}{prior_suffix}.pt"
        prior_stats = load_prior_stats(os.path.join(args.prior_dir, prior_name), device)
        print(f"[prior] {prior_name}: mu={prior_stats['mu'].item():.4f}  "
              f"sigma={prior_stats['sigma'].item():.4f}")

    # ── Source image ──────────────────────────────────────────────────────
    src_np   = np.load(args.source).astype(np.float32)
    x_source = torch.from_numpy(src_np)

    if x_source.ndim == 2:
        x_source = x_source[None, None]
    elif x_source.ndim == 3:
        x_source = x_source[None, :1]
    elif x_source.ndim == 4:
        x_source = x_source[:, :1]

    # Shared per-scene normalisation using IR10 statistics (matches DiffusionDataset).
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

    if use_dc:
        dc = ((flat - center) / scale).clamp(-10, 10).mean()
        x_source = x_source - dc
    else:
        dc = torch.zeros((), dtype=x_source.dtype)
    x_source = x_source * args.norm_gain
    print(f"[norm] center={center.item():.4f}  scale={scale.item():.4f}  "
          f"dc={dc.item():.4f}  dc_norm={'enabled' if use_dc else 'disabled'}  "
          f"norm_gain={args.norm_gain:g}")

    x_source = x_source.to(device)
    print(f"Source image:  shape={tuple(x_source.shape)}  "
          f"range=[{x_source.min().item():.3f}, {x_source.max().item():.3f}]")

    # ── Inference ─────────────────────────────────────────────────────────
    print(f"Sampling (FM Euler, N={cfg_inf.num_steps}) ...")

    direction = args.direction if args.train_mode == "bidirectional" else None
    result = sample_fm(
        model, x_source, cfg_inf, device,
        direction=direction,
        prior_stats=prior_stats,
    )

    # Restore dc offset.
    result = result / args.norm_gain + dc.to(device)

    # ── Save ──────────────────────────────────────────────────────────────
    out_np = result.squeeze().cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, out_np)
    print(f"Saved → {args.output}  shape={out_np.shape}  "
          f"range=[{out_np.min():.3f}, {out_np.max():.3f}]")


if __name__ == "__main__":
    main()
