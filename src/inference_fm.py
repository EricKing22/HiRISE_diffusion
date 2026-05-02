"""
Flow Matching inference via Euler ODE integration.

Generates a target-domain image from a source image using a trained
FMUNet velocity network.  Starts from x_0 ~ N(0, I) and integrates
the learned velocity field to x_1 (clean image).

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

sys.path.insert(0, os.path.dirname(__file__))
from config import FMModelConfig, FMInferenceConfig
from models.ir2red_fm import FMUNet
from diffusion.fm_utils import fm_euler_step

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# Full Euler ODE sampling loop
# =============================================================================

def sample_fm(
    model:    torch.nn.Module,
    x_source: torch.Tensor,       # [B, 1, H, W]
    cfg_inf:  FMInferenceConfig,
    device:   torch.device,
    verbose:  bool = True,
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

    if verbose:
        print(f"[FM] Euler ODE: {cfg_inf.num_steps} steps, dt={dt:.4f}")

    for step_idx in range(cfg_inf.num_steps):
        t_val   = step_idx * dt
        t_batch = torch.full((B,), t_val, device=device)

        with torch.no_grad():
            v_pred = model(x_t, x_source, t_batch)

        x_t = fm_euler_step(v_pred, x_t, dt)

        if verbose and step_idx % max(1, cfg_inf.num_steps // 10) == 0:
            print(f"  step {step_idx:>4}/{cfg_inf.num_steps}  t={t_val:.3f}  "
                  f"x range=[{x_t.min().item():.3f}, {x_t.max().item():.3f}]")

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
    parser.add_argument("--output",      default="../outputs/fm_result.npy",
                        help="Output path (.npy)")
    parser.add_argument("--num_steps",   type=int, default=0,
                        help="Euler ODE steps (0 = use FMInferenceConfig default)")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg_model = FMModelConfig()
    cfg_inf   = FMInferenceConfig()
    if args.num_steps > 0:
        cfg_inf = FMInferenceConfig(num_steps=args.num_steps)

    # ── Model ─────────────────────────────────────────────────────────────
    model = FMUNet(
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

    dc = ((flat - center) / scale).clamp(-10, 10).mean()
    x_source = x_source - dc
    print(f"[norm] center={center.item():.4f}  scale={scale.item():.4f}  dc={dc.item():.4f}")

    x_source = x_source.to(device)
    print(f"Source image:  shape={tuple(x_source.shape)}  "
          f"range=[{x_source.min().item():.3f}, {x_source.max().item():.3f}]")

    # ── Inference ─────────────────────────────────────────────────────────
    print(f"Sampling (FM Euler, N={cfg_inf.num_steps}) ...")

    result = sample_fm(model, x_source, cfg_inf, device)

    # Restore dc offset.
    result = result + dc.to(device)

    # ── Save ──────────────────────────────────────────────────────────────
    out_np = result.squeeze().cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, out_np)
    print(f"Saved → {args.output}  shape={out_np.shape}  "
          f"range=[{out_np.min():.3f}, {out_np.max():.3f}]")


if __name__ == "__main__":
    main()
