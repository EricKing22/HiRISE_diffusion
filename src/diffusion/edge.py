"""
Edge detection dispatcher: routes to Sobel or DexiNed based on mode.

Both detectors share the same I/O contract:
    Input : [B, 1, H, W]  — single-channel image
    Output: [B, 1, H, W]  — edge map normalised to [0, 1]
"""

import torch
import torch.nn as nn

from .process import sobel_edge


# =============================================================================
# DexiNed loader + wrapper
# =============================================================================

def load_dexined(weights_path: str, device: torch.device) -> nn.Module:
    """Load pretrained DexiNed, freeze all parameters, return on *device*."""
    from models.dexined import DexiNed

    model = DexiNed().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)

    # Handle DataParallel 'module.' prefix in state dict keys
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    print(f"  [edge] DexiNed loaded from {weights_path}")
    return model


@torch.no_grad()
def dexined_edge(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Compute edge map using pretrained DexiNed.

    Input : [B, 1, H, W]  — single-channel image (any value range)
    Output: [B, 1, H, W]  — edge magnitude, normalised to [0, 1]

    Preprocessing:
      1. Per-image min-max normalisation to [0, 1]
      2. Repeat 1→3 channels (DexiNed expects RGB)
    Postprocessing:
      1. Take fused output (last element of DexiNed's 7 outputs)
      2. Sigmoid (DexiNed outputs raw logits)
      3. Per-image min-max normalisation to [0, 1]
    """
    B = x.shape[0]

    # Normalise input per-image to [0, 1] for DexiNed
    flat = x.view(B, -1)
    mn = flat.min(dim=1).values[:, None, None, None]
    mx = flat.max(dim=1).values[:, None, None, None]
    x_01 = (x - mn) / (mx - mn + 1e-8)

    # 1ch → 3ch
    x_rgb = x_01.repeat(1, 3, 1, 1)

    # Forward pass — returns list of 7 tensors, last is fused
    outputs = model(x_rgb)
    edge = torch.sigmoid(outputs[-1])  # [B, 1, H, W]

    # Per-image normalisation to [0, 1]
    flat_e = edge.view(B, -1)
    mn_e = flat_e.min(dim=1).values[:, None, None, None]
    mx_e = flat_e.max(dim=1).values[:, None, None, None]
    return (edge - mn_e) / (mx_e - mn_e + 1e-8)


# =============================================================================
# Dispatcher
# =============================================================================

def compute_edge(
    x: torch.Tensor,
    mode: str = "sobel",
    dexined_model: nn.Module = None,
) -> torch.Tensor:
    """
    Compute edge map using the selected detector.

    Args:
        x:              [B, 1, H, W] input image
        mode:           "sobel" or "dexined"
        dexined_model:  pre-loaded DexiNed model (required when mode="dexined")

    Returns:
        [B, 1, H, W] edge map normalised to [0, 1]
    """
    if mode == "sobel":
        return sobel_edge(x)
    elif mode == "dexined":
        if dexined_model is None:
            raise ValueError("dexined_model must be provided when mode='dexined'")
        return dexined_edge(x, dexined_model)
    else:
        raise ValueError(f"Unknown edge mode: {mode!r}. Choose 'sobel' or 'dexined'.")
