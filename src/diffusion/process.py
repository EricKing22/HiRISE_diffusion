"""
DDPM forward process and utility functions.

q_sample  : adds noise to a clean image at timestep t (forward diffusion)
sobel_edge: differentiable Sobel edge detection for single-channel images
"""

import torch
import torch.nn.functional as F

from .scheduler import DDPMScheduler


def q_sample(
    scheduler: DDPMScheduler,
    x_start:   torch.Tensor,   # [B, 1, H, W]  clean image x_0
    t:         torch.Tensor,   # [B]            integer timesteps
    noise:     torch.Tensor,   # [B, 1, H, W]  pre-sampled Gaussian noise
) -> torch.Tensor:
    """
    Forward diffusion: q(x_t | x_0) = N(√ᾱ_t · x_0, (1 - ᾱ_t) · I)

    Returns x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε
    """
    sqrt_ab     = scheduler.gather(scheduler.sqrt_alpha_bar,    t)  # [B,1,1,1]
    sqrt_one_ab = scheduler.gather(scheduler.sqrt_one_minus_ab, t)  # [B,1,1,1]
    return sqrt_ab * x_start + sqrt_one_ab * noise


def sobel_edge(x: torch.Tensor) -> torch.Tensor:
    """
    Compute a Sobel edge map for a single-channel image tensor.

    Input : [B, 1, H, W]  — normalised grayscale image
    Output: [B, 1, H, W]  — edge magnitude, normalised to [0, 1]
    """
    # Sobel kernels
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)

    gx  = F.conv2d(x, kx, padding=1)
    gy  = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    # Normalise per image to [0, 1]
    B = mag.shape[0]
    mn = mag.view(B, -1).min(dim=1).values[:, None, None, None]
    mx = mag.view(B, -1).max(dim=1).values[:, None, None, None]
    return (mag - mn) / (mx - mn + 1e-8)
