"""
Flow Matching utilities for linear interpolation (rectified flow).

Convention
----------
    x_t = (1 - t) * noise + t * x_1

    t = 0  →  pure noise
    t = 1  →  clean data  x_1

Target velocity along the linear path:
    v = dx_t / dt = x_1 - noise          (constant w.r.t. t)

References
----------
Lipman et al. "Flow Matching for Generative Modeling" (2023)
Liu et al.   "Flow Straight and Fast" (2022)
"""

import torch


def fm_interpolate(
    x_1:   torch.Tensor,   # [B, 1, H, W]  clean target image
    noise: torch.Tensor,   # [B, 1, H, W]  sampled noise  ε ~ N(0, I)
    t:     torch.Tensor,   # [B]            continuous time in [0, 1]
) -> torch.Tensor:
    """
    Linear interpolation path (rectified flow):
        x_t = (1 - t) · noise  +  t · x_1
    """
    t = t[:, None, None, None]                    # [B, 1, 1, 1]
    return (1.0 - t) * noise + t * x_1


def fm_velocity_target(
    x_1:   torch.Tensor,   # [B, 1, H, W]  clean target
    noise: torch.Tensor,   # [B, 1, H, W]  sampled noise
) -> torch.Tensor:
    """
    Target velocity for the linear interpolation path:
        v = dx_t / dt = x_1 - noise
    """
    return x_1 - noise


def fm_euler_step(
    v_pred: torch.Tensor,  # [B, 1, H, W]  predicted velocity
    x_t:    torch.Tensor,  # [B, 1, H, W]  current state
    dt:     float,         # step size  (1 / num_steps)
) -> torch.Tensor:
    """
    Euler ODE integration step:
        x_{t+dt} = x_t + v_pred · dt
    """
    return x_t + v_pred * dt
