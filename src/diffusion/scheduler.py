"""
DDPM diffusion noise schedule.

Precomputes and stores all derived quantities (alpha, alpha_bar, sqrt terms)
needed for forward (q_sample) and reverse (mu prediction) passes.
"""

import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.01,
) -> torch.Tensor:
    """Linear schedule from beta_start to beta_end. CM-Diff uses beta_end=0.01."""
    return torch.linspace(beta_start, beta_end, timesteps)


class DDPMScheduler:
    """
    Precomputed DDPM noise schedule.

    All tensors are stored on CPU and moved to device on first use via .to(device).

    Attributes
    ----------
    betas             : [T]   β_t
    alphas            : [T]   α_t = 1 - β_t
    alpha_bar         : [T]   ᾱ_t = ∏_{s=1}^{t} α_s
    sqrt_alpha_bar    : [T]   √ᾱ_t          — forward process signal coefficient
    sqrt_one_minus_ab : [T]   √(1 - ᾱ_t)   — forward process noise coefficient
    posterior_var     : [T]   β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)   (fixed variance)
    """

    def __init__(
        self,
        timesteps:  int   = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 0.01,
    ) -> None:
        self.timesteps = timesteps

        betas     = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Pad alpha_bar with 1.0 at t=0 for posterior variance computation
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])

        self.betas              = betas
        self.alphas             = alphas
        self.alpha_bar          = alpha_bar
        self.alpha_bar_prev     = alpha_bar_prev
        self.sqrt_alpha_bar     = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_ab  = torch.sqrt(1.0 - alpha_bar)

        # Fixed posterior variance β̃_t (used in reverse process sampling)
        self.posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    def to(self, device: torch.device) -> "DDPMScheduler":
        """Move all schedule tensors to device (in-place)."""
        for attr in (
            "betas", "alphas", "alpha_bar", "alpha_bar_prev",
            "sqrt_alpha_bar", "sqrt_one_minus_ab", "posterior_var",
        ):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def gather(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Gather schedule values at timestep indices t and reshape to [B, 1, 1, 1]
        for broadcasting against [B, C, H, W] image tensors.
        """
        return tensor[t][:, None, None, None]
