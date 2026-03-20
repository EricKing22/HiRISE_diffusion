import torch


class DiffusionProcess:
    def __init__(self, timesteps: int = 1000) -> None:
        self.timesteps = timesteps

    def q_sample(self, x_start: torch.Tensor, noise: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise
