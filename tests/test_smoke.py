from hirise_diffusion.config import TrainConfig


def test_config_defaults() -> None:
    cfg = TrainConfig()
    assert cfg.timesteps == 1000
