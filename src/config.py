from pydantic import BaseModel


class TrainConfig(BaseModel):
    image_size: int = 64
    in_channels: int = 3
    base_channels: int = 64
    timesteps: int = 1000
    batch_size: int = 16
    lr: float = 1e-4
    epochs: int = 1
