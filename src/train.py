from hirise_diffusion.config import TrainConfig


def main() -> None:
    cfg = TrainConfig()
    print("Training template initialized with config:")
    print(cfg.model_dump())


if __name__ == "__main__":
    main()
