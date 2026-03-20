# HiRISE Diffusion

Starter template for a diffusion-model deep learning project with a `src` layout.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python -m hirise_diffusion.train
pytest
```

## Project layout

- `src/hirise_diffusion/data/`: dataset code and transforms
- `src/hirise_diffusion/models/`: model architecture modules
- `src/hirise_diffusion/diffusion/`: schedules and diffusion process logic
- `src/hirise_diffusion/train.py`: training entrypoint
- `src/hirise_diffusion/sample.py`: sampling/inference entrypoint

## Next steps

- Add your dataset in `src/hirise_diffusion/data/dataset.py`.
- Implement U-Net blocks in `src/hirise_diffusion/models/unet.py`.
- Expand diffusion schedule and reverse process in `src/hirise_diffusion/diffusion/`.
