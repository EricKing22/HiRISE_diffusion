from typing import Literal

from pydantic import BaseModel


class ModelConfig(BaseModel):
    # ── U-Net architecture (CM-Diff paper §6 / Table 7) ──────────────────────
    in_channels:    int = 3      # noisy target (1) + source image (1) + edge map (1)
    out_channels:   int = 1      # predicted noise ε  (single-channel grayscale)
    base_channels:  int = 128    # C; channel widths follow [C/2, C, C, 2C, 2C, 4C]
    num_res_blocks: int = 3      # ResBlocks per resolution level
    dropout:        float = 0.0

    # ── Edge detection ────────────────────────────────────────────────────────
    edge_mode:         Literal["sobel", "dexined"] = "sobel"
    dexined_weights:   str = "checkpoints/dexined_biped.pth"   # path to pretrained DexiNed .pth

    # ── Diffusion schedule ────────────────────────────────────────────────────
    timesteps:      int   = 1000
    beta_start:     float = 1e-4   # β₁ = 0.0001
    beta_end:       float = 2e-2   # β_T = 0.02  (standard DDPM; 0.01 is too small for non-zero-mean scene-robust normalisation)


class DataConfig(BaseModel):
    # ── Dataset paths ─────────────────────────────────────────────────────────
    # data_root is the directory that contains the paths stored in the CSV
    # (i.e. the prefix prepended to every row['Path'] value).
    # Default: <project_root>/data   — matches the original layout.
    # Change this if you move the files/ folder elsewhere.
    data_root:   str  = "/scratch_root/ed425/HiRISE/"           # empty = resolved at runtime to <project_root>/data
    csv_path:    str  = "/scratch_root/ed425/HiRISE/files/data_record_bin12.csv"           # empty = resolved at runtime to <project_root>/data/files/data_record_bin12.csv


class TrainConfig(BaseModel):
    # ── Data ──────────────────────────────────────────────────────────────────
    image_size:  int = 256
    batch_size:  int = 8       # paper Table 7

    # ── Optimiser (AdamW, paper Table 7) ─────────────────────────────────────
    lr:          float = 1e-4
    lr_decay:    float = 0.9   # multiplicative decay factor
    lr_decay_every: int = 5000 # decay every N steps
    total_steps: int = 100_000

    # ── Loss weights (paper Eq. 5 / Table 7) ─────────────────────────────────
    lambda_ir_to_red:  float = 1.0   # Direction A: IR10 → RED4
    lambda_red_to_ir:  float = 1.0   # Direction B: RED4 → IR10

    # ── Checkpoint & logging ──────────────────────────────────────────────────
    save_every:  int  = 5_000   # save checkpoint every N steps
    log_every:   int  = 500
    val_every:   int  = 1000    # run validation every N steps
    resume:      bool = False   # True = auto-resume from latest.pt if it exists


class InferenceConfig(BaseModel):
    # ── SCI constraint weights (paper Table 8, optimal λ = 20) ───────────────
    lambda_scl:  float = 0.0   # Statistical Constraint Loss weight (0 = disabled)
    lambda_ccl:  float = 0.0   # Channel Constraint Loss weight (0 = disabled)

    # ── Histogram bins for L_ccl soft histogram ───────────────────────────────
    hist_bins:   int   = 256

    # ── Sampling ──────────────────────────────────────────────────────────────
    timesteps:   int   = 1000  # number of denoising steps (same as training)


# =============================================================================
# Flow Matching configs
# =============================================================================

class FMModelConfig(BaseModel):
    # ── FM U-Net architecture (same backbone as DDPM, different I/O) ──────
    in_channels:    int   = 1      # just x_t (source injected via cross-attention)
    out_channels:   int   = 1      # predicted velocity v
    base_channels:  int   = 128    # same as DDPM
    num_res_blocks: int   = 3
    dropout:        float = 0.0
    t_scale:        float = 1000.0 # scale continuous t ∈ [0,1] before sinusoidal embedding

    # ── Edge detection (optional, disabled by default for FM baseline) ────
    edge_mode:         Literal["sobel", "dexined", "none"] = "none"
    dexined_weights:   str = "checkpoints/dexined_biped.pth"


class FMTrainConfig(BaseModel):
    # ── Data ──────────────────────────────────────────────────────────────
    image_size:     int   = 256
    batch_size:     int   = 8

    # ── Optimiser (same as DDPM for fair comparison) ─────────────────────
    lr:             float = 1e-4
    lr_decay:       float = 0.9
    lr_decay_every: int   = 5000
    total_steps:    int   = 100_000

    # ── Checkpoint & logging ──────────────────────────────────────────────
    save_every:     int   = 5_000
    log_every:      int   = 500
    val_every:      int   = 1000
    resume:         bool  = False


class FMInferenceConfig(BaseModel):
    # ── ODE solver ────────────────────────────────────────────────────────
    num_steps:      int   = 50     # Euler ODE steps (vs DDPM's 1000)
