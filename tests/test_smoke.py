"""
Smoke tests for CM-Diff model components.

Each test verifies that a module can be constructed and produces the expected
output shape, catching dimension mismatches early without requiring a GPU or
a full training run.
"""

import pytest
import torch

from config import ModelConfig, TrainConfig, InferenceConfig
from models.unet import (
    UNet,
    SinusoidalTimeEmbedding,
    DirectionEmbedding,
    ResBlock,
    CrossAttentionBlock,
    ModalityEncoder,
)

B = 2       # batch size used across all tests
C = 64      # smallest base_channels where C//2=32 is divisible by GroupNorm groups=32


# =============================================================================
# Config
# =============================================================================

def test_config_defaults() -> None:
    """ModelConfig carries the paper hyperparameters as defaults."""
    cfg = ModelConfig()
    assert cfg.timesteps   == 1000
    assert cfg.beta_start  == 1e-4
    assert cfg.beta_end    == 1e-2
    assert cfg.base_channels == 128


def test_train_config_defaults() -> None:
    cfg = TrainConfig()
    assert cfg.image_size  == 256
    assert cfg.batch_size  == 6
    assert cfg.lambda_ir_to_red == 1.0
    assert cfg.lambda_red_to_ir == 1.0


def test_inference_config_defaults() -> None:
    cfg = InferenceConfig()
    assert cfg.lambda_scl == 20.0
    assert cfg.lambda_ccl == 20.0


# =============================================================================
# SinusoidalTimeEmbedding
# =============================================================================

def test_time_embedding_shape() -> None:
    """Output should be [B, 4*base_channels]."""
    emb = SinusoidalTimeEmbedding(base_channels=C)
    t   = torch.randint(0, 1000, (B,))
    out = emb(t)
    assert out.shape == (B, C * 4), f"expected ({B}, {C * 4}), got {out.shape}"


def test_time_embedding_different_timesteps() -> None:
    """Different timesteps should produce different embeddings."""
    emb = SinusoidalTimeEmbedding(base_channels=C)
    t1  = torch.zeros(B, dtype=torch.long)
    t2  = torch.full((B,), 500, dtype=torch.long)
    assert not torch.allclose(emb(t1), emb(t2))


# =============================================================================
# DirectionEmbedding
# =============================================================================

def test_direction_embedding_shape() -> None:
    """Output should be [B, 4*base_channels]."""
    emb = DirectionEmbedding(base_channels=C)
    d   = torch.randint(0, 2, (B,))
    out = emb(d)
    assert out.shape == (B, C * 4), f"expected ({B}, {C * 4}), got {out.shape}"


def test_direction_embedding_two_classes() -> None:
    """Direction 0 and direction 1 must produce different embeddings."""
    emb = DirectionEmbedding(base_channels=C)
    d0  = torch.zeros(B, dtype=torch.long)
    d1  = torch.ones(B, dtype=torch.long)
    assert not torch.allclose(emb(d0), emb(d1))


# =============================================================================
# ResBlock
# =============================================================================

@pytest.mark.parametrize("in_ch,out_ch", [(C, C), (C, C * 2)])
def test_resblock_shape(in_ch: int, out_ch: int) -> None:
    """Output shape must match [B, out_channels, H, W] for both same and changed channel counts."""
    H, W   = 16, 16
    block  = ResBlock(in_ch, out_ch, emb_dim=C * 4)
    x      = torch.randn(B, in_ch, H, W)
    emb    = torch.randn(B, C * 4)
    out    = block(x, emb)
    assert out.shape == (B, out_ch, H, W), f"expected ({B},{out_ch},{H},{W}), got {out.shape}"


# =============================================================================
# CrossAttentionBlock  (CFC)
# =============================================================================

def test_cross_attention_shape() -> None:
    """Output must have the same shape as the main feature map x."""
    H, W        = 8, 8
    channels     = C * 2
    ctx_channels = C * 2
    block  = CrossAttentionBlock(channels, ctx_channels)
    x      = torch.randn(B, channels, H, W)
    ctx    = torch.randn(B, ctx_channels, H, W)
    out    = block(x, ctx)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"


def test_cross_attention_different_context_channels() -> None:
    """CrossAttentionBlock should handle context_channels ≠ channels."""
    H, W         = 8, 8
    channels      = C * 4   # 8×8 level: 512 at C=128
    ctx_channels  = C * 4
    block = CrossAttentionBlock(channels, ctx_channels)
    x     = torch.randn(B, channels, H, W)
    ctx   = torch.randn(B, ctx_channels, H, W)
    out   = block(x, ctx)
    assert out.shape == x.shape


# =============================================================================
# ModalityEncoder
# =============================================================================

def test_modality_encoder_output_scales() -> None:
    """ModalityEncoder must return features at all three CFC scales."""
    enc   = ModalityEncoder(in_channels=1, base_channels=C)
    x_src = torch.randn(B, 1, 256, 256)
    feats = enc(x_src)

    assert set(feats.keys()) == {'32', '16', '8'}, f"unexpected keys: {feats.keys()}"
    assert feats['32'].shape == (B, C * 2, 32, 32), f"feat_32: {feats['32'].shape}"
    assert feats['16'].shape == (B, C * 2, 16, 16), f"feat_16: {feats['16'].shape}"
    assert feats['8'].shape  == (B, C * 4,  8,  8), f"feat_8:  {feats['8'].shape}"


# =============================================================================
# Full UNet
# =============================================================================

@pytest.fixture
def unet_inputs():
    """Shared 3-channel input tensors for UNet tests (1ch each: target, source, edge)."""
    x_t       = torch.randn(B, 1, 256, 256)
    x_src     = torch.randn(B, 1, 256, 256)
    edge_map  = torch.randn(B, 1, 256, 256)
    direction = torch.tensor([0, 1])          # one sample per direction
    t         = torch.randint(0, 1000, (B,))
    return x_t, x_src, edge_map, direction, t


def test_unet_output_shape(unet_inputs) -> None:
    """UNet output must be [B, 1, H, W] — same spatial size as input, single channel."""
    model = UNet(base_channels=C)
    out   = model(*unet_inputs)
    assert out.shape == (B, 1, 256, 256), f"expected ({B},1,256,256), got {out.shape}"


def test_unet_output_is_finite(unet_inputs) -> None:
    """UNet output must contain no NaN or Inf values."""
    model = UNet(base_channels=C)
    out   = model(*unet_inputs)
    assert torch.isfinite(out).all(), "UNet output contains NaN or Inf"


def test_unet_both_directions(unet_inputs) -> None:
    """UNet must produce different outputs for direction=0 vs direction=1."""
    model     = UNet(base_channels=C)
    x_t, x_src, edge_map, _, t = unet_inputs

    out0 = model(x_t, x_src, edge_map, torch.zeros(B, dtype=torch.long), t)
    out1 = model(x_t, x_src, edge_map, torch.ones(B,  dtype=torch.long), t)
    assert not torch.allclose(out0, out1), "Both directions produced identical output"


def test_unet_gradients_flow(unet_inputs) -> None:
    """Loss.backward() must produce non-zero gradients for all model parameters."""
    model = UNet(base_channels=C)
    out   = model(*unet_inputs)
    loss  = out.mean()
    loss.backward()

    no_grad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not no_grad, f"No gradient for: {no_grad[:5]}"
