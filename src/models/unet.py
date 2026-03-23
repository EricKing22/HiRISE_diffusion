"""
CM-Diff U-Net backbone.

Architecture source: CM-Diff paper Section 3, Figure 2(a), Supplementary Figure 7.

────────────────────────────────────────────────────────────────────────────────
Overview
────────────────────────────────────────────────────────────────────────────────
Single U-Net shared by both IR↔RED4 translation directions.
  Input : 3-channel tensor  [noisy target (ch1) | source image (ch2) | edge map (ch3)]
  Output: predicted noise   ε ∈ ℝ^{1×H×W}

Encoder channel progression (6 resolution levels, input 256×256):
    Level 0   256×256    64 ch    ResBlock×3   conv-only
    Level 1   128×128   128 ch    ResBlock×3   conv-only
    Level 2    64×64    128 ch    ResBlock×3   conv-only
    Level 3    32×32    256 ch    ResBlock×3 + CFC CrossAttention   (n=1)
    Level 4    16×16    256 ch    ResBlock×3 + CFC CrossAttention   (n=2)
    Level 5     8×8    512 ch    ResBlock×3 + CFC CrossAttention   (n=3)
Bottleneck:    8×8    512 ch    ResBlock + CFC CrossAttention + ResBlock
Decoder: symmetric to encoder with skip connections concatenated at every level.

Conditioning injected into every ResBlock via AdaGN:
    emb = SinusoidalTimeEmbedding(t) + DirectionEmbedding(C)
    h   = (1 + scale) * GroupNorm(h) + shift     [scale, shift from Linear(emb)]

CFC cross-attention (encoder & decoder levels 3–5 only):
    Q  ← main U-Net feature map          (noisy target)
    K,V ← ModalityEncoder feature map    (source modality image)
    Both paths pre-normalised with GroupNorm before attention.
    Ref: Supplementary Fig. 7 "Attention Block" diagram.

ModalityEncoder (one per modality, i.e. one for IR, one for VIS):
    Lightweight encoder-only network built from plain ConvResBlocks (no time conditioning).
    Produces multi-scale features at the three CFC injection scales.
    Ref: Supplementary Fig. 7 "Extraction Block (ResBlock×3 C=128)".
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Time Embedding
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for the diffusion timestep t,
    followed by a two-layer MLP that expands to emb_dim = 4 * base_channels.

    Encoding:  [sin(t·ω₀), ..., sin(t·ω_{D/2}), cos(t·ω₀), ..., cos(t·ω_{D/2})]
               → [B, base_channels]
    MLP:       Linear → SiLU → Linear  →  [B, 4*base_channels]

    Reference: "Attention Is All You Need" §3.5 (Vaswani et al. 2017);
               DDPM (Ho et al. 2020).
    """

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.base_channels = base_channels
        emb_dim = base_channels * 4
        self.mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]  integer diffusion timesteps
        half = self.base_channels // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / (half - 1)
        )                                                   # [D/2]
        args = t[:, None].float() * freqs[None]             # [B, D/2]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # [B, base_channels]
        return self.mlp(emb)                                # [B, 4*base_channels]


# =============================================================================
# Direction Embedding  (TDG — Translation Direction Guidance)
# =============================================================================

class DirectionEmbedding(nn.Module):
    """
    Explicit translation direction signal (TDG).

    An nn.Embedding lookup table maps the binary direction label C ∈ {0, 1}
    to a learnable dense vector; a small MLP matches its dimension to the
    time embedding so the two can be summed before AdaGN injection.

        C = 0  →  IR10 → RED4
        C = 1  →  RED4 → IR10

    nn.Embedding is used (not Linear(scalar)) so each direction gets a
    fully independent, unconstrained embedding vector.

    """

    def __init__(self, base_channels: int, num_directions: int = 2) -> None:
        super().__init__()
        emb_dim = base_channels * 4
        self.table = nn.Embedding(num_directions, base_channels)
        self.mlp   = nn.Sequential(
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, direction: torch.Tensor) -> torch.Tensor:
        # direction: [B]  integer labels {0, 1}
        return self.mlp(self.table(direction))              # [B, 4*base_channels]


# =============================================================================
# ResBlock with AdaGN conditioning  (main U-Net)
# =============================================================================

class ResBlock(nn.Module):
    """
    Residual block with Adaptive Group Normalization (AdaGN) conditioning.

    Structure (Supplementary Fig. 7 "ResBlock" diagram):
        x → GroupNorm → SiLU → Conv3×3                          → h
        emb → SiLU → Linear(emb_dim → 2*C)  →  [scale, shift]
        h → GroupNorm → AdaGN: h = (1 + scale)*h + shift → SiLU → Conv3×3
        output = h + skip(x)

    The "FC-C-2C" in Fig. 7 is the Linear projection emb → [scale, shift].
    The "K3-S1-C/C" annotations refer to the Conv3×3 layers here.

    AdaGN lets the same weights handle all (t, direction) combinations
    without any structural change — the normalisation statistics adapt per call.

    Reference: CM-Diff §3.1; Improved DDPM §2 (Nichol & Dhariwal 2021).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        emb_dim:      int,
        num_groups:   int   = 32,
        dropout:      float = 0.0,
    ) -> None:
        super().__init__()

        # First conv path
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # AdaGN embedding projection: emb → scale + shift (2 × out_channels)
        # Leading SiLU activates the shared embedding before the per-block proj.
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_channels),
        )

        # Second conv path — norm comes before AdaGN modulation
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2  = nn.SiLU()
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection: 1×1 conv if channel width changes, else identity
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x:   [B, in_channels, H, W]
        # emb: [B, emb_dim]  combined time + direction embedding
        h = self.conv1(self.act1(self.norm1(x)))

        # AdaGN: split into scale and shift, broadcast over spatial dims
        emb_out        = self.emb_proj(emb)          # [B, 2*out_channels]
        scale, shift   = emb_out.chunk(2, dim=1)     # each [B, out_channels]
        scale = scale[:, :, None, None]               # [B, C, 1, 1]
        shift = shift[:, :, None, None]

        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(self.act2(h)))

        return h + self.skip(x)


# =============================================================================
# ConvResBlock  (plain residual block, no time/direction conditioning)
# =============================================================================

class ConvResBlock(nn.Module):
    """
    Plain residual block without time/direction conditioning.

    Used inside ModalityEncoder where no diffusion timestep embedding exists.
    Structure: GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3 + skip.

    Corresponds to the "Extraction Block (ResBlock×3 C=128)" in Supplementary Fig. 7.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        num_groups:   int = 32,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


# =============================================================================
# Self-Attention Block
# =============================================================================

class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention with residual connection.

    Q, K, V all originate from the same feature map.
    Used only inside ModalityEncoder (no cross-modal source available there).

    head_dim fixed at 64 following CM-Diff paper ("multi-head attention, 64 channels").
    """

    def __init__(self, channels: int, head_dim: int = 64) -> None:
        super().__init__()
        self.num_heads = max(1, channels // head_dim)
        self.head_dim  = channels // self.num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)                              # [B, heads, d, HW]

        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out  = torch.einsum('bhij,bhdj->bhdi', attn, v)      # [B, heads, d, HW]

        return x + self.proj(out.reshape(B, C, H, W))


# =============================================================================
# Cross-Attention Block  (CFC — Cross-Modality Feature Control)
# =============================================================================

class CrossAttentionBlock(nn.Module):
    """
    CFC cross-attention block (Supplementary Fig. 7 "Attention Block" diagram).

    Replaces self-attention at the 32×32, 16×16, and 8×8 levels of both
    the encoder and decoder (3 + 3 = 6 injection sites total).

    Structure (from Fig. 7):
        Q  path:  GroupNorm(x)       → Conv1×1_q  → Q
        K,V path: GroupNorm(context) → Conv1×1_kv → K, V
        out = x + Conv1×1_proj( MultiHeadAttention(Q, K, V) )

    Q  ← main U-Net feature map  (noisy target being denoised)
    K,V ← ModalityEncoder output  (source modality image)

    CFC ensures that at every scale where attention is applied, the denoising
    trajectory is continuously re-anchored to source-modality semantics.

    Reference: CM-Diff §3.2 / CFC, Eq. (3); Supplementary Fig. 7.
    """

    def __init__(
        self,
        channels:         int,
        context_channels: int,
        head_dim:         int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = max(1, channels // head_dim)
        self.head_dim  = channels // self.num_heads

        # Separate norms for the two input streams (Fig. 7: "GroupNorm" on both paths)
        self.norm_q  = nn.GroupNorm(32, channels)
        self.norm_kv = nn.GroupNorm(32, context_channels)

        # Q projection (from main U-Net), K/V projection (from modality encoder)
        self.conv_q  = nn.Conv2d(channels,         channels,     1)
        self.conv_kv = nn.Conv2d(context_channels, channels * 2, 1)
        self.proj    = nn.Conv2d(channels,         channels,     1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x:       [B, C,     H, W]  main U-Net feature (noisy target)
        # context: [B, C_ctx, H, W]  modality encoder feature (source image)
        B, C, H, W = x.shape

        q       = self.conv_q(self.norm_q(x))            # [B, C, H, W]
        k, v    = self.conv_kv(self.norm_kv(context)).chunk(2, dim=1)  # [B, C, H, W] each

        # Reshape for multi-head attention over flattened spatial positions
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out  = torch.einsum('bhij,bhdj->bhdi', attn, v)  # [B, heads, d, HW]

        return x + self.proj(out.reshape(B, C, H, W))


# =============================================================================
# Downsample / Upsample
# =============================================================================

class Downsample(nn.Module):
    """Strided 3×3 conv (stride=2) to halve spatial resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour 2× interpolation followed by a 3×3 conv to refine."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


# =============================================================================
# Modality Encoder  (lightweight encoder-only network for CFC context)
# =============================================================================

class ModalityEncoder(nn.Module):
    """
    Lightweight encoder-only network built from ConvResBlocks.

    Extracts multi-scale feature maps from a source-modality image to serve
    as K,V context in CFC cross-attention.  No decoder, no time conditioning.

    Two instances exist inside UNet:
        encoder_VIS — extracts VIS source features (IR→VIS direction, C=0)
        encoder_IR  — extracts IR  source features (VIS→IR direction, C=1)

    Output (Supplementary Fig. 7 "Extraction Block"):
        '32'  [B, 2C,  32, 32]   CFC n=1
        '16'  [B, 2C,  16, 16]   CFC n=2
        '8'   [B, 4C,   8,  8]   CFC n=3

    Channel counts match the main U-Net at the corresponding resolution so that
    the Conv1×1 projections inside CrossAttentionBlock are well-conditioned.

    Reference: CM-Diff §3.2; Supplementary Fig. 7 "Extraction Block (ResBlock×3 C=128)".
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 128) -> None:
        super().__init__()
        C = base_channels

        # Stem: 256×256, C//2 channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C // 2, 3, padding=1),
            nn.SiLU(),
        )

        # Progressive downsampling with ConvResBlocks (Extraction Blocks)
        # 256→128: C//2 → C
        self.stage1 = nn.Sequential(
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1),
            ConvResBlock(C, C),
        )
        # 128→64: C → C
        self.stage2 = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=2, padding=1),
            ConvResBlock(C, C),
        )
        # 64→32: C → 2C    ← feat_32 output (CFC n=1)
        self.stage3 = nn.Sequential(
            nn.Conv2d(C, C * 2, 3, stride=2, padding=1),
            ConvResBlock(C * 2, C * 2),
            ConvResBlock(C * 2, C * 2),
            ConvResBlock(C * 2, C * 2),   # ResBlock×3 matching Fig. 7
        )
        # 32→16: 2C → 2C   ← feat_16 output (CFC n=2)
        self.stage4 = nn.Sequential(
            nn.Conv2d(C * 2, C * 2, 3, stride=2, padding=1),
            ConvResBlock(C * 2, C * 2),
            ConvResBlock(C * 2, C * 2),
            ConvResBlock(C * 2, C * 2),
        )
        # 16→8: 2C → 4C    ← feat_8 output (CFC n=3)
        self.stage5 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1),
            ConvResBlock(C * 4, C * 4),
            ConvResBlock(C * 4, C * 4),
            ConvResBlock(C * 4, C * 4),
        )

    def forward(self, x: torch.Tensor) -> dict:
        h       = self.stem(x)
        h       = self.stage1(h)
        h       = self.stage2(h)
        feat_32 = self.stage3(h)
        feat_16 = self.stage4(feat_32)
        feat_8  = self.stage5(feat_16)
        return {'32': feat_32, '16': feat_16, '8': feat_8}


# =============================================================================
# Main U-Net
# =============================================================================

class UNet(nn.Module):
    """
    CM-Diff denoising U-Net.

    A single model handles both IR→VIS and VIS→IR via:
      (a) Channel position (implicit TDG): noisy target always in ch 1-3, source in ch 4-6.
      (b) Direction label embedding (explicit TDG): AdaGN conditioning in every ResBlock.

    ── Decoder channel accounting (skip = encoder output at same level) ────────
    Dec idx | Enc level | Input channels      | Output ch | Resolution
    --------|-----------|---------------------|-----------|----------
        0   |     5     | 512 + 512 = 1024    |    512    |  8×8
        1   |     4     | 512 + 256 =  768    |    256    | 16×16
        2   |     3     | 256 + 256 =  512    |    256    | 32×32
        3   |     2     | 256 + 128 =  384    |    128    | 64×64
        4   |     1     | 128 + 128 =  256    |    128    |128×128
        5   |     0     | 128 +  64 =  192    |     64    |256×256
    ────────────────────────────────────────────────────────────────────────────

    Parameters
    ----------
    in_channels     : 3  (1 noisy target + 1 source + 1 edge map)
    out_channels    : 1  (predicted noise ε, same spatial size as noisy target)
    base_channels   : C = 128  (paper Table 7)
    num_res_blocks  : 3 ResBlocks per resolution level  (paper §6)
    dropout         : dropout inside ResBlocks
    """

    # Encoder levels where CFC cross-attention is applied (32×32, 16×16, 8×8)
    _ATTN_LEVELS = {3, 4, 5}

    def __init__(
        self,
        in_channels:    int   = 3,
        out_channels:   int   = 1,
        base_channels:  int   = 128,
        num_res_blocks: int   = 3,
        dropout:        float = 0.0,
    ) -> None:
        super().__init__()
        C       = base_channels
        emb_dim = C * 4

        # Channel width at each of the 6 encoder resolution levels
        # index:   0     1    2     3      4      5
        # res:   256   128   64    32     16      8
        ch = [C // 2, C, C, C * 2, C * 2, C * 4]  # [64, 128, 128, 256, 256, 512]

        # Context channels from ModalityEncoder at each CFC level
        ctx_ch = {3: C * 2, 4: C * 2, 5: C * 4}

        # ── Conditioning embeddings ─────────────────────────────────────────
        self.time_emb      = SinusoidalTimeEmbedding(C)
        self.direction_emb = DirectionEmbedding(C)

        # ── Modality encoders for CFC ───────────────────────────────────────
        # encoder_IR:  source is IR10  (IR→RED4, direction=0)
        # encoder_RED: source is RED4  (RED4→IR, direction=1)
        self.encoder_IR  = ModalityEncoder(in_channels=1, base_channels=C)
        self.encoder_RED = ModalityEncoder(in_channels=1, base_channels=C)

        # ── Stem ─────────────────────────────────────────────────────────────
        # Project 9-channel input to the first feature width (C//2 = 64)
        self.stem = nn.Conv2d(in_channels, ch[0], 3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        # enc_res[level]       = ModuleList of ResBlocks
        # enc_attn[str(level)] = CrossAttentionBlock  (levels 3, 4, 5 only)
        # enc_down[i]          = Downsample  (5 total, between consecutive levels)
        self.enc_res  = nn.ModuleList()
        self.enc_attn = nn.ModuleDict()
        self.enc_down = nn.ModuleList()

        in_ch = ch[0]
        for level, out_ch in enumerate(ch):
            blocks = nn.ModuleList(
                [ResBlock(in_ch if i == 0 else out_ch, out_ch, emb_dim, dropout=dropout)
                 for i in range(num_res_blocks)]
            )
            self.enc_res.append(blocks)

            if level in self._ATTN_LEVELS:
                self.enc_attn[str(level)] = CrossAttentionBlock(out_ch, ctx_ch[level])

            if level < len(ch) - 1:
                self.enc_down.append(Downsample(out_ch))

            in_ch = out_ch

        # ── Bottleneck ────────────────────────────────────────────────────────
        # ResBlock → CFC CrossAttention (8×8) → ResBlock
        self.mid_res1 = ResBlock(ch[-1], ch[-1], emb_dim, dropout=dropout)
        self.mid_attn = CrossAttentionBlock(ch[-1], ctx_ch[5])
        self.mid_res2 = ResBlock(ch[-1], ch[-1], emb_dim, dropout=dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        # dec_res[i]       = ModuleList of ResBlocks for decoder level i
        # dec_attn[str(i)] = CrossAttentionBlock  (when mirroring enc levels 3, 4, 5)
        # dec_up[i]        = Upsample (for all decoder levels except the last)
        dec_in_ch  = [
            ch[5] + ch[5],  # dec 0: bottleneck(512) + skip_enc5(512) = 1024
            ch[5] + ch[4],  # dec 1: dec0_out(512)   + skip_enc4(256) =  768
            ch[4] + ch[3],  # dec 2: dec1_out(256)   + skip_enc3(256) =  512
            ch[3] + ch[2],  # dec 3: dec2_out(256)   + skip_enc2(128) =  384
            ch[2] + ch[1],  # dec 4: dec3_out(128)   + skip_enc1(128) =  256
            ch[1] + ch[0],  # dec 5: dec4_out(128)   + skip_enc0(64)  =  192
        ]
        dec_out_ch = list(reversed(ch))  # [512, 256, 256, 128, 128, 64]

        self.dec_res  = nn.ModuleList()
        self.dec_attn = nn.ModuleDict()
        self.dec_up   = nn.ModuleList()

        for dec_idx in range(len(ch)):
            enc_level  = len(ch) - 1 - dec_idx   # mirrored encoder level
            in_ch_d    = dec_in_ch[dec_idx]
            out_ch_d   = dec_out_ch[dec_idx]

            blocks = nn.ModuleList(
                [ResBlock(in_ch_d if i == 0 else out_ch_d, out_ch_d, emb_dim, dropout=dropout)
                 for i in range(num_res_blocks)]
            )
            self.dec_res.append(blocks)

            if enc_level in self._ATTN_LEVELS:
                self.dec_attn[str(dec_idx)] = CrossAttentionBlock(out_ch_d, ctx_ch[enc_level])

            if dec_idx < len(ch) - 1:          # no upsample after the very last level
                self.dec_up.append(Upsample(out_ch_d))

        # ── Output head ───────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(32, ch[0])
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(ch[0], out_channels, 3, padding=1)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x_t:       torch.Tensor,   # [B, 1, H, W]  noisy target image at timestep t
        x_src:     torch.Tensor,   # [B, 1, H, W]  source modality image (fixed)
        edge_map:  torch.Tensor,   # [B, 1, H, W]  Sobel edge map of source (fixed)
        direction: torch.Tensor,   # [B]            0 = IR10→RED4,  1 = RED4→IR10
        t:         torch.Tensor,   # [B]            integer diffusion timestep
    ) -> torch.Tensor:             # [B, 1, H, W]  predicted noise ε

        # ── Build 3-channel input ──────────────────────────────────────────
        # Implicit TDG: noisy target in ch1 marks which modality is being generated.
        # Source image in ch2 provides cross-modality guidance.
        # Edge map in ch3 provides structural priors (Sobel of source).
        z = torch.cat([x_t, x_src, edge_map], dim=1)       # [B, 3, H, W]

        # ── Combined conditioning embedding ───────────────────────────────
        # Explicit TDG: direction label fused with time embedding, injected via AdaGN.
        emb = self.time_emb(t) + self.direction_emb(direction)  # [B, 4C]

        # ── CFC context: extract multi-scale features from source image ───
        # direction=0 (IR10→RED4): source is IR10 → encoder_IR provides K, V
        # direction=1 (RED4→IR10): source is RED4 → encoder_RED provides K, V
        ctx_ir  = self.encoder_IR(x_src)    # {'32': ..., '16': ..., '8': ...}
        ctx_red = self.encoder_RED(x_src)

        def pick_ctx(scale_key: str) -> torch.Tensor:
            # Select context per sample: direction=0 uses IR context, direction=1 uses RED context.
            mask = (direction == 0)[:, None, None, None].float()
            return ctx_ir[scale_key] * mask + ctx_red[scale_key] * (1.0 - mask)

        _scale = {3: '32', 4: '16', 5: '8'}  # encoder level → context scale key

        # ── Encoder ───────────────────────────────────────────────────────
        h = self.stem(z)
        skips    = []
        down_idx = 0

        for level, blocks in enumerate(self.enc_res):
            for res in blocks:
                h = res(h, emb)

            # CFC cross-attention at levels 3, 4, 5 (32×32, 16×16, 8×8)
            if level in self._ATTN_LEVELS:
                h = self.enc_attn[str(level)](h, pick_ctx(_scale[level]))

            skips.append(h)                     # save before downsampling

            if level < len(self.enc_res) - 1:
                h = self.enc_down[down_idx](h)
                down_idx += 1

        # ── Bottleneck ────────────────────────────────────────────────────
        h = self.mid_res1(h, emb)
        h = self.mid_attn(h, pick_ctx('8'))     # CFC at 8×8
        h = self.mid_res2(h, emb)

        # ── Decoder ───────────────────────────────────────────────────────
        up_idx = 0

        for dec_idx, blocks in enumerate(self.dec_res):
            enc_level = len(self.enc_res) - 1 - dec_idx

            # Concatenate skip connection from the mirrored encoder level
            h = torch.cat([h, skips[enc_level]], dim=1)

            for res in blocks:
                h = res(h, emb)

            # CFC cross-attention at decoder levels mirroring enc levels 3, 4, 5
            if enc_level in self._ATTN_LEVELS:
                h = self.dec_attn[str(dec_idx)](h, pick_ctx(_scale[enc_level]))

            if dec_idx < len(self.dec_res) - 1:
                h = self.dec_up[up_idx](h)
                up_idx += 1

        # ── Output: predicted noise ε ──────────────────────────────────────
        return self.out_conv(self.out_act(self.out_norm(h)))  # [B, 3, H, W]
