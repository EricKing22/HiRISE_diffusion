"""
RED->IR single-direction DDPM U-Net.

Design goal
-----------
Provide the simplest *unidirectional* counterpart to `cm_diff_unet.UNet`:
    - keep timestep conditioning (diffusion time embedding)
    - remove translation-direction embedding entirely
    - keep source-conditioned CFC cross-attention (single source encoder)

I/O contract
------------
Inputs
    x_t      : noisy target at timestep t, shape [B, 1, H, W]
    x_src    : source modality image (RED), shape [B, 1, H, W]
    edge_map : Sobel edge map of source image, shape [B, 1, H, W]
    t        : diffusion timestep indices, shape [B]

Output
    eps_pred : predicted noise for DDPM objective, shape [B, 1, H, W]

Notes
-----
This model is still a *conditional* DDPM (conditioned on source+edge), but
it is single-direction by construction and does not consume a direction label.
"""

import torch
import torch.nn as nn

from .cm_diff_unet import (
    SinusoidalTimeEmbedding,
    ResBlock,
    CrossAttentionBlock,
    ModalityEncoder,
    Downsample,
    Upsample,
)


class UNet(nn.Module):
    """
    Unidirectional conditional DDPM U-Net (RED->IR).

    Architecture mirrors the CM-Diff backbone depths/channels so checkpoints
    and training behavior stay comparable, with only direction-conditioning
    removed.
    """

    _ATTN_LEVELS = {3, 4, 5}

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 128,
        num_res_blocks: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        C = base_channels
        emb_dim = C * 4

        # Encoder widths over 6 resolution levels:
        # 256->128->64->32->16->8 with channels [64, 128, 128, 256, 256, 512]
        ch = [C // 2, C, C, C * 2, C * 2, C * 4]
        # Source-context channels at CFC-attended levels (32/16/8).
        ctx_ch = {3: C * 2, 4: C * 2, 5: C * 4}

        # Time-only conditioning (no direction embedding in unidirectional mode).
        self.time_emb = SinusoidalTimeEmbedding(C)
        # Single source encoder: always encodes RED source features for CFC K/V.
        self.source_encoder = ModalityEncoder(in_channels=1, base_channels=C)

        # Stem consumes concatenated [x_t | x_src | edge] => 3 channels.
        self.stem = nn.Conv2d(in_channels, ch[0], 3, padding=1)

        # Encoder: ResBlocks per level + optional CFC + downsample between levels.
        self.enc_res = nn.ModuleList()
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

        # Bottleneck at 8x8: ResBlock -> CFC -> ResBlock.
        self.mid_res1 = ResBlock(ch[-1], ch[-1], emb_dim, dropout=dropout)
        self.mid_attn = CrossAttentionBlock(ch[-1], ctx_ch[5])
        self.mid_res2 = ResBlock(ch[-1], ch[-1], emb_dim, dropout=dropout)

        # Decoder channel accounting mirrors encoder skip concatenations.
        dec_in_ch = [
            ch[5] + ch[5],
            ch[5] + ch[4],
            ch[4] + ch[3],
            ch[3] + ch[2],
            ch[2] + ch[1],
            ch[1] + ch[0],
        ]
        dec_out_ch = list(reversed(ch))

        self.dec_res = nn.ModuleList()
        self.dec_attn = nn.ModuleDict()
        self.dec_up = nn.ModuleList()

        for dec_idx in range(len(ch)):
            enc_level = len(ch) - 1 - dec_idx
            in_ch_d = dec_in_ch[dec_idx]
            out_ch_d = dec_out_ch[dec_idx]

            blocks = nn.ModuleList(
                [ResBlock(in_ch_d if i == 0 else out_ch_d, out_ch_d, emb_dim, dropout=dropout)
                 for i in range(num_res_blocks)]
            )
            self.dec_res.append(blocks)

            if enc_level in self._ATTN_LEVELS:
                self.dec_attn[str(dec_idx)] = CrossAttentionBlock(out_ch_d, ctx_ch[enc_level])

            if dec_idx < len(ch) - 1:
                self.dec_up.append(Upsample(out_ch_d))

        # Prediction head: normalized activation then 3x3 projection to epsilon.
        self.out_norm = nn.GroupNorm(32, ch[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch[0], out_channels, 3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        x_src: torch.Tensor,
        edge_map: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # 1) Build model input with structural conditioning.
        z = torch.cat([x_t, x_src, edge_map], dim=1)
        # 2) Time embedding injected in all ResBlocks via AdaGN.
        emb = self.time_emb(t)

        # 3) Extract multi-scale source features used as CFC context.
        ctx = self.source_encoder(x_src)
        scale_map = {3: "32", 4: "16", 5: "8"}

        # 4) Encoder pass with optional CFC at levels {3,4,5}.
        h = self.stem(z)
        skips = []
        down_idx = 0

        for level, blocks in enumerate(self.enc_res):
            for res in blocks:
                h = res(h, emb)

            if level in self._ATTN_LEVELS:
                h = self.enc_attn[str(level)](h, ctx[scale_map[level]])

            skips.append(h)

            if level < len(self.enc_res) - 1:
                h = self.enc_down[down_idx](h)
                down_idx += 1

        # 5) Bottleneck processing at the lowest resolution.
        h = self.mid_res1(h, emb)
        h = self.mid_attn(h, ctx["8"])
        h = self.mid_res2(h, emb)

        # 6) Decoder with mirrored skips and optional CFC.
        up_idx = 0
        for dec_idx, blocks in enumerate(self.dec_res):
            enc_level = len(self.enc_res) - 1 - dec_idx
            h = torch.cat([h, skips[enc_level]], dim=1)

            for res in blocks:
                h = res(h, emb)

            if enc_level in self._ATTN_LEVELS:
                h = self.dec_attn[str(dec_idx)](h, ctx[scale_map[enc_level]])

            if dec_idx < len(self.dec_res) - 1:
                h = self.dec_up[up_idx](h)
                up_idx += 1

        # 7) Return predicted noise epsilon for DDPM training/inference.
        return self.out_conv(self.out_act(self.out_norm(h)))
