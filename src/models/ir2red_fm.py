"""
IR->RED Flow Matching velocity network.

Design goal
-----------
Provide a Flow Matching counterpart to `ir2red_ddpm.UNet`:
    - same U-Net backbone (channels, ResBlocks, CFC cross-attention)
    - simplified 1-channel input (just x_t; source via cross-attention only)
    - continuous time t ∈ [0, 1] scaled for the sinusoidal embedding
    - predicts velocity v instead of noise ε

I/O contract
------------
Inputs
    x_t   : interpolated sample at time t,  shape [B, 1, H, W]
    x_src : source modality image (IR),     shape [B, 1, H, W]
    t     : continuous time in [0, 1],      shape [B]

Output
    v_pred : predicted velocity field,      shape [B, 1, H, W]

Flow Matching formulation (rectified flow)
------------------------------------------
    x_t = (1 - t) · noise + t · x_1       (linear interpolation)
    v   = x_1 - noise                     (target velocity)

Bidirectional extension
-----------------------
To add direction conditioning later, import DirectionEmbedding from
cm_diff_unet and sum it with the time embedding in forward(), exactly
as the bidirectional DDPM model does.
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


class FMUNet(nn.Module):
    """
    Unidirectional conditional Flow Matching U-Net (IR->RED).

    Architecture mirrors the DDPM backbone (ir2red_ddpm.UNet) so that
    checkpoints and training behaviour stay comparable, with only the
    stem input width and time-embedding scaling changed.
    """

    _ATTN_LEVELS = {3, 4, 5}

    def __init__(
        self,
        in_channels:    int   = 1,
        out_channels:   int   = 1,
        base_channels:  int   = 128,
        num_res_blocks: int   = 3,
        dropout:        float = 0.0,
        t_scale:        float = 1000.0,
    ) -> None:
        super().__init__()
        C       = base_channels
        emb_dim = C * 4
        self.t_scale = t_scale

        # Encoder widths over 6 resolution levels:
        # 256->128->64->32->16->8 with channels [64, 128, 128, 256, 256, 512]
        ch = [C // 2, C, C, C * 2, C * 2, C * 4]
        # Source-context channels at CFC-attended levels (32/16/8).
        ctx_ch = {3: C * 2, 4: C * 2, 5: C * 4}

        # Time-only conditioning (no direction embedding in unidirectional mode).
        self.time_emb = SinusoidalTimeEmbedding(C)
        # Single source encoder: extracts multi-scale features for CFC K/V.
        self.source_encoder = ModalityEncoder(in_channels=1, base_channels=C)

        # Stem: only x_t enters (1 channel for FM baseline).
        self.stem = nn.Conv2d(in_channels, ch[0], 3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────
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

        # ── Bottleneck at 8×8 ────────────────────────────────────────────
        self.mid_res1 = ResBlock(ch[-1], ch[-1], emb_dim, dropout=dropout)
        self.mid_attn = CrossAttentionBlock(ch[-1], ctx_ch[5])
        self.mid_res2 = ResBlock(ch[-1], ch[-1], emb_dim, dropout=dropout)

        # ── Decoder ──────────────────────────────────────────────────────
        dec_in_ch = [
            ch[5] + ch[5],
            ch[5] + ch[4],
            ch[4] + ch[3],
            ch[3] + ch[2],
            ch[2] + ch[1],
            ch[1] + ch[0],
        ]
        dec_out_ch = list(reversed(ch))

        self.dec_res  = nn.ModuleList()
        self.dec_attn = nn.ModuleDict()
        self.dec_up   = nn.ModuleList()

        for dec_idx in range(len(ch)):
            enc_level = len(ch) - 1 - dec_idx
            in_ch_d   = dec_in_ch[dec_idx]
            out_ch_d  = dec_out_ch[dec_idx]

            blocks = nn.ModuleList(
                [ResBlock(in_ch_d if i == 0 else out_ch_d, out_ch_d, emb_dim, dropout=dropout)
                 for i in range(num_res_blocks)]
            )
            self.dec_res.append(blocks)

            if enc_level in self._ATTN_LEVELS:
                self.dec_attn[str(dec_idx)] = CrossAttentionBlock(out_ch_d, ctx_ch[enc_level])

            if dec_idx < len(ch) - 1:
                self.dec_up.append(Upsample(out_ch_d))

        # ── Output head ──────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(32, ch[0])
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(ch[0], out_channels, 3, padding=1)

    def forward(
        self,
        x_t:   torch.Tensor,   # [B, 1, H, W]  interpolated sample at time t
        x_src: torch.Tensor,   # [B, 1, H, W]  source modality image (IR)
        t:     torch.Tensor,   # [B]            continuous time in [0, 1]
    ) -> torch.Tensor:         # [B, 1, H, W]  predicted velocity v

        # 1) Stem — only the interpolated sample enters the U-Net.
        h = self.stem(x_t)

        # 2) Time embedding: scale continuous t to match sinusoidal frequency range.
        emb = self.time_emb(t * self.t_scale)

        # 3) Extract multi-scale source features for CFC cross-attention.
        ctx = self.source_encoder(x_src)
        scale_map = {3: "32", 4: "16", 5: "8"}

        # 4) Encoder pass with optional CFC at levels {3, 4, 5}.
        skips    = []
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

        # 5) Bottleneck.
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

        # 7) Predicted velocity v.
        return self.out_conv(self.out_act(self.out_norm(h)))
