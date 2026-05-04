"""
Bidirectional Flow Matching velocity network for HiRISE IR10 <-> RED4.

This is the Flow Matching counterpart to BidirectionalDDPMUNet. It keeps the
current FM image inputs, x_t and x_src, and adds an explicit direction label:

    direction = 0: IR10 -> RED4
    direction = 1: RED4 -> IR10

The direction embedding is summed with the continuous-time embedding and
injected through the existing AdaGN ResBlocks, matching CM-Diff's TDG pattern.
"""

import torch
import torch.nn as nn

from .cm_diff_unet import (
    SinusoidalTimeEmbedding,
    DirectionEmbedding,
    ResBlock,
    CrossAttentionBlock,
    ModalityEncoder,
    Downsample,
    Upsample,
)


class BidirectionalFMUNet(nn.Module):
    """
    Shared conditional rectified-flow U-Net for both IR->RED and RED->IR.

    Inputs:
        x_t       [B, 1, H, W] interpolated target/noise state
        x_src     [B, 1, H, W] source modality image
        direction [B]          0=IR10->RED4, 1=RED4->IR10
        t         [B]          continuous time in [0, 1]

    Output:
        v_pred    [B, 1, H, W] predicted velocity
    """

    _ATTN_LEVELS = {3, 4, 5}

    def __init__(
        self,
        in_channels:    int   = 2,
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

        ch     = [C // 2, C, C, C * 2, C * 2, C * 4]
        ctx_ch = {3: C * 2, 4: C * 2, 5: C * 4}

        self.time_emb      = SinusoidalTimeEmbedding(C)
        self.direction_emb = DirectionEmbedding(C)

        # Separate source encoders preserve modality-specific context features.
        self.encoder_IR  = ModalityEncoder(in_channels=1, base_channels=C)
        self.encoder_RED = ModalityEncoder(in_channels=1, base_channels=C)

        # Same FM image input as unidirectional FM: concat x_t and x_src.
        self.stem = nn.Conv2d(in_channels, ch[0], 3, padding=1)

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

        self.mid_res1 = ResBlock(ch[-1], ch[-1], emb_dim, dropout=dropout)
        self.mid_attn = CrossAttentionBlock(ch[-1], ctx_ch[5])
        self.mid_res2 = ResBlock(ch[-1], ch[-1], emb_dim, dropout=dropout)

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

        self.out_norm = nn.GroupNorm(32, ch[0])
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(ch[0], out_channels, 3, padding=1)

    def forward(
        self,
        x_t:       torch.Tensor,
        x_src:     torch.Tensor,
        direction: torch.Tensor,
        t:         torch.Tensor,
    ) -> torch.Tensor:
        z = torch.cat([x_t, x_src], dim=1)
        h = self.stem(z)

        emb = self.time_emb(t * self.t_scale) + self.direction_emb(direction)

        ctx_ir  = self.encoder_IR(x_src)
        ctx_red = self.encoder_RED(x_src)

        def pick_ctx(scale_key: str) -> torch.Tensor:
            mask = (direction == 0)[:, None, None, None].float()
            return ctx_ir[scale_key] * mask + ctx_red[scale_key] * (1.0 - mask)

        scale_map = {3: "32", 4: "16", 5: "8"}

        skips    = []
        down_idx = 0
        for level, blocks in enumerate(self.enc_res):
            for res in blocks:
                h = res(h, emb)

            if level in self._ATTN_LEVELS:
                h = self.enc_attn[str(level)](h, pick_ctx(scale_map[level]))

            skips.append(h)

            if level < len(self.enc_res) - 1:
                h = self.enc_down[down_idx](h)
                down_idx += 1

        h = self.mid_res1(h, emb)
        h = self.mid_attn(h, pick_ctx("8"))
        h = self.mid_res2(h, emb)

        up_idx = 0
        for dec_idx, blocks in enumerate(self.dec_res):
            enc_level = len(self.enc_res) - 1 - dec_idx
            h = torch.cat([h, skips[enc_level]], dim=1)

            for res in blocks:
                h = res(h, emb)

            if enc_level in self._ATTN_LEVELS:
                h = self.dec_attn[str(dec_idx)](h, pick_ctx(scale_map[enc_level]))

            if dec_idx < len(self.dec_res) - 1:
                h = self.dec_up[up_idx](h)
                up_idx += 1

        return self.out_conv(self.out_act(self.out_norm(h)))
