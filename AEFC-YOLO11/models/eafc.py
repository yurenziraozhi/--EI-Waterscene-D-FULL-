"""Enhancement-aware feature calibration modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class EAFC(nn.Module):
    """Fuse raw and enhanced features with a learned reliability map."""

    def __init__(self, channels: int, alpha_init: float = 0.02) -> None:
        super().__init__()
        alpha_init = min(max(alpha_init, 1e-4), 1.0 - 1e-4)
        self.attn = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self._init_near_raw(alpha_init)

    def _init_near_raw(self, alpha_init: float) -> None:
        last = self.attn[-2]
        if not isinstance(last, nn.Conv2d):
            return
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.constant_(last.bias, torch.logit(torch.tensor(alpha_init)).item())

    def forward(self, f_raw: torch.Tensor, f_enh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        diff = f_enh - f_raw
        alpha = self.attn(torch.cat([f_raw, f_enh, diff], dim=1))
        return f_raw + alpha * diff, alpha


class MultiScaleEAFC(nn.Module):
    """Apply EAFC to P3/P4/P5-style feature lists."""

    def __init__(self, channels: Sequence[int], alpha_init: float = 0.02) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(EAFC(ch, alpha_init=alpha_init) for ch in channels)

    def forward(
        self,
        raw_features: Sequence[torch.Tensor],
        enhanced_features: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if len(raw_features) != len(self.blocks) or len(enhanced_features) != len(self.blocks):
            raise ValueError("Feature count must match configured EAFC scales.")
        fused: list[torch.Tensor] = []
        attn_maps: list[torch.Tensor] = []
        for block, raw, enh in zip(self.blocks, raw_features, enhanced_features):
            out, attn = block(raw, enh)
            fused.append(out)
            attn_maps.append(attn)
        return fused, attn_maps
