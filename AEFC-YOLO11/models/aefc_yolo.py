"""High-level AEFC-YOLO11 building blocks.

This module intentionally keeps the Ultralytics integration thin. The UIAE and
EAFC modules are implemented independently so the first engineering milestone
can validate them before patching a YOLO11 backbone graph.
"""

from __future__ import annotations

import torch
from torch import nn

from .uiae import UIAE
from .eafc import MultiScaleEAFC


class AEFCEnhancementStem(nn.Module):
    """Image-level enhancement stem returning original and enhanced images."""

    def __init__(self) -> None:
        super().__init__()
        self.uiae = UIAE()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        enhanced, params = self.uiae(x)
        return {"raw": x, "enhanced": enhanced, "params": params}


class AEFCFeatureCalibrator(nn.Module):
    """Feature-level calibrator for raw/enhanced multi-scale features."""

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        self.eafc = MultiScaleEAFC(channels)

    def forward(
        self,
        raw_features: list[torch.Tensor],
        enhanced_features: list[torch.Tensor],
    ) -> dict[str, list[torch.Tensor]]:
        fused, attn = self.eafc(raw_features, enhanced_features)
        return {"features": fused, "attention": attn}
