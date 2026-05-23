"""Auxiliary losses for AEFC-YOLO11."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def parameter_regularization(params: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Keep UIAE parameters close to identity-like values."""

    if isinstance(params, dict):
        bpw = params["bpw"]
        kbl = params["kbl"]
        bpw_identity = torch.tensor(
            [0.2, 0.4, 0.6, 0.8] * 3,
            dtype=bpw.dtype,
            device=bpw.device,
        )
        kbl_identity = torch.full_like(kbl, 0.5)
        return torch.mean((bpw - bpw_identity.view(1, -1)) ** 2) + torch.mean((kbl - kbl_identity) ** 2)

    identity = torch.full_like(params, 0.5)
    return torch.mean((params - identity) ** 2)


def feature_consistency(f_normal: torch.Tensor, f_degraded: torch.Tensor) -> torch.Tensor:
    normal = F.adaptive_avg_pool2d(f_normal, 1).flatten(1)
    degraded = F.adaptive_avg_pool2d(f_degraded, 1).flatten(1)
    return torch.mean(1.0 - F.cosine_similarity(normal, degraded, dim=1))


def attention_smoothness(attn: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(attn[:, :, :, 1:] - attn[:, :, :, :-1]))
    dy = torch.mean(torch.abs(attn[:, :, 1:, :] - attn[:, :, :-1, :]))
    return dx + dy
