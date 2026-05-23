"""Online degradation operators for robust training."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def darken(x: torch.Tensor, gamma_range: tuple[float, float] = (1.5, 4.0)) -> torch.Tensor:
    gamma = random.uniform(*gamma_range)
    return torch.clamp(x, 0.0, 1.0) ** gamma


def add_fog(x: torch.Tensor, t_range: tuple[float, float] = (0.4, 0.9), atmosphere: float = 0.8) -> torch.Tensor:
    t = random.uniform(*t_range)
    return torch.clamp(x * t + atmosphere * (1.0 - t), 0.0, 1.0)


def add_noise(x: torch.Tensor, sigma_range: tuple[float, float] = (0.01, 0.05)) -> torch.Tensor:
    sigma = random.uniform(*sigma_range)
    return torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)


def blur(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pad = kernel_size // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)


def add_reflection(x: torch.Tensor, strength_range: tuple[float, float] = (0.1, 0.35)) -> torch.Tensor:
    strength = random.uniform(*strength_range)
    luminance = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    mask = torch.sigmoid(12.0 * (luminance - 0.65))
    return torch.clamp(x + strength * mask, 0.0, 1.0)


def add_rain(x: torch.Tensor, strength_range: tuple[float, float] = (0.05, 0.18)) -> torch.Tensor:
    strength = random.uniform(*strength_range)
    rain = torch.zeros_like(x[:, :1])
    _, _, h, w = rain.shape
    for offset in range(-h, w, 20):
        rr = torch.arange(h, device=x.device)
        cc = rr + offset
        valid = (cc >= 0) & (cc < w)
        rain[:, :, rr[valid], cc[valid]] = 1.0
    rain = blur(rain.repeat(1, x.shape[1], 1, 1), kernel_size=3)
    return torch.clamp(x + strength * rain, 0.0, 1.0)


def random_degradation(x: torch.Tensor) -> tuple[torch.Tensor, str]:
    ops = [
        ("dark", darken),
        ("fog", add_fog),
        ("rain", add_rain),
        ("blur", blur),
        ("noise", add_noise),
        ("reflection", add_reflection),
    ]
    name, fn = random.choice(ops)
    return fn(x), name
