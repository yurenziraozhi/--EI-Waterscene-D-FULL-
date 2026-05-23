"""Unified image-adaptive enhancement modules."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ParameterPredictor(nn.Module):
    """Predict BPW and KBL parameters from a low-resolution image."""

    def __init__(self, out_dim: int, ppn_size: int = 256) -> None:
        super().__init__()
        self.ppn_size = ppn_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=False),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=False),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=False),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.SiLU(inplace=False),
            nn.Linear(32, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_small = F.interpolate(
            x,
            size=(self.ppn_size, self.ppn_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.fc(self.features(x_small))


class UIAE(nn.Module):
    """BPW + KBL unified image-adaptive enhancement block."""

    def __init__(
        self,
        ppn_size: int = 256,
        kbl_kernel_size: int = 5,
        kbl_kernel_count: int = 2,
        alpha_kbl: float = 0.1,
        mode: str = "uiae",
    ) -> None:
        super().__init__()
        if mode not in {"bpw", "kbl", "uiae"}:
            raise ValueError(f"Unsupported UIAE mode: {mode}")
        if kbl_kernel_size % 2 == 0:
            raise ValueError("kbl_kernel_size must be odd.")
        self.mode = mode
        self.bpw_dim = 12
        self.kbl_kernel_size = kbl_kernel_size
        self.kbl_kernel_count = kbl_kernel_count
        self.kbl_dim = kbl_kernel_count * kbl_kernel_size * kbl_kernel_size * 3
        self.alpha_kbl = alpha_kbl
        self.ppn = ParameterPredictor(out_dim=self.bpw_dim + self.kbl_dim, ppn_size=ppn_size)
        self._init_identity_parameters()

        identity = torch.zeros(kbl_kernel_size, kbl_kernel_size, dtype=torch.float32)
        identity[kbl_kernel_size // 2, kbl_kernel_size // 2] = 1.0
        self.register_buffer("identity_kernel", identity.view(1, 1, kbl_kernel_size, kbl_kernel_size))

    def _init_identity_parameters(self) -> None:
        """Start close to identity enhancement for stable detector fine-tuning."""

        last_linear = self.ppn.fc[-2]
        if not isinstance(last_linear, nn.Linear):
            return
        bpw_identity = torch.tensor([0.2, 0.4, 0.6, 0.8] * 3, dtype=torch.float32)
        kbl_identity = torch.full((self.kbl_dim,), 0.5, dtype=torch.float32)
        target = torch.cat([bpw_identity, kbl_identity]).clamp(1e-4, 1 - 1e-4)
        bias = torch.logit(target)
        self.register_buffer("identity_params", target)
        nn.init.zeros_(last_linear.weight)
        with torch.no_grad():
            last_linear.bias.copy_(bias)

    @staticmethod
    def _bpw_filter(x: torch.Tensor, bpw_params: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        internal = bpw_params.view(batch, channels, 4, 1, 1)
        zeros = torch.zeros(batch, channels, 1, 1, 1, dtype=x.dtype, device=x.device)
        ones = torch.ones(batch, channels, 1, 1, 1, dtype=x.dtype, device=x.device)
        control = torch.cat([zeros, internal, ones], dim=2)

        t = x.clamp(0.0, 1.0).unsqueeze(2)
        one_minus_t = 1.0 - t
        basis = torch.cat(
            [
                one_minus_t**5,
                5.0 * t * one_minus_t**4,
                10.0 * t**2 * one_minus_t**3,
                10.0 * t**3 * one_minus_t**2,
                5.0 * t**4 * one_minus_t,
                t**5,
            ],
            dim=2,
        )
        return torch.sum(basis * control, dim=2).clamp(0.0, 1.0)

    def _kbl_filter(self, x: torch.Tensor, kbl_params: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        k = self.kbl_kernel_size
        kernels = kbl_params.view(batch, channels, self.kbl_kernel_count, k, k)
        residual = self.alpha_kbl * (kernels - 0.5).mean(dim=2)
        base = self.identity_kernel.to(dtype=x.dtype, device=x.device)
        kernels = base + residual.reshape(batch * channels, 1, k, k)

        x_grouped = x.reshape(1, batch * channels, height, width)
        out = F.conv2d(x_grouped, kernels, padding=k // 2, groups=batch * channels)
        return out.reshape(batch, channels, height, width).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        params = self.ppn(x)
        bpw_params = params[:, : self.bpw_dim]
        kbl_params = params[:, self.bpw_dim :]

        if self.mode == "bpw":
            x_enh = self._bpw_filter(x, bpw_params)
        elif self.mode == "kbl":
            x_enh = self._kbl_filter(x, kbl_params)
        else:
            x_bpw = self._bpw_filter(x, bpw_params)
            x_enh = self._kbl_filter(x_bpw, kbl_params)
        return x_enh, {
            "all": params,
            "bpw": bpw_params,
            "kbl": kbl_params,
        }
