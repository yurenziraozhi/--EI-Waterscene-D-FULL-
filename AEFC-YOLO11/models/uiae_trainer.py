"""Ultralytics trainer for the UIAE-only ablation."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn
from ultralytics.models.yolo.detect.train import DetectionTrainer

from .uiae import UIAE


_ORIGINAL_DDP = nn.parallel.DistributedDataParallel


def _ddp_find_unused_parameters(*args: Any, **kwargs: Any) -> nn.Module:
    kwargs["find_unused_parameters"] = True
    return _ORIGINAL_DDP(*args, **kwargs)


nn.parallel.DistributedDataParallel = _ddp_find_unused_parameters


def _trainer_arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _forward_with_uiae(self: nn.Module, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Run UIAE inside the YOLO forward path so DDP can track its gradients."""

    if isinstance(x, dict):
        if "img" not in x:
            return self._aefc_base_forward(x, *args, **kwargs)
        batch = dict(x)
        raw_img = batch["img"]
        enhanced, params = self.uiae(raw_img)
        self.uiae_last_params = {key: value.detach() for key, value in params.items()}
        batch["img"] = enhanced.clone()
        out = self._aefc_base_forward(batch, *args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]):
            identity = self.uiae.identity_params.to(dtype=params["all"].dtype, device=params["all"].device)
            param_loss = (params["all"] - identity.unsqueeze(0)).pow(2).mean()
            cons_loss = (enhanced - raw_img).pow(2).mean()
            aux_loss = self.uiae_lambda_param * param_loss + self.uiae_lambda_cons * cons_loss
            return out[0] + aux_loss, out[1]
        return out

    if torch.is_tensor(x):
        enhanced, params = self.uiae(x)
        self.uiae_last_params = {key: value.detach() for key, value in params.items()}
        return self._aefc_base_forward(enhanced.clone(), *args, **kwargs)

    return self._aefc_base_forward(x, *args, **kwargs)


class UIAETrainer(DetectionTrainer):
    """DetectionTrainer that enhances normalized images before YOLO forward."""

    def setup_model(self) -> dict | None:
        ckpt = super().setup_model()
        self._disable_inplace_ops()
        self._freeze_batchnorm_stats()
        self._attach_uiae()
        self._wrap_forward()
        return ckpt

    def _disable_inplace_ops(self) -> None:
        for module in self.model.modules():
            if hasattr(module, "inplace"):
                module.inplace = False

    def _freeze_batchnorm_stats(self) -> None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _attach_uiae(self) -> None:
        if hasattr(self.model, "uiae"):
            return

        uiae = UIAE(
            ppn_size=int(_trainer_arg(self.args, "uiae_ppn_size", 256)),
            kbl_kernel_size=int(_trainer_arg(self.args, "uiae_kbl_kernel_size", 5)),
            kbl_kernel_count=int(_trainer_arg(self.args, "uiae_kbl_kernel_count", 2)),
            alpha_kbl=float(_trainer_arg(self.args, "uiae_alpha_kbl", 0.1)),
            blend_init=float(_trainer_arg(self.args, "uiae_blend_init", 0.05)),
            mode="uiae",
        )
        self.model.add_module("uiae", uiae)
        self.model.uiae_enabled = True
        self.model.uiae_lambda_cons = float(_trainer_arg(self.args, "lambda_cons", 0.1))
        self.model.uiae_lambda_param = float(_trainer_arg(self.args, "lambda_param", 0.01))

    def _wrap_forward(self) -> None:
        if hasattr(self.model, "_aefc_base_forward"):
            return
        self.model._aefc_base_forward = self.model.forward
        self.model.forward = MethodType(_forward_with_uiae, self.model)

    def preprocess_batch(self, batch: dict) -> dict:
        self._freeze_batchnorm_stats()
        return super().preprocess_batch(batch)
