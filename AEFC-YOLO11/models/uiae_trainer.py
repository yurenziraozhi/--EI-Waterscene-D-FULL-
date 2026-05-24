"""Ultralytics trainer for the UIAE-only ablation."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn
from ultralytics.models.yolo.detect.train import DetectionTrainer

from .degradation import random_degradation
from .eafc import MultiScaleEAFC
from .uiae import UIAE


_ORIGINAL_DDP = nn.parallel.DistributedDataParallel


def _ddp_find_unused_parameters(*args: Any, **kwargs: Any) -> nn.Module:
    kwargs["find_unused_parameters"] = True
    return _ORIGINAL_DDP(*args, **kwargs)


nn.parallel.DistributedDataParallel = _ddp_find_unused_parameters


def _trainer_arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


DETECT_FEATURE_LAYERS = (16, 19, 22)
DETECT_FEATURE_CHANNELS = (256, 512, 512)


def _tensor_stats(prefix: str, value: torch.Tensor) -> dict[str, float | bool]:
    tensor = value.detach()
    finite = torch.isfinite(tensor)
    if not bool(finite.all()):
        tensor = tensor[finite]
    if tensor.numel() == 0:
        return {
            f"{prefix}_finite": False,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_absmax": float("nan"),
        }
    tensor = tensor.float()
    return {
        f"{prefix}_finite": bool(finite.all()),
        f"{prefix}_mean": float(tensor.mean().cpu()),
        f"{prefix}_std": float(tensor.std(unbiased=False).cpu()),
        f"{prefix}_min": float(tensor.min().cpu()),
        f"{prefix}_max": float(tensor.max().cpu()),
        f"{prefix}_absmax": float(tensor.abs().max().cpu()),
    }


def _prediction_stats(preds: Any) -> dict[str, float | bool | int]:
    tensors: list[torch.Tensor] = []

    def collect(value: Any) -> None:
        if torch.is_tensor(value):
            tensors.append(value.detach())
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(preds)
    if not tensors:
        return {"pred_tensor_count": 0}
    flat = torch.cat([tensor.float().reshape(-1) for tensor in tensors if tensor.numel() > 0])
    if flat.numel() == 0:
        return {"pred_tensor_count": len(tensors), "pred_empty": True}
    stats = _tensor_stats("pred", flat)
    stats["pred_tensor_count"] = len(tensors)
    stats["pred_zero_fraction"] = float((flat == 0).float().mean().cpu())
    return stats


def _forward_layers_until_detect(model: nn.Module, img: torch.Tensor) -> list[torch.Tensor | None]:
    outputs: list[torch.Tensor | None] = []
    x: Any = img
    for module in model.model[:-1]:
        if module.f != -1:
            x = outputs[module.f] if isinstance(module.f, int) else [x if j == -1 else outputs[j] for j in module.f]
        x = module(x)
        outputs.append(x if module.i in model.save else None)
    return outputs


def _aefc_predict(self: nn.Module, raw_img: torch.Tensor) -> tuple[Any, torch.Tensor]:
    if getattr(self, "uiae_frozen", False):
        with torch.no_grad():
            enhanced, params = self.uiae(raw_img)
    else:
        enhanced, params = self.uiae(raw_img)
    self.uiae_last_params = {key: value.detach() for key, value in params.items()}
    diagnostics: dict[str, Any] = {}
    diagnostics.update(_tensor_stats("input", raw_img))
    diagnostics.update(_tensor_stats("enhanced", enhanced))
    diagnostics.update(_tensor_stats("enh_delta", enhanced - raw_img))
    blend = params.get("blend")
    if blend is not None:
        diagnostics["uiae_blend"] = float(blend.detach().float().mean().cpu())

    if getattr(self, "eafc_enabled", False):
        raw_outputs = _forward_layers_until_detect(self, raw_img)
        enh_outputs = _forward_layers_until_detect(self, enhanced)
        raw_features = [raw_outputs[i] for i in DETECT_FEATURE_LAYERS]
        enh_features = [enh_outputs[i] for i in DETECT_FEATURE_LAYERS]
        fused_features, attn = self.eafc(raw_features, enh_features)
        self.eafc_last_attention = [value.detach() for value in attn]
        for idx, (raw_feature, enh_feature, fused_feature, attn_map) in enumerate(
            zip(raw_features, enh_features, fused_features, attn),
            start=3,
        ):
            diagnostics[f"eafc_p{idx}_alpha_mean"] = float(attn_map.detach().float().mean().cpu())
            diagnostics[f"eafc_p{idx}_alpha_min"] = float(attn_map.detach().float().min().cpu())
            diagnostics[f"eafc_p{idx}_alpha_max"] = float(attn_map.detach().float().max().cpu())
            diagnostics[f"eafc_p{idx}_delta_l1"] = float((enh_feature - raw_feature).detach().abs().mean().cpu())
            diagnostics[f"eafc_p{idx}_fused_delta_l1"] = float((fused_feature - raw_feature).detach().abs().mean().cpu())
        preds = self.model[-1](fused_features)
    else:
        preds = self._aefc_base_forward(enhanced.clone())
    diagnostics.update(_prediction_stats(preds))

    identity = self.uiae.identity_params.to(dtype=params["all"].dtype, device=params["all"].device)
    param_loss = (params["all"].detach() - identity.unsqueeze(0)).pow(2).mean()
    cons_loss = (enhanced.detach() - raw_img.detach()).pow(2).mean()
    aux_loss = raw_img.new_tensor(0.0)
    if not getattr(self, "uiae_frozen", False):
        param_loss = (params["all"] - identity.unsqueeze(0)).pow(2).mean()
        cons_loss = (enhanced - raw_img).pow(2).mean()
        aux_loss = self.uiae_lambda_param * param_loss + self.uiae_lambda_cons * cons_loss
    diagnostics["uiae_param_mse"] = float(param_loss.detach().cpu())
    diagnostics["uiae_cons_mse"] = float(cons_loss.detach().cpu())
    diagnostics["aux_loss"] = float(aux_loss.detach().cpu())
    self.aefc_last_diagnostics = diagnostics
    return preds, aux_loss


def _forward_with_aefc(self: nn.Module, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Run UIAE and optional EAFC inside the YOLO forward path."""

    if isinstance(x, dict):
        if "img" not in x:
            return self._aefc_base_forward(x, *args, **kwargs)
        batch = dict(x)
        preds, aux_loss = _aefc_predict(self, batch["img"])
        loss, loss_items = self.loss(batch, preds)
        return loss + aux_loss, loss_items

    if torch.is_tensor(x):
        preds, _ = _aefc_predict(self, x)
        return preds

    return self._aefc_base_forward(x, *args, **kwargs)


class UIAETrainer(DetectionTrainer):
    """DetectionTrainer that enhances normalized images before YOLO forward."""

    def setup_model(self) -> dict | None:
        ckpt = super().setup_model()
        self._disable_inplace_ops()
        self._freeze_batchnorm_stats()
        self._attach_uiae()
        self._attach_eafc()
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
        self.model.uiae_frozen = bool(_trainer_arg(self.args, "freeze_uiae", False))
        if self.model.uiae_frozen:
            self.model.uiae.eval()
            for param in self.model.uiae.parameters():
                param.requires_grad = False
        self.model.uiae_lambda_cons = float(_trainer_arg(self.args, "lambda_cons", 0.1))
        self.model.uiae_lambda_param = float(_trainer_arg(self.args, "lambda_param", 0.01))

    def _attach_eafc(self) -> None:
        if not bool(_trainer_arg(self.args, "use_eafc", False)):
            return
        if hasattr(self.model, "eafc"):
            return
        alpha_init = float(_trainer_arg(self.args, "eafc_alpha_init", 0.02))
        self.model.add_module("eafc", MultiScaleEAFC(DETECT_FEATURE_CHANNELS, alpha_init=alpha_init))
        self.model.eafc_enabled = True

    def _wrap_forward(self) -> None:
        if hasattr(self.model, "_aefc_base_forward"):
            return
        self.model._aefc_base_forward = self.model.forward
        self.model.forward = MethodType(_forward_with_aefc, self.model)

    def preprocess_batch(self, batch: dict) -> dict:
        self._freeze_batchnorm_stats()
        batch = super().preprocess_batch(batch)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.uiae_last_degradation = "none"
        if bool(_trainer_arg(self.args, "use_mdct", False)):
            p_degrade = float(_trainer_arg(self.args, "p_degrade", 0.5))
            if torch.rand((), device=batch["img"].device).item() < p_degrade:
                degraded, name = random_degradation(batch["img"])
                batch["img"] = degraded
                model.uiae_last_degradation = name
        return batch
