"""Ultralytics trainer for the UIAE-only ablation."""

from __future__ import annotations

import csv
import json
import os
from types import MethodType
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._diag_batch_i = 0
        self._diag_log_fp = None
        self._diag_csv_fp = None
        self._diag_csv = None
        self._setup_internal_logging()
        self.add_callback("on_train_batch_end", self._log_train_batch)
        self.add_callback("on_fit_epoch_end", self._log_fit_epoch)
        self.add_callback("teardown", self._close_internal_logging)

    def setup_model(self) -> dict | None:
        ckpt = super().setup_model()
        self._disable_inplace_ops()
        self._freeze_batchnorm_stats()
        self._attach_uiae()
        self._attach_eafc()
        self._wrap_forward()
        return ckpt

    def build_optimizer(self, model: nn.Module, name: str = "auto", lr: float = 0.001, momentum: float = 0.9,
                        decay: float = 1e-5, iterations: float = 1e5):
        if bool(_trainer_arg(self.args, "train_eafc_only", False)):
            unwrapped = unwrap_model(model)
            for param_name, param in unwrapped.named_parameters():
                param.requires_grad = param_name.startswith("eafc.")
            trainable = [param for param in unwrapped.parameters() if param.requires_grad]
            return torch.optim.AdamW(trainable, lr=lr, betas=(momentum, 0.999), weight_decay=decay)
        return super().build_optimizer(model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations)

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
        freeze_epochs = int(_trainer_arg(self.args, "freeze_uiae_epochs", 0))
        if bool(_trainer_arg(self.args, "freeze_uiae", False)):
            freeze_epochs = max(freeze_epochs, int(_trainer_arg(self.args, "epochs", 200)) + 1)
        self.model.uiae_freeze_epochs = freeze_epochs
        self.model.uiae_frozen = self.model.uiae_freeze_epochs > 0
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

    @staticmethod
    def _is_rank0() -> bool:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1"))
        return rank in {"-1", "0"}

    def _setup_internal_logging(self) -> None:
        if not self._is_rank0():
            return
        log_file = _trainer_arg(self.args, "log_file", None)
        log_dir = Path(str(_trainer_arg(self.args, "log_dir", "logs")))
        name = str(_trainer_arg(self.args, "name", "aefc_train")).replace("/", "_").replace("\\", "_")
        self._diag_log_path = Path(str(log_file)) if log_file else log_dir / f"{name}.log"
        self._diag_csv_path = log_dir / f"{name}_epoch_metrics.csv"
        self._diag_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._diag_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._diag_log_fp = self._diag_log_path.open("w", encoding="utf-8", buffering=1)
        self._diag_csv_fp = self._diag_csv_path.open("w", encoding="utf-8", newline="", buffering=1)
        self._diag_csv = csv.DictWriter(
            self._diag_csv_fp,
            fieldnames=[
                "epoch",
                "box_loss",
                "cls_loss",
                "dfl_loss",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "fitness",
                "uiae_blend",
                "enh_delta_absmax",
                "eafc_p3_alpha_mean",
                "eafc_p4_alpha_mean",
                "eafc_p5_alpha_mean",
                "pred_absmax",
                "pred_zero_fraction",
                "diag_alert",
            ],
        )
        self._diag_csv.writeheader()
        self._write_diag("trainer_log_start", {"log_file": str(self._diag_log_path), "csv_file": str(self._diag_csv_path)})

    def _close_internal_logging(self, *_: Any, **__: Any) -> None:
        if self._diag_log_fp is not None:
            self._diag_log_fp.close()
            self._diag_log_fp = None
        if self._diag_csv_fp is not None:
            self._diag_csv_fp.close()
            self._diag_csv_fp = None

    def _write_diag(self, event: str, payload: dict[str, Any]) -> None:
        if self._diag_log_fp is None:
            return
        record = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **payload}
        self._diag_log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _float_value(value: Any) -> float | str:
        if torch.is_tensor(value):
            if value.numel() == 1:
                return float(value.detach().cpu())
            return str([float(v) for v in value.detach().flatten().cpu().tolist()])
        if isinstance(value, (int, float)):
            return float(value)
        return "" if value is None else str(value)

    def _diag_alerts(self, diagnostics: dict[str, Any], metrics: dict[str, Any] | None = None) -> list[str]:
        alerts: list[str] = []
        for key in ("input_finite", "enhanced_finite", "pred_finite"):
            if diagnostics.get(key) is False:
                alerts.append(key.replace("_finite", "_nonfinite"))
        pred_zero = diagnostics.get("pred_zero_fraction")
        if isinstance(pred_zero, (int, float)) and pred_zero > 0.98:
            alerts.append("pred_mostly_zero")
        pred_absmax = diagnostics.get("pred_absmax")
        if isinstance(pred_absmax, (int, float)) and pred_absmax == 0:
            alerts.append("pred_all_zero")
        if metrics:
            if metrics.get("metrics/mAP50(B)", metrics.get("box.map50")) == 0:
                alerts.append("val_map50_zero")
            if metrics.get("metrics/mAP50-95(B)", metrics.get("box.map")) == 0:
                alerts.append("val_map50_95_zero")
        return alerts

    def _current_loss_items(self) -> dict[str, Any]:
        loss_items = getattr(self, "loss_items", None)
        if loss_items is None:
            return {}
        values = loss_items.detach().flatten().cpu().tolist() if torch.is_tensor(loss_items) else loss_items
        if isinstance(values, list):
            keys = ["box_loss", "cls_loss", "dfl_loss"]
            return {key: values[idx] for idx, key in enumerate(keys) if idx < len(values)}
        return {"loss_items": self._float_value(values)}

    def _log_train_batch(self, *_: Any, **__: Any) -> None:
        total = len(self.train_loader) if getattr(self, "train_loader", None) is not None else None
        interval = max(1, int(_trainer_arg(self.args, "log_interval", 100)))
        if self._diag_batch_i != 1 and self._diag_batch_i % interval != 0 and self._diag_batch_i != total:
            return
        model = unwrap_model(self.model)
        diagnostics = getattr(model, "aefc_last_diagnostics", {})
        payload = {
            "epoch": int(getattr(self, "epoch", 0)) + 1,
            "batch": self._diag_batch_i,
            "total_batches": total,
            "loss": self._float_value(getattr(self, "loss", None)),
            "loss_items": self._current_loss_items(),
            "last_degradation": getattr(model, "uiae_last_degradation", "unknown"),
            "diagnostics": diagnostics,
        }
        alerts = self._diag_alerts(diagnostics)
        if alerts:
            payload["diag_alert"] = alerts
        self._write_diag("train_batch", payload)

    def _maybe_update_freeze_state(self) -> None:
        model = unwrap_model(self.model)
        freeze_epochs = int(getattr(model, "uiae_freeze_epochs", 0))
        should_freeze = int(getattr(self, "epoch", 0)) < freeze_epochs
        if getattr(model, "uiae_frozen", False) == should_freeze:
            return
        model.uiae_frozen = should_freeze
        for param in model.uiae.parameters():
            param.requires_grad = not should_freeze
        if should_freeze:
            model.uiae.eval()
        else:
            model.uiae.train()
        self._write_diag(
            "freeze_state_update",
            {"epoch": int(getattr(self, "epoch", 0)) + 1, "uiae_frozen": should_freeze},
        )

    def _log_fit_epoch(self, *_: Any, **__: Any) -> None:
        metrics = self.metrics if isinstance(self.metrics, dict) else {}
        model = unwrap_model(self.model)
        diagnostics = getattr(model, "aefc_last_diagnostics", {})
        alerts = self._diag_alerts(diagnostics, metrics)
        payload = {
            "epoch": int(getattr(self, "epoch", 0)) + 1,
            "metrics": metrics,
            "diagnostics": diagnostics,
        }
        if alerts:
            payload["diag_alert"] = alerts
        self._write_diag("fit_epoch_end", payload)
        if self._diag_csv is not None:
            loss_items = self._current_loss_items()
            self._diag_csv.writerow(
                {
                    "epoch": int(getattr(self, "epoch", 0)) + 1,
                    "box_loss": loss_items.get("box_loss", ""),
                    "cls_loss": loss_items.get("cls_loss", ""),
                    "dfl_loss": loss_items.get("dfl_loss", ""),
                    "precision": metrics.get("metrics/precision(B)", metrics.get("box.mp", "")),
                    "recall": metrics.get("metrics/recall(B)", metrics.get("box.mr", "")),
                    "map50": metrics.get("metrics/mAP50(B)", metrics.get("box.map50", "")),
                    "map50_95": metrics.get("metrics/mAP50-95(B)", metrics.get("box.map", "")),
                    "fitness": metrics.get("fitness", ""),
                    "uiae_blend": diagnostics.get("uiae_blend", ""),
                    "enh_delta_absmax": diagnostics.get("enh_delta_absmax", ""),
                    "eafc_p3_alpha_mean": diagnostics.get("eafc_p3_alpha_mean", ""),
                    "eafc_p4_alpha_mean": diagnostics.get("eafc_p4_alpha_mean", ""),
                    "eafc_p5_alpha_mean": diagnostics.get("eafc_p5_alpha_mean", ""),
                    "pred_absmax": diagnostics.get("pred_absmax", ""),
                    "pred_zero_fraction": diagnostics.get("pred_zero_fraction", ""),
                    "diag_alert": ";".join(alerts),
                }
            )

    def _wrap_forward(self) -> None:
        if hasattr(self.model, "_aefc_base_forward"):
            return
        self.model._aefc_base_forward = self.model.forward
        self.model.forward = MethodType(_forward_with_aefc, self.model)

    def preprocess_batch(self, batch: dict) -> dict:
        self._freeze_batchnorm_stats()
        self._maybe_update_freeze_state()
        self._diag_batch_i += 1
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
