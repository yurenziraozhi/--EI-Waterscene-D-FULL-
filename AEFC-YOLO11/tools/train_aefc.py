"""DDP-friendly training entrypoint with file logging for WaterScenes YOLO11."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--cfg", type=Path, default=Path("configs/train_aefc.yaml"))
    cfg_args, remaining = cfg_parser.parse_known_args()
    cfg = load_yaml(cfg_args.cfg)

    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[cfg_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--data", default="configs/waterscenes_full.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--warmup-epochs", "--warmup_epochs", dest="warmup_epochs", type=float, default=3.0)
    parser.add_argument("--warmup-momentum", "--warmup_momentum", dest="warmup_momentum", type=float, default=0.8)
    parser.add_argument("--warmup-bias-lr", "--warmup_bias_lr", dest="warmup_bias_lr", type=float, default=0.1)
    parser.add_argument("--cos-lr", "--cos_lr", dest="cos_lr", type=str2bool, default=True)
    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/aefc_yolo11")
    parser.add_argument("--name", default="yolo11m_baseline")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--save-period", "--save_period", dest="save_period", type=int, default=-1)
    parser.add_argument("--cache", default=False)
    parser.add_argument("--plots", type=str2bool, default=True)
    parser.add_argument("--exist-ok", "--exist_ok", dest="exist_ok", action="store_true")
    parser.add_argument("--verbose", type=str2bool, default=False)

    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=100)

    parser.add_argument("--use-uiae", action="store_true")
    parser.add_argument("--use-eafc", action="store_true")
    parser.add_argument("--use-mdct", action="store_true")
    parser.add_argument("--uiae-ppn-size", "--uiae_ppn_size", dest="uiae_ppn_size", type=int, default=256)
    parser.add_argument(
        "--uiae-kbl-kernel-size",
        "--uiae_kbl_kernel_size",
        dest="uiae_kbl_kernel_size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--uiae-kbl-kernel-count",
        "--uiae_kbl_kernel_count",
        dest="uiae_kbl_kernel_count",
        type=int,
        default=2,
    )
    parser.add_argument("--uiae-alpha-kbl", "--uiae_alpha_kbl", dest="uiae_alpha_kbl", type=float, default=0.1)
    parser.add_argument("--uiae-blend-init", "--uiae_blend_init", dest="uiae_blend_init", type=float, default=0.05)
    parser.add_argument("--freeze-uiae", "--freeze_uiae", dest="freeze_uiae", type=str2bool, default=False)
    parser.add_argument("--freeze-uiae-epochs", "--freeze_uiae_epochs", dest="freeze_uiae_epochs", type=int, default=0)
    parser.add_argument("--eafc-alpha-init", "--eafc_alpha_init", dest="eafc_alpha_init", type=float, default=0.02)
    parser.add_argument("--lambda-cons", "--lambda_cons", dest="lambda_cons", type=float, default=0.1)
    parser.add_argument("--lambda-param", "--lambda_param", dest="lambda_param", type=float, default=0.01)
    parser.add_argument("--lambda-smooth", "--lambda_smooth", dest="lambda_smooth", type=float, default=0.005)
    parser.add_argument("--cons-start-epoch", "--cons_start_epoch", dest="cons_start_epoch", type=int, default=50)
    parser.add_argument("--mdct-start-epoch", "--mdct_start_epoch", dest="mdct_start_epoch", type=int, default=0)
    parser.add_argument("--mdct-warmup-epochs", "--mdct_warmup_epochs", dest="mdct_warmup_epochs", type=int, default=0)
    parser.add_argument("--freeze-detector", "--freeze_detector", dest="freeze_detector", type=str2bool, default=False)
    parser.add_argument("--freeze-backbone", "--freeze_backbone", dest="freeze_backbone", type=str2bool, default=False)
    parser.add_argument(
        "--condition-aware-enhancement",
        "--condition_aware_enhancement",
        dest="condition_aware_enhancement",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--adverse-lighting-list",
        "--adverse_lighting_list",
        dest="adverse_lighting_list",
        type=Path,
        default=Path("adverse_lighting.txt"),
    )
    parser.add_argument(
        "--adverse-weather-list",
        "--adverse_weather_list",
        dest="adverse_weather_list",
        type=Path,
        default=Path("adverse_weather.txt"),
    )

    parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args.cfg = cfg_args.cfg
    return args


def is_rank0() -> bool:
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1"))
    return rank in {"-1", "0"}


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return str(value)


class FileTrainLogger:
    """Rank-0 file logger for train and validation callbacks."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.enabled = is_rank0()
        self.log_interval = max(1, int(args.log_interval))
        self.log_file = self._resolve_log_file(args)
        self.csv_file = self._resolve_csv_file(args)
        self._fp = None
        self._csv_fp = None
        self._csv_writer = None
        if self.enabled:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self.log_file.open("w", encoding="utf-8", buffering=1)
            self.csv_file.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fp = self.csv_file.open("w", encoding="utf-8", newline="", buffering=1)
            self._csv_writer = csv.DictWriter(
                self._csv_fp,
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
                    "lr",
                    "uiae_blend",
                    "enh_delta_absmax",
                    "enh_delta_mean",
                    "eafc_p3_alpha_mean",
                    "eafc_p4_alpha_mean",
                    "eafc_p5_alpha_mean",
                    "pred_absmax",
                    "pred_zero_fraction",
                    "uiae_trainable_params",
                    "uiae_grad_norm",
                    "uiae_has_grad",
                    "eafc_trainable_params",
                    "eafc_grad_norm",
                    "eafc_has_grad",
                    "mdct_effective_p",
                    "condition_aware",
                    "adverse_fraction",
                    "adverse_count",
                    "normal_count",
                    "enh_delta_adverse_mean",
                    "enh_delta_normal_mean",
                    "diag_alert",
                ],
            )
            self._csv_writer.writeheader()

    def _resolve_log_file(self, args: argparse.Namespace) -> Path:
        if args.log_file is not None:
            return args.log_file
        safe_name = str(args.name).replace("/", "_").replace("\\", "_")
        return args.log_dir / f"{safe_name}.log"

    def _resolve_csv_file(self, args: argparse.Namespace) -> Path:
        safe_name = str(args.name).replace("/", "_").replace("\\", "_")
        return args.log_dir / f"{safe_name}_epoch_metrics.csv"

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
        if self._csv_fp is not None:
            self._csv_fp.close()

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self._fp is None:
            return
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **to_serializable(payload),
        }
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_start(self) -> None:
        self.write(
            "run_start",
            {
                "model": self.args.model,
                "pretrained_backbone": self.args.model,
                "data": self.args.data,
                "imgsz": self.args.imgsz,
                "epochs": self.args.epochs,
                "batch": self.args.batch,
                "device": self.args.device,
                "ddp": "," in str(self.args.device),
                "optimizer": self.args.optimizer,
                "lr0": self.args.lr0,
                "lrf": self.args.lrf,
                "weight_decay": self.args.weight_decay,
                "amp": self.args.amp,
                "seed": self.args.seed,
                "project": self.args.project,
                "name": self.args.name,
                "log_interval": self.log_interval,
                "jsonl_log_file": self.log_file,
                "epoch_metrics_csv": self.csv_file,
                "enabled_modules": {
                    "uiae": self.args.use_uiae,
                    "eafc": self.args.use_eafc,
                    "mdct": self.args.use_mdct,
                },
                "aefc_training": {
                    "freeze_uiae": self.args.freeze_uiae,
                    "freeze_uiae_epochs": self.args.freeze_uiae_epochs,
                    "uiae_blend_init": self.args.uiae_blend_init,
                    "eafc_alpha_init": self.args.eafc_alpha_init,
                    "p_degrade": getattr(self.args, "p_degrade", None),
                    "mdct_start_epoch": self.args.mdct_start_epoch,
                    "mdct_warmup_epochs": self.args.mdct_warmup_epochs,
                    "freeze_detector": self.args.freeze_detector,
                    "freeze_backbone": self.args.freeze_backbone,
                    "condition_aware_enhancement": self.args.condition_aware_enhancement,
                    "adverse_lighting_list": self.args.adverse_lighting_list,
                    "adverse_weather_list": self.args.adverse_weather_list,
                    "lambda_cons": self.args.lambda_cons,
                    "lambda_param": self.args.lambda_param,
                },
            },
        )

    @staticmethod
    def _len_loader(obj: Any, attr: str) -> int | None:
        loader = getattr(obj, attr, None)
        if loader is None:
            return None
        try:
            return len(loader)
        except TypeError:
            return None

    @staticmethod
    def _tensor_to_float(value: Any) -> float | list[float] | str:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return float(value.detach().cpu())
                return [float(v) for v in value.detach().flatten().cpu().tolist()]
        except Exception:
            pass
        if isinstance(value, (int, float)):
            return float(value)
        return str(value)

    def _loss_payload(self, trainer: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        names = getattr(trainer, "loss_names", None)
        loss_items = getattr(trainer, "loss_items", None)
        if names is not None and loss_items is not None:
            values = self._tensor_to_float(loss_items)
            if isinstance(values, list):
                payload["loss_items"] = dict(zip([str(n) for n in names], values))
            else:
                payload["loss_items"] = values
        elif loss_items is not None:
            payload["loss_items"] = self._tensor_to_float(loss_items)

        tloss = getattr(trainer, "tloss", None)
        if tloss is not None:
            payload["train_loss_running"] = self._tensor_to_float(tloss)
        payload.update(self._diagnostics_payload(trainer))
        return payload

    @staticmethod
    def _unwrapped_model(obj: Any) -> Any:
        model = getattr(obj, "model", None)
        if model is None:
            return None
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _metric_value(metrics: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _diagnostics_payload(self, trainer: Any) -> dict[str, Any]:
        model = self._unwrapped_model(trainer)
        if model is None:
            return {}
        diagnostics = getattr(model, "aefc_last_diagnostics", None)
        if not isinstance(diagnostics, dict):
            return {}
        alerts: list[str] = []
        for key in ("input_finite", "enhanced_finite", "pred_finite"):
            if diagnostics.get(key) is False:
                alerts.append(key.replace("_finite", "_nonfinite"))
        pred_absmax = diagnostics.get("pred_absmax")
        if isinstance(pred_absmax, (int, float)) and pred_absmax == 0:
            alerts.append("pred_all_zero")
        pred_zero = diagnostics.get("pred_zero_fraction")
        if isinstance(pred_zero, (int, float)) and pred_zero > 0.98:
            alerts.append("pred_mostly_zero")
        enh_absmax = diagnostics.get("enh_delta_absmax")
        if isinstance(enh_absmax, (int, float)) and enh_absmax > 0.5:
            alerts.append("enh_delta_large")
        payload = {
            "diagnostics": diagnostics,
            "last_degradation": getattr(model, "uiae_last_degradation", "unknown"),
        }
        if alerts:
            payload["diag_alert"] = alerts
        return payload

    def on_train_start(self, trainer: Any) -> None:
        self.write(
            "train_start",
            {
                "save_dir": str(getattr(trainer, "save_dir", "")),
                "train_batches": self._len_loader(trainer, "train_loader"),
                "args": vars(self.args),
            },
        )

    def on_train_batch_end(self, trainer: Any) -> None:
        batch_i = int(getattr(trainer, "batch_i", -1)) + 1
        total = self._len_loader(trainer, "train_loader")
        if batch_i != 1 and batch_i % self.log_interval != 0 and (total is None or batch_i != total):
            return
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        payload = {
            "epoch": epoch,
            "epochs": self.args.epochs,
            "batch": batch_i,
            "total_batches": total,
            "lr": getattr(trainer, "lr", None),
        }
        payload.update(self._loss_payload(trainer))
        self.write("train_batch", payload)

    def on_train_epoch_end(self, trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        payload = {"epoch": epoch, "epochs": self.args.epochs}
        payload.update(self._loss_payload(trainer))
        self.write("train_epoch_end", payload)

    def on_val_start(self, validator: Any) -> None:
        epoch = getattr(getattr(validator, "trainer", None), "epoch", None)
        self.write(
            "val_start",
            {
                "epoch": None if epoch is None else int(epoch) + 1,
                "val_batches": self._len_loader(validator, "dataloader"),
            },
        )

    def on_val_batch_end(self, validator: Any) -> None:
        batch_i = int(getattr(validator, "batch_i", -1)) + 1
        total = self._len_loader(validator, "dataloader")
        if batch_i != 1 and batch_i % self.log_interval != 0 and (total is None or batch_i != total):
            return
        epoch = getattr(getattr(validator, "trainer", None), "epoch", None)
        self.write(
            "val_batch",
            {
                "epoch": None if epoch is None else int(epoch) + 1,
                "batch": batch_i,
                "total_batches": total,
            },
        )

    def on_val_end(self, validator: Any) -> None:
        epoch = getattr(getattr(validator, "trainer", None), "epoch", None)
        metrics_obj = getattr(validator, "metrics", None)
        metrics: dict[str, Any] = {}
        if metrics_obj is not None:
            results_dict = getattr(metrics_obj, "results_dict", None)
            if isinstance(results_dict, dict):
                metrics.update(results_dict)
            for name in ("box.map50", "box.map", "box.mr", "box.mp"):
                root, _, leaf = name.partition(".")
                obj = getattr(metrics_obj, root, None)
                if obj is not None and hasattr(obj, leaf):
                    metrics[name] = getattr(obj, leaf)
        map50 = self._metric_value(metrics, "metrics/mAP50(B)", "box.map50")
        map5095 = self._metric_value(metrics, "metrics/mAP50-95(B)", "box.map")
        alerts: list[str] = []
        if map50 == 0.0:
            alerts.append("val_map50_zero")
        if map5095 == 0.0:
            alerts.append("val_map50_95_zero")
        event_payload = {
            "epoch": None if epoch is None else int(epoch) + 1,
            "metrics": metrics,
        }
        if alerts:
            event_payload["diag_alert"] = alerts
        self.write(
            "val_end",
            event_payload,
        )
        self.write_epoch_csv(validator, metrics)

    @staticmethod
    def _first_number(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list) and value:
            try:
                return float(value[0])
            except (TypeError, ValueError):
                return None
        try:
            import torch

            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return None
                return float(value.detach().flatten()[0].cpu())
        except Exception:
            return None
        return None

    def write_epoch_csv(self, validator: Any, metrics: dict[str, Any]) -> None:
        if not self.enabled or self._csv_writer is None:
            return
        trainer = getattr(validator, "trainer", None)
        epoch = getattr(trainer, "epoch", None)
        loss_items = self._tensor_to_float(getattr(trainer, "tloss", None))
        losses = loss_items if isinstance(loss_items, list) else []
        lr = getattr(trainer, "lr", None)
        if isinstance(lr, dict):
            lr_value = next(iter(lr.values()), None)
        elif isinstance(lr, (list, tuple)):
            lr_value = lr[0] if lr else None
        else:
            lr_value = lr

        row = {
            "epoch": "" if epoch is None else int(epoch) + 1,
            "box_loss": losses[0] if len(losses) > 0 else "",
            "cls_loss": losses[1] if len(losses) > 1 else "",
            "dfl_loss": losses[2] if len(losses) > 2 else "",
            "precision": metrics.get("metrics/precision(B)", metrics.get("box.mp", "")),
            "recall": metrics.get("metrics/recall(B)", metrics.get("box.mr", "")),
            "map50": metrics.get("metrics/mAP50(B)", metrics.get("box.map50", "")),
            "map50_95": metrics.get("metrics/mAP50-95(B)", metrics.get("box.map", "")),
            "fitness": metrics.get("fitness", ""),
            "lr": self._first_number(lr_value) if lr_value is not None else "",
        }
        diag = self._diagnostics_payload(trainer)
        diagnostics = diag.get("diagnostics", {}) if isinstance(diag, dict) else {}
        row.update(
            {
                "uiae_blend": diagnostics.get("uiae_blend", ""),
                "enh_delta_absmax": diagnostics.get("enh_delta_absmax", ""),
                "enh_delta_mean": diagnostics.get("enh_delta_mean", ""),
                "eafc_p3_alpha_mean": diagnostics.get("eafc_p3_alpha_mean", ""),
                "eafc_p4_alpha_mean": diagnostics.get("eafc_p4_alpha_mean", ""),
                "eafc_p5_alpha_mean": diagnostics.get("eafc_p5_alpha_mean", ""),
                "pred_absmax": diagnostics.get("pred_absmax", ""),
                "pred_zero_fraction": diagnostics.get("pred_zero_fraction", ""),
                "uiae_trainable_params": diagnostics.get("uiae_trainable_params", ""),
                "uiae_grad_norm": diagnostics.get("uiae_grad_norm", ""),
                "uiae_has_grad": diagnostics.get("uiae_has_grad", ""),
                "eafc_trainable_params": diagnostics.get("eafc_trainable_params", ""),
                "eafc_grad_norm": diagnostics.get("eafc_grad_norm", ""),
                "eafc_has_grad": diagnostics.get("eafc_has_grad", ""),
                "mdct_effective_p": diagnostics.get("mdct_effective_p", ""),
                "condition_aware": diagnostics.get("condition_aware", ""),
                "adverse_fraction": diagnostics.get("adverse_fraction", ""),
                "adverse_count": diagnostics.get("adverse_count", ""),
                "normal_count": diagnostics.get("normal_count", ""),
                "enh_delta_adverse_mean": diagnostics.get("enh_delta_adverse_mean", ""),
                "enh_delta_normal_mean": diagnostics.get("enh_delta_normal_mean", ""),
                "diag_alert": ";".join(diag.get("diag_alert", [])) if isinstance(diag, dict) else "",
            }
        )
        self._csv_writer.writerow(row)


def assert_supported(args: argparse.Namespace) -> None:
    if (args.use_eafc or args.use_mdct) and not args.use_uiae:
        raise SystemExit("EAFC/MDCT currently use the AEFC trainer path. Enable --use-uiae together with them.")


def export_aefc_env_args(args: argparse.Namespace) -> None:
    custom_keys = [
        "use_uiae",
        "use_eafc",
        "use_mdct",
        "uiae_ppn_size",
        "uiae_kbl_kernel_size",
        "uiae_kbl_kernel_count",
        "uiae_alpha_kbl",
        "uiae_blend_init",
        "freeze_uiae",
        "freeze_uiae_epochs",
        "eafc_alpha_init",
        "p_degrade",
        "p_dark",
        "p_fog",
        "p_rain",
        "p_blur",
        "p_noise",
        "p_reflection",
        "lambda_cons",
        "lambda_param",
        "lambda_smooth",
        "cons_start_epoch",
        "mdct_start_epoch",
        "mdct_warmup_epochs",
        "freeze_detector",
        "freeze_backbone",
        "condition_aware_enhancement",
        "adverse_lighting_list",
        "adverse_weather_list",
        "log_dir",
        "log_file",
        "log_interval",
        "name",
        "epochs",
    ]
    payload = {key: to_serializable(getattr(args, key)) for key in custom_keys if hasattr(args, key)}
    os.environ["AEFC_TRAIN_ARGS"] = json.dumps(payload, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    assert_supported(args)
    export_aefc_env_args(args)

    try:
        from ultralytics import YOLO
        from models.uiae_trainer import UIAETrainer
    except ImportError as exc:
        raise SystemExit(f"Import failed: {exc!r}") from exc

    logger = FileTrainLogger(args)
    logger.log_start()

    model = YOLO(args.model)
    model.add_callback("on_train_start", logger.on_train_start)
    model.add_callback("on_train_batch_end", logger.on_train_batch_end)
    model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)
    model.add_callback("on_val_start", logger.on_val_start)
    model.add_callback("on_val_batch_end", logger.on_val_batch_end)
    model.add_callback("on_val_end", logger.on_val_end)

    start = time.time()
    try:
        trainer_cls = UIAETrainer if args.use_uiae else None
        results = model.train(
            trainer=trainer_cls,
            data=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            warmup_epochs=args.warmup_epochs,
            warmup_momentum=args.warmup_momentum,
            warmup_bias_lr=args.warmup_bias_lr,
            cos_lr=args.cos_lr,
            amp=args.amp,
            seed=args.seed,
            project=args.project,
            name=args.name,
            patience=args.patience,
            save_period=args.save_period,
            cache=args.cache,
            plots=args.plots,
            exist_ok=args.exist_ok,
            verbose=args.verbose,
            val=True,
        )
        logger.write(
            "run_end",
            {
                "status": "success",
                "elapsed_sec": round(time.time() - start, 2),
                "results": str(results),
            },
        )
    except Exception as exc:
        logger.write(
            "run_end",
            {
                "status": "failed",
                "elapsed_sec": round(time.time() - start, 2),
                "error": repr(exc),
            },
        )
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
