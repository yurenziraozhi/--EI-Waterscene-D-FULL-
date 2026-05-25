"""Validate AEFC-YOLO11 checkpoints with condition-aware UIAE/EAFC enabled."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/aefc_yolo11_eval")
    parser.add_argument("--name", required=True)
    parser.add_argument("--plots", type=str2bool, default=False)
    parser.add_argument("--save-json", "--save_json", dest="save_json", type=str2bool, default=False)
    parser.add_argument("--condition-aware-enhancement", "--condition_aware_enhancement", type=str2bool, default=True)
    parser.add_argument("--adverse-lighting-list", "--adverse_lighting_list", type=Path, default=Path("adverse_lighting.txt"))
    parser.add_argument("--adverse-weather-list", "--adverse_weather_list", type=Path, default=Path("adverse_weather.txt"))
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import unwrap_model

    from models.uiae_trainer import (
        _adverse_mask_from_files,
        _forward_with_aefc,
        _read_id_list,
    )

    # Checkpoints saved after MethodType forward wrapping need this class attribute
    # registered before torch.load() unpickles the DetectionModel.
    DetectionModel._forward_with_aefc = _forward_with_aefc

    adverse_ids = set()
    if args.condition_aware_enhancement:
        for list_path in (args.adverse_lighting_list, args.adverse_weather_list):
            adverse_ids.update(_read_id_list(resolve_project_path(list_path)))
        if not adverse_ids:
            raise FileNotFoundError(
                "No adverse image ids were loaded. Put adverse_lighting.txt and "
                "adverse_weather.txt in AEFC-YOLO11 or pass explicit list paths."
            )

    yolo = YOLO(args.weights)
    aefc_model = unwrap_model(yolo.model)
    aefc_model.condition_aware_enhancement = bool(args.condition_aware_enhancement)
    aefc_model.adverse_image_ids = adverse_ids
    aefc_model.uiae_condition_available = False

    original_preprocess = DetectionValidator.preprocess

    def preprocess_with_condition_mask(self: DetectionValidator, batch: dict[str, Any]) -> dict[str, Any]:
        batch = original_preprocess(self, batch)
        model = unwrap_model(aefc_model)
        model.uiae_adverse_mask = _adverse_mask_from_files(
            model,
            batch.get("im_file"),
            int(batch["img"].shape[0]),
            batch["img"].device,
        )
        return batch

    DetectionValidator.preprocess = preprocess_with_condition_mask
    try:
        metrics = yolo.val(
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            plots=args.plots,
            save_json=args.save_json,
        )
    finally:
        DetectionValidator.preprocess = original_preprocess

    print(metrics)


if __name__ == "__main__":
    main()
