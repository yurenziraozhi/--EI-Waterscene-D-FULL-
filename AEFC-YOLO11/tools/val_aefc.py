"""Validation command helper for AEFC-YOLO11 checkpoints."""

from __future__ import annotations

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        "yolo",
        "detect",
        "val",
        f"model={args.weights}",
        f"data={args.data}",
        f"imgsz={args.imgsz}",
        f"split={args.split}",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
