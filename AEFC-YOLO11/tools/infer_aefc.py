"""Inference placeholder for future enhanced/attention visualizations."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-enhanced", action="store_true")
    parser.add_argument("--save-attention", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(
        "AEFC inference visualization is not wired yet. "
        "Use Ultralytics baseline prediction until the model graph is integrated: "
        f"yolo detect predict model={args.weights} source={args.source} imgsz={args.imgsz}"
    )


if __name__ == "__main__":
    main()
