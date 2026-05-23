"""Visualize UIAE enhancement results for selected images."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="Future UIAE/AEFC checkpoint path.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/visualize_enhancement"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        "Enhancement visualization needs a trained UIAE/AEFC checkpoint. "
        f"Output directory prepared: {args.out}"
    )


if __name__ == "__main__":
    main()
