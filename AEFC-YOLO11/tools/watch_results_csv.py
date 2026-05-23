"""Mirror Ultralytics results.csv into logs during or after training."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory containing Ultralytics results.csv.")
    parser.add_argument("--search-root", type=Path, default=Path("runs"), help="Root used to find results.csv.")
    parser.add_argument("--name", default="yolo11m_baseline")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def find_results_csv(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        candidate = args.run_dir / "results.csv"
        if not candidate.exists():
            raise FileNotFoundError(f"results.csv not found: {candidate}")
        return candidate

    candidates = sorted(
        args.search_root.rglob("results.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No results.csv found under {args.search_root}")

    named = [p for p in candidates if args.name in str(p.parent)]
    return named[0] if named else candidates[0]


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def mirror_once(src: Path, dst_csv: Path, dst_log: Path, last_epoch: str | None) -> str | None:
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_csv)

    rows: list[dict[str, str]] = []
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                rows.append(normalize_row(row))

    if not rows:
        return last_epoch

    latest = rows[-1]
    epoch = latest.get("epoch", "")
    if epoch == last_epoch:
        return last_epoch

    event = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": "epoch_metrics",
        "source": str(src),
        "epoch": epoch,
        "metrics": latest,
    }
    with dst_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return epoch


def main() -> None:
    args = parse_args()
    dst_csv = args.log_dir / f"{args.name}_epoch_metrics.csv"
    dst_log = args.log_dir / f"{args.name}.log"
    last_epoch: str | None = None

    while True:
        try:
            src = find_results_csv(args)
            last_epoch = mirror_once(src, dst_csv, dst_log, last_epoch)
        except FileNotFoundError:
            pass

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
