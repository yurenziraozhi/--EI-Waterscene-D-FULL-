"""Prepare WaterScenes files for Ultralytics YOLO training and testing."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def read_ids(list_path: Path) -> list[str]:
    ids: list[str] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item:
            continue
        stem = Path(item).stem
        ids.append(stem)
    return ids


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def reset_split(out_root: Path, split: str) -> None:
    for kind in ("images", "labels"):
        target = (out_root / kind / split).resolve()
        expected_root = (out_root / kind).resolve()
        if expected_root not in target.parents:
            raise ValueError(f"Refusing to clear path outside output root: {target}")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def materialize_split(
    ids: list[str],
    image_dir: Path,
    label_dir: Path,
    out_root: Path,
    split: str,
    mode: str,
) -> dict[str, int]:
    reset_split(out_root, split)
    unique_ids = unique_keep_order(ids)
    missing_images: list[str] = []
    missing_labels: list[str] = []

    for stem in unique_ids:
        image = find_image(image_dir, stem)
        label = label_dir / f"{stem}.txt"
        if image is None:
            missing_images.append(stem)
            continue
        if not label.exists():
            missing_labels.append(stem)
            continue
        link_or_copy(image, out_root / "images" / split / image.name, mode)
        link_or_copy(label, out_root / "labels" / split / label.name, mode)

    if missing_images or missing_labels:
        report = out_root / f"missing_{split}.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "\n".join(
                [
                    "[missing_images]",
                    *missing_images,
                    "",
                    "[missing_labels]",
                    *missing_labels,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        raise FileNotFoundError(
            f"{split}: missing_images={len(missing_images)}, "
            f"missing_labels={len(missing_labels)}. See {report}"
        )

    return {
        "listed": len(ids),
        "unique": len(unique_ids),
        "duplicates": len(ids) - len(unique_ids),
    }


def write_yaml(path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "path: waterscenes_yolo",
                "train: images/train",
                f"val: images/{split}",
                f"test: images/{split}",
                "",
                "names:",
                "  0: pier",
                "  1: buoy",
                "  2: sailor",
                "  3: ship",
                "  4: boat",
                "  5: vessel",
                "  6: kayak",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".."), help="Source root containing image/ and detection/yolo/.")
    parser.add_argument("--image-dir", default="image")
    parser.add_argument("--label-dir", default="detection/yolo")
    parser.add_argument("--train-list", default="train.txt")
    parser.add_argument("--val-list", default="val.txt")
    parser.add_argument("--test-list", default="test.txt")
    parser.add_argument("--lighting-list", default="adverse_lighting.txt")
    parser.add_argument("--weather-list", default="adverse_weather.txt")
    parser.add_argument("--out", type=Path, default=Path("datasets/waterscenes_yolo"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    image_dir = root / args.image_dir
    label_dir = root / args.label_dir
    out_root = args.out.resolve()

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    train_ids = read_ids(root / args.train_list)
    val_ids = read_ids(root / args.val_list)
    test_ids = read_ids(root / args.test_list)

    split_sets = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }
    overlaps = {
        "train_val": split_sets["train"] & split_sets["val"],
        "train_test": split_sets["train"] & split_sets["test"],
        "val_test": split_sets["val"] & split_sets["test"],
    }
    bad_overlaps = {name: ids for name, ids in overlaps.items() if ids}
    if bad_overlaps:
        detail = ", ".join(f"{name}={len(ids)}" for name, ids in bad_overlaps.items())
        raise ValueError(f"Train/val/test lists must be disjoint, got overlaps: {detail}")

    source_ids = {path.stem for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS}
    listed_ids = set().union(*split_sets.values())
    missing_from_lists = source_ids - listed_ids
    extra_in_lists = listed_ids - source_ids
    if missing_from_lists or extra_in_lists:
        report = out_root / "split_coverage_report.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "\n".join(
                [
                    "[missing_from_train_val_test]",
                    *sorted(missing_from_lists),
                    "",
                    "[listed_but_no_source_image]",
                    *sorted(extra_in_lists),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        raise ValueError(
            f"Split coverage mismatch: missing_from_lists={len(missing_from_lists)}, "
            f"extra_in_lists={len(extra_in_lists)}. See {report}"
        )

    train_stats = materialize_split(train_ids, image_dir, label_dir, out_root, "train", args.mode)
    val_stats = materialize_split(val_ids, image_dir, label_dir, out_root, "val", args.mode)
    test_stats = materialize_split(test_ids, image_dir, label_dir, out_root, "test", args.mode)

    lighting_ids = read_ids(root / args.lighting_list)
    lighting_stats = materialize_split(lighting_ids, image_dir, label_dir, out_root, "adverse_lighting", args.mode)

    weather_ids = read_ids(root / args.weather_list)
    weather_stats = materialize_split(weather_ids, image_dir, label_dir, out_root, "adverse_weather", args.mode)

    write_yaml(args.configs_dir / "waterscenes_full.yaml", "val")
    write_yaml(args.configs_dir / "waterscenes_adverse_lighting.yaml", "adverse_lighting")
    write_yaml(args.configs_dir / "waterscenes_adverse_weather.yaml", "adverse_weather")

    full_yaml = args.configs_dir / "waterscenes_full.yaml"
    full_yaml.write_text(
        "\n".join(
            [
                "path: waterscenes_yolo",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: pier",
                "  1: buoy",
                "  2: sailor",
                "  3: ship",
                "  4: boat",
                "  5: vessel",
                "  6: kayak",
                "",
            ]
        ),
        encoding="utf-8",
    )

    for name, stats in [
        ("train", train_stats),
        ("val", val_stats),
        ("test", test_stats),
        ("adverse_lighting", lighting_stats),
        ("adverse_weather", weather_stats),
    ]:
        print(
            f"{name}: listed={stats['listed']} unique={stats['unique']} "
            f"duplicates={stats['duplicates']}"
        )


if __name__ == "__main__":
    main()
