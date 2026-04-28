from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


SUPPORTED_IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gather_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    image_files = [
        p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIX
    ]
    pairs: list[tuple[Path, Path]] = []
    missing_labels = 0
    for img in image_files:
        label = img.with_suffix(".txt")
        if label.exists():
            pairs.append((img, label))
        else:
            missing_labels += 1
    print(f"找到图片: {len(image_files)}")
    print(f"找到图片+标注配对: {len(pairs)}")
    print(f"缺失标注图片: {missing_labels}")
    return pairs


def ensure_structure(dataset_root: Path) -> None:
    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_items(
    items: list[tuple[Path, Path]], train_ratio: float, val_ratio: float, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    random.seed(seed)
    random.shuffle(items)
    total = len(items)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    train_items = items[:train_count]
    val_items = items[train_count : train_count + val_count]
    test_items = items[train_count + val_count :]
    return train_items, val_items, test_items


def copy_split(items: list[tuple[Path, Path]], split: str, dataset_root: Path) -> None:
    for img, lbl in items:
        img_target = dataset_root / "images" / split / img.name
        lbl_target = dataset_root / "labels" / split / lbl.name
        shutil.copy2(img, img_target)
        shutil.copy2(lbl, lbl_target)


def main() -> None:
    parser = argparse.ArgumentParser(description="从平铺目录导入YOLO数据并切分train/val/test")
    parser.add_argument("--source", type=Path, required=True, help="源目录：图片和同名txt标注放在一起")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear", action="store_true", help="导入前清空dataset/images和dataset/labels")
    args = parser.parse_args()

    source = args.source.resolve()
    dataset_root = args.dataset_root.resolve()
    if not source.exists():
        raise FileNotFoundError(f"源目录不存在: {source}")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("比例错误，需满足: train_ratio > 0, val_ratio >= 0, train_ratio + val_ratio < 1")

    if args.clear:
        shutil.rmtree(dataset_root / "images", ignore_errors=True)
        shutil.rmtree(dataset_root / "labels", ignore_errors=True)

    ensure_structure(dataset_root)
    pairs = gather_pairs(source)
    if not pairs:
        print("未找到可用数据对，请确认源目录存在图片和同名txt标注。")
        return

    train_items, val_items, test_items = split_items(
        pairs, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    copy_split(train_items, "train", dataset_root)
    copy_split(val_items, "val", dataset_root)
    copy_split(test_items, "test", dataset_root)

    print("\n导入完成:")
    print(f"train: {len(train_items)}")
    print(f"val:   {len(val_items)}")
    print(f"test:  {len(test_items)}")
    print(f"目标目录: {dataset_root}")


if __name__ == "__main__":
    main()
