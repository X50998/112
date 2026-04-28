from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SUPPORTED_IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_stems(image_dir: Path) -> list[str]:
    stems: list[str] = []
    if not image_dir.exists():
        return stems
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIX:
            stems.append(p.stem)
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description="按 images/train|val|test 将 labels 根目录分集")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--move", action="store_true", help="将根目录标签移动到分集目录（默认复制）")
    parser.add_argument("--fill-empty", action="store_true", help="为缺失标签补空txt")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    image_root = dataset_root / "images"
    label_root = dataset_root / "labels"
    if not image_root.exists() or not label_root.exists():
        raise FileNotFoundError(f"目录不存在: {image_root} 或 {label_root}")

    copied = 0
    moved = 0
    missing = 0
    for split in ("train", "val", "test"):
        stems = image_stems(image_root / split)
        target_dir = label_root / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            src = label_root / f"{stem}.txt"
            dst = target_dir / f"{stem}.txt"
            if src.exists():
                if args.move:
                    shutil.move(str(src), str(dst))
                    moved += 1
                else:
                    shutil.copy2(src, dst)
                    copied += 1
            else:
                missing += 1
                if args.fill_empty:
                    dst.write_text("", encoding="utf-8")

    print(f"copied={copied}, moved={moved}, missing={missing}, fill_empty={args.fill_empty}")


if __name__ == "__main__":
    main()
