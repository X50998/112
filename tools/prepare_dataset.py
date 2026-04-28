from __future__ import annotations

import argparse
from pathlib import Path


SUPPORTED_IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIX)


def write_yaml(dataset_root: Path, class_name: str, yaml_path: Path) -> None:
    content = "\n".join(
        [
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    yaml_path.write_text(content, encoding="utf-8")


def validate_pairs(dataset_root: Path, split: str) -> tuple[int, int]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    image_files = [
        p for p in image_dir.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIX
    ] if image_dir.exists() else []
    missing_labels = 0
    for img in image_files:
        rel = img.relative_to(image_dir).with_suffix(".txt")
        if not (label_dir / rel).exists():
            missing_labels += 1
    return len(image_files), missing_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="为YOLO训练准备电动车入梯数据集配置")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--class-name", type=str, default="电动车")
    parser.add_argument("--yaml", type=Path, default=Path("dataset/elevator_ebike.yaml"))
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    yaml_path = args.yaml.resolve()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        img_cnt, missing = validate_pairs(dataset_root, split)
        print(f"[{split}] 图像数量: {img_cnt}, 缺失标注: {missing}")

    write_yaml(dataset_root, args.class_name, yaml_path)
    print(f"\n已生成数据集配置: {yaml_path}")
    print("目录规范示例:")
    print("dataset/")
    print("  images/train/*.jpg")
    print("  images/val/*.jpg")
    print("  images/test/*.jpg")
    print("  labels/train/*.txt")
    print("  labels/val/*.txt")
    print("  labels/test/*.txt")
    print("\n标注格式: class x_center y_center width height (归一化坐标)")


if __name__ == "__main__":
    main()
