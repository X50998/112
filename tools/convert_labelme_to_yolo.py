from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORTED_IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def shape_to_yolo_line(shape: dict, image_w: int, image_h: int, class_id: int) -> str | None:
    points = shape.get("points") or []
    if not points:
        return None

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    if x_max <= x_min or y_max <= y_min:
        return None

    x_center = ((x_min + x_max) / 2.0) / image_w
    y_center = ((y_min + y_max) / 2.0) / image_h
    width = (x_max - x_min) / image_w
    height = (y_max - y_min) / image_h

    x_center = clamp01(x_center)
    y_center = clamp01(y_center)
    width = clamp01(width)
    height = clamp01(height)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_one_json(json_path: Path, target_label: str, class_id: int) -> tuple[Path, int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_w = int(data.get("imageWidth") or 0)
    image_h = int(data.get("imageHeight") or 0)
    if image_w <= 0 or image_h <= 0:
        raise ValueError(f"图片尺寸缺失或无效: {json_path}")

    lines: list[str] = []
    for shape in data.get("shapes", []):
        if str(shape.get("label", "")).strip() != target_label:
            continue
        yolo_line = shape_to_yolo_line(shape, image_w=image_w, image_h=image_h, class_id=class_id)
        if yolo_line:
            lines.append(yolo_line)

    image_name = data.get("imagePath")
    if image_name:
        label_path = json_path.parent / Path(image_name).with_suffix(".txt").name
    else:
        label_path = json_path.with_suffix(".txt")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return label_path, len(lines)


def create_empty_labels_for_unannotated(source_dir: Path) -> int:
    created = 0
    for image_path in source_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIX:
            continue
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            continue
        txt_path.write_text("", encoding="utf-8")
        created += 1
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="将Labelme JSON标注转换为YOLO TXT标注")
    parser.add_argument("--source", type=Path, required=True, help="包含jpg/json的源目录")
    parser.add_argument("--label-name", type=str, default="电动车", help="需要转换的标签名称")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO类别ID")
    parser.add_argument(
        "--create-empty",
        action="store_true",
        help="为无标注图片创建空txt（作为负样本）",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"源目录不存在: {source}")

    json_files = list(source.rglob("*.json"))
    total_boxes = 0
    converted = 0
    for json_file in json_files:
        try:
            _, n = convert_one_json(json_file, target_label=args.label_name, class_id=args.class_id)
            converted += 1
            total_boxes += n
        except Exception as exc:
            print(f"[WARN] 转换失败: {json_file} -> {exc}")

    empty_created = 0
    if args.create_empty:
        empty_created = create_empty_labels_for_unannotated(source)

    print(f"JSON文件总数: {len(json_files)}")
    print(f"成功转换JSON: {converted}")
    print(f"生成目标框数量: {total_boxes}")
    print(f"补充空标签数量: {empty_created}")


if __name__ == "__main__":
    main()
