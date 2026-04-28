from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def count_positive_samples(label_dir: Path) -> tuple[int, int]:
    """返回(非空标签文件数, 总目标框数)。"""
    if not label_dir.exists():
        return 0, 0
    positive_images = 0
    total_boxes = 0
    for txt_file in label_dir.rglob("*.txt"):
        lines = [line for line in txt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            positive_images += 1
            total_boxes += len(lines)
    return positive_images, total_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练电动车入梯YOLOv11检测模型")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="基础模型或预训练权重路径")
    parser.add_argument("--data", type=str, default="dataset/elevator_ebike.yaml", help="数据集yaml路径")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="训练设备：auto(自动), cpu, 0/1 等GPU编号",
    )
    parser.add_argument("--project", type=str, default="runs/elevator_ebike")
    parser.add_argument("--name", type=str, default="yolo11_train")
    parser.add_argument("--copy-best-to", type=str, default="weights/best.pt", help="训练结束后复制best.pt到此路径")
    parser.add_argument(
        "--min-positive-images",
        type=int,
        default=30,
        help="最小正样本图片数阈值，低于此值会告警",
    )
    parser.add_argument(
        "--strict-data-check",
        action="store_true",
        help="开启后，正样本低于阈值时直接中止训练",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据集配置文件: {data_path}")
    data_cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_root = Path(data_cfg.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (data_path.parent / dataset_root).resolve()
    split_missing = []
    for split_key in ("train", "val"):
        split_rel = data_cfg.get(split_key)
        split_dir = (dataset_root / str(split_rel)).resolve()
        if not split_dir.exists():
            split_missing.append(str(split_dir))
    if split_missing:
        missing_text = "\n".join(split_missing)
        raise FileNotFoundError(
            "数据集目录缺失，训练无法开始。缺失路径:\n"
            f"{missing_text}\n\n"
            "请先整理数据集目录，或执行:\n"
            "venv\\Scripts\\python tools\\import_flat_dataset.py --source <你的图片目录> --clear"
        )

    # 训练前小数据告警：避免训练出几乎不可用的模型
    train_image_dir = (dataset_root / str(data_cfg.get("train"))).resolve()
    val_image_dir = (dataset_root / str(data_cfg.get("val"))).resolve()
    train_label_dir = train_image_dir.parent.parent / "labels" / train_image_dir.name
    val_label_dir = val_image_dir.parent.parent / "labels" / val_image_dir.name

    train_pos, train_boxes = count_positive_samples(train_label_dir)
    val_pos, val_boxes = count_positive_samples(val_label_dir)
    print(
        f"数据集统计: train正样本={train_pos}, train目标框={train_boxes}, "
        f"val正样本={val_pos}, val目标框={val_boxes}"
    )
    if train_pos < args.min_positive_images:
        warn_msg = (
            f"警告: train正样本仅 {train_pos}，低于阈值 {args.min_positive_images}。"
            "模型可能出现漏检严重、mAP极低。"
        )
        if args.strict_data_check:
            raise RuntimeError(f"{warn_msg} 已启用 --strict-data-check，训练中止。")
        print(warn_msg)
    if val_pos == 0:
        print("警告: val中没有正样本，验证指标将不可靠。建议补充验证集正样本。")

    device = args.device
    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"自动选择训练设备: {device}")
    elif args.device != "cpu" and not torch.cuda.is_available():
        print(
            f"检测到当前环境无CUDA，但收到 device={args.device}，将自动回退为 cpu。"
        )
        device = "cpu"

    model = YOLO(args.weights)
    train_result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        workers=2,
        cache=False,
    )

    best_path = Path(train_result.save_dir) / "weights" / "best.pt"
    print("\n训练结束。最佳模型通常位于:")
    print(str(best_path))

    if best_path.exists():
        target = Path(args.copy_best_to)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, target)
        print(f"已自动复制 best.pt 到: {target}")
    else:
        print("未找到 best.pt，请检查训练日志是否报错。")


if __name__ == "__main__":
    main()
