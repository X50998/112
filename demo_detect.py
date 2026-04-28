from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.detect_engine import EbikeDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="电动车入梯检测演示（命令行）")
    parser.add_argument("--model", type=str, default="weights/best.pt")
    parser.add_argument("--source", type=str, default="0", help="摄像头id或视频路径")
    parser.add_argument("--target-class", type=str, default="电动车")
    parser.add_argument("--conf", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型: {model_path}")

    source = int(args.source) if args.source.isdigit() else args.source
    detector = EbikeDetector(
        model_path=str(model_path), target_class_name=args.target_class, conf_thres=args.conf
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"视频源打开失败: {source}")

    print("按 q 退出演示窗口")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, matched, max_conf = detector.detect(frame)
        if matched:
            cv2.putText(
                annotated,
                f"ALARM! 电动车 max={max_conf:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
        cv2.imshow("Ebike Elevator Alarm Demo", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
