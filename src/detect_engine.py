from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List

import cv2
from ultralytics import YOLO

from src.config import ALARM_IMAGE_DIR, ALARM_LOG_PATH, ensure_runtime_dirs


@dataclass
class DetectionItem:
    cls_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]


class AlarmRecorder:
    def __init__(self, cooldown_sec: float = 2.0) -> None:
        ensure_runtime_dirs()
        self.cooldown_sec = cooldown_sec
        self.last_alarm_ts = 0.0
        self._init_log()

    def _init_log(self) -> None:
        if not ALARM_LOG_PATH.exists():
            with ALARM_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "confidence", "snapshot"])

    def try_alarm(self, frame, max_conf: float) -> Path | None:
        now = time.time()
        if now - self.last_alarm_ts < self.cooldown_sec:
            return None

        self.last_alarm_ts = now
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = ALARM_IMAGE_DIR / f"alarm_{stamp}.jpg"
        cv2.imwrite(str(snap_path), frame)
        self._log(max_conf, snap_path)
        self._beep()
        return snap_path

    @staticmethod
    def _beep() -> None:
        try:
            import winsound

            winsound.Beep(1800, 300)
            winsound.Beep(1800, 300)
        except Exception:
            pass

    @staticmethod
    def _log(confidence: float, snapshot: Path) -> None:
        with ALARM_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), f"{confidence:.4f}", str(snapshot)])


class EbikeDetector:
    def __init__(
        self,
        model_path: str,
        target_class_name: str = "电动车",
        conf_thres: float = 0.35,
    ) -> None:
        self.model = YOLO(model_path)
        self.target_class_name = target_class_name
        self.conf_thres = conf_thres
        self.alarm = AlarmRecorder()

    def detect(self, frame) -> tuple[Any, List[DetectionItem], float]:
        result = self.model.predict(source=frame, conf=self.conf_thres, verbose=False)[0]
        names = result.names
        boxes = result.boxes

        matched: List[DetectionItem] = []
        max_conf = 0.0
        if boxes is None:
            return frame, matched, max_conf

        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(box.conf[0].item())
            if cls_name != self.target_class_name:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            max_conf = max(max_conf, conf)
            matched.append(DetectionItem(cls_name=cls_name, confidence=conf, xyxy=(x1, y1, x2, y2)))

        annotated = frame.copy()
        for item in matched:
            x1, y1, x2, y2 = item.xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"{item.cls_name} {item.confidence:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if matched:
            self.alarm.try_alarm(annotated, max_conf)

        return annotated, matched, max_conf
