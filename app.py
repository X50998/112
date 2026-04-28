from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.config import ALARM_LOG_PATH, DEFAULT_MODEL_PATH
from src.detect_engine import EbikeDetector


class DetectThread(QThread):
    frame_ready = Signal(QImage)
    status_ready = Signal(str)
    alarm_ready = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.running = False
        self.source: str | int = 0
        self.model_path = str(DEFAULT_MODEL_PATH)
        self.conf_thres = 35
        self.target_class = "电动车"

    def configure(self, source: str | int, model_path: str, conf_thres: int, target_class: str) -> None:
        self.source = source
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.target_class = target_class.strip() or "电动车"

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        if not Path(self.model_path).exists():
            self.status_ready.emit(f"模型不存在: {self.model_path}")
            return

        detector = EbikeDetector(
            model_path=self.model_path,
            target_class_name=self.target_class,
            conf_thres=max(0.01, self.conf_thres / 100),
        )

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status_ready.emit(f"视频源打开失败: {self.source}")
            return

        self.running = True
        self.status_ready.emit("检测已启动")

        while self.running:
            ok, frame = cap.read()
            if not ok:
                self.status_ready.emit("视频流结束或读取失败")
                break

            annotated, matched, max_conf = detector.detect(frame)
            if matched:
                self.alarm_ready.emit(
                    f"检测到电动车! 数量={len(matched)}, 最高置信度={max_conf:.2f}"
                )
            else:
                self.status_ready.emit("正常：未检测到电动车")

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(image)

        cap.release()
        self.running = False
        self.status_ready.emit("检测已停止")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("电动车入梯预警平台 (YOLOv11 + PySide6)")
        self.resize(1200, 760)
        self.thread = DetectThread()
        self._build_ui()
        self._bind_signals()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QGridLayout(root)

        self.video_label = QLabel("等待启动检测...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(900, 600)
        self.video_label.setStyleSheet("background:#111;color:#ddd;border:1px solid #333;")

        self.model_edit = QLineEdit(str(DEFAULT_MODEL_PATH))
        self.source_edit = QLineEdit("0")
        self.class_edit = QLineEdit("电动车")
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(1, 100)
        self.conf_spin.setValue(35)

        self.btn_model = QPushButton("选择模型")
        self.btn_video = QPushButton("选择视频文件")
        self.btn_image = QPushButton("选择图片并检测")
        self.btn_start = QPushButton("开始检测")
        self.btn_stop = QPushButton("停止检测")
        self.btn_stop.setEnabled(False)
        self.btn_open_log = QPushButton("打开报警日志目录")

        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)

        form = QVBoxLayout()
        form.addWidget(QLabel("模型权重(.pt):"))
        form.addWidget(self.model_edit)
        form.addWidget(self.btn_model)
        form.addWidget(QLabel("视频源(0为摄像头或本地视频路径):"))
        form.addWidget(self.source_edit)
        form.addWidget(self.btn_video)
        form.addWidget(self.btn_image)
        form.addWidget(QLabel("目标类别名:"))
        form.addWidget(self.class_edit)
        form.addWidget(QLabel("置信度阈值(%)"))
        form.addWidget(self.conf_spin)

        ops = QHBoxLayout()
        ops.addWidget(self.btn_start)
        ops.addWidget(self.btn_stop)
        form.addLayout(ops)
        form.addWidget(self.btn_open_log)
        form.addWidget(QLabel("状态与报警日志:"))
        form.addWidget(self.status_box)

        layout.addWidget(self.video_label, 0, 0)
        layout.addLayout(form, 0, 1)
        layout.setColumnStretch(0, 4)
        layout.setColumnStretch(1, 2)

    def _bind_signals(self) -> None:
        self.btn_model.clicked.connect(self.pick_model)
        self.btn_video.clicked.connect(self.pick_video)
        self.btn_image.clicked.connect(self.detect_image)
        self.btn_start.clicked.connect(self.start_detect)
        self.btn_stop.clicked.connect(self.stop_detect)
        self.btn_open_log.clicked.connect(self.open_log_dir)
        self.thread.frame_ready.connect(self.show_frame)
        self.thread.status_ready.connect(self.append_status)
        self.thread.alarm_ready.connect(self.append_alarm)

    def pick_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "Model (*.pt)")
        if file_path:
            self.model_edit.setText(file_path)

    def pick_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            self.source_edit.setText(file_path)

    def start_detect(self) -> None:
        if self.thread.isRunning():
            return

        model_path = self.model_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "提示", "请先选择模型文件")
            return

        source_text = self.source_edit.text().strip()
        source: str | int = int(source_text) if source_text.isdigit() else source_text
        self.thread.configure(
            source=source,
            model_path=model_path,
            conf_thres=self.conf_spin.value(),
            target_class=self.class_edit.text(),
        )
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def detect_image(self) -> None:
        if self.thread.isRunning():
            QMessageBox.warning(self, "提示", "请先停止视频检测，再进行图片检测")
            return

        model_path = self.model_edit.text().strip()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "提示", "模型文件不存在，请先选择正确的 .pt 文件")
            return

        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Image (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if not image_path:
            return

        frame = cv2.imread(image_path)
        if frame is None:
            QMessageBox.warning(self, "提示", "图片读取失败，请检查文件")
            return

        detector = EbikeDetector(
            model_path=model_path,
            target_class_name=self.class_edit.text().strip() or "电动车",
            conf_thres=max(0.01, self.conf_spin.value() / 100),
        )
        annotated, matched, max_conf = detector.detect(frame)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.show_frame(image)

        if matched:
            self.append_alarm(
                f"图片检测到电动车! 数量={len(matched)}, 最高置信度={max_conf:.2f}, 文件={image_path}"
            )
        else:
            self.append_status(f"图片检测未发现电动车: {image_path}")

    def stop_detect(self) -> None:
        self.thread.stop()
        self.thread.wait(3000)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def open_log_dir(self) -> None:
        log_dir = ALARM_LOG_PATH.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        QMessageBox.information(self, "日志目录", str(log_dir))

    def show_frame(self, image: QImage) -> None:
        pix = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def append_status(self, msg: str) -> None:
        self.status_box.append(f"[INFO] {msg}")

    def append_alarm(self, msg: str) -> None:
        self.status_box.append(f"[ALARM] {msg}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
