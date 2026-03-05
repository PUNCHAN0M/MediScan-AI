"""
PillCameraWidget — Camera feed with YOLO-based pill segmentation
================================================================
Uses ``YOLODetector`` to detect and segment pills in realtime.
Shows annotated camera feed + segmented pill crops below.
Supports BBOX/CENTER draw mode toggle.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QComboBox,
)


def _enum_cameras(max_test: int = 5) -> list[int]:
    """Return indices of available cameras."""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available or [0]


class PillCameraWidget(QWidget):
    """Camera with YOLO pill segmentation, bbox/center toggle, and crop strip.

    Signals
    -------
    crops_ready(list)
        Emits list of cropped pill ``np.ndarray`` (BGR, 256×256) each frame.
    """

    crops_ready = pyqtSignal(list)
    started = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._yolo = None
        self._cap = None
        self._running = False
        self._draw_mode = "bbox"  # "bbox" | "center"
        self._last_crops: list[np.ndarray] = []

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._grab)

        self._strip_pool: list[QLabel] = []   # reuse pill labels

        self._init_ui()

    def _init_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # Camera selector
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self._cam_combo = QComboBox()
        for idx in _enum_cameras():
            self._cam_combo.addItem(f"Camera {idx}", idx)
        saved = self._settings.get("camera_index", 0)
        ci = self._cam_combo.findData(saved)
        if ci >= 0:
            self._cam_combo.setCurrentIndex(ci)
        cam_row.addWidget(self._cam_combo, stretch=1)
        lay.addLayout(cam_row)

        # Buttons: START / BBOX toggle
        btn_row = QHBoxLayout()

        self._btn_toggle_cam = QPushButton("START CAMERA")
        self._btn_toggle_cam.setCheckable(True)
        self._btn_toggle_cam.setStyleSheet(
            "QPushButton{background:#1976D2; color:white; font-weight:bold;"
            " padding:8px 16px; border:2px solid #1565C0; border-radius:4px;}"
            "QPushButton:checked{background:#ff1744; border-color:#d50000;}"
            "QPushButton:hover{opacity:0.9;}"
        )
        self._btn_toggle_cam.toggled.connect(self._toggle_camera)
        btn_row.addWidget(self._btn_toggle_cam)

        self._btn_toggle_draw = QPushButton("BBOX")
        self._btn_toggle_draw.setCheckable(True)
        self._btn_toggle_draw.setStyleSheet(
            "QPushButton{background:#555; color:white; font-weight:bold;"
            " padding:8px 16px; border:2px solid #333; border-radius:4px;}"
            "QPushButton:checked{background:#2962ff; border-color:#0039cb;}"
        )
        self._btn_toggle_draw.clicked.connect(self._toggle_draw)
        btn_row.addWidget(self._btn_toggle_draw)

        lay.addLayout(btn_row)

        # Camera preview label
        self._cam_label = QLabel("Camera Off")
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setStyleSheet(
            "background:#1a1a1a; color:#888; font-size:14px; border:2px solid #444;"
        )
        self._cam_label.setMinimumSize(300, 200)
        self._cam_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        lay.addWidget(self._cam_label, stretch=3)

        # Segmented pill strip
        strip_label = QLabel("Segmented Pills")
        strip_label.setStyleSheet("font-weight:bold; font-size:11px; padding:2px;")
        lay.addWidget(strip_label)

        self._pill_scroll = QScrollArea()
        self._pill_scroll.setWidgetResizable(True)
        self._pill_scroll.setMinimumHeight(100)
        self._pill_scroll.setMaximumHeight(130)
        self._pill_scroll.setStyleSheet("background:#222; border:1px solid #444;")

        self._pill_strip = QWidget()
        self._pill_strip_layout = QHBoxLayout(self._pill_strip)
        self._pill_strip_layout.setContentsMargins(4, 4, 4, 4)
        self._pill_strip_layout.setSpacing(4)
        self._pill_strip_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._pill_scroll.setWidget(self._pill_strip)
        lay.addWidget(self._pill_scroll, stretch=0)

    # ── public API ───────────────────────────────────────

    @property
    def last_crops(self) -> list[np.ndarray]:
        return list(self._last_crops)

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Start camera + YOLO segmentation."""
        if self._running:
            return
        cam_idx = self._cam_combo.currentData()
        if cam_idx is None:
            cam_idx = self._settings.get("camera_index", 0)
        self._cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(cam_idx)
        if not self._cap.isOpened():
            self._cam_label.setText("Cannot open camera")
            self._btn_toggle_cam.setChecked(False)
            return
        self._load_yolo()
        self._running = True
        self._timer.start()
        self.started.emit()

    def stop(self):
        """Stop camera."""
        self._timer.stop()
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._cam_label.setText("Camera Off")
        # Clear pill strip and crops cache
        self._last_crops.clear()
        self._clear_pill_strip()
        self.stopped.emit()

    def _clear_pill_strip(self):
        """Hide all pills in the strip (reuse pool)."""
        for lbl in self._strip_pool:
            lbl.setVisible(False)

    # ── internal ─────────────────────────────────────────

    def _load_yolo(self):
        """Lazy-load YOLODetector model."""
        if self._yolo is not None:
            return
        try:
            from detection.yolo_detector import YOLODetector
            import torch

            model_path = self._settings.get(
                "segmentation_model",
                "model/detection/pill-detection-best-2.onnx",
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._yolo = YOLODetector(
                model_path=model_path,
                img_size=self._settings.get("seg_img_size", 512),
                conf=self._settings.get("seg_conf", 0.5),
                iou=self._settings.get("seg_iou", 0.6),
                device=device,
                target_size=self._settings.get("img_size", 256),
                pad=self._settings.get("seg_pad", 5),
                bg_value=0,
                retina_masks=True,
                enable_tracking=False,
            )
        except Exception as e:
            print(f"[PillCameraWidget] Failed to load YOLODetector: {e}")
            self._yolo = None

    def _toggle_camera(self, checked: bool):
        if checked:
            self.start()
            self._btn_toggle_cam.setText("STOP CAMERA")
        else:
            self.stop()
            self._btn_toggle_cam.setText("START CAMERA")

    def _toggle_draw(self):
        if self._btn_toggle_draw.isChecked():
            self._draw_mode = "center"
            self._btn_toggle_draw.setText("CENTER")
        else:
            self._draw_mode = "bbox"
            self._btn_toggle_draw.setText("BBOX")

    def _grab(self):
        if not self._cap or not self._cap.isOpened():
            return
        ok, frame = self._cap.read()
        if not ok:
            return

        crops = []
        infos = []
        display = frame.copy()

        if self._yolo is not None:
            try:
                crops, infos = self._yolo.detect_and_crop(frame)
                # Draw annotations on display
                for info in infos:
                    x1, y1, x2, y2 = info["bbox"]
                    conf = info.get("conf", 0.0)
                    color = (0, 255, 0)

                    if self._draw_mode == "bbox":
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        label = f"{conf:.2f}"
                        cv2.putText(display, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                    cv2.LINE_AA)
                    else:  # center
                        cx = int(info["center"][0])
                        cy = int(info["center"][1])
                        cv2.circle(display, (cx, cy), 8, color, -1)
                        cv2.circle(display, (cx, cy), 10, (255, 255, 255), 1)
            except Exception as e:
                print(f"[PillCameraWidget] YOLO error: {e}")

        self._last_crops = crops
        self.crops_ready.emit(crops)

        # Show annotated frame
        self._show_frame(display)
        # Update pill strip
        self._update_pill_strip(crops)

    def _show_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self._cam_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._cam_label.setPixmap(scaled)

    def _update_pill_strip(self, crops: list):
        """Update pill strip — reuse QLabel widgets from pool."""
        needed = min(len(crops), 20)

        # Grow pool if needed
        while len(self._strip_pool) < needed:
            lbl = QLabel()
            lbl.setFixedSize(80, 80)
            lbl.setStyleSheet("border:1px solid #555;")
            self._pill_strip_layout.addWidget(lbl)
            self._strip_pool.append(lbl)

        # Update visible labels
        for i in range(needed):
            self._strip_pool[i].setPixmap(self._cv2pixmap(crops[i], 80))
            if not self._strip_pool[i].isVisible():
                self._strip_pool[i].setVisible(True)

        # Hide surplus
        for i in range(needed, len(self._strip_pool)):
            if self._strip_pool[i].isVisible():
                self._strip_pool[i].setVisible(False)

    @staticmethod
    def _cv2pixmap(img: np.ndarray, size: int) -> QPixmap:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        h, w = rgb.shape[:2]
        ch = 3 if rgb.ndim == 3 else 1
        fmt = QImage.Format.Format_RGB888 if ch == 3 else QImage.Format.Format_Grayscale8
        qimg = QImage(rgb.data, w, h, ch * w, fmt)
        return QPixmap.fromImage(qimg).scaled(
            size, size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
