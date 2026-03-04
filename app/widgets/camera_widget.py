"""
CameraWidget — OpenCV camera feed inside a QLabel
==================================================
Provides START / STOP / single-frame capture.
Emits ``frame_ready(np.ndarray)`` on every grabbed frame and
``captured(np.ndarray)`` when the user clicks *Capture*.
"""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class CameraWidget(QWidget):
    """Live camera preview with capture support."""

    frame_ready = pyqtSignal(np.ndarray)   # every grabbed frame
    captured    = pyqtSignal(np.ndarray)   # single-shot capture

    def __init__(self, camera_index: int = 0, fps: int = 30, parent=None):
        super().__init__(parent)
        self._camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._last_frame: np.ndarray | None = None

        # Timer drives the grab loop
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / fps))
        self._timer.timeout.connect(self._grab)

        # UI
        self._label = QLabel("Camera Off")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background:#222; color:#aaa; font-size:14px;")
        self._label.setMinimumSize(320, 240)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label)

    # ── public API ───────────────────────────────────────
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_frame(self) -> np.ndarray | None:
        return self._last_frame

    def set_camera_index(self, idx: int) -> None:
        was_running = self._running
        if was_running:
            self.stop()
        self._camera_index = idx
        if was_running:
            self.start()

    def start(self) -> bool:
        if self._running:
            return True
        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            self._label.setText("Cannot open camera")
            return False
        self._running = True
        self._timer.start()
        return True

    def stop(self) -> None:
        self._timer.stop()
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._label.setText("Camera Off")

    def toggle(self) -> None:
        if self._running:
            self.stop()
        else:
            self.start()

    def capture(self) -> np.ndarray | None:
        """Grab current frame and emit ``captured`` signal."""
        if self._last_frame is not None:
            self.captured.emit(self._last_frame.copy())
            return self._last_frame.copy()
        return None

    # ── internal ─────────────────────────────────────────
    def _grab(self) -> None:
        if not self._cap or not self._cap.isOpened():
            return
        ok, frame = self._cap.read()
        if not ok:
            return
        self._last_frame = frame
        self.frame_ready.emit(frame)
        self._show(frame)

    def _show(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    # ── cleanup ──────────────────────────────────────────
    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
