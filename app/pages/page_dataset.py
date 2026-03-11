"""
Page: Dataset — Manage training data & bad data + collect pills via camera
==========================================================================
Layout
------
Two sub-tabs:  Training Data (data_train_defection)  |  Bad Data (data_bad)
Each tab:
    Left  : camera feed + detected pills grid
    Right : class/subclass tree  + create/rename/delete  + Save button

Camera mutex:
    Emits ``camera_started`` / ``camera_stopped`` signals so MainWindow
    can close the camera on other pages (Verify ↔ Dataset).
"""
from __future__ import annotations

import cv2
import os
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QListWidget, QListWidgetItem,
    QGroupBox, QSizePolicy, QScrollArea, QMessageBox,
    QTabWidget, QInputDialog, QFileDialog, QFrame,
)

from detection.yolo_detector import YOLODetector as YOLOTracking


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def _enum_cameras(max_test: int = 5) -> list[int]:
    available: list[int] = []
    prev = os.environ.get("OPENCV_LOG_LEVEL", "")
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    try:
        for i in range(max_test):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
    finally:
        if prev:
            os.environ["OPENCV_LOG_LEVEL"] = prev
        else:
            os.environ.pop("OPENCV_LOG_LEVEL", None)
    return available or [0]


def _digital_zoom(frame: np.ndarray, zoom: float = 1.5) -> np.ndarray:
    if zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    nw, nh = int(w / zoom), int(h / zoom)
    x1, y1 = (w - nw) // 2, (h - nh) // 2
    return cv2.resize(frame[y1:y1 + nh, x1:x1 + nw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────
#  Single data-root panel (reused for both tabs)
# ─────────────────────────────────────────────
class _DataPanel(QWidget):
    """
    One panel managing a data root directory.

    Features:
      - Browse main_class / sub_class in a two-level list
      - Create / Rename / Delete main_class and sub_class
      - Camera feed → YOLO crop → save pills into selected sub_class
    """

    camera_started = pyqtSignal()
    camera_stopped = pyqtSignal()

    def __init__(self, root_dir: Path, settings: dict, label: str = "Data", parent=None):
        super().__init__(parent)
        self._root = root_dir
        self._root.mkdir(parents=True, exist_ok=True)
        self._settings = settings
        self._label = label

        self._cap: Optional[cv2.VideoCapture] = None
        self._yolo: Optional[YOLOTracking] = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._process_frame)
        self._running = False

        # Capture buffer: list of (crop_img, widget_frame)
        self._capture_buffer: list[np.ndarray] = []

        self._selected_main: Optional[str] = None
        self._selected_sub: Optional[str] = None

        self._init_ui()
        self._refresh_tree()

    # ══════════════════ UI ══════════════════════════════
    def _init_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)
        main.setSpacing(6)

        # ── Left: camera feed only ──
        self._cam_label = QLabel("Camera Off")
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setStyleSheet(
            "background:#1a1a1a; color:#888; font-size:14px; border:2px solid #444;"
        )
        self._cam_label.setMinimumSize(420, 300)
        self._cam_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )

        # ── Right column ──
        right = QVBoxLayout()
        right.setSpacing(6)
        right.setContentsMargins(0, 0, 0, 0)

        # ── Camera controls (top-right, like page_verify) ──
        cam_group = QGroupBox("Camera")
        cam_group.setStyleSheet(
            "QGroupBox{font-size:12px; font-weight:bold; "
            "border:1px solid #aaa; padding:6px; padding-top:18px;}"
        )
        cam_lay = QVBoxLayout(cam_group)
        self._cam_combo = QComboBox()
        for idx in _enum_cameras():
            self._cam_combo.addItem(f"Camera {idx}", idx)
        cam_lay.addWidget(self._cam_combo)

        btn_row = QHBoxLayout()
        self._btn_cam = QPushButton("START")
        self._btn_cam.setCheckable(True)
        self._btn_cam.setStyleSheet(
            "QPushButton{background:#1976D2; color:white; font-weight:bold;"
            " padding:8px 16px; border:2px solid #1565C0; border-radius:4px;}"
            "QPushButton:checked{background:#ff1744; border-color:#d50000;}"
        )
        self._btn_cam.toggled.connect(self._toggle_camera)
        btn_row.addWidget(self._btn_cam)

        self._btn_capture = QPushButton("Capture")
        self._btn_capture.setStyleSheet(
            "QPushButton{background:#F57C00; color:white; font-weight:bold;"
            " padding:8px 16px; border:2px solid #E65100; border-radius:4px;}"
            "QPushButton:hover{background:#FB8C00;}"
        )
        self._btn_capture.clicked.connect(self._capture)
        btn_row.addWidget(self._btn_capture)
        cam_lay.addLayout(btn_row)

        self._lbl_detect = QLabel("Detected: 0")
        self._lbl_detect.setStyleSheet("font-size:11px; font-weight:bold; padding:2px;")
        cam_lay.addWidget(self._lbl_detect)
        right.addWidget(cam_group)

        # ── Capture buffer strip ──
        cap_group = QGroupBox("Captured (pending save)")
        cap_group.setStyleSheet(
            "QGroupBox{font-size:11px; font-weight:bold; "
            "border:1px solid #F57C00; padding:4px; padding-top:16px;}"
        )
        cap_lay = QVBoxLayout(cap_group)

        cap_scroll = QScrollArea()
        cap_scroll.setWidgetResizable(True)
        cap_scroll.setMinimumHeight(90)
        cap_scroll.setMaximumHeight(120)
        self._cap_inner = QWidget()
        self._cap_layout = QHBoxLayout(self._cap_inner)
        self._cap_layout.setContentsMargins(4, 4, 4, 4)
        self._cap_layout.setSpacing(6)
        self._cap_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        cap_scroll.setWidget(self._cap_inner)
        cap_lay.addWidget(cap_scroll)

        save_row = QHBoxLayout()
        self._lbl_captured = QLabel("0 captured")
        self._lbl_captured.setStyleSheet("font-weight:bold; font-size:11px;")
        save_row.addWidget(self._lbl_captured)
        save_row.addStretch()
        self._btn_save = QPushButton("Save All")
        self._btn_save.setStyleSheet(
            "QPushButton{background:#388E3C; color:white; font-weight:bold;"
            " padding:6px 14px; border-radius:3px;}"
            "QPushButton:hover{background:#43A047;}"
        )
        self._btn_save.clicked.connect(self._save_buffer)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setStyleSheet(
            "QPushButton{background:#757575; color:white; font-weight:bold;"
            " padding:6px 12px; border-radius:3px;}"
        )
        self._btn_clear.clicked.connect(self._clear_buffer)
        save_row.addWidget(self._btn_save)
        save_row.addWidget(self._btn_clear)
        cap_lay.addLayout(save_row)
        right.addWidget(cap_group)

        # ── Main class list ──
        mc_group = QGroupBox("Main Class")
        mc_group.setStyleSheet(
            "QGroupBox{font-size:12px; font-weight:bold; border:1px solid #aaa; padding:6px; padding-top:18px;}"
        )
        mc_lay = QVBoxLayout(mc_group)
        self._main_list = QListWidget()
        self._main_list.currentItemChanged.connect(self._on_main_selected)
        mc_lay.addWidget(self._main_list)

        mc_btns = QHBoxLayout()
        self._btn_add_main = QPushButton("+ Add")
        self._btn_add_main.clicked.connect(self._add_main_class)
        self._btn_rename_main = QPushButton("Rename")
        self._btn_rename_main.clicked.connect(self._rename_main_class)
        self._btn_del_main = QPushButton("Delete")
        self._btn_del_main.setStyleSheet("color:red;")
        self._btn_del_main.clicked.connect(self._delete_main_class)
        mc_btns.addWidget(self._btn_add_main)
        mc_btns.addWidget(self._btn_rename_main)
        mc_btns.addWidget(self._btn_del_main)
        mc_lay.addLayout(mc_btns)
        right.addWidget(mc_group)

        # ── Sub class list ──
        sc_group = QGroupBox("Sub Class")
        sc_group.setStyleSheet(
            "QGroupBox{font-size:12px; font-weight:bold; border:1px solid #aaa; padding:6px; padding-top:18px;}"
        )
        sc_lay = QVBoxLayout(sc_group)
        self._sub_list = QListWidget()
        self._sub_list.currentItemChanged.connect(self._on_sub_selected)
        sc_lay.addWidget(self._sub_list)

        sc_btns = QHBoxLayout()
        self._btn_add_sub = QPushButton("+ Add")
        self._btn_add_sub.clicked.connect(self._add_sub_class)
        self._btn_rename_sub = QPushButton("Rename")
        self._btn_rename_sub.clicked.connect(self._rename_sub_class)
        self._btn_del_sub = QPushButton("Delete")
        self._btn_del_sub.setStyleSheet("color:red;")
        self._btn_del_sub.clicked.connect(self._delete_sub_class)
        sc_btns.addWidget(self._btn_add_sub)
        sc_btns.addWidget(self._btn_rename_sub)
        sc_btns.addWidget(self._btn_del_sub)
        sc_lay.addLayout(sc_btns)
        right.addWidget(sc_group)

        # ── Info ──
        self._lbl_path = QLabel(f"Root: {self._root}")
        self._lbl_path.setStyleSheet("font-size:10px; color:#666; padding:4px;")
        self._lbl_images = QLabel("Images: —")
        self._lbl_images.setStyleSheet("font-size:11px; font-weight:bold; padding:2px;")
        right.addWidget(self._lbl_path)
        right.addWidget(self._lbl_images)
        right.addStretch()

        right_w = QWidget()
        right_w.setLayout(right)
        right_w.setFixedWidth(310)

        main.addWidget(self._cam_label, stretch=3)
        main.addWidget(right_w, stretch=0)

    # ══════════════════ Tree ════════════════════════════
    def _refresh_tree(self):
        self._main_list.clear()
        self._sub_list.clear()
        self._selected_main = None
        self._selected_sub = None
        if not self._root.exists():
            return
        for d in sorted(self._root.iterdir()):
            if d.is_dir():
                self._main_list.addItem(d.name)
        self._lbl_images.setText("Images: —")

    def _on_main_selected(self, current: Optional[QListWidgetItem], _prev):
        self._sub_list.clear()
        self._selected_sub = None
        if current is None:
            self._selected_main = None
            return
        self._selected_main = current.text()
        main_dir = self._root / self._selected_main
        if main_dir.exists():
            subs = sorted(d for d in main_dir.iterdir() if d.is_dir())
            if subs:
                for s in subs:
                    count = self._count_images(s)
                    self._sub_list.addItem(f"{s.name}  ({count})")
            else:
                # No subdirs — count images directly
                count = self._count_images(main_dir)
                self._lbl_images.setText(f"Images: {count}")

    def _on_sub_selected(self, current: Optional[QListWidgetItem], _prev):
        if current is None:
            self._selected_sub = None
            self._lbl_images.setText("Images: —")
            return
        raw = current.text().split("  (")[0]
        self._selected_sub = raw
        sub_dir = self._root / (self._selected_main or "") / self._selected_sub
        count = self._count_images(sub_dir)
        self._lbl_images.setText(f"Images: {count}")

    @staticmethod
    def _count_images(d: Path) -> int:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        if not d.exists():
            return 0
        return sum(1 for f in d.rglob("*") if f.suffix.lower() in exts)

    # ══════════════════ CRUD ════════════════════════════
    def _add_main_class(self):
        name, ok = QInputDialog.getText(self, "New Main Class", "Class name:")
        if ok and name.strip():
            path = self._root / name.strip()
            path.mkdir(parents=True, exist_ok=True)
            self._refresh_tree()

    def _rename_main_class(self):
        if not self._selected_main:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename Main Class", "New name:", text=self._selected_main,
        )
        if ok and new_name.strip() and new_name.strip() != self._selected_main:
            src = self._root / self._selected_main
            dst = self._root / new_name.strip()
            if dst.exists():
                QMessageBox.warning(self, "Error", f"'{new_name.strip()}' already exists.")
                return
            src.rename(dst)
            self._refresh_tree()

    def _delete_main_class(self):
        if not self._selected_main:
            return
        reply = QMessageBox.question(
            self, "Delete Main Class",
            f"Delete '{self._selected_main}' and ALL its contents?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            path = self._root / self._selected_main
            if path.exists():
                shutil.rmtree(path)
            self._refresh_tree()

    def _add_sub_class(self):
        if not self._selected_main:
            QMessageBox.information(self, "Info", "Select a main class first.")
            return
        name, ok = QInputDialog.getText(self, "New Sub Class", "Subclass name:")
        if ok and name.strip():
            path = self._root / self._selected_main / name.strip()
            path.mkdir(parents=True, exist_ok=True)
            # Refresh sub-list
            item = self._main_list.currentItem()
            if item:
                self._on_main_selected(item, None)

    def _rename_sub_class(self):
        if not self._selected_main or not self._selected_sub:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename Sub Class", "New name:", text=self._selected_sub,
        )
        if ok and new_name.strip() and new_name.strip() != self._selected_sub:
            src = self._root / self._selected_main / self._selected_sub
            dst = self._root / self._selected_main / new_name.strip()
            if dst.exists():
                QMessageBox.warning(self, "Error", f"'{new_name.strip()}' already exists.")
                return
            src.rename(dst)
            item = self._main_list.currentItem()
            if item:
                self._on_main_selected(item, None)

    def _delete_sub_class(self):
        if not self._selected_main or not self._selected_sub:
            return
        reply = QMessageBox.question(
            self, "Delete Sub Class",
            f"Delete '{self._selected_sub}' and ALL its images?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            path = self._root / self._selected_main / self._selected_sub
            if path.exists():
                shutil.rmtree(path)
            item = self._main_list.currentItem()
            if item:
                self._on_main_selected(item, None)

    # ══════════════════ Camera ══════════════════════════
    def _init_yolo(self):
        if self._yolo is not None:
            return
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = str(self._settings.get("segmentation_model",
                         "weights/detection/pill-detection-best-2.onnx"))
        self._yolo = YOLOTracking(
            model_path=model_path,
            img_size=1280,
            conf=0.25,
            iou=0.6,
            device=device,
            target_size=256,
            pad=5,
            bg_value=0,
            retina_masks=True,
            enable_tracking=False,
        )

    def _toggle_camera(self, checked: bool):
        if checked:
            self._start_camera()
            self._btn_cam.setText("STOP Camera")
        else:
            self._stop_camera()
            self._btn_cam.setText("START Camera")

    def _start_camera(self):
        if self._running:
            return
        cam_idx = self._cam_combo.currentData()
        if cam_idx is None:
            cam_idx = 0

        self._cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(cam_idx)
        if not self._cap.isOpened():
            self._cam_label.setText("Cannot open camera")
            self._btn_cam.setChecked(False)
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self._init_yolo()
        self._running = True
        self._timer.start()
        self.camera_started.emit()

    def _stop_camera(self):
        self._timer.stop()
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._cam_label.setText("Camera Off")
        self._lbl_detect.setText("Detected: 0")
        self.camera_stopped.emit()

    @property
    def is_camera_running(self) -> bool:
        return self._running

    def force_stop_camera(self):
        """Called externally to release camera for another page."""
        if self._running:
            self._btn_cam.setChecked(False)  # triggers _toggle_camera(False)

    def _process_frame(self):
        if not self._cap or not self._cap.isOpened():
            return
        ok, frame = self._cap.read()
        if not ok:
            return
        frame = _digital_zoom(frame, zoom=1.5)
        self._current_frame = frame  # remember for capture

        if self._yolo is not None:
            try:
                crops, infos, preview = self._yolo.process_frame(
                    frame, draw=True, grid=False,
                )
                self._current_crops = list(crops)
                self._lbl_detect.setText(f"Detected: {len(crops)}")
                display = preview if preview is not None else frame
                self._show_frame(display)
                return
            except Exception as e:
                print(f"[DatasetPage] YOLO error: {e}")

        self._current_frame = frame
        self._current_crops = []
        self._show_frame(frame)

    # ══════════════════ Capture / Save ══════════════════
    def _capture(self):
        """Buffer current detected crops for review before saving."""
        crops = getattr(self, "_current_crops", [])
        if not crops:
            QMessageBox.information(self, "Capture", "No pills detected in current frame.")
            return
        for crop in crops:
            self._add_to_buffer(crop)

    def _add_to_buffer(self, crop: np.ndarray):
        idx = len(self._capture_buffer)
        self._capture_buffer.append(crop.copy())

        # Thumbnail frame with delete button
        frame_w = QFrame()
        frame_w.setFixedSize(72, 88)
        frame_w.setStyleSheet("border:1px solid #F57C00; background:#fff8f0;")
        flay = QVBoxLayout(frame_w)
        flay.setContentsMargins(2, 2, 2, 2)
        flay.setSpacing(1)

        thumb = QLabel()
        thumb.setFixedSize(64, 64)
        thumb.setPixmap(self._cv2pixmap(crop, 64))
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        flay.addWidget(thumb)

        btn_del = QPushButton("✕")
        btn_del.setFixedHeight(18)
        btn_del.setStyleSheet(
            "QPushButton{background:#ff1744; color:white; font-size:10px;"
            " font-weight:bold; border:none; border-radius:2px;}"
            "QPushButton:hover{background:#d50000;}"
        )
        # Use default arg to capture current index at lambda creation time
        buf_idx = idx
        btn_del.clicked.connect(lambda _, i=buf_idx: self._remove_from_buffer(i))
        flay.addWidget(btn_del)

        # Store widget reference on frame for later hiding
        frame_w._buf_index = buf_idx  # type: ignore[attr-defined]
        self._cap_layout.addWidget(frame_w)
        self._lbl_captured.setText(f"{len(self._capture_buffer)} captured")

    def _remove_from_buffer(self, idx: int):
        """Mark slot as removed (None) and hide its widget."""
        if idx < len(self._capture_buffer):
            self._capture_buffer[idx] = None  # type: ignore[call-overload]
        # Hide the widget in the layout
        for i in range(self._cap_layout.count()):
            w = self._cap_layout.itemAt(i).widget()
            if w is not None and getattr(w, "_buf_index", -1) == idx:
                w.setVisible(False)
                break
        remaining = sum(1 for c in self._capture_buffer if c is not None)
        self._lbl_captured.setText(f"{remaining} captured")

    def _clear_buffer(self):
        self._capture_buffer.clear()
        while self._cap_layout.count():
            item = self._cap_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._lbl_captured.setText("0 captured")

    def _save_buffer(self):
        valid = [c for c in self._capture_buffer if c is not None]
        if not valid:
            QMessageBox.information(self, "Info", "Nothing to save — capture some pills first.")
            return
        target = self._get_save_dir()
        if target is None:
            QMessageBox.information(
                self, "Info",
                "Select a main class (and optionally a sub class) before saving.",
            )
            return
        target.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, crop in enumerate(valid):
            cv2.imwrite(str(target / f"pill_{ts}_{i:03d}.png"), crop)
        saved = len(valid)
        self._clear_buffer()
        # Refresh counts
        item = self._main_list.currentItem()
        if item:
            self._on_main_selected(item, None)
        QMessageBox.information(self, "Saved", f"Saved {saved} pill(s) to:\n{target}")

    def _get_save_dir(self) -> Optional[Path]:
        if not self._selected_main:
            return None
        if self._selected_sub:
            return self._root / self._selected_main / self._selected_sub
        return self._root / self._selected_main

    # ══════════════════ Helpers ═════════════════════════
    @staticmethod
    def _cv2pixmap(img: np.ndarray, size: int) -> QPixmap:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        ch = 3 if rgb.ndim == 3 else 1
        fmt = (
            QImage.Format.Format_RGB888 if ch == 3
            else QImage.Format.Format_Grayscale8
        )
        qimg = QImage(rgb.data, w, h, ch * w, fmt).copy()
        return QPixmap.fromImage(qimg).scaled(
            size, size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

    def _show_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self._cam_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._cam_label.setPixmap(scaled)

    def cleanup(self):
        self._stop_camera()


# ─────────────────────────────────────────────
#  Dataset Page (tabs: Training Data + Bad Data)
# ─────────────────────────────────────────────
class DatasetPage(QWidget):
    """
    Top-level dataset management page.

    Two sub-tabs:
            1. Training Data  → ``data/``
      2. Bad Data        → ``data_bad/``

    Signals ``camera_started`` / ``camera_stopped`` so MainWindow
    can enforce the camera mutex.
    """

    camera_started = pyqtSignal()
    camera_stopped = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabBar::tab{padding:8px 30px; font-size:14px; font-weight:bold;}"
            "QTabBar::tab:selected{background:#4CAF50; color:white;}"
        )

        settings_dict = (
            settings.all() if hasattr(settings, "all") else dict(settings)
        )

        self._train_panel = _DataPanel(
            root_dir=Path("data"),
            settings=settings_dict,
            label="Training Data",
        )
        self._bad_panel = _DataPanel(
            root_dir=Path("data_bad"),
            settings=settings_dict,
            label="Bad Data",
        )

        self._tabs.addTab(self._train_panel, "Training Data")
        self._tabs.addTab(self._bad_panel, "Bad Data")
        lay.addWidget(self._tabs)

        # Forward camera signals
        self._train_panel.camera_started.connect(self._on_train_cam_start)
        self._train_panel.camera_stopped.connect(self.camera_stopped.emit)
        self._bad_panel.camera_started.connect(self._on_bad_cam_start)
        self._bad_panel.camera_stopped.connect(self.camera_stopped.emit)

    def _on_train_cam_start(self):
        # Stop bad panel camera if running
        self._bad_panel.force_stop_camera()
        self.camera_started.emit()

    def _on_bad_cam_start(self):
        # Stop train panel camera if running
        self._train_panel.force_stop_camera()
        self.camera_started.emit()

    @property
    def is_camera_running(self) -> bool:
        return self._train_panel.is_camera_running or self._bad_panel.is_camera_running

    def force_stop_camera(self):
        """Called from MainWindow to release camera."""
        self._train_panel.force_stop_camera()
        self._bad_panel.force_stop_camera()

    def cleanup(self):
        self._train_panel.cleanup()
        self._bad_panel.cleanup()
