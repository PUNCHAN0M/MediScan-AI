"""
Page: Classifier — Realtime pill defect detection
==================================================
Left  : camera feed with bounding-box / centre overlay + segmented pills
Right : count info, START/STOP toggle, BBOX/CENTER toggle,
        class scroller, selected-classes panel

Architecture:
    QTimer (main/Qt thread): cap.read() → YOLO detect (sync, fast ~20-50 ms)
                             → draw current bboxes + cached labels → pixmap
    Background thread      : feature extraction + scoring (async, ~100-500 ms)
                             → updates {track_id: status} map

Benefits:
    - Bboxes appear/disappear INSTANTLY with YOLO detections.
    - Labels (NORMAL/ANOMALY) update once scoring finishes.
    - Covering a pill → YOLO stops detecting → bbox gone immediately.
    - No stale bboxes, no stutter.
"""
from __future__ import annotations

import cv2
import threading
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QListWidget, QListWidgetItem,
    QGroupBox, QFrame, QSplitter, QSizePolicy, QScrollArea,
    QGridLayout, QMessageBox,
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


# ─────────────────────────────────────────────
# Async Scoring Worker (background thread)
# ─────────────────────────────────────────────
class _ScoringWorker:
    """
    Background thread for feature extraction + anomaly scoring.

    YOLO detection stays in the Qt thread for instant bbox response.
    Only scoring (slow part) runs here.
    """

    def __init__(self, inspector: Any):
        self._inspector = inspector

        # Input slot (latest-only)
        self._input_lock = threading.Lock()
        self._input_slot: Optional[tuple] = None   # (crops, infos)

        # Output: per-track scorings
        self._result_lock = threading.Lock()
        self._status_map: Dict[int, Dict[str, Any]] = {}
        self._crops_map: Dict[int, np.ndarray] = {}

        self._new_input = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, crops: list, infos: list) -> None:
        """Submit YOLO detection batch for scoring (replaces pending)."""
        with self._input_lock:
            self._input_slot = (list(crops), list(infos))
        self._new_input.set()

    @property
    def status_map(self) -> Dict[int, Dict[str, Any]]:
        with self._result_lock:
            return dict(self._status_map)

    @property
    def crops_map(self) -> Dict[int, np.ndarray]:
        with self._result_lock:
            return dict(self._crops_map)

    def stop(self) -> None:
        self._stop_event.set()
        self._new_input.set()
        self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._new_input.wait()
            self._new_input.clear()
            if self._stop_event.is_set():
                break

            with self._input_lock:
                data = self._input_slot
                self._input_slot = None
            if data is None:
                continue

            try:
                crops, infos = data
                results = self._inspector.score_pills(crops, infos)

                new_status: Dict[int, Dict[str, Any]] = {}
                new_crops: Dict[int, np.ndarray] = {}
                for r, crop in zip(results, crops):
                    tid = r["id"]
                    new_status[tid] = {
                        "status": r["status"],
                        "class_scores": r.get("class_scores", {}),
                        "normal_from": r.get("normal_from", []),
                    }
                    if tid >= 0:
                        new_crops[tid] = crop

                with self._result_lock:
                    self._status_map = new_status
                    self._crops_map = new_crops
            except Exception as e:
                print(f"[ScoringWorker] {e}")


class ClassifierPage(QWidget):
    """Realtime pill anomaly classifier page."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._inspector = None
        self._worker: Optional[_ScoringWorker] = None
        self._running = False
        self._draw_mode = "bbox"   # "bbox" | "center"
        self._cap = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._process_frame)

        self._selected_classes: list[str] = []
        self._applied_classes: list[str] = []   # classes currently loaded in inspector
        self._all_classes: list[str] = []
        self._init_ui()
        self._load_classes()

    # ─────────────────── UI ──────────────────────────────
    def _init_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(6, 6, 6, 6)

        # ── Left column ──────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(4)

        # Camera feed
        self._cam_label = QLabel("Camera Off")
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setStyleSheet("background:#1a1a1a; color:#888; font-size:16px; border:2px solid #444;")
        self._cam_label.setMinimumSize(480, 360)
        self._cam_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.addWidget(self._cam_label, stretch=3)

        # Segmented pills grid (below camera)
        seg_label = QLabel("Segmented Pills")
        seg_label.setStyleSheet("font-weight:bold; font-size:12px; padding:2px;")
        left.addWidget(seg_label)

        self._pill_scroll = QScrollArea()
        self._pill_scroll.setWidgetResizable(True)
        self._pill_scroll.setMinimumHeight(140)
        self._pill_scroll.setMaximumHeight(180)
        self._pill_scroll.setStyleSheet("background:#222; border:1px solid #444;")

        self._pill_grid_widget = QWidget()
        self._pill_grid_layout = QHBoxLayout(self._pill_grid_widget)
        self._pill_grid_layout.setContentsMargins(4, 4, 4, 4)
        self._pill_grid_layout.setSpacing(4)
        self._pill_grid_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._pill_scroll.setWidget(self._pill_grid_widget)
        left.addWidget(self._pill_scroll, stretch=0)

        # Normal / Defect pill rows
        nd_row = QHBoxLayout()

        # Normal pills (styled like Segmented Pills)
        normal_col = QVBoxLayout()
        normal_label = QLabel("Normal")
        normal_label.setStyleSheet("font-weight:bold; font-size:12px; padding:2px; color:#00c853;")
        normal_col.addWidget(normal_label)

        self._normal_scroll = QScrollArea()
        self._normal_scroll.setWidgetResizable(True)
        self._normal_scroll.setMinimumHeight(100)
        self._normal_scroll.setMaximumHeight(130)
        self._normal_scroll.setStyleSheet("background:#1b2e1b; border:2px solid #00c853;")
        normal_inner = QWidget()
        self._normal_layout = QHBoxLayout(normal_inner)
        self._normal_layout.setContentsMargins(4, 4, 4, 4)
        self._normal_layout.setSpacing(4)
        self._normal_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._normal_scroll.setWidget(normal_inner)
        normal_col.addWidget(self._normal_scroll)
        nd_row.addLayout(normal_col)

        # Defect pills (styled like Segmented Pills)
        defect_col = QVBoxLayout()
        defect_label = QLabel("Defect")
        defect_label.setStyleSheet("font-weight:bold; font-size:12px; padding:2px; color:#ff1744;")
        defect_col.addWidget(defect_label)

        self._defect_scroll = QScrollArea()
        self._defect_scroll.setWidgetResizable(True)
        self._defect_scroll.setMinimumHeight(100)
        self._defect_scroll.setMaximumHeight(130)
        self._defect_scroll.setStyleSheet("background:#2e1b1b; border:2px solid #ff1744;")
        defect_inner = QWidget()
        self._defect_layout = QHBoxLayout(defect_inner)
        self._defect_layout.setContentsMargins(4, 4, 4, 4)
        self._defect_layout.setSpacing(4)
        self._defect_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._defect_scroll.setWidget(defect_inner)
        defect_col.addWidget(self._defect_scroll)
        nd_row.addLayout(defect_col)

        left.addLayout(nd_row)

        left_w = QWidget()
        left_w.setLayout(left)

        # ── Right column: controls ───────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)

        # Camera selector
        cam_group = QGroupBox("Camera")
        cam_group.setStyleSheet("QGroupBox{font-size:12px; font-weight:bold; border:1px solid #aaa; padding:6px; padding-top:18px;}")
        cam_lay = QVBoxLayout(cam_group)
        self._cam_combo = QComboBox()
        for idx in _enum_cameras():
            self._cam_combo.addItem(f"Camera {idx}", idx)
        saved_idx = self._settings.get("camera_index", 0)
        combo_idx = self._cam_combo.findData(saved_idx)
        if combo_idx >= 0:
            self._cam_combo.setCurrentIndex(combo_idx)
        cam_lay.addWidget(self._cam_combo)
        right.addWidget(cam_group)

        # Info box
        info_box = QGroupBox()
        info_box.setStyleSheet("QGroupBox{background:#e8e8e8; border:1px solid #aaa; padding:10px;}")
        info_lay = QVBoxLayout(info_box)

        self._lbl_count  = QLabel("Count : 0")
        self._lbl_normal = QLabel("Normal : 0")
        self._lbl_defect = QLabel("Defect : 0")
        for lbl in (self._lbl_count, self._lbl_normal, self._lbl_defect):
            lbl.setStyleSheet("font-size:15px; font-weight:bold; padding:4px 0;")
            info_lay.addWidget(lbl)
        right.addWidget(info_box)

        # Toggle buttons row
        btn_row = QHBoxLayout()

        self._btn_toggle_cam = QPushButton("START")
        self._btn_toggle_cam.setCheckable(True)
        self._btn_toggle_cam.setStyleSheet(
            "QPushButton{background:#1976D2; color:white; font-weight:bold; padding:8px 16px; border:2px solid #1565C0; border-radius:4px;}"
            "QPushButton:checked{background:#ff1744; border-color:#d50000;}"
            "QPushButton:hover{opacity:0.9;}"
        )
        self._btn_toggle_cam.toggled.connect(self._toggle_camera)

        self._btn_toggle_draw = QPushButton("BBOX")
        self._btn_toggle_draw.setCheckable(True)
        self._btn_toggle_draw.setStyleSheet(
            "QPushButton{background:#555; color:white; font-weight:bold; padding:8px 16px; border:2px solid #333; border-radius:4px;}"
            "QPushButton:checked{background:#2962ff; border-color:#0039cb; color:white;}"
        )
        self._btn_toggle_draw.clicked.connect(self._toggle_draw_mode)

        btn_row.addWidget(self._btn_toggle_cam)
        btn_row.addWidget(self._btn_toggle_draw)
        right.addLayout(btn_row)

        # Class selector
        cls_group = QGroupBox("Classes")
        cls_group.setStyleSheet("QGroupBox{font-size:13px; font-weight:bold; border:1px solid #aaa; padding:8px; padding-top:18px;}")
        cls_lay = QVBoxLayout(cls_group)

        self._class_search = QLineEdit()
        self._class_search.setPlaceholderText("Search : ....")
        self._class_search.textChanged.connect(self._filter_classes)
        cls_lay.addWidget(self._class_search)

        self._class_list = QListWidget()
        self._class_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._class_list.itemSelectionChanged.connect(self._on_class_selection)
        cls_lay.addWidget(self._class_list)

        right.addWidget(cls_group, stretch=1)

        # Selected classes panel
        sel_group = QGroupBox("Selected Classes")
        sel_group.setStyleSheet(
            "QGroupBox{font-size:12px; font-weight:bold; border:2px solid #FF6A00; background:#fff3e0;"
            " padding:6px; padding-top:18px;}"
        )
        sel_lay = QVBoxLayout(sel_group)
        self._selected_list = QListWidget()
        self._selected_list.setMaximumHeight(120)
        self._selected_list.setStyleSheet("background:#fff3e0; font-size:12px;")
        sel_lay.addWidget(self._selected_list)

        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setStyleSheet(
            "QPushButton{background:#FF6A00; color:white; font-weight:bold; padding:6px 12px; border-radius:3px;}"
            "QPushButton:hover{background:#FF8F00;}"
        )
        self._btn_apply.clicked.connect(self._reload_inspector)
        self._btn_apply.setVisible(False)  # Hidden until selection changes
        sel_lay.addWidget(self._btn_apply)

        right.addWidget(sel_group)

        # Layout
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(320)

        main.addWidget(left_w, stretch=3)
        main.addWidget(right_widget, stretch=0)

    # ─────────────────── class list ──────────────────────
    def _load_classes(self):
        """Scan data_train_defection for main class names.

        Each main_class.pth holds all subclasses inside it,
        so the inspector only needs the parent name.
        """
        root = Path("data_train_defection")
        model_dir = Path("model/patchcore_resnet")
        self._all_classes = []
        if root.exists():
            for main_dir in sorted(root.iterdir()):
                if main_dir.is_dir():
                    # Only list classes that have a trained model
                    if (model_dir / f"{main_dir.name}.pth").exists():
                        self._all_classes.append(main_dir.name)
        self._populate_class_list(self._all_classes)

    def _populate_class_list(self, classes: list[str]):
        self._class_list.clear()
        for cls in classes:
            self._class_list.addItem(cls)

    def _filter_classes(self, text: str):
        text = text.lower()
        filtered = [c for c in self._all_classes if text in c.lower()]
        self._populate_class_list(filtered)

    def _on_class_selection(self):
        self._selected_classes = [
            item.text() for item in self._class_list.selectedItems()
        ]
        self._selected_list.clear()
        for cls in self._selected_classes:
            self._selected_list.addItem(cls)

        # Show Apply button only if selection differs from applied
        if sorted(self._selected_classes) != sorted(self._applied_classes):
            self._btn_apply.setVisible(True)
        else:
            self._btn_apply.setVisible(False)

    def _reload_inspector(self):
        """Reload PillInspector with currently selected classes."""
        # Stop existing worker before replacing inspector
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

        self._inspector = None
        self._load_inspector()
        self._applied_classes = list(self._selected_classes)
        self._btn_apply.setVisible(False)

        # Restart worker if camera is running
        if self._running and self._inspector is not None:
            self._worker = _ScoringWorker(self._inspector)

        QMessageBox.information(self, "Inspector",
                                f"Reloaded with {len(self._selected_classes)} class(es).")

    # ─────────────────── camera control ──────────────────
    def _toggle_camera(self, checked: bool):
        if checked:
            self._start()
            self._btn_toggle_cam.setText("STOP")
        else:
            self._stop()
            self._btn_toggle_cam.setText("START")

    def _toggle_draw_mode(self):
        """Toggle BBOX / CENTER draw mode using button checked state."""
        if self._btn_toggle_draw.isChecked():
            self._draw_mode = "center"
            self._btn_toggle_draw.setText("CENTER")
        else:
            self._draw_mode = "bbox"
            self._btn_toggle_draw.setText("BBOX")

    def _start(self):
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

        if self._inspector is None:
            self._load_inspector()

        # Create background scoring worker
        if self._inspector is not None:
            self._worker = _ScoringWorker(self._inspector)

        self._running = True
        self._timer.start()

    def _stop(self):
        self._timer.stop()
        self._running = False

        # Stop inference thread
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

        if self._cap:
            self._cap.release()
            self._cap = None
        self._cam_label.setText("Camera Off")

        # Clear counts
        self._lbl_count.setText("Count : 0")
        self._lbl_normal.setText("Normal : 0")
        self._lbl_defect.setText("Defect : 0")
        self._lbl_normal.setStyleSheet("font-size:15px; font-weight:bold; padding:4px 0;")
        self._lbl_defect.setStyleSheet("font-size:15px; font-weight:bold; padding:4px 0;")

        # Clear segmented pills
        while self._pill_grid_layout.count():
            item = self._pill_grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        # Clear normal pills
        while self._normal_layout.count():
            item = self._normal_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        # Clear defect pills
        while self._defect_layout.count():
            item = self._defect_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

    # ─────────────────── inspector ───────────────────────
    def _load_inspector(self):
        """Lazy-load PillInspector from new pipeline."""
        try:
            from pipeline.infer_pipeline import PillInspector, InspectorConfig
            classes = self._selected_classes or self._all_classes
            config = InspectorConfig(compare_classes=list(classes))
            self._inspector = PillInspector(config)
        except Exception as e:
            print(f"[ClassifierPage] Failed to load inspector: {e}")
            self._inspector = None

    # ─────────────────── frame processing ────────────────
    def _process_frame(self):
        """
        QTimer callback — runs at ~30 fps.

        YOLO detection is **synchronous** (fast ~20-50 ms) for instant
        bbox response.  Scoring runs in ``_ScoringWorker`` — labels
        update once scoring finishes.  Bboxes appear/disappear
        instantly regardless of scoring latency.
        """
        if not self._cap or not self._cap.isOpened():
            return
        ok, frame = self._cap.read()
        if not ok:
            return

        count = 0
        normal_count = 0
        defect_count = 0
        display = frame.copy()
        normal_crops = []
        defect_crops = []
        all_crops_list = []

        if self._inspector is not None and self._worker is not None:
            try:
                # Phase 1: synchronous YOLO detection (fast)
                crops, infos = self._inspector.detect_frame(frame)

                # Submit crops for async scoring (non-blocking)
                if crops:
                    self._worker.submit(crops, infos)

                # Merge: current YOLO bboxes + cached scoring labels
                scored = self._worker.status_map
                crops_map = self._worker.crops_map

                all_crops_list = list(crops_map.values()) if crops_map else [c for c in crops]
                count = len(infos)

                for i, info in enumerate(infos):
                    tid = info.get("track_id", -1)
                    bbox = info.get("bbox", None)
                    center = info.get("center", None)

                    cached = scored.get(tid)
                    status = cached["status"] if cached else "SCORING"

                    crop_img = crops_map.get(tid) if crops_map else (crops[i] if i < len(crops) else None)

                    if status == "ANOMALY":
                        defect_count += 1
                        color = (0, 0, 255)
                        if crop_img is not None:
                            defect_crops.append(crop_img)
                    elif status == "NORMAL":
                        normal_count += 1
                        color = (0, 255, 0)
                        if crop_img is not None:
                            normal_crops.append(crop_img)
                    else:
                        # SCORING / PENDING — neutral color
                        color = (200, 200, 0)

                    # Draw annotations
                    if self._draw_mode == "bbox" and bbox:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display, status, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    elif self._draw_mode == "center":
                        if center:
                            cx, cy = int(center[0]), int(center[1])
                        elif bbox:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        else:
                            continue
                        cv2.circle(display, (cx, cy), 8, color, -1)
                        cv2.circle(display, (cx, cy), 10, (255, 255, 255), 1)
            except Exception as e:
                print(f"[ClassifierPage] Inference error: {e}")

        # Update labels
        self._lbl_count.setText(f"Count : {count}")
        self._lbl_normal.setText(f"Normal : {normal_count}")
        self._lbl_defect.setText(f"Defect : {defect_count}")

        self._lbl_normal.setStyleSheet("font-size:15px; font-weight:bold; color:green; padding:4px 0;")
        self._lbl_defect.setStyleSheet(
            f"font-size:15px; font-weight:bold; color:{'red' if defect_count > 0 else '#333'}; padding:4px 0;"
        )

        # Display camera view (our own annotated display)
        self._show_frame(display)

        # Update segmented pill grid & normal/defect rows
        self._update_pill_grid(all_crops_list)
        self._update_category_pills(normal_crops, defect_crops)

    def _update_pill_grid(self, crops: list):
        """Show all segmented pills in the horizontal strip."""
        # Clear previous
        while self._pill_grid_layout.count():
            item = self._pill_grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        for crop in crops[:20]:  # limit to 20
            lbl = QLabel()
            lbl.setFixedSize(80, 80)
            lbl.setStyleSheet("border:1px solid #555;")
            pix = self._cv2pixmap(crop, 80)
            lbl.setPixmap(pix)
            self._pill_grid_layout.addWidget(lbl)

    def _update_category_pills(self, normals: list, defects: list):
        """Update normal/defect pill rows."""
        # Clear normal
        while self._normal_layout.count():
            item = self._normal_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        for crop in normals[:10]:
            lbl = QLabel()
            lbl.setFixedSize(60, 60)
            lbl.setStyleSheet("border:1px solid #00c853;")
            lbl.setPixmap(self._cv2pixmap(crop, 60))
            self._normal_layout.addWidget(lbl)

        # Clear defect
        while self._defect_layout.count():
            item = self._defect_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        for crop in defects[:10]:
            lbl = QLabel()
            lbl.setFixedSize(60, 60)
            lbl.setStyleSheet("border:1px solid #ff1744;")
            lbl.setPixmap(self._cv2pixmap(crop, 60))
            self._defect_layout.addWidget(lbl)

    @staticmethod
    def _cv2pixmap(img: np.ndarray, size: int) -> QPixmap:
        """Convert BGR cv2 image to QPixmap scaled to *size*."""
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

    def _show_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self._cam_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._cam_label.setPixmap(scaled)

    # ─────────────────── cleanup ─────────────────────────
    def cleanup(self):
        self._stop()

    def hideEvent(self, event):
        super().hideEvent(event)

