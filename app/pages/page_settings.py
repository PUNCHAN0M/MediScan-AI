"""
Page: Settings — All configuration knobs
=========================================
Left  : UI Preferences, Colors, YOLO, Tracking
Right : PatchCore, Layers, Color Features
"""
from __future__ import annotations

from pathlib import Path
from functools import partial

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox,
    QGridLayout, QScrollArea, QFrame, QSizePolicy, QApplication,
    QButtonGroup, QRadioButton, QColorDialog,
)

from app.settings_manager import SettingsManager


def _scan_models(directory: str, extensions: set[str]) -> list[str]:
    """Return list of model files in *directory*."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(f.name for f in d.iterdir() if f.is_file() and f.suffix.lower() in extensions)


class SettingsPage(QWidget):
    """Full settings page — mirrors app_settings.json."""

    settings_saved = pyqtSignal()

    def __init__(self, settings: SettingsManager, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._widgets: dict[str, QWidget] = {}
        self._init_ui()
        self._load_values()

    # ─────────────────── Build UI ────────────────────────
    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QHBoxLayout(scroll_widget)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_layout.setSpacing(16)

        # ── Left column ──────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(12)

        # --- UI Preferences ---
        ui_group = self._group("---- UI Preferences ----")
        ui_lay = QGridLayout()
        row = 0

        # Camera index
        ui_lay.addWidget(QLabel("CAMERA INDEX"), row, 0)
        self._w_camera = QComboBox()
        for i in range(5):
            self._w_camera.addItem(str(i))
        self._widgets["camera_index"] = self._w_camera
        ui_lay.addWidget(self._w_camera, row, 1)
        row += 1

        # Font size
        ui_lay.addWidget(QLabel("Font Size"), row, 0)
        font_row = QHBoxLayout()
        self._font_group = QButtonGroup(self)
        for label, val in [("เล็ก", 11), ("กลาง", 14), ("ใหญ่", 18)]:
            rb = QRadioButton(label)
            rb.setProperty("font_val", val)
            self._font_group.addButton(rb)
            font_row.addWidget(rb)
        self._widgets["font_size_group"] = self._font_group
        font_w = QWidget()
        font_w.setLayout(font_row)
        ui_lay.addWidget(font_w, row, 1)
        row += 1

        ui_group.setLayout(ui_lay)
        left.addWidget(ui_group)

        # --- Colors ---
        color_grp = self._group("---- Colors ----")
        color_grp_lay = QGridLayout()
        crow = 0

        self._color_btns: dict[str, QPushButton] = {}
        for label, key, default in [
            ("Background Color", "background_color", "#ffffff"),
            ("Text Color", "text_color", "#000000"),
            ("Primary Color", "primary_color", "#33FF00"),
            ("Secondary Color", "secondary_color", "#FF6A00"),
        ]:
            color_grp_lay.addWidget(QLabel(label), crow, 0)
            btn = QPushButton()
            btn.setFixedSize(80, 28)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._color_btns[key] = btn
            btn.clicked.connect(partial(self._pick_color, key))
            self._widgets[key] = btn
            color_grp_lay.addWidget(btn, crow, 1)
            crow += 1

        color_grp.setLayout(color_grp_lay)
        left.addWidget(color_grp)

        # --- YOLO ---
        yolo_group = self._group("---- YOLO ----")
        yolo_lay = QGridLayout()
        row = 0

        # Segmentation model
        yolo_lay.addWidget(QLabel("SEGMENTATION_MODEL"), row, 0)
        self._w_seg_model = QComboBox()
        for m in _scan_models("model/detection", {".pt", ".onnx", ".engine"}):
            self._w_seg_model.addItem(m)
        self._widgets["segmentation_model"] = self._w_seg_model
        yolo_lay.addWidget(self._w_seg_model, row, 1)
        row += 1

        yolo_lay.addWidget(QLabel("SEGMENTATION_IMG_SIZE"), row, 0)
        self._w_seg_img = self._spin(128, 2048, 64)
        self._widgets["seg_img_size"] = self._w_seg_img
        yolo_lay.addWidget(self._w_seg_img, row, 1)
        row += 1

        yolo_lay.addWidget(QLabel("SEGMENTATION_CONF"), row, 0)
        self._w_seg_conf = self._dspin(0.01, 1.0, 0.05)
        self._widgets["seg_conf"] = self._w_seg_conf
        yolo_lay.addWidget(self._w_seg_conf, row, 1)
        row += 1

        yolo_lay.addWidget(QLabel("SEGMENTATION_IOU"), row, 0)
        self._w_seg_iou = self._dspin(0.01, 1.0, 0.05)
        self._widgets["seg_iou"] = self._w_seg_iou
        yolo_lay.addWidget(self._w_seg_iou, row, 1)
        row += 1

        yolo_lay.addWidget(QLabel("SEGMENTATION_PAD"), row, 0)
        self._w_seg_pad = self._spin(0, 50, 1)
        self._widgets["seg_pad"] = self._w_seg_pad
        yolo_lay.addWidget(self._w_seg_pad, row, 1)
        row += 1

        # Detection model
        yolo_lay.addWidget(QLabel(""), row, 0)
        row += 1
        yolo_lay.addWidget(QLabel("DETECTION_MODEL"), row, 0)
        self._w_det_model = QComboBox()
        for m in _scan_models("model/detection", {".pt", ".onnx", ".engine"}):
            self._w_det_model.addItem(m)
        self._widgets["detection_model"] = self._w_det_model
        yolo_lay.addWidget(self._w_det_model, row, 1)
        row += 1

        yolo_lay.addWidget(QLabel("DETECTION_IMG_SIZE"), row, 0)
        self._w_det_img = self._spin(128, 2048, 64)
        self._widgets["det_img_size"] = self._w_det_img
        yolo_lay.addWidget(self._w_det_img, row, 1)
        row += 1

        yolo_lay.addWidget(QLabel("DETECTION_CONF"), row, 0)
        self._w_det_conf = self._dspin(0.01, 1.0, 0.05)
        self._widgets["det_conf"] = self._w_det_conf
        yolo_lay.addWidget(self._w_det_conf, row, 1)
        row += 1

        yolo_lay.addWidget(QLabel("DETECTION_IOU"), row, 0)
        self._w_det_iou = self._dspin(0.01, 1.0, 0.05)
        self._widgets["det_iou"] = self._w_det_iou
        yolo_lay.addWidget(self._w_det_iou, row, 1)
        row += 1

        yolo_group.setLayout(yolo_lay)
        left.addWidget(yolo_group)

        # --- Tracking ---
        track_group = self._group("---- Tracking ----")
        track_lay = QGridLayout()
        row = 0

        track_lay.addWidget(QLabel("Max Distance"), row, 0)
        self._w_track_dist = self._dspin(1.0, 500.0, 5.0)
        self._widgets["track_max_dist"] = self._w_track_dist
        track_lay.addWidget(self._w_track_dist, row, 1)
        row += 1

        track_lay.addWidget(QLabel("IoU Threshold"), row, 0)
        self._w_track_iou = self._dspin(0.01, 1.0, 0.05)
        self._widgets["track_iou_thr"] = self._w_track_iou
        track_lay.addWidget(self._w_track_iou, row, 1)
        row += 1

        track_lay.addWidget(QLabel("Max Age"), row, 0)
        self._w_track_age = self._spin(1, 100, 1)
        self._widgets["track_max_age"] = self._w_track_age
        track_lay.addWidget(self._w_track_age, row, 1)
        row += 1

        track_group.setLayout(track_lay)
        left.addWidget(track_group)
        left.addStretch()

        # ── Right column ─────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(12)

        # --- PatchCore ---
        pc_group = self._group("----- PatchCore -----")
        pc_lay = QGridLayout()
        row = 0

        # Backbone dropdown
        pc_lay.addWidget(QLabel("BACKBONE"), row, 0)
        self._w_backbone = QComboBox()
        for m in _scan_models("model/backbone", {".pth"}):
            self._w_backbone.addItem(m)
        self._widgets["backbone_model"] = self._w_backbone
        pc_lay.addWidget(self._w_backbone, row, 1)
        row += 1

        pc_lay.addWidget(QLabel("Coreset Ratio"), row, 0)
        self._w_coreset = self._dspin(0.01, 1.0, 0.05)
        self._widgets["coreset_ratio"] = self._w_coreset
        pc_lay.addWidget(self._w_coreset, row, 1)
        row += 1

        pc_lay.addWidget(QLabel("K-Nearest"), row, 0)
        self._w_knn = self._spin(1, 50, 1)
        self._widgets["k_nearest"] = self._w_knn
        pc_lay.addWidget(self._w_knn, row, 1)
        row += 1

        pc_lay.addWidget(QLabel("Fallback Threshold"), row, 0)
        self._w_fallback = self._dspin(0.0, 10.0, 0.05)
        self._widgets["fallback_thr"] = self._w_fallback
        pc_lay.addWidget(self._w_fallback, row, 1)
        row += 1

        pc_lay.addWidget(QLabel("CORESET_MIN_KEEP"), row, 0)
        self._w_cs_min = self._spin(100, 100000, 100)
        self._widgets["coreset_min_keep"] = self._w_cs_min
        pc_lay.addWidget(self._w_cs_min, row, 1)
        row += 1

        pc_lay.addWidget(QLabel("CORESET_MAX_KEEP"), row, 0)
        self._w_cs_max = self._spin(1000, 100000, 1000)
        self._widgets["coreset_max_keep"] = self._w_cs_max
        pc_lay.addWidget(self._w_cs_max, row, 1)
        row += 1

        pc_lay.addWidget(QLabel("GRID_SIZE"), row, 0)
        self._w_grid = self._spin(4, 64, 4)
        self._widgets["grid_size"] = self._w_grid
        pc_lay.addWidget(self._w_grid, row, 1)
        row += 1

        pc_lay.addWidget(QLabel("IMG_SIZE"), row, 0)
        self._w_img_size = self._spin(64, 1024, 32)
        self._widgets["img_size"] = self._w_img_size
        pc_lay.addWidget(self._w_img_size, row, 1)
        row += 1

        pc_lay.addWidget(QLabel("Fine-tune Step"), row, 0)
        self._w_steps = self._spin(1, 10000, 10)
        self._widgets["n_steps"] = self._w_steps
        pc_lay.addWidget(self._w_steps, row, 1)
        row += 1

        pc_lay.addWidget(QLabel("SEED"), row, 0)
        self._w_seed = self._spin(0, 99999, 1)
        self._widgets["seed"] = self._w_seed
        pc_lay.addWidget(self._w_seed, row, 1)
        row += 1

        pc_group.setLayout(pc_lay)
        right.addWidget(pc_group)

        # --- Layers ---
        layer_group = self._group("LAYERS")
        layer_lay = QVBoxLayout()

        self._w_layer1 = QCheckBox("enable layer 1")
        self._w_layer2 = QCheckBox("enable layer 2")
        self._w_layer3 = QCheckBox("enable layer 3")
        self._widgets["enable_layer1"] = self._w_layer1
        self._widgets["enable_layer2"] = self._w_layer2
        self._widgets["enable_layer3"] = self._w_layer3

        layer_lay.addWidget(self._w_layer1)
        layer_lay.addWidget(self._w_layer2)
        layer_lay.addWidget(self._w_layer3)

        layer_group.setLayout(layer_lay)
        right.addWidget(layer_group)

        # --- Color Features ---
        color_group = self._group("---- Color Features ----")
        color_lay = QGridLayout()
        row = 0

        self._w_rgb = QCheckBox("Enable RGB Features")
        self._widgets["use_color"] = self._w_rgb
        color_lay.addWidget(self._w_rgb, row, 0, 1, 2)
        row += 1

        self._w_hsv = QCheckBox("Enable HSV Features")
        self._widgets["use_hsv"] = self._w_hsv
        color_lay.addWidget(self._w_hsv, row, 0, 1, 2)
        row += 1

        color_lay.addWidget(QLabel("Color Weight"), row, 0)
        self._w_cw = self._dspin(0.0, 10.0, 0.1)
        self._widgets["color_weight"] = self._w_cw
        color_lay.addWidget(self._w_cw, row, 1)
        row += 1

        color_group.setLayout(color_lay)
        right.addWidget(color_group)
        right.addStretch()

        # ── Save button ──────────────────────────────────
        btn_save = QPushButton("💾  Save Settings")
        btn_save.setStyleSheet(
            "QPushButton{background:#00c853; color:white; font-weight:bold; padding:10px 30px; font-size:14px; border-radius:4px;}"
            "QPushButton:hover{background:#00e676;}"
        )
        btn_save.clicked.connect(self._save)

        # ── Assemble columns ─────────────────────────────
        left_w = QWidget()
        left_w.setLayout(left)
        right_w = QWidget()
        right_w.setLayout(right)

        scroll_layout.addWidget(left_w, stretch=1)
        scroll_layout.addWidget(right_w, stretch=1)
        scroll.setWidget(scroll_widget)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(scroll)
        root.addWidget(btn_save)

    # ─────────────────── helpers ─────────────────────────
    @staticmethod
    def _group(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet(
            "QGroupBox{background:#e8e8e8; border:1px solid #bbb; border-radius:4px;"
            " padding:12px; padding-top:22px; font-weight:bold;}"
        )
        return g

    @staticmethod
    def _spin(lo: int, hi: int, step: int) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setMinimumWidth(100)
        return sb

    @staticmethod
    def _dspin(lo: float, hi: float, step: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setDecimals(3)
        sb.setMinimumWidth(100)
        return sb

    def _pick_color(self, key: str):
        """Open a QColorDialog for the given color *key*."""
        current = QColor(self._settings.get(key, "#ffffff"))
        color = QColorDialog.getColor(current, self, f"Select {key}")
        if color.isValid():
            hex_val = color.name()
            self._color_btns[key].setStyleSheet(
                f"background:{hex_val}; border:2px solid #555; border-radius:4px;"
            )

    def _set_color_btn_style(self, key: str, hex_val: str):
        btn = self._color_btns.get(key)
        if btn:
            btn.setStyleSheet(
                f"background:{hex_val}; border:2px solid #555; border-radius:4px;"
            )

    # ─────────────────── load / save ─────────────────────
    def _load_values(self):
        s = self._settings

        # Camera
        self._w_camera.setCurrentText(str(s.get("camera_index", 0)))

        # Font size radio
        font_val = s.get("font_size", 14)
        for btn in self._font_group.buttons():
            if btn.property("font_val") == font_val:
                btn.setChecked(True)

        # YOLO
        seg_model = Path(s.get("segmentation_model", "")).name
        idx = self._w_seg_model.findText(seg_model)
        if idx >= 0:
            self._w_seg_model.setCurrentIndex(idx)

        det_model = Path(s.get("detection_model", "")).name
        idx = self._w_det_model.findText(det_model)
        if idx >= 0:
            self._w_det_model.setCurrentIndex(idx)

        self._w_seg_img.setValue(s.get("seg_img_size", 512))
        self._w_seg_conf.setValue(s.get("seg_conf", 0.5))
        self._w_seg_iou.setValue(s.get("seg_iou", 0.6))
        self._w_seg_pad.setValue(s.get("seg_pad", 5))
        self._w_det_img.setValue(s.get("det_img_size", 512))
        self._w_det_conf.setValue(s.get("det_conf", 0.5))
        self._w_det_iou.setValue(s.get("det_iou", 0.6))

        # Tracking
        self._w_track_dist.setValue(s.get("track_max_dist", 80.0))
        self._w_track_iou.setValue(s.get("track_iou_thr", 0.8))
        self._w_track_age.setValue(s.get("track_max_age", 10))

        # PatchCore
        bb_model = Path(s.get("backbone_model", "")).name
        idx = self._w_backbone.findText(bb_model)
        if idx >= 0:
            self._w_backbone.setCurrentIndex(idx)

        self._w_coreset.setValue(s.get("coreset_ratio", 0.25))
        self._w_knn.setValue(s.get("k_nearest", 3))
        self._w_fallback.setValue(s.get("fallback_thr", 0.5))
        self._w_cs_min.setValue(s.get("coreset_min_keep", 1000))
        self._w_cs_max.setValue(s.get("coreset_max_keep", 20000))
        self._w_grid.setValue(s.get("grid_size", 16))
        self._w_img_size.setValue(s.get("img_size", 256))
        self._w_steps.setValue(s.get("n_steps", 351))
        self._w_seed.setValue(s.get("seed", 42))

        # Layers
        self._w_layer1.setChecked(s.get("enable_layer1", True))
        self._w_layer2.setChecked(s.get("enable_layer2", True))
        self._w_layer3.setChecked(s.get("enable_layer3", True))

        # Color
        self._w_rgb.setChecked(s.get("use_color", True))
        self._w_hsv.setChecked(s.get("use_hsv", True))
        self._w_cw.setValue(s.get("color_weight", 1.5))

        # UI Colors
        for key, default in [
            ("background_color", "#ffffff"),
            ("text_color", "#000000"),
            ("primary_color", "#33FF00"),
            ("secondary_color", "#FF6A00"),
        ]:
            hex_val = s.get(key, default)
            self._set_color_btn_style(key, hex_val)

    def _save(self):
        s = self._settings

        s["camera_index"] = int(self._w_camera.currentText())

        # Font size
        checked_btn = self._font_group.checkedButton()
        if checked_btn:
            font_val = checked_btn.property("font_val")
            s["font_size"] = font_val
            app = QApplication.instance()
            if app:
                f = app.font()
                f.setPointSize(font_val)
                app.setFont(f)

        # YOLO models (store as relative path)
        s["segmentation_model"] = f"model\\detection\\{self._w_seg_model.currentText()}"
        s["detection_model"] = f"model\\detection\\{self._w_det_model.currentText()}"
        s["seg_img_size"] = self._w_seg_img.value()
        s["seg_conf"] = self._w_seg_conf.value()
        s["seg_iou"] = self._w_seg_iou.value()
        s["seg_pad"] = self._w_seg_pad.value()
        s["det_img_size"] = self._w_det_img.value()
        s["det_conf"] = self._w_det_conf.value()
        s["det_iou"] = self._w_det_iou.value()

        # Tracking
        s["track_max_dist"] = self._w_track_dist.value()
        s["track_iou_thr"] = self._w_track_iou.value()
        s["track_max_age"] = self._w_track_age.value()

        # PatchCore
        s["backbone_model"] = f"model\\backbone\\{self._w_backbone.currentText()}"
        s["coreset_ratio"] = self._w_coreset.value()
        s["k_nearest"] = self._w_knn.value()
        s["fallback_thr"] = self._w_fallback.value()
        s["coreset_min_keep"] = self._w_cs_min.value()
        s["coreset_max_keep"] = self._w_cs_max.value()
        s["grid_size"] = self._w_grid.value()
        s["img_size"] = self._w_img_size.value()
        s["n_steps"] = self._w_steps.value()
        s["seed"] = self._w_seed.value()

        # Layers
        s["enable_layer1"] = self._w_layer1.isChecked()
        s["enable_layer2"] = self._w_layer2.isChecked()
        s["enable_layer3"] = self._w_layer3.isChecked()

        # Color
        s["use_color"] = self._w_rgb.isChecked()
        s["use_hsv"] = self._w_hsv.isChecked()
        s["color_weight"] = self._w_cw.value()

        # UI Colors
        for key in ("background_color", "text_color", "primary_color", "secondary_color"):
            btn = self._color_btns.get(key)
            if btn:
                # Extract color from stylesheet
                ss = btn.styleSheet()
                if "background:" in ss:
                    hex_val = ss.split("background:")[1].split(";")[0].strip()
                    s[key] = hex_val

        s.save()
        self.settings_saved.emit()
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Saved", "Settings saved to app_settings.json")
