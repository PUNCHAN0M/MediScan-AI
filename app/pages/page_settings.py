"""
Page: Settings — Simplified UI preferences
============================================
Camera Index, Font Size, Background Color, Primary Color, Secondary Color.
"""
from __future__ import annotations

from functools import partial

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QApplication,
    QButtonGroup, QRadioButton, QColorDialog, QMessageBox,
)


class SettingsPage(QWidget):
    """Simplified settings — UI preferences only."""

    settings_saved = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._color_btns: dict[str, QPushButton] = {}
        self._init_ui()
        self._load_values()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)
        root.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── Camera ───────────────────────────────────────
        cam_group = self._group("Camera")
        cam_lay = QGridLayout()

        cam_lay.addWidget(QLabel("Camera Index"), 0, 0)
        self._w_camera = QComboBox()
        for i in range(5):
            self._w_camera.addItem(str(i))
        cam_lay.addWidget(self._w_camera, 0, 1)

        cam_group.setLayout(cam_lay)
        root.addWidget(cam_group)

        # ── Font ─────────────────────────────────────────
        font_group = self._group("Font")
        font_lay = QGridLayout()

        font_lay.addWidget(QLabel("Font Size"), 0, 0)
        font_row = QWidget()
        font_row_lay = QVBoxLayout(font_row)
        font_row_lay.setContentsMargins(0, 0, 0, 0)
        self._font_group = QButtonGroup(self)
        for label, val in [("เล็ก", 11), ("กลาง", 14), ("ใหญ่", 18)]:
            rb = QRadioButton(label)
            rb.setProperty("font_val", val)
            self._font_group.addButton(rb)
            font_row_lay.addWidget(rb)
        font_lay.addWidget(font_row, 0, 1)

        font_group.setLayout(font_lay)
        root.addWidget(font_group)

        # ── Colors ───────────────────────────────────────
        color_group = self._group("Colors")
        color_lay = QGridLayout()
        row = 0
        for label, key, default in [
            ("Background Color", "background_color", "#ffffff"),
            ("Primary Color", "primary_color", "#33FF00"),
            ("Secondary Color", "secondary_color", "#FF6A00"),
        ]:
            color_lay.addWidget(QLabel(label), row, 0)
            btn = QPushButton()
            btn.setFixedSize(80, 28)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._color_btns[key] = btn
            btn.clicked.connect(partial(self._pick_color, key))
            color_lay.addWidget(btn, row, 1)
            row += 1

        color_group.setLayout(color_lay)
        root.addWidget(color_group)

        # ── Save ─────────────────────────────────────────
        root.addStretch()
        btn_save = QPushButton("💾  Save Settings")
        btn_save.setStyleSheet(
            "QPushButton{background:#00c853; color:white; font-weight:bold;"
            " padding:10px 30px; font-size:14px; border-radius:4px;}"
            "QPushButton:hover{background:#00e676;}"
        )
        btn_save.clicked.connect(self._save)
        root.addWidget(btn_save)

    # ─────────────────── helpers ─────────────────────────
    @staticmethod
    def _group(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet(
            "QGroupBox{background:#e8e8e8; border:1px solid #bbb;"
            " border-radius:4px; padding:12px; padding-top:22px;"
            " font-weight:bold;}"
        )
        return g

    def _pick_color(self, key: str):
        current = QColor(self._settings.get(key, "#ffffff"))
        color = QColorDialog.getColor(current, self, f"Select {key}")
        if color.isValid():
            self._color_btns[key].setStyleSheet(
                f"background:{color.name()}; border:2px solid #555;"
                " border-radius:4px;"
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
        self._w_camera.setCurrentText(str(s.get("camera_index", 0)))

        font_val = s.get("font_size", 14)
        for btn in self._font_group.buttons():
            if btn.property("font_val") == font_val:
                btn.setChecked(True)

        for key, default in [
            ("background_color", "#ffffff"),
            ("primary_color", "#33FF00"),
            ("secondary_color", "#FF6A00"),
        ]:
            self._set_color_btn_style(key, s.get(key, default))

    def _save(self):
        s = self._settings

        s["camera_index"] = int(self._w_camera.currentText())

        checked_btn = self._font_group.checkedButton()
        if checked_btn:
            font_val = checked_btn.property("font_val")
            s["font_size"] = font_val
            app = QApplication.instance()
            if app:
                f = app.font()
                f.setPointSize(font_val)
                app.setFont(f)

        for key in ("background_color", "primary_color", "secondary_color"):
            btn = self._color_btns.get(key)
            if btn:
                ss = btn.styleSheet()
                if "background:" in ss:
                    hex_val = ss.split("background:")[1].split(";")[0].strip()
                    s[key] = hex_val

        s.save()
        self.settings_saved.emit()
        QMessageBox.information(self, "Saved", "Settings saved.")
