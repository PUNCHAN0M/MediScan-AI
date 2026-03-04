"""
ImageGrid — Scrollable, selectable thumbnail grid
==================================================
Displays images as a grid of clickable thumbnails.
Supports multi-select (checkbox at top-right), delete, Select All / Deselect.

Signals
-------
``selection_changed(list[str])`` — emits list of selected file paths.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Sequence

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QIcon, QPainter, QColor, QFont, QPen
from PyQt6.QtWidgets import (
    QWidget, QScrollArea, QGridLayout, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QFrame, QSizePolicy,
)


# ─────────────────────────────────────────────────────────
#  Single thumbnail card
# ─────────────────────────────────────────────────────────
class _ThumbCard(QFrame):
    """One thumbnail with an overlay checkbox."""

    toggled = pyqtSignal(str, bool)   # (filepath, checked)

    def __init__(self, filepath: str, thumb_size: int = 100, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self._thumb_size = thumb_size
        self._checked = False
        self.setFixedSize(thumb_size + 4, thumb_size + 4)
        self.setStyleSheet("border:1px solid #555; background:#222;")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Load and scale image
        self._pixmap = self._load(filepath, thumb_size)

    def _load(self, path: str, size: int) -> QPixmap:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            pix = QPixmap(size, size)
            pix.fill(QColor(60, 60, 60))
            return pix
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(
            size, size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    @property
    def checked(self) -> bool:
        return self._checked

    @checked.setter
    def checked(self, v: bool):
        self._checked = v
        self.update()

    # ── paint ────────────────────────────────────────────
    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        # draw thumbnail centred
        x = (self.width() - self._pixmap.width()) // 2
        y = (self.height() - self._pixmap.height()) // 2
        p.drawPixmap(x, y, self._pixmap)
        # checkbox indicator at top-right
        sz = 18
        rx = self.width() - sz - 3
        ry = 3
        if self._checked:
            p.setBrush(QColor(255, 60, 60, 200))
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawEllipse(rx, ry, sz, sz)
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            p.drawText(rx, ry, sz, sz, Qt.AlignmentFlag.AlignCenter, "✕")
        else:
            p.setBrush(QColor(200, 200, 200, 120))
            p.setPen(QPen(QColor(100, 100, 100), 1))
            p.drawEllipse(rx, ry, sz, sz)
        p.end()

    def mousePressEvent(self, event):
        self._checked = not self._checked
        self.toggled.emit(self.filepath, self._checked)
        self.update()


# ─────────────────────────────────────────────────────────
#  ImageGrid — main widget
# ─────────────────────────────────────────────────────────
class ImageGrid(QWidget):
    """Scrollable thumbnail grid with multi-select and delete.

    If *flexible* is ``True`` (default), columns auto-adapt to widget width.
    """

    selection_changed = pyqtSignal(list)

    def __init__(
        self,
        columns: int = 5,
        thumb_size: int = 100,
        show_toolbar: bool = True,
        flexible: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._columns = columns
        self._thumb_size = thumb_size
        self._flexible = flexible
        self._cards: list[_ThumbCard] = []

        # ── toolbar ──
        self._toolbar = QWidget()
        tb_lay = QHBoxLayout(self._toolbar)
        tb_lay.setContentsMargins(0, 0, 0, 0)

        self.btn_select_all = QPushButton("Select All")
        self.btn_deselect   = QPushButton("Deselect")
        self.btn_delete     = QPushButton("🗑 Delete Selected")
        self.btn_delete.setStyleSheet("color:red; font-weight:bold;")

        tb_lay.addWidget(self.btn_select_all)
        tb_lay.addWidget(self.btn_deselect)
        tb_lay.addWidget(self.btn_delete)
        tb_lay.addStretch()

        self.btn_select_all.clicked.connect(self.select_all)
        self.btn_deselect.clicked.connect(self.deselect_all)
        self.btn_delete.clicked.connect(self._delete_selected)

        if not show_toolbar:
            self._toolbar.hide()

        # ── grid inside scroll ──
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(4)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._grid_widget)
        self._scroll.setStyleSheet("background:#333;")

        # ── main layout ──
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._toolbar)
        lay.addWidget(self._scroll)

    # ── responsive columns ───────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._flexible:
            self._recalc_columns()

    def showEvent(self, event):
        super().showEvent(event)
        if self._flexible:
            # Delay recalc so viewport has its final geometry
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self._recalc_columns)

    def _recalc_columns(self):
        """Recalculate column count based on scroll-area width."""
        avail = self._scroll.viewport().width() - 8
        card_w = self._thumb_size + 4 + 4  # card size + spacing
        new_cols = max(1, avail // card_w)
        if new_cols != self._columns:
            self._columns = new_cols
            self._relayout()

    # ── public API ───────────────────────────────────────
    @property
    def columns(self) -> int:
        return self._columns

    @columns.setter
    def columns(self, val: int):
        self._columns = max(1, val)
        self._relayout()

    def set_thumb_size(self, size: int) -> None:
        self._thumb_size = size

    def load_paths(self, paths: Sequence[str]) -> None:
        """Replace grid contents with *paths*."""
        self.clear()
        for p in paths:
            self._add_card(str(p))
        self._relayout()

    def add_image_path(self, path: str) -> None:
        self._add_card(path)
        self._relayout()

    def add_image_paths(self, paths: Sequence[str]) -> None:
        for p in paths:
            self._add_card(str(p))
        self._relayout()

    def add_cv_image(self, img: np.ndarray, save_path: str) -> None:
        """Save *img* (BGR) to *save_path*, then add to grid."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, img)
        self._add_card(save_path)
        self._relayout()

    def clear(self) -> None:
        for c in self._cards:
            c.setParent(None)
            c.deleteLater()
        self._cards.clear()

    def selected_paths(self) -> List[str]:
        return [c.filepath for c in self._cards if c.checked]

    def select_all(self) -> None:
        for c in self._cards:
            c.checked = True
        self.selection_changed.emit(self.selected_paths())

    def deselect_all(self) -> None:
        for c in self._cards:
            c.checked = False
        self.selection_changed.emit(self.selected_paths())

    def count(self) -> int:
        return len(self._cards)

    # ── internal ─────────────────────────────────────────
    def _add_card(self, path: str) -> None:
        card = _ThumbCard(path, self._thumb_size)
        card.toggled.connect(self._on_toggled)
        self._cards.append(card)

    def _on_toggled(self, path: str, checked: bool):
        self.selection_changed.emit(self.selected_paths())

    def _relayout(self) -> None:
        # Remove all items from grid
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        for idx, card in enumerate(self._cards):
            r, c = divmod(idx, self._columns)
            self._grid_layout.addWidget(card, r, c)

    def _delete_selected(self) -> None:
        sel = [c for c in self._cards if c.checked]
        for card in sel:
            try:
                Path(card.filepath).unlink(missing_ok=True)
            except Exception:
                pass
            card.setParent(None)
            card.deleteLater()
            self._cards.remove(card)
        self._relayout()
        self.selection_changed.emit(self.selected_paths())
