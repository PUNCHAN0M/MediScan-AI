"""
MediScan-AI — Main Window
==========================
Top tab-bar: Classifier | Dataset | Setting
Colors driven by settings: primary_color for main tabs, text_color, background_color.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabBar, QStackedWidget, QLabel, QApplication,
)

from app.settings_manager import SettingsManager
from app.pages.page_classifier import ClassifierPage
from app.pages.page_dataset import DatasetPage
from app.pages.page_settings import SettingsPage


class MainWindow(QMainWindow):
    """Root window housing all 3 top-level pages."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MediScan-AI")
        self.resize(1280, 800)

        self._settings = SettingsManager()

        # ── Read theme colours from settings ─────────────
        bg_color = self._settings.get("background_color", "#ffffff")
        text_color = self._settings.get("text_color", "#000000")
        primary_color = self._settings.get("primary_color", "#33FF00")
        font_size = self._settings.get("font_size", 14)

        # Apply saved font size
        app = QApplication.instance()
        if app:
            f = app.font()
            f.setPointSize(font_size)
            app.setFont(f)

        # ── Apply global stylesheet ──────────────────────
        self._apply_global_style(bg_color, text_color)

        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        # ── Top bar: tabs only ───────────────────────────
        top_bar = QWidget()
        top_bar.setObjectName("topBar")
        top_bar.setStyleSheet(
            f"#topBar{{background:{bg_color}; border-bottom:2px solid #aaa;}}"
        )
        top_lay = QHBoxLayout(top_bar)
        top_lay.setContentsMargins(0, 0, 8, 0)
        top_lay.setSpacing(0)

        self._tab_bar = QTabBar()
        self._tab_bar.setExpanding(False)
        self._tab_bar.addTab("Classifier")
        self._tab_bar.addTab("Dataset")
        self._tab_bar.addTab("Setting")
        self._tab_bar.setStyleSheet(f"""
            QTabBar::tab {{
                padding: 10px 40px;
                font-size: 18px;
                font-weight: bold;
                min-width: 150px;
                background: #d0d0d0;
                color: {text_color};
                border: 2px solid #aaa;
                border-bottom: none;
            }}
            QTabBar::tab:selected {{
                background: {primary_color};
                color: {text_color};
            }}
            QTabBar::tab:hover:!selected {{
                background: #bbb;
            }}
        """)
        top_lay.addWidget(self._tab_bar)
        top_lay.addStretch()

        root_lay.addWidget(top_bar)

        # ── Stacked pages ────────────────────────────────
        self._stack = QStackedWidget()
        self._classifier_page = ClassifierPage(self._settings)
        self._dataset_page = DatasetPage(self._settings)
        self._settings_page = SettingsPage(self._settings)

        self._stack.addWidget(self._classifier_page)
        self._stack.addWidget(self._dataset_page)
        self._stack.addWidget(self._settings_page)

        root_lay.addWidget(self._stack)

        # Connections
        self._tab_bar.currentChanged.connect(self._switch_page)
        self._settings_page.settings_saved.connect(self._refresh_theme)

    def _apply_global_style(self, bg: str, text: str):
        """Set global stylesheet for the entire window."""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg};
                color: {text};
            }}
            QLabel {{
                color: {text};
            }}
            QGroupBox {{
                color: {text};
            }}
            QGroupBox::title {{
                color: {text};
            }}
            QLineEdit {{
                color: {text};
                background: white;
                border: 1px solid #aaa;
                padding: 4px;
            }}
            QComboBox {{
                color: {text};
                background: white;
                border: 1px solid #aaa;
                padding: 4px;
            }}
            QTreeWidget {{
                color: {text};
                background: white;
            }}
            QListWidget {{
                color: {text};
                background: white;
            }}
            QCheckBox {{
                color: {text};
            }}
            QRadioButton {{
                color: {text};
            }}
            QSpinBox {{
                color: {text};
                background: white;
            }}
            QDoubleSpinBox {{
                color: {text};
                background: white;
            }}
            QScrollArea {{
                border: none;
            }}
        """)

    def _switch_page(self, idx: int):
        self._stack.setCurrentIndex(idx)

    def _refresh_theme(self):
        """Re-read settings and apply theme changes live."""
        bg = self._settings.get("background_color", "#ffffff")
        text = self._settings.get("text_color", "#000000")
        primary = self._settings.get("primary_color", "#33FF00")
        secondary = self._settings.get("secondary_color", "#FF6A00")
        font_size = self._settings.get("font_size", 14)

        # Global colours
        self._apply_global_style(bg, text)

        # Top bar background
        top_bar = self.findChild(QWidget, "topBar")
        if top_bar:
            top_bar.setStyleSheet(
                f"#topBar{{background:{bg}; border-bottom:2px solid #aaa;}}"
            )

        # Main tab bar
        self._tab_bar.setStyleSheet(f"""
            QTabBar::tab {{
                padding: 10px 40px;
                font-size: 18px;
                font-weight: bold;
                min-width: 150px;
                background: #d0d0d0;
                color: {text};
                border: 2px solid #aaa;
                border-bottom: none;
            }}
            QTabBar::tab:selected {{
                background: {primary};
                color: {text};
            }}
            QTabBar::tab:hover:!selected {{
                background: #bbb;
            }}
        """)

        # Dataset sub-tab bar
        ds_tab_bar = self._dataset_page._tab_bar
        ds_tab_bar.setStyleSheet(f"""
            QTabBar::tab {{
                padding: 8px 30px;
                font-size: 15px;
                font-weight: bold;
                min-width: 120px;
                background: #ddd;
                color: {text};
                border: 1px solid #aaa;
            }}
            QTabBar::tab:selected {{
                background: {secondary};
                color: white;
            }}
        """)

        # Font
        app = QApplication.instance()
        if app:
            f = app.font()
            f.setPointSize(font_size)
            app.setFont(f)

    def closeEvent(self, event):
        self._classifier_page.cleanup()
        self._dataset_page.cleanup()
        super().closeEvent(event)
