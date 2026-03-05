"""
MediScan-AI — Main Window
==========================
Two tabs: Verify | Setting
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabBar, QStackedWidget, QApplication,
)

from app.settings_manager import SettingsManager
from app.pages.page_verify import VerifyPage
from app.pages.page_settings import SettingsPage


class MainWindow(QMainWindow):
    """Root window with Verify and Setting pages."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MediScan-AI")
        self.resize(1280, 800)

        self._settings = SettingsManager()

        bg_color = self._settings.get("background_color", "#ffffff")
        text_color = self._settings.get("text_color", "#000000")
        primary_color = self._settings.get("primary_color", "#33FF00")
        font_size = self._settings.get("font_size", 14)

        app = QApplication.instance()
        if app:
            f = app.font()
            f.setPointSize(font_size)
            app.setFont(f)

        self._apply_global_style(bg_color, text_color)

        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        # ── Top bar ──────────────────────────────────────
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
        self._tab_bar.addTab("Verify")
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

        # ── Pages ────────────────────────────────────────
        self._stack = QStackedWidget()
        self._verify_page = VerifyPage(self._settings)
        self._settings_page = SettingsPage(self._settings)

        self._stack.addWidget(self._verify_page)
        self._stack.addWidget(self._settings_page)
        root_lay.addWidget(self._stack)

        self._tab_bar.currentChanged.connect(self._switch_page)
        self._settings_page.settings_saved.connect(self._refresh_theme)

    def _apply_global_style(self, bg: str, text: str):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg};
                color: {text};
            }}
            QLabel {{ color: {text}; }}
            QGroupBox {{ color: {text}; }}
            QGroupBox::title {{ color: {text}; }}
            QLineEdit {{
                color: {text}; background: white;
                border: 1px solid #aaa; padding: 4px;
            }}
            QComboBox {{
                color: {text}; background: white;
                border: 1px solid #aaa; padding: 4px;
            }}
            QListWidget {{
                color: {text}; background: white;
            }}
            QCheckBox {{ color: {text}; }}
            QRadioButton {{ color: {text}; }}
            QSpinBox {{
                color: {text}; background: white;
            }}
            QDoubleSpinBox {{
                color: {text}; background: white;
            }}
            QScrollArea {{ border: none; }}
        """)

    def _switch_page(self, idx: int):
        self._stack.setCurrentIndex(idx)

    def _refresh_theme(self):
        bg = self._settings.get("background_color", "#ffffff")
        text = self._settings.get("text_color", "#000000")
        primary = self._settings.get("primary_color", "#33FF00")
        font_size = self._settings.get("font_size", 14)

        self._apply_global_style(bg, text)

        top_bar = self.findChild(QWidget, "topBar")
        if top_bar:
            top_bar.setStyleSheet(
                f"#topBar{{background:{bg}; border-bottom:2px solid #aaa;}}"
            )

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

        app = QApplication.instance()
        if app:
            f = app.font()
            f.setPointSize(font_size)
            app.setFont(f)

    def closeEvent(self, event):
        self._verify_page.cleanup()
        super().closeEvent(event)
