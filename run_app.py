#!/usr/bin/env python3
"""
MediScan-AI — PyQt6 Desktop Application
========================================
Entry point.  Run with:  ``python run_app.py``
"""
import sys
import os
from pathlib import Path

# Suppress OpenCV DSHOW warnings
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Default font
    font = QFont("Segoe UI", 14)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
