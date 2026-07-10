"""
ScriptedElite AI - Precision target lock
Capture → detect → sticky lock → virtual controller tracking.
"""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from pathlib import Path

from ui.theme import apply_theme
from ui.main_window import MainWindow

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    apply_theme(app)
    if LOGO_PATH.exists():
        app.setWindowIcon(QIcon(str(LOGO_PATH)))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
