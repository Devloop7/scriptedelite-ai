"""ScriptedElite brand theme — electric blue on deep navy."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication

BG = "#060910"
BG_PANEL = "#0a101c"
BG_CARD = "#0e1628"
BG_ELEVATED = "#121c32"
BORDER = "#1c2d4a"
BORDER_HOT = "#00b4ff"
ACCENT = "#00c8ff"
ACCENT_DIM = "#0077aa"
TEXT = "#e8eef8"
TEXT_MUTED = "#7a879e"
GREEN = "#22c55e"
RED = "#ef4444"
ORANGE = "#f59e0b"
GOLD = "#fbbf24"


def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    app.setStyleSheet(f"""
    QMainWindow, QWidget {{
        background: {BG};
        color: {TEXT};
        font-family: "Segoe UI", "Bahnschrift", system-ui, sans-serif;
        font-size: 13px;
    }}
    QToolTip {{
        background: {BG_ELEVATED};
        color: {TEXT};
        border: 1px solid {BORDER_HOT};
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 12px;
    }}
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        background: {BG_PANEL};
        top: -1px;
        padding: 8px;
    }}
    QTabBar::tab {{
        background: {BG_CARD};
        color: {TEXT_MUTED};
        border: 1px solid {BORDER};
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 10px 18px;
        margin-right: 2px;
        font-weight: 600;
        min-width: 72px;
    }}
    QTabBar::tab:selected {{
        background: {BG_PANEL};
        color: {ACCENT};
        border-color: {BORDER_HOT};
    }}
    QTabBar::tab:hover:!selected {{
        color: {TEXT};
        border-color: {ACCENT_DIM};
    }}
    QGroupBox {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        margin-top: 14px;
        padding: 14px 12px 12px 12px;
        font-weight: 700;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 8px;
        color: {ACCENT};
        font-size: 11px;
        letter-spacing: 1.1px;
        font-weight: 800;
    }}
    QLabel#valueBadge {{
        background: #0a1a2e;
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 3px 10px;
        color: {ACCENT};
        font-weight: 700;
        font-size: 12px;
        min-width: 48px;
    }}
    QLabel#statPill {{
        background: {BG_ELEVATED};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 6px 12px;
        color: {TEXT};
        font-size: 11px;
        font-weight: 600;
    }}
    QPushButton {{
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #162038, stop:1 #0f1728);
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 8px 14px;
        color: {TEXT};
        font-weight: 600;
    }}
    QPushButton:hover {{
        border-color: {ACCENT};
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1a2a4a, stop:1 #122036);
    }}
    QPushButton:pressed {{ background: #0a1424; }}
    QPushButton:disabled {{ color: #4a5568; border-color: #1a2030; }}
    QPushButton#primaryBtn {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0066aa, stop:1 #00a0d0);
        border: 1px solid {ACCENT};
        color: white;
        font-weight: 700;
        padding: 11px 20px;
    }}
    QPushButton#primaryBtn:hover {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0080cc, stop:1 #00c8ff);
    }}
    QPushButton#dangerBtn {{
        background: #2a1010;
        border: 1px solid {RED};
        color: {RED};
    }}
    QPushButton#aimToggle {{
        font-size: 14px;
        padding: 14px 20px;
        font-weight: 800;
        letter-spacing: 0.5px;
        border-radius: 10px;
    }}
    QPushButton#aimToggle[on="true"] {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0a2e1a, stop:1 #0d3a22);
        border: 2px solid {GREEN};
        color: {GREEN};
    }}
    QPushButton#aimToggle[on="false"] {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2a1010, stop:1 #1a0c0c);
        border: 2px solid {RED};
        color: {RED};
    }}
    QPushButton#ghostBtn {{
        background: transparent;
        border: 1px dashed {BORDER};
        color: {TEXT_MUTED};
        padding: 6px 12px;
    }}
    QPushButton#ghostBtn:hover {{
        border-color: {ACCENT};
        color: {ACCENT};
    }}
    QPushButton#segBtn {{
        background: {BG_ELEVATED};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 8px 14px;
        font-weight: 700;
    }}
    QPushButton#segBtn[active="true"] {{
        background: #0a3050;
        border: 1px solid {BORDER_HOT};
        color: {ACCENT};
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        background: #0a1420;
        border: 1px solid {BORDER};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #00d4ff, stop:1 #0088cc);
        border: 1px solid #004466;
        width: 16px; height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::sub-page:horizontal {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #005588, stop:1 {ACCENT});
        border-radius: 3px;
    }}
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
        background: #0a1220;
        border: 1px solid {BORDER};
        border-radius: 7px;
        padding: 6px 10px;
        color: {TEXT};
        selection-background-color: {ACCENT_DIM};
    }}
    QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {ACCENT_DIM};
    }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox QAbstractItemView {{
        background: {BG_CARD};
        border: 1px solid {BORDER_HOT};
        selection-background-color: #0a3050;
        color: {TEXT};
    }}
    QCheckBox {{ spacing: 8px; color: {TEXT}; }}
    QCheckBox::indicator {{
        width: 17px; height: 17px;
        border-radius: 4px;
        border: 1px solid {BORDER};
        background: #0a1220;
    }}
    QCheckBox::indicator:checked {{
        background: {ACCENT};
        border-color: {ACCENT};
    }}
    QScrollArea {{ border: none; background: transparent; }}
    QStatusBar {{
        background: #04060c;
        color: {TEXT_MUTED};
        border-top: 1px solid {BORDER};
        padding: 4px;
    }}
    QFrame#previewFrame {{
        background: #04060c;
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}
    QListWidget {{
        background: #0a1220;
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 4px;
        outline: none;
    }}
    QListWidget::item {{
        padding: 8px 10px;
        border-radius: 6px;
        margin: 1px 0;
    }}
    QListWidget::item:selected {{
        background: #0a3050;
        color: {ACCENT};
        border: 1px solid {BORDER_HOT};
    }}
    QListWidget::item:hover:!selected {{
        background: #121c30;
    }}
    QSplitter::handle {{
        background: {BORDER};
        width: 2px;
    }}
    """)
