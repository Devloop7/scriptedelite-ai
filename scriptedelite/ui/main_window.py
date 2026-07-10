"""
ScriptedElite main window — tactical control surface.
Tabs: Lock | Setup | More. Live feed is the primary status display.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QSplitter, QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)
from pynput import keyboard, mouse

from core.capture import WindowInfo, list_open_windows
from core.color_detector import parse_hex_color
from core.config import AppConfig
from core.worker import AimWorker
from ui.theme import ACCENT, BORDER, GOLD, GREEN, RED, TEXT_MUTED
from ui.widgets import badge, mini_slider, slider_row

BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScriptedElite AI  ·  Precision Target Lock")
        self.resize(1380, 880)
        self.setMinimumSize(1000, 680)

        if LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(LOGO_PATH)))

        self.cfg = AppConfig.load()
        self.worker: AimWorker | None = None
        self.worker_thread: threading.Thread | None = None
        self._syncing = False
        self._window_cache: list[WindowInfo] = []

        self._build_ui()
        self._load_cfg_to_ui()
        self._wire_signals()
        self._setup_hotkeys()
        self._refresh_window_list()

    # ══════════════════════════════════════════════════════════════════════
    # Layout
    # ══════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 8)
        root.setSpacing(8)

        # Header
        header = QHBoxLayout()
        header.setSpacing(12)
        logo_lab = QLabel()
        if LOGO_PATH.exists():
            pix = QPixmap(str(LOGO_PATH)).scaled(52, 52, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_lab.setPixmap(pix)
        logo_lab.setFixedSize(52, 52)
        header.addWidget(logo_lab)

        titles = QVBoxLayout()
        titles.setSpacing(1)
        t1 = QLabel("SCRIPTED ELITE")
        t1.setStyleSheet(f"font-size: 20px; font-weight: 900; color: {ACCENT}; letter-spacing: 2px;")
        t2 = QLabel("Acquire  →  Lock  →  Hold")
        t2.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")
        titles.addWidget(t1)
        titles.addWidget(t2)
        header.addLayout(titles)
        header.addStretch()

        self.lock_badge = QLabel("NO TARGET")
        self._style_badge(self.lock_badge, "idle")
        header.addWidget(self.lock_badge)

        self.live_badge = QLabel("● ENGINE OFF")
        self._style_badge(self.live_badge, "off")
        header.addWidget(self.live_badge)
        root.addLayout(header)

        # Body: controls | feed
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(6)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_lock(), "  Lock  ")
        self.tabs.addTab(self._tab_setup(), "  Setup  ")
        self.tabs.addTab(self._tab_more(), "  More  ")
        left_l.addWidget(self.tabs, 1)

        # Bottom bar: engine + aim always reachable
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.start_btn = QPushButton("▶  Start Engine")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_engine)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_engine)
        bar.addWidget(self.start_btn)
        bar.addWidget(self.stop_btn)
        left_l.addLayout(bar)

        self.aim_btn = QPushButton("AIM OFF  ·  F")
        self.aim_btn.setObjectName("aimToggle")
        self.aim_btn.setProperty("on", "false")
        self.aim_btn.setCursor(Qt.PointingHandCursor)
        self.aim_btn.clicked.connect(self.toggle_aim)
        self.aim_btn.setToolTip("Master aim. Hold activation (L2 / right mouse) to track locked target.")
        left_l.addWidget(self.aim_btn)

        split.addWidget(left)

        # Live feed
        right = QFrame()
        right.setObjectName("previewFrame")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 10, 10, 10)
        rl.setSpacing(6)

        ph = QHBoxLayout()
        pt = QLabel("LIVE FEED")
        pt.setStyleSheet(f"font-weight: 800; color: {ACCENT}; letter-spacing: 1.2px; font-size: 11px;")
        ph.addWidget(pt)
        ph.addStretch()
        self.show_prev = QCheckBox("Preview")
        self.show_prev.setChecked(True)
        ph.addWidget(self.show_prev)
        self.draw_zone_cb = QCheckBox("Zone")
        self.draw_zone_cb.setChecked(True)
        self.draw_zone_cb.setToolTip("Draw FOV circle and crosshair center.")
        ph.addWidget(self.draw_zone_cb)
        rl.addLayout(ph)

        self.preview = QLabel("Start Engine to begin capture and sticky lock.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(460, 360)
        self.preview.setStyleSheet(
            f"color: {TEXT_MUTED}; background: #04060c; border-radius: 10px; padding: 16px;"
        )
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rl.addWidget(self.preview, 1)

        self.stats_lab = QLabel("Waiting…")
        self.stats_lab.setObjectName("statPill")
        self.stats_lab.setWordWrap(True)
        rl.addWidget(self.stats_lab)

        split.addWidget(right)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 5)
        root.addWidget(split, 1)

        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        self.sb.showMessage("Select window → Start Engine → Enable AIM → Hold activation")

    def _style_badge(self, lab: QLabel, kind: str):
        styles = {
            "idle": f"background:#121018; color:{TEXT_MUTED}; border:1px solid {BORDER}; "
                    f"border-radius:8px; padding:7px 12px; font-weight:800;",
            "off": f"background:#1a1010; color:{RED}; border:1px solid {RED}; "
                   f"border-radius:8px; padding:7px 12px; font-weight:800;",
            "on": f"background:#0a2a18; color:{GREEN}; border:1px solid {GREEN}; "
                  f"border-radius:8px; padding:7px 12px; font-weight:800;",
            "track": f"background:#1a1808; color:{GOLD}; border:1px solid {GOLD}; "
                     f"border-radius:8px; padding:7px 12px; font-weight:800;",
            "coast": f"background:#1a1810; color:{GOLD}; border:1px solid {GOLD}; "
                     f"border-radius:8px; padding:7px 12px; font-weight:800;",
        }
        lab.setStyleSheet(styles.get(kind, styles["idle"]))

    # ── Lock tab ─────────────────────────────────────────────────────────
    def _tab_lock(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setSpacing(8)

        # FOV + aim part
        top = QHBoxLayout()
        g_fov = QGroupBox("FOV")
        fl = QVBoxLayout(g_fov)
        self.zone_radius = QSlider(Qt.Horizontal)
        self.zone_radius.setRange(40, 700)
        self.zone_badge = badge("200")
        fl.addWidget(slider_row(
            "Zone Radius",
            "Acquire range from crosshair (green circle). Hold expands after lock.",
            self.zone_radius, self.zone_badge,
        ))
        top.addWidget(g_fov, 1)

        g_part = QGroupBox("AIM POINT")
        pl = QVBoxLayout(g_part)
        row = QHBoxLayout()
        self.part_head = QPushButton("HEAD")
        self.part_chest = QPushButton("CHEST")
        self.part_body = QPushButton("BODY")
        for b in (self.part_head, self.part_chest, self.part_body):
            b.setObjectName("segBtn")
            b.setProperty("active", "false")
            b.setCursor(Qt.PointingHandCursor)
            b.setCheckable(False)
            row.addWidget(b)
        self.part_head.clicked.connect(lambda: self._set_part("head"))
        self.part_chest.clicked.connect(lambda: self._set_part("chest"))
        self.part_body.clicked.connect(lambda: self._set_part("body"))
        pl.addLayout(row)
        self.aim_fine = QSlider(Qt.Horizontal)
        self.aim_fine.setRange(-10, 10)
        self.aim_fine_badge = badge("0")
        pl.addWidget(slider_row(
            "Fine Tune",
            "Vertical nudge (− higher, + lower).",
            self.aim_fine, self.aim_fine_badge,
        ))
        top.addWidget(g_part)
        lay.addLayout(top)

        g_track = QGroupBox("TRACKING")
        tl = QVBoxLayout(g_track)
        self.smooth = QSlider(Qt.Horizontal)
        self.smooth.setRange(0, 85)
        self.smooth_badge = badge("35%")
        tl.addWidget(slider_row(
            "Smoothness",
            "Higher = more damping, softer settle, less snap.",
            self.smooth, self.smooth_badge,
        ))
        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(10, 250)
        self.strength_badge = badge("100%")
        tl.addWidget(slider_row(
            "Strength",
            "Overall aim gain. Lower if overshooting; raise if too weak.",
            self.strength, self.strength_badge,
        ))
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 8.0)
        self.speed.setSingleStep(0.1)
        self.speed.setDecimals(1)
        self.speed.setToolTip("How fast the stick closes large gaps.")
        row_s = QHBoxLayout()
        lab_s = QLabel("Speed")
        lab_s.setStyleSheet("font-weight: 700;")
        row_s.addWidget(lab_s)
        row_s.addStretch()
        row_s.addWidget(self.speed)
        tl.addLayout(row_s)
        lay.addWidget(g_track)

        g_hold = QGroupBox("LOCK HOLD")
        hl = QFormLayout(g_hold)
        self.lock_hold = QSpinBox()
        self.lock_hold.setRange(8, 60)
        self.lock_hold.setToolTip("Missed frames to coast before release.")
        hl.addRow("Hold Frames", self.lock_hold)
        self.shake_red = QCheckBox("Shake reduction")
        self.shake_red.setChecked(True)
        self.shake_red.setToolTip("Extra settle filter when already near the aim point.")
        hl.addRow(self.shake_red)
        lay.addWidget(g_hold)

        lay.addStretch()
        scroll.setWidget(host)
        return scroll

    def _set_part(self, part: str):
        for name, btn in (("head", self.part_head), ("chest", self.part_chest), ("body", self.part_body)):
            btn.setProperty("active", "true" if name == part else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self.on_config_changed()

    def _current_part(self) -> str:
        if self.part_chest.property("active") == "true" or self.part_chest.property("active") is True:
            return "chest"
        if self.part_body.property("active") == "true" or self.part_body.property("active") is True:
            return "body"
        return "head"

    # ── Setup tab ────────────────────────────────────────────────────────
    def _tab_setup(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setSpacing(8)

        g = QGroupBox("CAPTURE")
        gl = QFormLayout(g)
        self.cap_mode = QComboBox()
        self.cap_mode.addItem("Application Window", "window")
        self.cap_mode.addItem("Chiaki (keyword)", "chiaki")
        self.cap_mode.addItem("Capture Card", "capture_card")
        self.cap_mode.addItem("Desktop", "desktop")
        gl.addRow("Source", self.cap_mode)
        lay.addWidget(g)

        self.win_group = QGroupBox("WINDOWS")
        wl = QVBoxLayout(self.win_group)
        row = QHBoxLayout()
        self.refresh_wins_btn = QPushButton("↻  Refresh")
        self.refresh_wins_btn.setObjectName("ghostBtn")
        self.refresh_wins_btn.clicked.connect(self._refresh_window_list)
        row.addWidget(self.refresh_wins_btn)
        row.addStretch()
        self.win_count_lab = QLabel("")
        self.win_count_lab.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        row.addWidget(self.win_count_lab)
        wl.addLayout(row)
        self.win_list = QListWidget()
        self.win_list.setMinimumHeight(150)
        self.win_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.win_list.itemSelectionChanged.connect(self._on_window_selected)
        wl.addWidget(self.win_list)
        self.selected_win_lab = QLabel("Selected: none")
        self.selected_win_lab.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
        wl.addWidget(self.selected_win_lab)
        self.chiaki_win = QLineEdit("Chiaki")
        self.chiaki_win.setPlaceholderText("Chiaki keyword")
        wl.addWidget(self.chiaki_win)
        lay.addWidget(self.win_group)

        g2 = QGroupBox("OPTIONS")
        g2l = QFormLayout(g2)
        self.cap_dev = QSpinBox()
        self.cap_dev.setRange(0, 10)
        g2l.addRow("Capture Card #", self.cap_dev)
        self.region_size = QSpinBox()
        self.region_size.setRange(0, 2160)
        self.region_size.setSingleStep(40)
        self.region_size.setSpecialValueText("Full window")
        g2l.addRow("Region Size", self.region_size)
        self.use_gpu = QCheckBox("GPU (CUDA)")
        self.use_gpu.setChecked(True)
        g2l.addRow(self.use_gpu)
        lay.addWidget(g2)

        g3 = QGroupBox("CALIBRATION")
        g3l = QFormLayout(g3)
        self.cal_x = QSpinBox(); self.cal_x.setRange(-300, 300)
        self.cal_y = QSpinBox(); self.cal_y.setRange(-300, 300)
        self.cal_x.setToolTip("Shift center X onto in-game crosshair.")
        self.cal_y.setToolTip("Shift center Y onto in-game crosshair.")
        g3l.addRow("Calibrate X", self.cal_x)
        g3l.addRow("Calibrate Y", self.cal_y)
        lay.addWidget(g3)

        g4 = QGroupBox("ACTIVATION")
        g4l = QFormLayout(g4)
        self.gated = QCheckBox("Require activation button")
        self.gated.setChecked(True)
        g4l.addRow(self.gated)
        self.act_btn = QComboBox()
        self.act_btn.addItems([
            "l2", "r2", "lt", "rt", "lb", "rb", "l1", "r1",
            "a", "b", "x", "y", "ls", "rs", "none",
        ])
        g4l.addRow("Controller", self.act_btn)
        self.ads_key = QComboBox()
        self.ads_key.setEditable(True)
        self.ads_key.addItems(["right", "left", "middle", "ctrl", "shift", "alt", "c", "v", "x"])
        g4l.addRow("Mouse / Key", self.ads_key)
        self.toggle_key = QLineEdit("f")
        self.toggle_key.setMaxLength(1)
        g4l.addRow("Master Toggle", self.toggle_key)
        self.ctrl_platform = QComboBox()
        self.ctrl_platform.addItems(["playstation", "xbox"])
        g4l.addRow("Labels", self.ctrl_platform)
        lay.addWidget(g4)

        g5 = QGroupBox("STATUS")
        g5l = QVBoxLayout(g5)
        self.cap_status = QLabel("Capture: idle")
        self.cap_status.setStyleSheet(f"color: {TEXT_MUTED};")
        self.pad_status = QLabel("Pad: —")
        self.pad_status.setStyleSheet(f"color: {TEXT_MUTED};")
        g5l.addWidget(self.cap_status)
        g5l.addWidget(self.pad_status)
        lay.addWidget(g5)

        lay.addStretch()
        scroll.setWidget(host)
        return scroll

    # ── More tab ─────────────────────────────────────────────────────────
    def _tab_more(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setSpacing(8)

        g = QGroupBox("DETECTION")
        gl = QFormLayout(g)
        self.engine = QComboBox()
        self.engine.addItems(["yolo", "color", "hybrid"])
        gl.addRow("Engine", self.engine)
        self.conf = QSlider(Qt.Horizontal)
        self.conf.setRange(15, 85)
        self.conf_badge = badge("42%")
        gl.addRow("Confidence", mini_slider(self.conf, self.conf_badge))
        self.draw_boxes = QCheckBox("Draw boxes")
        self.draw_boxes.setChecked(True)
        gl.addRow(self.draw_boxes)
        self.priority = QComboBox()
        self.priority.addItems(["closest", "highest_conf"])
        self.priority.setToolTip("Only for acquiring a new target.")
        gl.addRow("Acquire Priority", self.priority)
        lay.addWidget(g)

        g2 = QGroupBox("COLOR ENGINE")
        g2l = QFormLayout(g2)
        color_row = QHBoxLayout()
        self.color_swatch = QLabel()
        self.color_swatch.setFixedSize(32, 24)
        self.color_swatch.setStyleSheet(
            "background:#FF00FA; border:1px solid #2a3a5c; border-radius:4px;"
        )
        color_row.addWidget(self.color_swatch)
        self.enemy_color_edit = QLineEdit("#FF00FA")
        self.enemy_color_edit.setMaxLength(16)
        color_row.addWidget(self.enemy_color_edit, 1)
        self.pick_color_btn = QPushButton("Pick…")
        self.pick_color_btn.clicked.connect(self._pick_enemy_color)
        color_row.addWidget(self.pick_color_btn)
        g2l.addRow("Enemy Color", color_row)
        preset_row = QHBoxLayout()
        for label, hx in [
            ("Mag", "#FF00FA"), ("Red", "#FF2020"), ("Org", "#FF8800"),
            ("Grn", "#00FF00"), ("Yel", "#FFE600"), ("Cyn", "#00E5FF"),
        ]:
            b = QPushButton(label)
            b.setToolTip(hx)
            b.clicked.connect(lambda checked=False, h=hx: self._set_enemy_color_hex(h))
            preset_row.addWidget(b)
        g2l.addRow(preset_row)
        self.color_tol = QSlider(Qt.Horizontal)
        self.color_tol.setRange(10, 100)
        self.color_tol_badge = badge("55")
        g2l.addRow("Tolerance", mini_slider(self.color_tol, self.color_tol_badge))
        self.color_off = QDoubleSpinBox()
        self.color_off.setRange(0, 200)
        self.color_off.setSuffix(" px")
        g2l.addRow("Aim Offset Y", self.color_off)
        lay.addWidget(g2)

        g3 = QGroupBox("RESPONSE")
        g3l = QFormLayout(g3)
        self.stick_resp = QSlider(Qt.Horizontal)
        self.stick_resp.setRange(20, 120)
        self.stick_resp_badge = badge("55")
        g3l.addRow(
            "Stick Response",
            mini_slider(self.stick_resp, self.stick_resp_badge),
        )
        self.stick_resp.setToolTip(
            "Px of camera motion per full stick per frame. "
            "Raise if game is sensitive (overshoots); lower if sluggish."
        )
        self.adaptive_stick = QCheckBox("Adaptive stick response")
        self.adaptive_stick.setToolTip("Estimate stick→pixel response while aiming.")
        g3l.addRow(self.adaptive_stick)
        self.max_dist = QSpinBox(); self.max_dist.setRange(50, 1400)
        self.fps_limit = QSpinBox(); self.fps_limit.setRange(30, 240)
        g3l.addRow("Max Distance", self.max_dist)
        g3l.addRow("FPS Limit", self.fps_limit)
        lay.addWidget(g3)

        g4 = QGroupBox("PREDICTION")
        g4l = QFormLayout(g4)
        self.use_pred = QCheckBox("Lead moving targets")
        self.use_pred.setToolTip("Small Kalman lead. Off by default for pure lock.")
        g4l.addRow(self.use_pred)
        self.pred_strength = QSlider(Qt.Horizontal)
        self.pred_strength.setRange(0, 100)
        self.pred_badge = badge("35%")
        g4l.addRow("Predict Strength", mini_slider(self.pred_strength, self.pred_badge))
        self.humanize = QSlider(Qt.Horizontal)
        self.humanize.setRange(0, 40)
        self.human_badge = badge("0")
        g4l.addRow("Humanization", mini_slider(self.humanize, self.human_badge))
        lay.addWidget(g4)

        lay.addStretch()
        scroll.setWidget(host)
        return scroll

    # ── Color helpers ────────────────────────────────────────────────────
    def _set_enemy_color_hex(self, hx: str):
        self.enemy_color_edit.setText(hx.upper() if hx.startswith("#") else f"#{hx.upper()}")
        self._update_color_swatch()
        self.on_config_changed()

    def _pick_enemy_color(self):
        r, g, b = parse_hex_color(self.enemy_color_edit.text())
        col = QColorDialog.getColor(QColor(r, g, b), self, "Select Enemy Color")
        if col.isValid():
            self.enemy_color_edit.setText(col.name().upper())
            self._update_color_swatch()
            self.on_config_changed()

    def _update_color_swatch(self):
        r, g, b = parse_hex_color(self.enemy_color_edit.text())
        self.color_swatch.setStyleSheet(
            f"background:rgb({r},{g},{b}); border:1px solid #2a3a5c; border-radius:4px;"
        )

    # ── Window list ──────────────────────────────────────────────────────
    def _refresh_window_list(self):
        self._window_cache = list_open_windows()
        prev_hwnd = int(getattr(self.cfg, "window_hwnd", 0) or 0)
        prev_title = (self.cfg.window_title or self.cfg.chiaki_window or "").strip()

        self.win_list.blockSignals(True)
        self.win_list.clear()
        select_row = -1
        for i, w in enumerate(self._window_cache):
            item = QListWidgetItem(w.label)
            item.setData(Qt.UserRole, w.hwnd)
            item.setData(Qt.UserRole + 1, w.title)
            item.setToolTip(f"HWND {w.hwnd}\n{w.title}\n{w.width}x{w.height}")
            self.win_list.addItem(item)
            if prev_hwnd and w.hwnd == prev_hwnd:
                select_row = i
            elif select_row < 0 and prev_title and prev_title.lower() in w.title.lower():
                select_row = i
        self.win_list.blockSignals(False)

        self.win_count_lab.setText(f"{len(self._window_cache)} windows")
        if select_row >= 0:
            self.win_list.setCurrentRow(select_row)
            self._apply_selected_window_label()
        elif self.cfg.window_title:
            self.selected_win_lab.setText(f"Selected: {self.cfg.window_title}")
        else:
            self.selected_win_lab.setText("Selected: none")

    def _on_window_selected(self):
        self._apply_selected_window_label()
        self.on_config_changed()

    def _apply_selected_window_label(self):
        item = self.win_list.currentItem()
        if not item:
            return
        title = item.data(Qt.UserRole + 1) or item.text()
        self.selected_win_lab.setText(f"Selected: {title}")
        if self.cap_mode.currentData() == "chiaki":
            self.chiaki_win.setText(title)

    def _selected_window(self) -> tuple[str, int]:
        item = self.win_list.currentItem()
        if item:
            return str(item.data(Qt.UserRole + 1) or ""), int(item.data(Qt.UserRole) or 0)
        return self.cfg.window_title or "", int(self.cfg.window_hwnd or 0)

    def _on_cap_mode_changed(self, *_):
        mode = self.cap_mode.currentData() or self.cap_mode.currentText()
        self.chiaki_win.setEnabled(mode == "chiaki")
        self.cap_dev.setEnabled(mode == "capture_card")
        if mode in ("window", "chiaki") and self.win_list.count() == 0:
            self._refresh_window_list()
        self.on_config_changed()

    # ── Config sync ──────────────────────────────────────────────────────
    def _load_cfg_to_ui(self):
        self._syncing = True
        c = self.cfg

        mode = c.capture_mode or "window"
        idx = self.cap_mode.findData(mode)
        self.cap_mode.setCurrentIndex(idx if idx >= 0 else 0)

        self.chiaki_win.setText(c.chiaki_window or c.window_title or "Chiaki")
        self.cap_dev.setValue(c.capture_device)
        self.region_size.setValue(getattr(c, "region_size", 0) or 0)
        self.use_gpu.setChecked(getattr(c, "use_gpu", True))
        self.cal_x.setValue(c.cal_x)
        self.cal_y.setValue(c.cal_y)

        self.zone_radius.setValue(c.zone_radius)
        self.zone_badge.setText(str(c.zone_radius))
        part = (c.target_part or "head").lower()
        self._set_part_ui(part)

        fine = int(round(float(getattr(c, "aim_fine_tune", 0.0)) * 100))
        self.aim_fine.setValue(max(-10, min(10, fine)))
        self.aim_fine_badge.setText(str(self.aim_fine.value()))

        self.smooth.setValue(int(c.smoothing * 100))
        self.smooth_badge.setText(f"{int(c.smoothing * 100)}%")
        self.strength.setValue(int(c.strength * 100))
        self.strength_badge.setText(f"{int(c.strength * 100)}%")
        self.speed.setValue(c.acquisition_speed)
        self.lock_hold.setValue(int(getattr(c, "lock_hold_frames", 28)))
        self.shake_red.setChecked(getattr(c, "shake_reduction", True))

        self.use_pred.setChecked(c.use_linear_prediction)
        self.pred_strength.setValue(int(getattr(c, "prediction_strength", 0.35) * 100))
        self.pred_badge.setText(f"{self.pred_strength.value()}%")
        self.priority.setCurrentText(getattr(c, "target_priority", "closest"))
        self.humanize.setValue(int(c.humanization * 100))
        self.human_badge.setText(str(self.humanize.value()))
        self.max_dist.setValue(c.max_distance)
        self.fps_limit.setValue(c.fps_limit)
        self.stick_resp.setValue(int(getattr(c, "stick_response", 55.0)))
        self.stick_resp_badge.setText(str(self.stick_resp.value()))
        self.adaptive_stick.setChecked(getattr(c, "adaptive_stick", False))

        self.engine.setCurrentText(c.engine)
        self.conf.setValue(int(c.confidence * 100))
        self.conf_badge.setText(f"{int(c.confidence * 100)}%")
        self.draw_boxes.setChecked(getattr(c, "draw_boxes", True))
        self.color_off.setValue(c.color_head_offset)
        self.enemy_color_edit.setText(getattr(c, "enemy_color", "#FF00FA"))
        self.color_tol.setValue(int(getattr(c, "color_tolerance", 55.0)))
        self.color_tol_badge.setText(str(self.color_tol.value()))
        self._update_color_swatch()

        self.gated.setChecked(c.gated_aim)
        aidx = self.act_btn.findText((c.activation_button or "l2").lower())
        self.act_btn.setCurrentIndex(max(0, aidx))
        self.ads_key.setCurrentText(c.ads_key or "right")
        self.toggle_key.setText((c.toggle_key or "f")[:1])
        self.ctrl_platform.setCurrentText(getattr(c, "controller_platform", "playstation"))

        self.show_prev.setChecked(c.show_preview)
        self.draw_zone_cb.setChecked(c.draw_zone)
        self._syncing = False
        self._on_cap_mode_changed()
        tk = (c.toggle_key or "f").upper()
        self.aim_btn.setText(f"AIM OFF  ·  {tk}")

    def _set_part_ui(self, part: str):
        for name, btn in (("head", self.part_head), ("chest", self.part_chest), ("body", self.part_body)):
            btn.setProperty("active", "true" if name == part else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _wire_signals(self):
        def bind(w, sig):
            getattr(w, sig).connect(self.on_config_changed)

        for w, sig in [
            (self.chiaki_win, "textChanged"),
            (self.cap_dev, "valueChanged"),
            (self.region_size, "valueChanged"),
            (self.use_gpu, "stateChanged"),
            (self.cal_x, "valueChanged"),
            (self.cal_y, "valueChanged"),
            (self.zone_radius, "valueChanged"),
            (self.aim_fine, "valueChanged"),
            (self.smooth, "valueChanged"),
            (self.strength, "valueChanged"),
            (self.speed, "valueChanged"),
            (self.lock_hold, "valueChanged"),
            (self.use_pred, "stateChanged"),
            (self.pred_strength, "valueChanged"),
            (self.priority, "currentTextChanged"),
            (self.shake_red, "stateChanged"),
            (self.humanize, "valueChanged"),
            (self.max_dist, "valueChanged"),
            (self.fps_limit, "valueChanged"),
            (self.stick_resp, "valueChanged"),
            (self.adaptive_stick, "stateChanged"),
            (self.engine, "currentTextChanged"),
            (self.conf, "valueChanged"),
            (self.draw_boxes, "stateChanged"),
            (self.color_off, "valueChanged"),
            (self.enemy_color_edit, "textChanged"),
            (self.color_tol, "valueChanged"),
            (self.gated, "stateChanged"),
            (self.act_btn, "currentTextChanged"),
            (self.ads_key, "currentTextChanged"),
            (self.toggle_key, "textChanged"),
            (self.ctrl_platform, "currentTextChanged"),
            (self.show_prev, "stateChanged"),
            (self.draw_zone_cb, "stateChanged"),
        ]:
            bind(w, sig)

        self.cap_mode.currentIndexChanged.connect(self._on_cap_mode_changed)
        self.zone_radius.valueChanged.connect(lambda v: self.zone_badge.setText(str(v)))
        self.smooth.valueChanged.connect(lambda v: self.smooth_badge.setText(f"{v}%"))
        self.strength.valueChanged.connect(lambda v: self.strength_badge.setText(f"{v}%"))
        self.pred_strength.valueChanged.connect(lambda v: self.pred_badge.setText(f"{v}%"))
        self.humanize.valueChanged.connect(lambda v: self.human_badge.setText(str(v)))
        self.conf.valueChanged.connect(lambda v: self.conf_badge.setText(f"{v}%"))
        self.color_tol.valueChanged.connect(lambda v: self.color_tol_badge.setText(str(v)))
        self.aim_fine.valueChanged.connect(lambda v: self.aim_fine_badge.setText(str(v)))
        self.stick_resp.valueChanged.connect(lambda v: self.stick_resp_badge.setText(str(v)))
        self.enemy_color_edit.textChanged.connect(lambda *_: self._update_color_swatch())
        self.toggle_key.textChanged.connect(self._update_aim_label)

    def _update_aim_label(self, *_):
        if self.worker and self.worker.aim_enabled:
            return
        tk = (self.toggle_key.text() or "f").upper()[:1] or "F"
        self.aim_btn.setText(f"AIM OFF  ·  {tk}")

    def on_config_changed(self, *_):
        if self._syncing:
            return
        c = self.cfg
        mode = self.cap_mode.currentData()
        if mode is None:
            mode = self.cap_mode.currentText()
        c.capture_mode = str(mode)

        title, hwnd = self._selected_window()
        if c.capture_mode == "chiaki":
            c.chiaki_window = self.chiaki_win.text().strip() or "Chiaki"
            c.window_title = title or c.chiaki_window
            c.window_hwnd = hwnd if title else 0
        elif c.capture_mode == "window":
            c.window_title = title or self.chiaki_win.text().strip()
            c.window_hwnd = hwnd
            c.chiaki_window = c.window_title or c.chiaki_window
        else:
            c.window_title = title or c.window_title
            c.window_hwnd = hwnd

        c.capture_device = self.cap_dev.value()
        c.region_size = self.region_size.value()
        c.use_gpu = self.use_gpu.isChecked()
        c.cal_x = self.cal_x.value()
        c.cal_y = self.cal_y.value()

        c.zone_radius = self.zone_radius.value()
        c.target_part = self._current_part()
        c.aim_fine_tune = self.aim_fine.value() / 100.0
        c.target_offset = c.offset_for_part()

        c.smoothing = self.smooth.value() / 100.0
        c.strength = self.strength.value() / 100.0
        c.acquisition_speed = self.speed.value()
        c.lock_hold_frames = self.lock_hold.value()
        c.use_linear_prediction = self.use_pred.isChecked()
        c.prediction_strength = self.pred_strength.value() / 100.0
        c.target_priority = self.priority.currentText()
        c.shake_reduction = self.shake_red.isChecked()
        c.humanization = self.humanize.value() / 100.0
        c.max_distance = self.max_dist.value()
        c.fps_limit = self.fps_limit.value()
        c.stick_response = float(self.stick_resp.value())
        c.adaptive_stick = self.adaptive_stick.isChecked()

        c.engine = self.engine.currentText()
        c.confidence = self.conf.value() / 100.0
        c.draw_boxes = self.draw_boxes.isChecked()
        c.color_head_offset = self.color_off.value()
        r, g, b = parse_hex_color(self.enemy_color_edit.text())
        c.enemy_color = f"#{r:02X}{g:02X}{b:02X}"
        c.color_tolerance = float(self.color_tol.value())

        c.gated_aim = self.gated.isChecked()
        c.activation_button = self.act_btn.currentText()
        c.ads_key = self.ads_key.currentText().strip().lower()
        c.toggle_key = (self.toggle_key.text() or "f").strip().lower()[:1] or "f"
        c.controller_platform = self.ctrl_platform.currentText()

        c.show_preview = self.show_prev.isChecked()
        c.draw_zone = self.draw_zone_cb.isChecked()

        if self.worker:
            self.worker.cfg = c
            self.worker._sync_color_cfg()
            self.worker._sync_controller_cfg()

    # ── Engine ───────────────────────────────────────────────────────────
    def toggle_aim(self):
        if not self.worker or not self.worker.running:
            self.start_engine()
            QTimer.singleShot(400, lambda: self._set_aim(True))
            return
        self._set_aim(not self.worker.aim_enabled)

    def _set_aim(self, on: bool):
        if not self.worker:
            return
        self.worker.set_aim_enabled(on)
        tk = (self.cfg.toggle_key or "f").upper()
        if on:
            self.aim_btn.setText(f"AIM ON  ·  Hold activation  ·  {tk}")
            self.aim_btn.setProperty("on", "true")
        else:
            self.aim_btn.setText(f"AIM OFF  ·  {tk}")
            self.aim_btn.setProperty("on", "false")
        self.aim_btn.style().unpolish(self.aim_btn)
        self.aim_btn.style().polish(self.aim_btn)

    def start_engine(self):
        if self.worker and self.worker.running:
            return
        self.on_config_changed()
        self.cfg.save()
        self.worker = AimWorker(self.cfg)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_update.connect(self.on_status)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.live_badge.setText("● ENGINE ON")
        self._style_badge(self.live_badge, "on")
        self.sb.showMessage("Engine running — sticky lock armed")

    def stop_engine(self):
        if self.worker:
            self.worker.stop()
            self._set_aim(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.live_badge.setText("● ENGINE OFF")
        self._style_badge(self.live_badge, "off")
        self.lock_badge.setText("NO TARGET")
        self._style_badge(self.lock_badge, "idle")
        self.preview.setText("Engine stopped.")
        self.sb.showMessage("Engine stopped.")

    @Slot(object, dict)
    def on_frame(self, frame, stats: dict):
        locked = stats.get("locked")
        target = stats.get("target")
        sticky = stats.get("sticky")
        coasting = stats.get("coasting")
        if locked:
            self.lock_badge.setText("● ON POINT")
            self._style_badge(self.lock_badge, "on")
        elif coasting:
            self.lock_badge.setText("● COAST")
            self._style_badge(self.lock_badge, "coast")
        elif sticky or target:
            self.lock_badge.setText("● TRACKING")
            self._style_badge(self.lock_badge, "track")
        else:
            self.lock_badge.setText("NO TARGET")
            self._style_badge(self.lock_badge, "idle")

        tid = stats.get("track_id")
        tid_s = f"#{tid}" if tid is not None else "—"
        lock_s = "ON-POINT" if locked else ("COAST" if coasting else ("TRACK" if target else "—"))
        self.stats_lab.setText(
            f"FPS {stats.get('fps', 0)}  ·  "
            f"Dets {stats.get('dets', 0)}  ·  "
            f"{lock_s}  ·  "
            f"ID {tid_s}  ·  "
            f"Age {stats.get('age', 0)}  ·  "
            f"Act {'HELD' if stats.get('activation') else '—'}  ·  "
            f"Pad {'yes' if stats.get('pad') else 'no'}  ·  "
            f"{stats.get('res', '?')}  ·  "
            f"{stats.get('capture', '')}"
        )
        self.cap_status.setText(f"Capture: {stats.get('capture', '—')}")
        self.pad_status.setText(
            f"Physical pad: {'connected' if stats.get('pad') else 'not detected'}"
        )

        if not self.cfg.show_preview or frame is None:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            rgb = np.ascontiguousarray(rgb)
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            target_sz = self.preview.size()
            self.preview.setPixmap(
                pix.scaled(target_sz * 0.98, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        except Exception:
            pass

    @Slot(str)
    def on_status(self, msg: str):
        self.sb.showMessage(msg, 5000)

    # ── Hotkeys ──────────────────────────────────────────────────────────
    def _setup_hotkeys(self):
        def current_toggle():
            return (self.cfg.toggle_key or "f").lower()

        def current_ads():
            return (self.cfg.ads_key or "right").lower()

        def on_kb_press(key):
            try:
                ch = getattr(key, "char", None)
                if ch and ch.lower() == current_toggle():
                    QTimer.singleShot(0, self.toggle_aim)
                name = getattr(key, "name", None)
                ads = current_ads()
                if ads not in ("right", "left", "middle") and name and name.lower() == ads:
                    if self.worker:
                        QTimer.singleShot(0, lambda: self.worker.set_ads_held(True))
            except Exception:
                pass

        def on_kb_release(key):
            try:
                name = getattr(key, "name", None)
                ads = current_ads()
                if ads not in ("right", "left", "middle") and name and name.lower() == ads:
                    if self.worker:
                        QTimer.singleShot(0, lambda: self.worker.set_ads_held(False))
            except Exception:
                pass

        self.kb_listener = keyboard.Listener(on_press=on_kb_press, on_release=on_kb_release)
        self.kb_listener.daemon = True
        self.kb_listener.start()

        def on_mouse_click(x, y, button, pressed):
            try:
                btn = str(button).split(".")[-1].lower()
                if btn == current_ads():
                    if self.worker:
                        QTimer.singleShot(0, lambda p=pressed: self.worker.set_ads_held(p))
            except Exception:
                pass

        self.mouse_listener = mouse.Listener(on_click=on_mouse_click)
        self.mouse_listener.daemon = True
        self.mouse_listener.start()

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
        try:
            if hasattr(self, "kb_listener") and self.kb_listener:
                self.kb_listener.stop()
            if hasattr(self, "mouse_listener") and self.mouse_listener:
                self.mouse_listener.stop()
        except Exception:
            pass
        self.on_config_changed()
        self.cfg.save()
        e.accept()
