"""
ScriptedElite AI - PS5 Optimization
Chiaki capture → YOLO detection → virtual controller tracking.
"""
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPainter, QPen, QLinearGradient, QBrush
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QSpinBox,
    QFormLayout, QStatusBar, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QLineEdit, QFrame, QGridLayout, QSizePolicy, QScrollArea, QButtonGroup,
    QRadioButton, QToolTip, QSplitter, QMessageBox, QColorDialog,
)

from core.config import AppConfig
from core.capture import ScreenCapture
from core.detector import YOLODetector
from core.color_detector import ColorSignatureDetector, parse_hex_color
from core.predictor import LinearPredictor
from core.controller import AimController
from core.tracker import StickyTargetTracker, StickyColorTracker
from core.recoil import RecoilMatrix

from pynput import keyboard, mouse

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
LOGO_PATH = ASSETS_DIR / "logo.png"

# ─── Scripted Elite brand palette (matches logo: electric blue / silver) ───
BG          = "#070b14"
BG_PANEL    = "#0c1220"
BG_CARD     = "#10182a"
BORDER      = "#1a2a48"
BORDER_HOT  = "#00b4ff"
ACCENT      = "#00c8ff"
ACCENT_DIM  = "#0077aa"
TEXT        = "#e8eef8"
TEXT_MUTED  = "#7a879e"
GREEN       = "#22c55e"
RED         = "#ef4444"
ORANGE      = "#f59e0b"
PURPLE      = "#a855f7"


def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    app.setStyleSheet(f"""
    QMainWindow, QWidget {{
        background: {BG};
        color: {TEXT};
        font-family: "Segoe UI", "Inter", Arial;
        font-size: 13px;
    }}
    QToolTip {{
        background: {BG_CARD};
        color: {TEXT};
        border: 1px solid {BORDER_HOT};
        padding: 6px 10px;
        border-radius: 4px;
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
        padding: 10px 22px;
        margin-right: 3px;
        font-weight: 600;
        min-width: 90px;
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
        left: 14px;
        padding: 0 8px;
        color: {ACCENT};
        font-size: 12px;
        letter-spacing: 0.5px;
    }}
    QLabel#sectionHint {{
        color: {TEXT_MUTED};
        font-size: 11px;
        font-weight: 400;
    }}
    QLabel#valueBadge {{
        background: #0a1a2e;
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 2px 10px;
        color: {ACCENT};
        font-weight: 700;
        font-size: 12px;
        min-width: 48px;
    }}
    QLabel#statusOk {{ color: {GREEN}; font-weight: 600; }}
    QLabel#statusWarn {{ color: {ORANGE}; font-weight: 600; }}
    QLabel#statusErr {{ color: {RED}; font-weight: 600; }}

    QPushButton {{
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #162038, stop:1 #0f1728);
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 9px 18px;
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
        font-size: 15px;
        padding: 14px 24px;
        font-weight: 800;
        letter-spacing: 0.5px;
        border-radius: 10px;
    }}
    QPushButton#aimToggle[on="true"] {{
        background: #0a2e1a;
        border: 2px solid {GREEN};
        color: {GREEN};
    }}
    QPushButton#aimToggle[on="false"] {{
        background: #2a1010;
        border: 2px solid {RED};
        color: {RED};
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
        width: 16px;
        height: 16px;
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
        border-radius: 6px;
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
        width: 18px; height: 18px;
        border-radius: 4px;
        border: 1px solid {BORDER};
        background: #0a1220;
    }}
    QCheckBox::indicator:checked {{
        background: {ACCENT};
        border-color: {ACCENT};
    }}
    QRadioButton {{ spacing: 8px; color: {TEXT}; }}
    QRadioButton::indicator {{
        width: 16px; height: 16px;
        border-radius: 8px;
        border: 1px solid {BORDER};
        background: #0a1220;
    }}
    QRadioButton::indicator:checked {{
        background: {ACCENT};
        border-color: {ACCENT};
    }}

    QScrollArea {{ border: none; background: transparent; }}
    QStatusBar {{
        background: #050810;
        color: {TEXT_MUTED};
        border-top: 1px solid {BORDER};
    }}
    QTableWidget {{
        background: #0a1220;
        border: 1px solid {BORDER};
        border-radius: 6px;
        gridline-color: {BORDER};
    }}
    QHeaderView::section {{
        background: {BG_CARD};
        color: {ACCENT};
        border: 1px solid {BORDER};
        padding: 6px;
        font-weight: 700;
    }}
    QFrame#divider {{
        background: {BORDER};
        max-height: 1px;
        min-height: 1px;
    }}
    QFrame#previewFrame {{
        background: #050810;
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}
    """)


# ═══════════════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════════════
class AimWorker(QObject):
    frame_ready = Signal(object, dict)   # np.ndarray, stats
    status_update = Signal(str)

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self.capture: ScreenCapture | None = None
        self.yolo: YOLODetector | None = None
        self.color_det = ColorSignatureDetector()
        self.color_det.set_enemy_color(getattr(cfg, "enemy_color", "#FF00FA"))
        self.color_det.set_tolerance(getattr(cfg, "color_tolerance", 55.0))
        self.predictor = LinearPredictor(history_len=6, prediction_ms=cfg.prediction_ms)
        self.controller = AimController()
        # Strong sticky lock — once acquired, hold the same player hard
        self.tracker = StickyTargetTracker(
            max_miss_frames=22,
            match_iou=0.05,
            match_center_px=220.0,
            aim_smooth_far=0.35,
            aim_smooth_near=0.82,
            near_px=50.0,
            hold_expand=1.85,
        )
        self.color_tracker = StickyColorTracker(max_miss_frames=18, stick_radius=200.0)
        self.running = False
        self.aim_enabled = False
        self.ads_held = False          # keyboard / mouse gate
        self._last_fps_t = time.time()
        self._frame_count = 0
        self.stats = {
            "dets": 0, "fps": 0, "engine": cfg.engine,
            "gated": False, "target": False, "capture": "",
            "pad": False, "activation": False, "locked": False, "sticky": False,
        }

    def _sync_color_cfg(self):
        self.color_det.set_enemy_color(getattr(self.cfg, "enemy_color", "#FF00FA"))
        self.color_det.set_tolerance(getattr(self.cfg, "color_tolerance", 55.0))

    @Slot(bool)
    def set_aim_enabled(self, enabled: bool):
        self.aim_enabled = enabled
        self.controller.set_enabled(enabled)
        if not enabled:
            self.controller.reset()
            self.predictor.reset()
            self.tracker.reset()
            self.color_tracker.reset()

    @Slot(bool)
    def set_ads_held(self, held: bool):
        self.ads_held = held

    def _activation_active(self) -> bool:
        """True when user is holding the configured activation input."""
        if not self.cfg.gated_aim:
            return True

        # Physical controller trigger / button (primary for PS5 / Chiaki)
        btn = (self.cfg.activation_button or "").strip().lower()
        if btn and btn not in ("none", "off", "-"):
            if self.controller.is_activation_held(btn):
                return True

        # Keyboard / mouse fallback
        if self.ads_held:
            return True
        return False

    def run(self):
        self.running = True
        try:
            self.capture = ScreenCapture(
                mode=self.cfg.capture_mode,
                window_title=self.cfg.chiaki_window or "Chiaki",
                capture_device=self.cfg.capture_device,
                region_size=getattr(self.cfg, "region_size", 0) or 0,
            )
            if self.cfg.engine in ("yolo", "hybrid"):
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
                self.yolo = YOLODetector(str(MODEL_PATH), use_gpu=self.cfg.use_gpu)
            self.status_update.emit("Engine started")
        except Exception as e:
            self.status_update.emit(f"Init failed: {e}")
            self.running = False
            return

        while self.running:
            t0 = time.time()
            try:
                # Hot-reload capture settings that matter mid-run
                if self.capture:
                    if self.capture.mode != self.cfg.capture_mode:
                        self.capture.set_mode(
                            self.cfg.capture_mode,
                            window_title=self.cfg.chiaki_window,
                            device=self.cfg.capture_device,
                        )
                    elif self.cfg.capture_mode == "chiaki":
                        self.capture.window_title = self.cfg.chiaki_window or "Chiaki"

                frame, _left, _top = self.capture.grab_region()
                if frame is None or frame.size == 0:
                    time.sleep(0.005)
                    continue

                fh, fw = frame.shape[:2]
                # ★ Frame-local crosshair (THIS is the correct center)
                screen_cx = fw / 2.0 + self.cfg.cal_x
                screen_cy = fh / 2.0 + self.cfg.cal_y

                best = None
                engine_used = self.cfg.engine
                offset = self.cfg.offset_for_part()

                # Keep enemy color in sync every frame (cheap)
                self._sync_color_cfg()

                # ── YOLO ──
                dets = []
                zone_r = max(10, int(self.cfg.zone_radius))
                max_dist = min(self.cfg.max_distance, zone_r)

                if self.cfg.engine in ("yolo", "hybrid") and self.yolo:
                    dets = self.yolo.detect(frame, self.cfg.confidence)

                    # HARD STICKY: same player until truly gone
                    best = self.tracker.update(
                        dets,
                        (screen_cx, screen_cy),
                        max_dist,
                        offset,
                        priority=self.cfg.target_priority,
                    )

                    if self.cfg.draw_boxes:
                        vis = self.yolo.draw_detections(frame, dets, best, offset)
                    else:
                        vis = frame.copy()

                    if best and self.cfg.use_linear_prediction and not best.get("coasting"):
                        self.predictor.update(best["aim_x"], best["aim_y"])
                        pred = self.predictor.predict(self.cfg.prediction_strength)
                        if pred:
                            px, py = pred
                            best["aim_x"] = best["aim_x"] * 0.70 + px * 0.30
                            best["aim_y"] = best["aim_y"] * 0.70 + py * 0.30
                            best["dx"] = best["aim_x"] - screen_cx
                            best["dy"] = best["aim_y"] - screen_cy
                            best["dist"] = (best["dx"] ** 2 + best["dy"] ** 2) ** 0.5
                            cv2.circle(vis, (int(px), int(py)), 4, (255, 180, 0), -1)
                    elif not best:
                        self.predictor.reset()
                else:
                    vis = frame.copy()
                    if self.tracker.active:
                        self.tracker.reset()

                # ── Color path: enemy-color only + sticky blob lock ──
                if self.cfg.engine == "color" or (self.cfg.engine == "hybrid" and best is None):
                    hits = self.color_det.detect_all(frame, self.cfg.color_head_offset)
                    ch = self.color_tracker.update(
                        hits, (screen_cx, screen_cy), max_dist, self.cfg.color_head_offset,
                    )
                    if ch:
                        best = ch
                        vis = self.color_det.draw(frame if self.cfg.engine == "color" else vis, ch)
                        engine_used = "color"
                    elif self.cfg.engine == "color":
                        # Still draw mode hint
                        pass
                elif self.cfg.engine != "color" and self.color_tracker.active:
                    # Leaving color mode mid-run
                    self.color_tracker.reset()

                # ── Draw detection zone circle + center X ──
                if self.cfg.draw_zone:
                    cz = (int(round(screen_cx)), int(round(screen_cy)))
                    cv2.circle(vis, cz, zone_r, (0, 255, 80), 2, cv2.LINE_AA)
                    arm = 12
                    green = (0, 255, 80)
                    cv2.line(vis, (cz[0] - arm, cz[1]), (cz[0] + arm, cz[1]), green, 2, cv2.LINE_AA)
                    cv2.line(vis, (cz[0], cz[1] - arm), (cz[0], cz[1] + arm), green, 2, cv2.LINE_AA)
                    cv2.circle(vis, cz, 3, green, -1, cv2.LINE_AA)

                # Aim line + sticky marker
                if best:
                    color = (0, 200, 255) if not best.get("coasting") else (180, 180, 80)
                    cv2.line(
                        vis,
                        (int(screen_cx), int(screen_cy)),
                        (int(best["aim_x"]), int(best["aim_y"])),
                        color, 1, cv2.LINE_AA,
                    )
                    # Smoothed aim point (what we actually track)
                    cv2.circle(vis, (int(best["aim_x"]), int(best["aim_y"])), 6, (0, 0, 255), -1, cv2.LINE_AA)
                    if best.get("sticky"):
                        cv2.putText(
                            vis, "LOCK" if self.controller.is_locked else "TRACK",
                            (int(best["aim_x"]) + 10, int(best["aim_y"]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA,
                        )

                # ── Activation + tracking ──
                activation = self._activation_active()
                should_aim = self.aim_enabled and activation

                if should_aim and best:
                    smooth = self.cfg.smoothing
                    if self.cfg.shake_reduction:
                        smooth = min(0.85, smooth + 0.10)
                    self.controller.move_to_target(
                        best["dx"], best["dy"],
                        smoothing=smooth,
                        strength=self.cfg.strength,
                        speed=self.cfg.acquisition_speed / 3.0,
                        humanization=self.cfg.humanization,
                    )
                else:
                    # Release stick when not actively aiming, but KEEP sticky
                    # target identity while activation is only briefly released.
                    self.controller.reset()
                    if not should_aim and not activation:
                        # Full clear only when activation fully released and no target
                        if not best:
                            self.tracker.reset()
                            self.color_tracker.reset()
                            self.predictor.reset()
                    elif not best:
                        self.predictor.reset()

                # FPS
                self._frame_count += 1
                now = time.time()
                if now - self._last_fps_t >= 1.0:
                    fps = self._frame_count
                    self._frame_count = 0
                    self._last_fps_t = now
                else:
                    fps = self.stats.get("fps", 0)

                self.stats = {
                    "dets": len(dets) if dets else (1 if best else 0),
                    "fps": int(fps),
                    "engine": engine_used,
                    "gated": should_aim,
                    "target": best is not None,
                    "capture": self.capture.get_status() if self.capture else "",
                    "pad": self.controller.physical_connected(),
                    "activation": activation,
                    "locked": self.controller.is_locked,
                    "sticky": bool(best and best.get("sticky")),
                    "res": f"{fw}x{fh}",
                }
                self.frame_ready.emit(vis, dict(self.stats))

            except Exception as e:
                self.status_update.emit(f"Error: {str(e)[:90]}")
                time.sleep(0.03)

            elapsed = time.time() - t0
            target = 1.0 / max(30, min(int(self.cfg.fps_limit or 120), 240))
            if elapsed < target:
                time.sleep(target - elapsed)

        if self.capture:
            self.capture.close()
        self.controller.reset()
        self.status_update.emit("Engine stopped")

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════════
# UI helpers
# ═══════════════════════════════════════════════════════════════════════════
def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("sectionHint")
    lab.setWordWrap(True)
    return lab


def _badge(initial: str = "—") -> QLabel:
    lab = QLabel(initial)
    lab.setObjectName("valueBadge")
    lab.setAlignment(Qt.AlignCenter)
    return lab


def _slider_row(title: str, tip: str, slider: QSlider, badge: QLabel) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 4, 0, 4)
    lay.setSpacing(4)
    top = QHBoxLayout()
    t = QLabel(title)
    t.setStyleSheet("font-weight: 700;")
    t.setToolTip(tip)
    top.addWidget(t)
    top.addStretch()
    top.addWidget(badge)
    lay.addLayout(top)
    slider.setToolTip(tip)
    lay.addWidget(slider)
    h = _hint(tip)
    lay.addWidget(h)
    return w


# ═══════════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScriptedElite AI  •  Precision Control Execution")
        self.resize(1280, 860)
        self.setMinimumSize(980, 700)

        if LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(LOGO_PATH)))

        self.cfg = AppConfig.load()
        self.worker: AimWorker | None = None
        self.worker_thread: threading.Thread | None = None
        self.recoil = RecoilMatrix()
        self._syncing = False

        self._build_ui()
        self._load_cfg_to_ui()
        self._wire_signals()
        self._setup_hotkeys()

    # ── Header ────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 8)
        root.setSpacing(10)

        # Header bar
        header = QHBoxLayout()
        header.setSpacing(12)

        logo_lab = QLabel()
        if LOGO_PATH.exists():
            pix = QPixmap(str(LOGO_PATH)).scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_lab.setPixmap(pix)
        logo_lab.setFixedSize(56, 56)
        header.addWidget(logo_lab)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        t1 = QLabel("SCRIPTED ELITE")
        t1.setStyleSheet(f"font-size: 20px; font-weight: 900; color: {ACCENT}; letter-spacing: 2px;")
        t2 = QLabel("PS5 AI  ·  Chiaki Capture  ·  YOLO Tracking")
        t2.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")
        titles.addWidget(t1)
        titles.addWidget(t2)
        header.addLayout(titles)
        header.addStretch()

        self.live_badge = QLabel("● ENGINE OFF")
        self.live_badge.setStyleSheet(
            f"background:#1a1010; color:{RED}; border:1px solid {RED}; "
            f"border-radius:8px; padding:8px 14px; font-weight:800;"
        )
        header.addWidget(self.live_badge)
        root.addLayout(header)

        # Master aim toggle
        self.aim_btn = QPushButton("AIM DISABLED  —  Press F or click to enable")
        self.aim_btn.setObjectName("aimToggle")
        self.aim_btn.setProperty("on", "false")
        self.aim_btn.setCursor(Qt.PointingHandCursor)
        self.aim_btn.clicked.connect(self.toggle_aim)
        self.aim_btn.setToolTip(
            "Master switch. When ON and you hold the activation button "
            "(default L2 / right mouse), tracking follows the detected target."
        )
        root.addWidget(self.aim_btn)

        # Body: tabs + preview side by side
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # Left: tabs
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_capture(), "  Capture  ")
        self.tabs.addTab(self._tab_aim(), "  Aim  ")
        self.tabs.addTab(self._tab_model(), "  Model  ")
        self.tabs.addTab(self._tab_activation(), "  Activation  ")
        self.tabs.addTab(self._tab_cronus(), "  Cronus  ")
        left_l.addWidget(self.tabs)

        # Engine controls
        eng = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Engine")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_engine)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_engine)
        eng.addWidget(self.start_btn)
        eng.addWidget(self.stop_btn)
        left_l.addLayout(eng)

        split.addWidget(left)

        # Right: live preview
        right = QFrame()
        right.setObjectName("previewFrame")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 10, 10, 10)

        ph = QHBoxLayout()
        pt = QLabel("LIVE FEED")
        pt.setStyleSheet(f"font-weight: 800; color: {ACCENT}; letter-spacing: 1px;")
        ph.addWidget(pt)
        ph.addStretch()
        self.show_prev = QCheckBox("Show Preview")
        self.show_prev.setChecked(True)
        ph.addWidget(self.show_prev)
        self.draw_zone_cb = QCheckBox("Zone + X")
        self.draw_zone_cb.setChecked(True)
        self.draw_zone_cb.setToolTip("Draw the detection circle and center crosshair on the feed.")
        ph.addWidget(self.draw_zone_cb)
        rl.addLayout(ph)

        self.preview = QLabel("Start the engine to capture Chiaki / desktop and run detection.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(420, 320)
        self.preview.setStyleSheet(f"color: {TEXT_MUTED}; background: #050810; border-radius: 8px;")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rl.addWidget(self.preview, 1)

        self.stats_lab = QLabel("FPS: —   Detections: —   Target: —   Activation: —")
        self.stats_lab.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        rl.addWidget(self.stats_lab)

        split.addWidget(right)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 4)
        root.addWidget(split, 1)

        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        self.sb.showMessage("ScriptedElite AI ready  ·  Configure Capture → Start Engine → Enable Aim → Hold activation")

    # ── Capture tab ───────────────────────────────────────────────────────
    def _tab_capture(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)

        g = QGroupBox("SCREEN CAPTURE")
        gl = QFormLayout(g)
        gl.setSpacing(10)

        self.cap_mode = QComboBox()
        self.cap_mode.addItems(["chiaki", "capture_card", "desktop"])
        self.cap_mode.setToolTip(
            "chiaki — capture the Chiaki Remote Play window (recommended for PS5)\n"
            "capture_card — HDMI capture device\n"
            "desktop — primary monitor center / full screen"
        )
        gl.addRow("Source", self.cap_mode)

        self.chiaki_win = QLineEdit("Chiaki")
        self.chiaki_win.setPlaceholderText("Window title contains… (e.g. Chiaki)")
        self.chiaki_win.setToolTip(
            "Partial match for the Chiaki window title. Leave as 'Chiaki' unless you renamed it."
        )
        gl.addRow("Chiaki Window", self.chiaki_win)

        self.cap_dev = QSpinBox()
        self.cap_dev.setRange(0, 10)
        self.cap_dev.setToolTip("Capture card device index (usually 0).")
        gl.addRow("Capture Card #", self.cap_dev)

        self.region_size = QSpinBox()
        self.region_size.setRange(0, 2160)
        self.region_size.setSingleStep(40)
        self.region_size.setSpecialValueText("Full window")
        self.region_size.setToolTip(
            "0 = capture the entire Chiaki window (recommended).\n"
            "Set a value (e.g. 720) to center-crop a square FOV region."
        )
        gl.addRow("Region Size (0=full)", self.region_size)

        self.use_gpu = QCheckBox("GPU Acceleration (CUDA)")
        self.use_gpu.setToolTip("Use NVIDIA CUDA for YOLO if available. Falls back to CPU automatically.")
        gl.addRow(self.use_gpu)

        lay.addWidget(g)

        g2 = QGroupBox("STATUS")
        g2l = QVBoxLayout(g2)
        self.cap_status = QLabel("Capture: idle")
        self.cap_status.setObjectName("sectionHint")
        self.pad_status = QLabel("Physical pad: —")
        self.pad_status.setObjectName("sectionHint")
        g2l.addWidget(self.cap_status)
        g2l.addWidget(self.pad_status)
        g2l.addWidget(_hint(
            "1. Open Chiaki and connect to your PS5 (fullscreen recommended).\n"
            "2. Set Source to 'chiaki' and confirm the window name.\n"
            "3. Start Engine — the green circle + X should sit on your crosshair."
        ))
        lay.addWidget(g2)

        g3 = QGroupBox("CROSSHAIR CALIBRATION")
        g3l = QFormLayout(g3)
        self.cal_x = QSpinBox(); self.cal_x.setRange(-200, 200)
        self.cal_y = QSpinBox(); self.cal_y.setRange(-200, 200)
        self.cal_x.setToolTip("Shift the detection center horizontally if the X is slightly off your crosshair.")
        self.cal_y.setToolTip("Shift the detection center vertically if the X is slightly off your crosshair.")
        g3l.addRow("Calibrate X (px)", self.cal_x)
        g3l.addRow("Calibrate Y (px)", self.cal_y)
        g3l.addRow(_hint("Nudge until the green X lines up with your in-game crosshair."))
        lay.addWidget(g3)
        lay.addStretch()
        return w

    # ── Aim tab ───────────────────────────────────────────────────────────
    def _tab_aim(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setSpacing(8)

        # FOV / Zone
        top = QHBoxLayout()

        g_fov = QGroupBox("FOV  ·  DETECTION ZONE")
        fl = QVBoxLayout(g_fov)
        self.zone_radius = QSlider(Qt.Horizontal)
        self.zone_radius.setRange(40, 600)
        self.zone_badge = _badge("200")
        fl.addWidget(_slider_row(
            "Zone Radius (px)",
            "How far from the crosshair a target may be before tracking engages. "
            "Matches the green circle on the live feed.",
            self.zone_radius, self.zone_badge,
        ))
        top.addWidget(g_fov, 1)

        g_zone = QGroupBox("ZONE  ·  AIM POINT")
        zl = QVBoxLayout(g_zone)
        zl.addWidget(_hint("Where on the player box to aim."))
        self.zone_head = QRadioButton("HEAD")
        self.zone_chest = QRadioButton("CHEST")
        self.zone_body = QRadioButton("BODY")
        self.zone_head.setChecked(True)
        self.zone_group = QButtonGroup(self)
        for b in (self.zone_head, self.zone_chest, self.zone_body):
            self.zone_group.addButton(b)
            zl.addWidget(b)
        top.addWidget(g_zone)
        lay.addLayout(top)

        # Tracking core
        g_track = QGroupBox("TRACKING")
        tl = QVBoxLayout(g_track)

        self.smooth = QSlider(Qt.Horizontal)
        self.smooth.setRange(0, 90)
        self.smooth_badge = _badge("35%")
        tl.addWidget(_slider_row(
            "Tracking Smoothness",
            "Higher = smoother, less snappy motion. Lower = faster snaps to target. "
            "Recommended 25–50% for natural look.",
            self.smooth, self.smooth_badge,
        ))

        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(10, 250)
        self.strength_badge = _badge("100%")
        tl.addWidget(_slider_row(
            "Tracking Strength",
            "Overall aim force / gain. Raise if tracking feels weak; lower if it overshoots.",
            self.strength, self.strength_badge,
        ))

        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 8.0)
        self.speed.setSingleStep(0.1)
        self.speed.setDecimals(1)
        self.speed.setToolTip(
            "Tracking speed (acquisition). How quickly the stick moves toward the target. "
            "3.0–4.5 is a solid starting range."
        )
        row_s = QHBoxLayout()
        row_s.addWidget(QLabel("Tracking Speed"))
        row_s.addStretch()
        row_s.addWidget(self.speed)
        tl.addLayout(row_s)
        tl.addWidget(_hint("How quickly aim closes the gap to the target. Independent of smoothness."))

        lay.addWidget(g_track)

        # Prediction / priority
        g_pred = QGroupBox("PREDICTION & PRIORITY")
        pl = QFormLayout(g_pred)
        self.use_pred = QCheckBox("Linear Prediction")
        self.use_pred.setToolTip("Lead moving targets based on recent velocity.")
        pl.addRow(self.use_pred)
        self.pred_strength = QSlider(Qt.Horizontal)
        self.pred_strength.setRange(0, 100)
        self.pred_badge = _badge("50%")
        pl.addRow("Predict Strength", self._mini_slider(self.pred_strength, self.pred_badge))
        self.priority = QComboBox()
        self.priority.addItems(["closest", "highest_conf"])
        self.priority.setToolTip("closest = nearest to crosshair; highest_conf = most confident YOLO box.")
        pl.addRow("Target Priority", self.priority)
        self.shake_red = QCheckBox("Shake Reduction")
        self.shake_red.setToolTip("Adds a touch of extra smoothing to reduce micro-jitter.")
        self.shake_red.setChecked(True)
        pl.addRow(self.shake_red)
        self.humanize = QSlider(Qt.Horizontal)
        self.humanize.setRange(0, 50)
        self.human_badge = _badge("0")
        pl.addRow("Humanization", self._mini_slider(self.humanize, self.human_badge))
        lay.addWidget(g_pred)

        g_lim = QGroupBox("LIMITS")
        ll = QFormLayout(g_lim)
        self.max_dist = QSpinBox(); self.max_dist.setRange(50, 1200)
        self.max_dist.setToolTip("Hard max distance (px) for selecting a target (also limited by FOV zone).")
        self.fps_limit = QSpinBox(); self.fps_limit.setRange(30, 240)
        self.fps_limit.setToolTip("Cap processing rate to save CPU/GPU.")
        ll.addRow("Max Distance (px)", self.max_dist)
        ll.addRow("FPS Limit", self.fps_limit)
        lay.addWidget(g_lim)

        lay.addStretch()
        scroll.setWidget(host)
        return scroll

    def _mini_slider(self, slider: QSlider, badge: QLabel) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(badge)
        return w

    # ── Model tab ─────────────────────────────────────────────────────────
    def _tab_model(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        g = QGroupBox("DETECTION MODE")
        gl = QVBoxLayout(g)
        gl.addWidget(_hint(
            "YOLO — neural person detection. Sticky-locks the first valid player in zone.\n"
            "Color — tracks ONLY your configured enemy color (ignores other team colors).\n"
            "Hybrid — YOLO sticky first; falls back to enemy-color sticky if no person box."
        ))
        self.engine = QComboBox()
        self.engine.addItems(["yolo", "color", "hybrid"])
        gl.addWidget(self.engine)
        lay.addWidget(g)

        g2 = QGroupBox("YOLO SETTINGS")
        g2l = QFormLayout(g2)
        self.conf = QSlider(Qt.Horizontal)
        self.conf.setRange(15, 85)
        self.conf_badge = _badge("42%")
        g2l.addRow("Confidence", self._mini_slider(self.conf, self.conf_badge))
        g2l.addRow(_hint("Higher = fewer false detections, may miss distant targets. 35–55% typical."))
        self.draw_boxes = QCheckBox("Draw bounding boxes on preview")
        self.draw_boxes.setChecked(True)
        g2l.addRow(self.draw_boxes)
        lay.addWidget(g2)

        g3 = QGroupBox("ENEMY COLOR  ·  COLOR / HYBRID MODE")
        g3l = QFormLayout(g3)
        g3l.addRow(_hint(
            "Set the exact enemy highlight / nameplate / outline color so teammates "
            "are ignored. Use the picker or paste a hex code (e.g. #FF00FA)."
        ))

        # Color swatch + picker + hex
        color_row = QHBoxLayout()
        self.color_swatch = QLabel()
        self.color_swatch.setFixedSize(36, 28)
        self.color_swatch.setStyleSheet(
            "background:#FF00FA; border:1px solid #2a3a5c; border-radius:4px;"
        )
        color_row.addWidget(self.color_swatch)

        self.enemy_color_edit = QLineEdit("#FF00FA")
        self.enemy_color_edit.setPlaceholderText("#RRGGBB")
        self.enemy_color_edit.setMaxLength(16)
        self.enemy_color_edit.setToolTip("Enemy color as #RRGGBB or R,G,B")
        color_row.addWidget(self.enemy_color_edit, 1)

        self.pick_color_btn = QPushButton("Pick Color…")
        self.pick_color_btn.setCursor(Qt.PointingHandCursor)
        self.pick_color_btn.setToolTip("Open a color picker and set the enemy color.")
        self.pick_color_btn.clicked.connect(self._pick_enemy_color)
        color_row.addWidget(self.pick_color_btn)
        g3l.addRow("Enemy Color", color_row)

        # Presets
        preset_row = QHBoxLayout()
        for label, hx in [
            ("Magenta", "#FF00FA"),
            ("Red", "#FF2020"),
            ("Orange", "#FF8800"),
            ("Green", "#00FF00"),
            ("Yellow", "#FFE600"),
            ("Cyan", "#00E5FF"),
        ]:
            b = QPushButton(label)
            b.setToolTip(hx)
            b.clicked.connect(lambda checked=False, h=hx: self._set_enemy_color_hex(h))
            preset_row.addWidget(b)
        g3l.addRow("Presets", preset_row)

        self.color_tol = QSlider(Qt.Horizontal)
        self.color_tol.setRange(10, 100)
        self.color_tol.setValue(55)
        self.color_tol_badge = _badge("55")
        self.color_tol.setToolTip(
            "How loosely to match the enemy color under different lighting. "
            "Lower = stricter (fewer false locks). Higher = more forgiving."
        )
        g3l.addRow("Color Tolerance", self._mini_slider(self.color_tol, self.color_tol_badge))

        self.color_off = QDoubleSpinBox()
        self.color_off.setRange(0, 200)
        self.color_off.setSuffix(" px")
        self.color_off.setToolTip("Pixels downward from the color tag center to estimate head/chest.")
        g3l.addRow("Head Offset Y", self.color_off)
        g3l.addRow(_hint(
            "Color mode sticky-locks the first matching enemy blob and stays on it.\n"
            "Teammate colors that do not match this setting are ignored."
        ))
        lay.addWidget(g3)
        lay.addStretch()
        return w

    def _set_enemy_color_hex(self, hx: str):
        self.enemy_color_edit.setText(hx.upper() if hx.startswith("#") else f"#{hx.upper()}")
        self._update_color_swatch()
        self.on_config_changed()

    def _pick_enemy_color(self):
        r, g, b = parse_hex_color(self.enemy_color_edit.text())
        initial = QColor(r, g, b)
        col = QColorDialog.getColor(initial, self, "Select Enemy Color")
        if col.isValid():
            hx = col.name().upper()  # #RRGGBB
            self.enemy_color_edit.setText(hx)
            self._update_color_swatch()
            self.on_config_changed()

    def _update_color_swatch(self):
        r, g, b = parse_hex_color(self.enemy_color_edit.text())
        self.color_swatch.setStyleSheet(
            f"background:rgb({r},{g},{b}); border:1px solid #2a3a5c; border-radius:4px;"
        )

    # ── Activation tab ────────────────────────────────────────────────────
    def _tab_activation(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        g = QGroupBox("ACTIVATION  ·  WHEN TRACKING RUNS")
        gl = QFormLayout(g)
        gl.setSpacing(12)

        self.gated = QCheckBox("Require activation button (recommended)")
        self.gated.setChecked(True)
        self.gated.setToolTip(
            "When enabled, tracking only moves the stick while you hold the activation button. "
            "Disable only for testing."
        )
        gl.addRow(self.gated)

        self.act_btn = QComboBox()
        self.act_btn.addItems([
            "l2", "r2", "lt", "rt",
            "lb", "rb", "l1", "r1",
            "a", "b", "x", "y",
            "ls", "rs", "none",
        ])
        self.act_btn.setToolTip(
            "Physical controller button that enables tracking while held.\n"
            "PlayStation: L2 (aim) / R2 (fire) are typical.\n"
            "Xbox aliases: LT = L2, RT = R2, LB = L1, RB = R1.\n"
            "Requires the pad to be visible to Windows (XInput)."
        )
        gl.addRow("Controller Button", self.act_btn)

        self.ads_key = QComboBox()
        self.ads_key.setEditable(True)
        self.ads_key.addItems(["right", "left", "middle", "ctrl", "shift", "alt", "c", "v", "x"])
        self.ads_key.setToolTip(
            "Keyboard / mouse fallback for activation.\n"
            "right / left / middle = mouse buttons.\n"
            "Or a key name (ctrl, shift, c, …)."
        )
        gl.addRow("Keyboard / Mouse", self.ads_key)

        self.toggle_key = QLineEdit("f")
        self.toggle_key.setMaxLength(1)
        self.toggle_key.setToolTip("Key that toggles master AIM on/off.")
        gl.addRow("Master Toggle Key", self.toggle_key)

        plat = QComboBox()
        plat.addItems(["playstation", "xbox"])
        self.ctrl_platform = plat
        gl.addRow("Controller Labels", self.ctrl_platform)

        lay.addWidget(g)

        info = QGroupBox("HOW ACTIVATION WORKS")
        il = QVBoxLayout(info)
        il.addWidget(_hint(
            "1. Click Start Engine so capture + YOLO are running.\n"
            "2. Enable AIM (button above or hotkey F).\n"
            "3. Hold your activation button (default L2 / right mouse).\n"
            "4. While held, the system tracks the closest player inside the green zone\n"
            "   and moves the virtual right stick to follow them.\n"
            "5. Release the button → stick centers, tracking stops.\n\n"
            "Chiaki must receive the virtual Xbox 360 pad from ViGEm. Install ViGEmBus\n"
            "if the controller does not appear. Your physical pad is still read for L2/R2."
        ))
        lay.addWidget(info)
        lay.addStretch()
        return w

    # ── Cronus tab ────────────────────────────────────────────────────────
    def _tab_cronus(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("RECOIL PROFILES (Cronus / ZPU notes)")
        gl = QVBoxLayout(g)
        self.wep = QComboBox()
        self.wep.addItems(self.recoil.list_weapons() or ["AR-15", "SMG"])
        gl.addWidget(QLabel("Weapon"))
        gl.addWidget(self.wep)
        self.rec_table = QTableWidget(8, 2)
        self.rec_table.setHorizontalHeaderLabels(["Vertical", "Horizontal"])
        gl.addWidget(self.rec_table)
        saveb = QPushButton("Save Recoil Profile")
        saveb.clicked.connect(self.save_recoil)
        gl.addWidget(saveb)
        gl.addWidget(_hint("Reference tables for Cronus Zen scripts. Not applied to live stick output yet."))
        lay.addWidget(g)
        lay.addStretch()
        return w

    # ── Config sync ───────────────────────────────────────────────────────
    def _load_cfg_to_ui(self):
        self._syncing = True
        c = self.cfg
        self.cap_mode.setCurrentText(c.capture_mode)
        self.chiaki_win.setText(c.chiaki_window or "Chiaki")
        self.cap_dev.setValue(c.capture_device)
        self.region_size.setValue(getattr(c, "region_size", 0) or 0)
        self.use_gpu.setChecked(getattr(c, "use_gpu", True))
        self.cal_x.setValue(c.cal_x)
        self.cal_y.setValue(c.cal_y)

        self.zone_radius.setValue(c.zone_radius)
        self.zone_badge.setText(str(c.zone_radius))
        part = (c.target_part or "head").lower()
        {"head": self.zone_head, "chest": self.zone_chest, "body": self.zone_body}.get(
            part, self.zone_head
        ).setChecked(True)

        self.smooth.setValue(int(c.smoothing * 100))
        self.smooth_badge.setText(f"{int(c.smoothing * 100)}%")
        self.strength.setValue(int(c.strength * 100))
        self.strength_badge.setText(f"{int(c.strength * 100)}%")
        self.speed.setValue(c.acquisition_speed)
        self.use_pred.setChecked(c.use_linear_prediction)
        self.pred_strength.setValue(int(getattr(c, "prediction_strength", 0.5) * 100))
        self.pred_badge.setText(f"{self.pred_strength.value()}%")
        self.priority.setCurrentText(getattr(c, "target_priority", "closest"))
        self.shake_red.setChecked(getattr(c, "shake_reduction", True))
        self.humanize.setValue(int(c.humanization * 100))
        self.human_badge.setText(str(self.humanize.value()))
        self.max_dist.setValue(c.max_distance)
        self.fps_limit.setValue(c.fps_limit)

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
        idx = self.act_btn.findText((c.activation_button or "l2").lower())
        self.act_btn.setCurrentIndex(max(0, idx))
        self.ads_key.setCurrentText(c.ads_key or "right")
        self.toggle_key.setText((c.toggle_key or "f")[:1])
        self.ctrl_platform.setCurrentText(getattr(c, "controller_platform", "playstation"))

        self.show_prev.setChecked(c.show_preview)
        self.draw_zone_cb.setChecked(c.draw_zone)
        self.wep.setCurrentText(c.current_weapon)
        self._load_recoil_table()
        self._syncing = False

    def _wire_signals(self):
        def bind(w, sig):
            getattr(w, sig).connect(self.on_config_changed)

        for w, sig in [
            (self.cap_mode, "currentTextChanged"),
            (self.chiaki_win, "textChanged"),
            (self.cap_dev, "valueChanged"),
            (self.region_size, "valueChanged"),
            (self.use_gpu, "stateChanged"),
            (self.cal_x, "valueChanged"),
            (self.cal_y, "valueChanged"),
            (self.zone_radius, "valueChanged"),
            (self.zone_head, "toggled"),
            (self.zone_chest, "toggled"),
            (self.zone_body, "toggled"),
            (self.smooth, "valueChanged"),
            (self.strength, "valueChanged"),
            (self.speed, "valueChanged"),
            (self.use_pred, "stateChanged"),
            (self.pred_strength, "valueChanged"),
            (self.priority, "currentTextChanged"),
            (self.shake_red, "stateChanged"),
            (self.humanize, "valueChanged"),
            (self.max_dist, "valueChanged"),
            (self.fps_limit, "valueChanged"),
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
            (self.wep, "currentTextChanged"),
        ]:
            bind(w, sig)

        # Live badge text for sliders
        self.zone_radius.valueChanged.connect(lambda v: self.zone_badge.setText(str(v)))
        self.smooth.valueChanged.connect(lambda v: self.smooth_badge.setText(f"{v}%"))
        self.strength.valueChanged.connect(lambda v: self.strength_badge.setText(f"{v}%"))
        self.pred_strength.valueChanged.connect(lambda v: self.pred_badge.setText(f"{v}%"))
        self.humanize.valueChanged.connect(lambda v: self.human_badge.setText(str(v)))
        self.conf.valueChanged.connect(lambda v: self.conf_badge.setText(f"{v}%"))
        self.color_tol.valueChanged.connect(lambda v: self.color_tol_badge.setText(str(v)))
        self.enemy_color_edit.textChanged.connect(lambda *_: self._update_color_swatch())
        self.wep.currentTextChanged.connect(self._on_weapon_changed)

    def on_config_changed(self, *_):
        if self._syncing:
            return
        c = self.cfg
        c.capture_mode = self.cap_mode.currentText()
        c.chiaki_window = self.chiaki_win.text().strip() or "Chiaki"
        c.capture_device = self.cap_dev.value()
        c.region_size = self.region_size.value()
        c.use_gpu = self.use_gpu.isChecked()
        c.cal_x = self.cal_x.value()
        c.cal_y = self.cal_y.value()

        c.zone_radius = self.zone_radius.value()
        if self.zone_head.isChecked():
            c.target_part = "head"
        elif self.zone_chest.isChecked():
            c.target_part = "chest"
        else:
            c.target_part = "body"
        c.target_offset = c.offset_for_part()

        c.smoothing = self.smooth.value() / 100.0
        c.strength = self.strength.value() / 100.0
        c.acquisition_speed = self.speed.value()
        c.use_linear_prediction = self.use_pred.isChecked()
        c.prediction_strength = self.pred_strength.value() / 100.0
        c.target_priority = self.priority.currentText()
        c.shake_reduction = self.shake_red.isChecked()
        c.humanization = self.humanize.value() / 100.0
        c.max_distance = self.max_dist.value()
        c.fps_limit = self.fps_limit.value()

        c.engine = self.engine.currentText()
        c.confidence = self.conf.value() / 100.0
        c.draw_boxes = self.draw_boxes.isChecked()
        c.color_head_offset = self.color_off.value()
        # Normalize enemy color to #RRGGBB
        r, g, b = parse_hex_color(self.enemy_color_edit.text())
        c.enemy_color = f"#{r:02X}{g:02X}{b:02X}"
        c.color_tolerance = float(self.color_tol.value())
        c.sticky_aim = True

        c.gated_aim = self.gated.isChecked()
        c.activation_button = self.act_btn.currentText()
        c.ads_key = self.ads_key.currentText().strip().lower()
        c.toggle_key = (self.toggle_key.text() or "f").strip().lower()[:1] or "f"
        c.controller_platform = self.ctrl_platform.currentText()

        c.show_preview = self.show_prev.isChecked()
        c.draw_zone = self.draw_zone_cb.isChecked()
        c.current_weapon = self.wep.currentText()

        if self.worker:
            self.worker.cfg = c
            self.worker.predictor.prediction_ms = c.prediction_ms
            self.worker._sync_color_cfg()

    # ── Recoil helpers ────────────────────────────────────────────────────
    def _load_recoil_table(self):
        w = self.recoil.get(self.wep.currentText())
        n = max(len(w.vertical), len(w.horizontal), 8)
        self.rec_table.setRowCount(n)
        for i in range(n):
            v = w.vertical[i] if i < len(w.vertical) else 0
            h = w.horizontal[i] if i < len(w.horizontal) else 0
            self.rec_table.setItem(i, 0, QTableWidgetItem(str(v)))
            self.rec_table.setItem(i, 1, QTableWidgetItem(str(h)))

    def _on_weapon_changed(self, *_):
        self._load_recoil_table()
        self.on_config_changed()

    def save_recoil(self):
        w = self.recoil.get(self.wep.currentText())
        v, h = [], []
        for row in range(self.rec_table.rowCount()):
            try:
                v.append(int(self.rec_table.item(row, 0).text() if self.rec_table.item(row, 0) else 0))
            except Exception:
                v.append(0)
            try:
                h.append(int(self.rec_table.item(row, 1).text() if self.rec_table.item(row, 1) else 0))
            except Exception:
                h.append(0)
        w.vertical = v
        w.horizontal = h
        self.recoil.add_or_update(w)
        self.sb.showMessage(f"Recoil profile saved: {w.name}", 2500)

    # ── Engine control ────────────────────────────────────────────────────
    def toggle_aim(self):
        if not self.worker or not self.worker.running:
            self.start_engine()
            # Enable aim after short delay so worker is up
            QTimer.singleShot(400, lambda: self._set_aim(True))
            return
        self._set_aim(not self.worker.aim_enabled)

    def _set_aim(self, on: bool):
        if not self.worker:
            return
        self.worker.set_aim_enabled(on)
        if on:
            self.aim_btn.setText("AIM ENABLED  —  Hold activation button to track")
            self.aim_btn.setProperty("on", "true")
        else:
            self.aim_btn.setText("AIM DISABLED  —  Press F or click to enable")
            self.aim_btn.setProperty("on", "false")
        self.aim_btn.style().unpolish(self.aim_btn)
        self.aim_btn.style().polish(self.aim_btn)

    def start_engine(self):
        if self.worker and self.worker.running:
            return
        self.on_config_changed()  # flush UI → cfg
        self.cfg.save()
        self.worker = AimWorker(self.cfg)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_update.connect(self.on_status)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.live_badge.setText("● ENGINE ON")
        self.live_badge.setStyleSheet(
            f"background:#0a2a18; color:{GREEN}; border:1px solid {GREEN}; "
            f"border-radius:8px; padding:8px 14px; font-weight:800;"
        )
        self.sb.showMessage("Engine running…")

    def stop_engine(self):
        if self.worker:
            self.worker.stop()
            self._set_aim(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.live_badge.setText("● ENGINE OFF")
        self.live_badge.setStyleSheet(
            f"background:#1a1010; color:{RED}; border:1px solid {RED}; "
            f"border-radius:8px; padding:8px 14px; font-weight:800;"
        )
        self.preview.setText("Engine stopped.")
        self.sb.showMessage("Engine stopped.")

    @Slot(object, dict)
    def on_frame(self, frame, stats: dict):
        lock_s = "ON-POINT" if stats.get("locked") else ("TRACK" if stats.get("target") else "—")
        sticky_s = "yes" if stats.get("sticky") else "no"
        self.stats_lab.setText(
            f"FPS: {stats.get('fps', 0)}   "
            f"Dets: {stats.get('dets', 0)}   "
            f"Target: {'YES' if stats.get('target') else 'no'}   "
            f"Sticky: {sticky_s}   "
            f"Aim: {lock_s}   "
            f"Act: {'HELD' if stats.get('activation') else '—'}   "
            f"Pad: {'yes' if stats.get('pad') else 'no'}   "
            f"Res: {stats.get('res', '?')}"
        )
        self.cap_status.setText(f"Capture: {stats.get('capture', '—')}")
        self.pad_status.setText(
            f"Physical pad (XInput): {'connected' if stats.get('pad') else 'not detected'}"
        )

        if not self.cfg.show_preview or frame is None:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            # Ensure contiguous for QImage
            rgb = np.ascontiguousarray(rgb)
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            target = self.preview.size()
            self.preview.setPixmap(
                pix.scaled(target * 0.98, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        except Exception:
            pass

    @Slot(str)
    def on_status(self, msg: str):
        self.sb.showMessage(msg, 5000)

    # ── Hotkeys ───────────────────────────────────────────────────────────
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


def main():
    # High-DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
