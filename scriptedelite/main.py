"""
ScriptedElite AI - Precision target lock
Capture → detect → sticky lock → virtual controller tracking.
"""
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot
from PySide6.QtGui import QPixmap, QImage, QIcon, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QSpinBox,
    QFormLayout, QStatusBar, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QLineEdit, QFrame, QSizePolicy, QScrollArea, QButtonGroup,
    QRadioButton, QSplitter, QColorDialog, QListWidget, QListWidgetItem,
    QAbstractItemView,
)

from core.config import AppConfig
from core.capture import ScreenCapture, list_open_windows, WindowInfo
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

# ─── Brand palette ───
BG          = "#060910"
BG_PANEL    = "#0a101c"
BG_CARD     = "#0e1628"
BG_ELEVATED = "#121c32"
BORDER      = "#1c2d4a"
BORDER_HOT  = "#00b4ff"
ACCENT      = "#00c8ff"
ACCENT_DIM  = "#0077aa"
TEXT        = "#e8eef8"
TEXT_MUTED  = "#7a879e"
GREEN       = "#22c55e"
RED         = "#ef4444"
ORANGE      = "#f59e0b"
GOLD        = "#fbbf24"


def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    app.setStyleSheet(f"""
    QMainWindow, QWidget {{
        background: {BG};
        color: {TEXT};
        font-family: "Segoe UI", "Inter", system-ui, Arial;
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
        border-radius: 12px;
        background: {BG_PANEL};
        top: -1px;
        padding: 10px;
    }}
    QTabBar::tab {{
        background: {BG_CARD};
        color: {TEXT_MUTED};
        border: 1px solid {BORDER};
        border-bottom: none;
        border-top-left-radius: 9px;
        border-top-right-radius: 9px;
        padding: 11px 20px;
        margin-right: 3px;
        font-weight: 600;
        min-width: 88px;
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
        border-radius: 12px;
        margin-top: 16px;
        padding: 16px 14px 14px 14px;
        font-weight: 700;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 14px;
        padding: 0 10px;
        color: {ACCENT};
        font-size: 11px;
        letter-spacing: 1.2px;
        font-weight: 800;
    }}
    QLabel#sectionHint {{
        color: {TEXT_MUTED};
        font-size: 11.5px;
        font-weight: 400;
        line-height: 1.35;
    }}
    QLabel#heroHint {{
        color: {TEXT_MUTED};
        font-size: 12px;
        background: {BG_ELEVATED};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 12px 14px;
    }}
    QLabel#valueBadge {{
        background: #0a1a2e;
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 3px 12px;
        color: {ACCENT};
        font-weight: 700;
        font-size: 12px;
        min-width: 52px;
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
        padding: 9px 16px;
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
        padding: 12px 22px;
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
        padding: 16px 24px;
        font-weight: 800;
        letter-spacing: 0.6px;
        border-radius: 12px;
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
        padding: 7px 12px;
    }}
    QPushButton#ghostBtn:hover {{
        border-color: {ACCENT};
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
        padding: 7px 11px;
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
    QRadioButton {{ spacing: 8px; color: {TEXT}; font-weight: 600; }}
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
        background: #04060c;
        color: {TEXT_MUTED};
        border-top: 1px solid {BORDER};
        padding: 4px;
    }}
    QTableWidget {{
        background: #0a1220;
        border: 1px solid {BORDER};
        border-radius: 8px;
        gridline-color: {BORDER};
    }}
    QHeaderView::section {{
        background: {BG_CARD};
        color: {ACCENT};
        border: 1px solid {BORDER};
        padding: 6px;
        font-weight: 700;
    }}
    QFrame#previewFrame {{
        background: #04060c;
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}
    QFrame#lockCard {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0c1a28, stop:1 #0a1420);
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


# ═══════════════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════════════
class AimWorker(QObject):
    frame_ready = Signal(object, dict)
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
        hold = max(12, int(getattr(cfg, "lock_hold_frames", 28)))
        self.tracker = StickyTargetTracker(
            max_miss_frames=hold,
            match_iou=0.04,
            match_center_px=260.0,
            aim_smooth_far=0.18,
            aim_smooth_near=0.42,
            near_px=40.0,
            hold_expand=2.2,
        )
        self.color_tracker = StickyColorTracker(max_miss_frames=max(14, hold - 6), stick_radius=240.0)
        self.running = False
        self.aim_enabled = False
        self.ads_held = False
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

    def _rebuild_tracker_hold(self):
        hold = max(12, int(getattr(self.cfg, "lock_hold_frames", 28)))
        was_active = self.tracker.active
        # Don't reset identity mid-lock; only update hold frames
        self.tracker.max_miss_frames = hold
        self.color_tracker.max_miss_frames = max(14, hold - 6)
        if not was_active:
            pass

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
        if not self.cfg.gated_aim:
            return True
        btn = (self.cfg.activation_button or "").strip().lower()
        if btn and btn not in ("none", "off", "-"):
            if self.controller.is_activation_held(btn):
                return True
        if self.ads_held:
            return True
        return False

    def run(self):
        self.running = True
        try:
            mode = self.cfg.capture_mode or "window"
            title = self.cfg.window_title or self.cfg.chiaki_window or ""
            self.capture = ScreenCapture(
                mode=mode,  # type: ignore
                window_title=title,
                window_hwnd=int(getattr(self.cfg, "window_hwnd", 0) or 0),
                capture_device=self.cfg.capture_device,
                region_size=getattr(self.cfg, "region_size", 0) or 0,
            )
            if self.cfg.engine in ("yolo", "hybrid"):
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
                self.yolo = YOLODetector(str(MODEL_PATH), use_gpu=self.cfg.use_gpu)
            self.status_update.emit("Engine started — sticky lock armed")
        except Exception as e:
            self.status_update.emit(f"Init failed: {e}")
            self.running = False
            return

        while self.running:
            t0 = time.time()
            try:
                self._rebuild_tracker_hold()
                if self.capture:
                    want_title = self.cfg.window_title or self.cfg.chiaki_window or ""
                    want_hwnd = int(getattr(self.cfg, "window_hwnd", 0) or 0)
                    if self.capture.mode != self.cfg.capture_mode:
                        self.capture.set_mode(
                            self.cfg.capture_mode,
                            window_title=want_title,
                            window_hwnd=want_hwnd,
                            device=self.cfg.capture_device,
                        )
                    elif self.cfg.capture_mode in ("window", "chiaki"):
                        if (
                            self.capture.window_title != want_title
                            or self.capture.window_hwnd != want_hwnd
                        ):
                            self.capture.set_window(want_title, want_hwnd)

                frame, _left, _top = self.capture.grab_region()
                if frame is None or frame.size == 0:
                    time.sleep(0.005)
                    continue

                fh, fw = frame.shape[:2]
                screen_cx = fw / 2.0 + self.cfg.cal_x
                screen_cy = fh / 2.0 + self.cfg.cal_y

                best = None
                engine_used = self.cfg.engine
                offset = self.cfg.offset_for_part()
                self._sync_color_cfg()

                dets = []
                zone_r = max(10, int(self.cfg.zone_radius))
                max_dist = min(self.cfg.max_distance, zone_r)

                # ── YOLO sticky lock ──
                if self.cfg.engine in ("yolo", "hybrid") and self.yolo:
                    dets = self.yolo.detect(frame, self.cfg.confidence)
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

                    # Prediction only while tracking a live (non-coast) lock.
                    # Blended lightly so it never tears the aim off the box point.
                    if (
                        best
                        and self.cfg.use_linear_prediction
                        and not best.get("coasting")
                        and best.get("age", 0) >= 4
                    ):
                        self.predictor.prediction_ms = self.cfg.prediction_ms
                        self.predictor.update(best["aim_x"], best["aim_y"])
                        pred = self.predictor.predict(self.cfg.prediction_strength)
                        if pred:
                            px, py = pred
                            # Cap lead so we stay near the detection point
                            lead_x = px - best["aim_x"]
                            lead_y = py - best["aim_y"]
                            lead_mag = (lead_x ** 2 + lead_y ** 2) ** 0.5
                            max_lead = 28.0
                            if lead_mag > max_lead:
                                s = max_lead / lead_mag
                                lead_x *= s
                                lead_y *= s
                            blend = 0.22 * float(self.cfg.prediction_strength)
                            best["aim_x"] = best["aim_x"] + lead_x * blend
                            best["aim_y"] = best["aim_y"] + lead_y * blend
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

                # ── Color sticky fallback ──
                if self.cfg.engine == "color" or (self.cfg.engine == "hybrid" and best is None):
                    hits = self.color_det.detect_all(frame, self.cfg.color_head_offset)
                    ch = self.color_tracker.update(
                        hits, (screen_cx, screen_cy), max_dist, self.cfg.color_head_offset,
                    )
                    if ch:
                        best = ch
                        vis = self.color_det.draw(
                            frame if self.cfg.engine == "color" else vis, ch
                        )
                        engine_used = "color"
                elif self.cfg.engine != "color" and self.color_tracker.active:
                    self.color_tracker.reset()

                # ── Zone + crosshair ──
                if self.cfg.draw_zone:
                    cz = (int(round(screen_cx)), int(round(screen_cy)))
                    cv2.circle(vis, cz, zone_r, (0, 255, 80), 2, cv2.LINE_AA)
                    arm = 12
                    green = (0, 255, 80)
                    cv2.line(vis, (cz[0] - arm, cz[1]), (cz[0] + arm, cz[1]), green, 2, cv2.LINE_AA)
                    cv2.line(vis, (cz[0], cz[1] - arm), (cz[0], cz[1] + arm), green, 2, cv2.LINE_AA)
                    cv2.circle(vis, cz, 3, green, -1, cv2.LINE_AA)

                if best:
                    locked_on = self.controller.is_locked
                    if locked_on:
                        color = (0, 255, 120)
                    elif best.get("coasting"):
                        color = (180, 180, 80)
                    else:
                        color = (0, 200, 255)
                    cv2.line(
                        vis,
                        (int(screen_cx), int(screen_cy)),
                        (int(best["aim_x"]), int(best["aim_y"])),
                        color, 2, cv2.LINE_AA,
                    )
                    cv2.circle(
                        vis, (int(best["aim_x"]), int(best["aim_y"])),
                        7, (0, 0, 255), -1, cv2.LINE_AA,
                    )
                    cv2.circle(
                        vis, (int(best["aim_x"]), int(best["aim_y"])),
                        12, color, 2, cv2.LINE_AA,
                    )
                    label = "LOCK" if locked_on else ("HOLD" if best.get("coasting") else "TRACK")
                    cv2.putText(
                        vis, label,
                        (int(best["aim_x"]) + 12, int(best["aim_y"]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                    )

                # ── Activation + tracking ──
                activation = self._activation_active()
                should_aim = self.aim_enabled and activation

                if should_aim and best:
                    smooth = self.cfg.smoothing
                    if self.cfg.shake_reduction:
                        # Extra smoothing only near the aim point (reduces jitter, keeps speed far)
                        if best.get("dist", 999) < 55:
                            smooth = min(0.80, smooth + 0.12)
                        else:
                            smooth = min(0.70, smooth + 0.04)
                    self.controller.move_to_target(
                        best["dx"], best["dy"],
                        smoothing=smooth,
                        strength=self.cfg.strength,
                        speed=self.cfg.acquisition_speed / 3.0,
                        humanization=self.cfg.humanization,
                    )
                else:
                    # Release stick but KEEP sticky target identity while locked track exists
                    self.controller.release_stick()
                    if not activation and not best:
                        self.tracker.reset()
                        self.color_tracker.reset()
                        self.predictor.reset()
                    elif not best:
                        self.predictor.reset()
                    # Full controller state wipe only when aim master is off
                    if not self.aim_enabled:
                        self.controller.reset()

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
                    "age": int(best.get("age", 0)) if best else 0,
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


def _hero(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("heroHint")
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
    lay.addWidget(_hint(tip))
    return w


# ═══════════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScriptedElite AI  ·  Precision Target Lock")
        self.resize(1360, 900)
        self.setMinimumSize(1040, 720)

        if LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(LOGO_PATH)))

        self.cfg = AppConfig.load()
        self.worker: AimWorker | None = None
        self.worker_thread: threading.Thread | None = None
        self.recoil = RecoilMatrix()
        self._syncing = False
        self._window_cache: list[WindowInfo] = []

        self._build_ui()
        self._load_cfg_to_ui()
        self._wire_signals()
        self._setup_hotkeys()
        self._refresh_window_list()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 8)
        root.setSpacing(10)

        # ── Header ──
        header = QHBoxLayout()
        header.setSpacing(14)

        logo_lab = QLabel()
        if LOGO_PATH.exists():
            pix = QPixmap(str(LOGO_PATH)).scaled(58, 58, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_lab.setPixmap(pix)
        logo_lab.setFixedSize(58, 58)
        header.addWidget(logo_lab)

        titles = QVBoxLayout()
        titles.setSpacing(2)
        t1 = QLabel("SCRIPTED ELITE")
        t1.setStyleSheet(f"font-size: 21px; font-weight: 900; color: {ACCENT}; letter-spacing: 2.5px;")
        t2 = QLabel("Acquire  →  Lock  →  Hold  ·  Sticky target tracking")
        t2.setStyleSheet(f"font-size: 12px; color: {TEXT_MUTED};")
        titles.addWidget(t1)
        titles.addWidget(t2)
        header.addLayout(titles)
        header.addStretch()

        self.lock_badge = QLabel("NO TARGET")
        self.lock_badge.setStyleSheet(
            f"background:#121018; color:{TEXT_MUTED}; border:1px solid {BORDER}; "
            f"border-radius:8px; padding:8px 14px; font-weight:800; letter-spacing:0.5px;"
        )
        header.addWidget(self.lock_badge)

        self.live_badge = QLabel("● ENGINE OFF")
        self.live_badge.setStyleSheet(
            f"background:#1a1010; color:{RED}; border:1px solid {RED}; "
            f"border-radius:8px; padding:8px 14px; font-weight:800;"
        )
        header.addWidget(self.live_badge)
        root.addLayout(header)

        # Master aim
        self.aim_btn = QPushButton("AIM DISABLED  —  Press F or click to enable")
        self.aim_btn.setObjectName("aimToggle")
        self.aim_btn.setProperty("on", "false")
        self.aim_btn.setCursor(Qt.PointingHandCursor)
        self.aim_btn.clicked.connect(self.toggle_aim)
        self.aim_btn.setToolTip(
            "Master switch. When ON and you hold the activation button "
            "(default L2 / right mouse), aim sticks to the locked target point."
        )
        root.addWidget(self.aim_btn)

        # Body
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_guide(), "  Guide  ")
        self.tabs.addTab(self._tab_capture(), "  Capture  ")
        self.tabs.addTab(self._tab_aim(), "  Aim Lock  ")
        self.tabs.addTab(self._tab_model(), "  Detection  ")
        self.tabs.addTab(self._tab_activation(), "  Activation  ")
        self.tabs.addTab(self._tab_cronus(), "  Recoil  ")
        left_l.addWidget(self.tabs)

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

        # Preview panel
        right = QFrame()
        right.setObjectName("previewFrame")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(8)

        ph = QHBoxLayout()
        pt = QLabel("LIVE FEED")
        pt.setStyleSheet(f"font-weight: 800; color: {ACCENT}; letter-spacing: 1.5px; font-size: 12px;")
        ph.addWidget(pt)
        ph.addStretch()
        self.show_prev = QCheckBox("Preview")
        self.show_prev.setChecked(True)
        ph.addWidget(self.show_prev)
        self.draw_zone_cb = QCheckBox("Zone + X")
        self.draw_zone_cb.setChecked(True)
        self.draw_zone_cb.setToolTip("Draw FOV circle and crosshair center on the feed.")
        ph.addWidget(self.draw_zone_cb)
        rl.addLayout(ph)

        self.preview = QLabel(
            "Start the engine to capture your selected window\n"
            "and run detection with sticky target lock."
        )
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(440, 340)
        self.preview.setStyleSheet(
            f"color: {TEXT_MUTED}; background: #04060c; border-radius: 10px; padding: 20px;"
        )
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rl.addWidget(self.preview, 1)

        # Stats row
        stats_row = QHBoxLayout()
        self.stats_lab = QLabel("Waiting for engine…")
        self.stats_lab.setObjectName("statPill")
        self.stats_lab.setWordWrap(True)
        stats_row.addWidget(self.stats_lab, 1)
        rl.addLayout(stats_row)

        split.addWidget(right)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 4)
        root.addWidget(split, 1)

        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        self.sb.showMessage(
            "Ready  ·  Capture a window → Start Engine → Enable Aim → Hold activation to lock"
        )

    # ── Guide tab ─────────────────────────────────────────────────────────
    def _tab_guide(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setSpacing(12)

        lay.addWidget(_hero(
            "<b style='color:#00c8ff'>What this system does</b><br><br>"
            "ScriptedElite finds a player inside your FOV zone, <b>commits to that single target</b>, "
            "moves aim to your configured point (head / chest / body), and <b>holds the lock</b> "
            "as consistently as possible. It will not hop to another player while the current one "
            "is still trackable."
        ))

        g = QGroupBox("QUICK START")
        gl = QVBoxLayout(g)
        gl.addWidget(_hint(
            "1. Open the game / Chiaki / app you want to capture.\n"
            "2. Go to Capture → pick Source = Application Window → Refresh → select the window.\n"
            "3. Click Start Engine. Confirm the green zone + X sit on your crosshair.\n"
            "4. Aim Lock → choose HEAD / CHEST / BODY and set FOV radius.\n"
            "5. Activation → set controller button (default L2) or mouse (right).\n"
            "6. Enable AIM (button or F). Hold activation → lock engages on first valid target.\n"
            "7. Release activation → stick centers; re-hold to track the same sticky target if still visible."
        ))
        lay.addWidget(g)

        g2 = QGroupBox("HOW LOCK WORKS")
        g2l = QVBoxLayout(g2)
        g2l.addWidget(_hint(
            "• ACQUIRE — first valid detection inside the green FOV is chosen (closest by default).\n"
            "• COMMIT — that player is sticky-tracked; other boxes are ignored.\n"
            "• AIM POINT — red dot is the configured body offset (head/chest/body + fine tune).\n"
            "• HOLD — when aim is on the point, stick is zeroed (true lock). Target movement re-engages.\n"
            "• COAST — brief YOLO dropouts keep the last aim so the lock does not blink off.\n"
            "• RELEASE — only after many missed frames or the track leaves the expanded hold zone."
        ))
        lay.addWidget(g2)

        g3 = QGroupBox("TUNING TIPS")
        g3l = QVBoxLayout(g3)
        g3l.addWidget(_hint(
            "• Overshooting past the target → lower Tracking Strength or raise Smoothness.\n"
            "• Too slow to catch up → raise Tracking Speed or lower Smoothness slightly.\n"
            "• Lock jitters on the head → enable Shake Reduction; raise Confidence a bit.\n"
            "• Loses target when they strafe → raise FOV / Max Distance; raise Lock Hold Frames.\n"
            "• Wrong window content → Capture → Refresh list and re-select; use Full window (0)."
        ))
        lay.addWidget(g3)

        g4 = QGroupBox("REQUIREMENTS")
        g4l = QVBoxLayout(g4)
        g4l.addWidget(_hint(
            "• Windows + ViGEmBus (virtual Xbox 360 pad for stick output).\n"
            "• NVIDIA CUDA recommended for YOLO performance.\n"
            "• Physical pad visible to XInput if using L2/R2 activation.\n"
            "• Model file: assets/model.pt"
        ))
        lay.addWidget(g4)
        lay.addStretch()
        scroll.setWidget(host)
        return scroll

    # ── Capture tab ───────────────────────────────────────────────────────
    def _tab_capture(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setSpacing(10)

        g = QGroupBox("CAPTURE SOURCE")
        gl = QFormLayout(g)
        gl.setSpacing(10)

        self.cap_mode = QComboBox()
        self.cap_mode.addItem("Application Window", "window")
        self.cap_mode.addItem("Chiaki (keyword)", "chiaki")
        self.cap_mode.addItem("Capture Card", "capture_card")
        self.cap_mode.addItem("Desktop (primary monitor)", "desktop")
        self.cap_mode.setToolTip(
            "Application Window — pick any open window (game, Chrome, Chiaki, …)\n"
            "Chiaki — auto-find a window titled like Chiaki\n"
            "Capture Card — HDMI capture device index\n"
            "Desktop — full primary monitor"
        )
        gl.addRow("Source", self.cap_mode)
        lay.addWidget(g)

        # Window picker
        self.win_group = QGroupBox("OPEN WINDOWS  ·  SELECT CAPTURE TARGET")
        wl = QVBoxLayout(self.win_group)
        wl.addWidget(_hint(
            "Click a window below to use it as the detection source. "
            "The client area (game content) is captured when possible."
        ))
        row = QHBoxLayout()
        self.refresh_wins_btn = QPushButton("↻  Refresh List")
        self.refresh_wins_btn.setObjectName("ghostBtn")
        self.refresh_wins_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_wins_btn.clicked.connect(self._refresh_window_list)
        row.addWidget(self.refresh_wins_btn)
        row.addStretch()
        self.win_count_lab = QLabel("")
        self.win_count_lab.setObjectName("sectionHint")
        row.addWidget(self.win_count_lab)
        wl.addLayout(row)

        self.win_list = QListWidget()
        self.win_list.setMinimumHeight(180)
        self.win_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.win_list.itemSelectionChanged.connect(self._on_window_selected)
        wl.addWidget(self.win_list)

        self.selected_win_lab = QLabel("Selected: none")
        self.selected_win_lab.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
        wl.addWidget(self.selected_win_lab)

        self.chiaki_win = QLineEdit("Chiaki")
        self.chiaki_win.setPlaceholderText("Chiaki keyword (only for Chiaki source mode)")
        self.chiaki_win.setToolTip("Used when Source is Chiaki — matches window titles containing this text.")
        wl.addWidget(_hint("Chiaki keyword (Chiaki mode only):"))
        wl.addWidget(self.chiaki_win)
        lay.addWidget(self.win_group)

        g2 = QGroupBox("CAPTURE OPTIONS")
        g2l = QFormLayout(g2)
        self.cap_dev = QSpinBox()
        self.cap_dev.setRange(0, 10)
        self.cap_dev.setToolTip("Capture card device index (usually 0).")
        g2l.addRow("Capture Card #", self.cap_dev)

        self.region_size = QSpinBox()
        self.region_size.setRange(0, 2160)
        self.region_size.setSingleStep(40)
        self.region_size.setSpecialValueText("Full window")
        self.region_size.setToolTip(
            "0 = entire selected window (recommended).\n"
            "e.g. 720 = center-crop a square FOV region."
        )
        g2l.addRow("Region Size", self.region_size)

        self.use_gpu = QCheckBox("GPU Acceleration (CUDA)")
        self.use_gpu.setToolTip("Use NVIDIA CUDA for YOLO when available.")
        g2l.addRow(self.use_gpu)
        lay.addWidget(g2)

        g3 = QGroupBox("CROSSHAIR CALIBRATION")
        g3l = QFormLayout(g3)
        self.cal_x = QSpinBox(); self.cal_x.setRange(-300, 300)
        self.cal_y = QSpinBox(); self.cal_y.setRange(-300, 300)
        self.cal_x.setToolTip("Shift detection center horizontally onto your in-game crosshair.")
        self.cal_y.setToolTip("Shift detection center vertically onto your in-game crosshair.")
        g3l.addRow("Calibrate X (px)", self.cal_x)
        g3l.addRow("Calibrate Y (px)", self.cal_y)
        g3l.addRow(_hint("Nudge until the green X lines up with your crosshair on the live feed."))
        lay.addWidget(g3)

        g4 = QGroupBox("STATUS")
        g4l = QVBoxLayout(g4)
        self.cap_status = QLabel("Capture: idle")
        self.cap_status.setObjectName("sectionHint")
        self.pad_status = QLabel("Physical pad: —")
        self.pad_status.setObjectName("sectionHint")
        g4l.addWidget(self.cap_status)
        g4l.addWidget(self.pad_status)
        lay.addWidget(g4)
        lay.addStretch()
        scroll.setWidget(host)
        return scroll

    # ── Aim tab ───────────────────────────────────────────────────────────
    def _tab_aim(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setSpacing(8)

        lay.addWidget(_hero(
            "<b style='color:#00c8ff'>Target lock controls</b> — once a target is acquired inside the FOV, "
            "the system sticks to that player and holds the configured aim point. "
            "Speed / smoothness change how fast you close the gap; they do not break the sticky identity."
        ))

        top = QHBoxLayout()
        g_fov = QGroupBox("FOV  ·  DETECTION ZONE")
        fl = QVBoxLayout(g_fov)
        self.zone_radius = QSlider(Qt.Horizontal)
        self.zone_radius.setRange(40, 700)
        self.zone_badge = _badge("200")
        fl.addWidget(_slider_row(
            "Zone Radius (px)",
            "How far from the crosshair a new target may be acquired. "
            "Matches the green circle. Once locked, hold expands beyond this automatically.",
            self.zone_radius, self.zone_badge,
        ))
        top.addWidget(g_fov, 1)

        g_zone = QGroupBox("AIM POINT")
        zl = QVBoxLayout(g_zone)
        zl.addWidget(_hint("Where on the player box the red lock-dot sits."))
        self.zone_head = QRadioButton("HEAD")
        self.zone_chest = QRadioButton("CHEST")
        self.zone_body = QRadioButton("BODY")
        self.zone_head.setChecked(True)
        self.zone_head.setToolTip("Aim near the top of the detection box (~12% down).")
        self.zone_chest.setToolTip("Aim at upper-mid box (~30% down).")
        self.zone_body.setToolTip("Aim at mid box (~45% down).")
        self.zone_group = QButtonGroup(self)
        for b in (self.zone_head, self.zone_chest, self.zone_body):
            self.zone_group.addButton(b)
            zl.addWidget(b)
        top.addWidget(g_zone)
        lay.addLayout(top)

        g_fine = QGroupBox("AIM FINE-TUNE")
        flay = QVBoxLayout(g_fine)
        self.aim_fine = QSlider(Qt.Horizontal)
        self.aim_fine.setRange(-10, 10)
        self.aim_fine.setValue(0)
        self.aim_fine_badge = _badge("0")
        flay.addWidget(_slider_row(
            "Vertical Fine Tune",
            "Small offset on top of Head/Chest/Body (− = higher, + = lower). "
            "Use this to land the red dot exactly on the hitbox you want.",
            self.aim_fine, self.aim_fine_badge,
        ))
        lay.addWidget(g_fine)

        g_track = QGroupBox("TRACKING RESPONSE")
        tl = QVBoxLayout(g_track)

        self.smooth = QSlider(Qt.Horizontal)
        self.smooth.setRange(0, 85)
        self.smooth_badge = _badge("32%")
        tl.addWidget(_slider_row(
            "Tracking Smoothness",
            "Higher = softer motion, less snap. Lower = more aggressive close-in. "
            "Does not release the sticky target. Typical: 25–45%.",
            self.smooth, self.smooth_badge,
        ))

        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(10, 250)
        self.strength_badge = _badge("105%")
        tl.addWidget(_slider_row(
            "Tracking Strength",
            "Overall stick gain. Raise if aim feels weak; lower if it overshoots the lock point.",
            self.strength, self.strength_badge,
        ))

        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 8.0)
        self.speed.setSingleStep(0.1)
        self.speed.setDecimals(1)
        self.speed.setToolTip(
            "How quickly the stick closes the gap to the locked aim point. "
            "Independent of smoothness. 2.5–4.5 is a solid range."
        )
        row_s = QHBoxLayout()
        lab_s = QLabel("Tracking Speed")
        lab_s.setStyleSheet("font-weight: 700;")
        row_s.addWidget(lab_s)
        row_s.addStretch()
        row_s.addWidget(self.speed)
        tl.addLayout(row_s)
        tl.addWidget(_hint(
            "Speed of closing onto the lock point. Higher = faster acquire; "
            "the anti-overshoot layer still prevents flying past the target."
        ))
        lay.addWidget(g_track)

        g_lock = QGroupBox("LOCK HOLD")
        ll = QFormLayout(g_lock)
        self.lock_hold = QSpinBox()
        self.lock_hold.setRange(8, 60)
        self.lock_hold.setToolTip(
            "How many missed detection frames to keep the sticky lock (coast) before releasing. "
            "Higher = more stubborn lock through blinks/occlusion."
        )
        ll.addRow("Lock Hold Frames", self.lock_hold)
        ll.addRow(_hint(
            "Sticky identity is always on. This only controls how long to coast through YOLO dropouts."
        ))
        self.shake_red = QCheckBox("Shake Reduction (near-target extra filter)")
        self.shake_red.setToolTip("Adds mild extra smoothing when already close to the aim point.")
        self.shake_red.setChecked(True)
        ll.addRow(self.shake_red)
        lay.addWidget(g_lock)

        g_pred = QGroupBox("PREDICTION  ·  OPTIONAL")
        pl = QFormLayout(g_pred)
        self.use_pred = QCheckBox("Lead moving targets (capped)")
        self.use_pred.setToolTip(
            "Slight forward lead based on target velocity. "
            "Capped so it cannot pull far off the detection point. Off by default for pure lock."
        )
        pl.addRow(self.use_pred)
        self.pred_strength = QSlider(Qt.Horizontal)
        self.pred_strength.setRange(0, 100)
        self.pred_badge = _badge("35%")
        pl.addRow("Predict Strength", self._mini_slider(self.pred_strength, self.pred_badge))
        self.priority = QComboBox()
        self.priority.addItems(["closest", "highest_conf"])
        self.priority.setToolTip(
            "Only used when acquiring a NEW target. "
            "Once locked, priority is ignored — the same player is held."
        )
        pl.addRow("Acquire Priority", self.priority)
        self.humanize = QSlider(Qt.Horizontal)
        self.humanize.setRange(0, 40)
        self.human_badge = _badge("0")
        pl.addRow("Humanization", self._mini_slider(self.humanize, self.human_badge))
        pl.addRow(_hint("Humanization is disabled near the lock point so it cannot break hold accuracy."))
        lay.addWidget(g_pred)

        g_lim = QGroupBox("LIMITS")
        lim = QFormLayout(g_lim)
        self.max_dist = QSpinBox(); self.max_dist.setRange(50, 1400)
        self.max_dist.setToolTip("Hard max distance (px) for acquire; also limited by FOV zone.")
        self.fps_limit = QSpinBox(); self.fps_limit.setRange(30, 240)
        self.fps_limit.setToolTip("Cap processing rate to save CPU/GPU.")
        lim.addRow("Max Distance (px)", self.max_dist)
        lim.addRow("FPS Limit", self.fps_limit)
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        lay = QVBoxLayout(host)

        g = QGroupBox("DETECTION MODE")
        gl = QVBoxLayout(g)
        gl.addWidget(_hint(
            "YOLO — neural person detection with hard sticky lock (recommended).\n"
            "Color — tracks only your configured enemy color blobs.\n"
            "Hybrid — YOLO sticky first; color sticky if no person box."
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
        g2l.addRow(_hint("Higher = fewer false detections; may miss distant targets. 35–55% typical."))
        self.draw_boxes = QCheckBox("Draw bounding boxes on preview")
        self.draw_boxes.setChecked(True)
        g2l.addRow(self.draw_boxes)
        lay.addWidget(g2)

        g3 = QGroupBox("ENEMY COLOR  ·  COLOR / HYBRID")
        g3l = QFormLayout(g3)
        g3l.addRow(_hint(
            "Set the enemy highlight / outline color so teammates are ignored in color mode."
        ))

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
        color_row.addWidget(self.enemy_color_edit, 1)
        self.pick_color_btn = QPushButton("Pick Color…")
        self.pick_color_btn.setCursor(Qt.PointingHandCursor)
        self.pick_color_btn.clicked.connect(self._pick_enemy_color)
        color_row.addWidget(self.pick_color_btn)
        g3l.addRow("Enemy Color", color_row)

        preset_row = QHBoxLayout()
        for label, hx in [
            ("Magenta", "#FF00FA"), ("Red", "#FF2020"), ("Orange", "#FF8800"),
            ("Green", "#00FF00"), ("Yellow", "#FFE600"), ("Cyan", "#00E5FF"),
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
        g3l.addRow("Color Tolerance", self._mini_slider(self.color_tol, self.color_tol_badge))

        self.color_off = QDoubleSpinBox()
        self.color_off.setRange(0, 200)
        self.color_off.setSuffix(" px")
        self.color_off.setToolTip("Pixels downward from the color blob center to the aim point.")
        g3l.addRow("Aim Offset Y", self.color_off)
        lay.addWidget(g3)
        lay.addStretch()
        scroll.setWidget(host)
        return scroll

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

    # ── Activation tab ────────────────────────────────────────────────────
    def _tab_activation(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        g = QGroupBox("ACTIVATION  ·  WHEN TRACKING MOVES THE STICK")
        gl = QFormLayout(g)
        gl.setSpacing(12)

        self.gated = QCheckBox("Require activation button (recommended)")
        self.gated.setChecked(True)
        self.gated.setToolTip(
            "When on, stick only moves while you hold the activation input. "
            "Sticky target identity can still be held while re-pressing."
        )
        gl.addRow(self.gated)

        self.act_btn = QComboBox()
        self.act_btn.addItems([
            "l2", "r2", "lt", "rt", "lb", "rb", "l1", "r1",
            "a", "b", "x", "y", "ls", "rs", "none",
        ])
        self.act_btn.setToolTip(
            "Physical controller button held to drive aim.\n"
            "PlayStation: L2 / R2. Xbox: LT / RT. Requires XInput."
        )
        gl.addRow("Controller Button", self.act_btn)

        self.ads_key = QComboBox()
        self.ads_key.setEditable(True)
        self.ads_key.addItems(["right", "left", "middle", "ctrl", "shift", "alt", "c", "v", "x"])
        self.ads_key.setToolTip("Mouse button or key as activation fallback.")
        gl.addRow("Keyboard / Mouse", self.ads_key)

        self.toggle_key = QLineEdit("f")
        self.toggle_key.setMaxLength(1)
        self.toggle_key.setToolTip("Key that toggles master AIM on/off.")
        gl.addRow("Master Toggle Key", self.toggle_key)

        self.ctrl_platform = QComboBox()
        self.ctrl_platform.addItems(["playstation", "xbox"])
        gl.addRow("Controller Labels", self.ctrl_platform)
        lay.addWidget(g)

        info = QGroupBox("HOW TO USE")
        il = QVBoxLayout(info)
        il.addWidget(_hint(
            "1. Start Engine so capture + detection are live.\n"
            "2. Enable AIM (button or hotkey F).\n"
            "3. Hold activation (default L2 / right mouse).\n"
            "4. First valid player inside the green zone is locked — TRACK → LOCK on the red point.\n"
            "5. Release activation → stick centers. Sticky identity may remain if the target is still seen.\n\n"
            "ViGEmBus must be installed so the virtual Xbox 360 pad appears for Chiaki / the game."
        ))
        lay.addWidget(info)
        lay.addStretch()
        return w

    # ── Recoil tab ────────────────────────────────────────────────────────
    def _tab_cronus(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("RECOIL PROFILES  ·  REFERENCE ONLY")
        gl = QVBoxLayout(g)
        gl.addWidget(_hint(
            "These tables are reference data for Cronus Zen / external scripts. "
            "They are not applied to the live virtual stick in this build."
        ))
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
        lay.addWidget(g)
        lay.addStretch()
        return w

    # ── Window list ───────────────────────────────────────────────────────
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
            item.setToolTip(f"HWND {w.hwnd}\n{w.title}\n{w.width}x{w.height} @ ({w.left},{w.top})")
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
            self.selected_win_lab.setText("Selected: none — pick a window above")

    def _on_window_selected(self):
        self._apply_selected_window_label()
        self.on_config_changed()

    def _apply_selected_window_label(self):
        item = self.win_list.currentItem()
        if not item:
            return
        title = item.data(Qt.UserRole + 1) or item.text()
        hwnd = int(item.data(Qt.UserRole) or 0)
        self.selected_win_lab.setText(f"Selected: {title}")
        # Keep chiaki field in sync for keyword mode convenience
        if self.cap_mode.currentData() == "chiaki":
            self.chiaki_win.setText(title)

    def _selected_window(self) -> tuple[str, int]:
        item = self.win_list.currentItem()
        if item:
            return str(item.data(Qt.UserRole + 1) or ""), int(item.data(Qt.UserRole) or 0)
        return self.cfg.window_title or "", int(self.cfg.window_hwnd or 0)

    def _on_cap_mode_changed(self, *_):
        mode = self.cap_mode.currentData() or self.cap_mode.currentText()
        is_win = mode in ("window", "chiaki")
        self.win_group.setVisible(True)  # always show list; useful for both
        self.chiaki_win.setEnabled(mode == "chiaki")
        self.cap_dev.setEnabled(mode == "capture_card")
        if is_win and self.win_list.count() == 0:
            self._refresh_window_list()
        self.on_config_changed()

    # ── Config sync ───────────────────────────────────────────────────────
    def _load_cfg_to_ui(self):
        self._syncing = True
        c = self.cfg

        mode = c.capture_mode or "window"
        idx = self.cap_mode.findData(mode)
        if idx < 0:
            # legacy plain text
            idx = max(0, self.cap_mode.findText(mode, Qt.MatchContains))
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
        {"head": self.zone_head, "chest": self.zone_chest, "body": self.zone_body}.get(
            part, self.zone_head
        ).setChecked(True)

        fine = int(round(float(getattr(c, "aim_fine_tune", 0.0)) * 100))
        self.aim_fine.setValue(max(-10, min(10, fine)))
        self.aim_fine_badge.setText(str(self.aim_fine.value()))

        self.smooth.setValue(int(c.smoothing * 100))
        self.smooth_badge.setText(f"{int(c.smoothing * 100)}%")
        self.strength.setValue(int(c.strength * 100))
        self.strength_badge.setText(f"{int(c.strength * 100)}%")
        self.speed.setValue(c.acquisition_speed)
        self.lock_hold.setValue(int(getattr(c, "lock_hold_frames", 28)))
        self.use_pred.setChecked(c.use_linear_prediction)
        self.pred_strength.setValue(int(getattr(c, "prediction_strength", 0.35) * 100))
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
        aidx = self.act_btn.findText((c.activation_button or "l2").lower())
        self.act_btn.setCurrentIndex(max(0, aidx))
        self.ads_key.setCurrentText(c.ads_key or "right")
        self.toggle_key.setText((c.toggle_key or "f")[:1])
        self.ctrl_platform.setCurrentText(getattr(c, "controller_platform", "playstation"))

        self.show_prev.setChecked(c.show_preview)
        self.draw_zone_cb.setChecked(c.draw_zone)
        self.wep.setCurrentText(c.current_weapon)
        self._load_recoil_table()
        self._syncing = False
        self._on_cap_mode_changed()

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
            (self.zone_head, "toggled"),
            (self.zone_chest, "toggled"),
            (self.zone_body, "toggled"),
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

        self.cap_mode.currentIndexChanged.connect(self._on_cap_mode_changed)

        self.zone_radius.valueChanged.connect(lambda v: self.zone_badge.setText(str(v)))
        self.smooth.valueChanged.connect(lambda v: self.smooth_badge.setText(f"{v}%"))
        self.strength.valueChanged.connect(lambda v: self.strength_badge.setText(f"{v}%"))
        self.pred_strength.valueChanged.connect(lambda v: self.pred_badge.setText(f"{v}%"))
        self.humanize.valueChanged.connect(lambda v: self.human_badge.setText(str(v)))
        self.conf.valueChanged.connect(lambda v: self.conf_badge.setText(f"{v}%"))
        self.color_tol.valueChanged.connect(lambda v: self.color_tol_badge.setText(str(v)))
        self.aim_fine.valueChanged.connect(lambda v: self.aim_fine_badge.setText(str(v)))
        self.enemy_color_edit.textChanged.connect(lambda *_: self._update_color_swatch())
        self.wep.currentTextChanged.connect(self._on_weapon_changed)

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
            c.window_title = c.chiaki_window
            # Prefer list selection if user picked one
            if title:
                c.window_title = title
                c.window_hwnd = hwnd
            else:
                c.window_hwnd = 0
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
        if self.zone_head.isChecked():
            c.target_part = "head"
        elif self.zone_chest.isChecked():
            c.target_part = "chest"
        else:
            c.target_part = "body"
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

        c.engine = self.engine.currentText()
        c.confidence = self.conf.value() / 100.0
        c.draw_boxes = self.draw_boxes.isChecked()
        c.color_head_offset = self.color_off.value()
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
            QTimer.singleShot(400, lambda: self._set_aim(True))
            return
        self._set_aim(not self.worker.aim_enabled)

    def _set_aim(self, on: bool):
        if not self.worker:
            return
        self.worker.set_aim_enabled(on)
        if on:
            self.aim_btn.setText("AIM ENABLED  —  Hold activation to lock target")
            self.aim_btn.setProperty("on", "true")
        else:
            self.aim_btn.setText("AIM DISABLED  —  Press F or click to enable")
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
        self.live_badge.setStyleSheet(
            f"background:#0a2a18; color:{GREEN}; border:1px solid {GREEN}; "
            f"border-radius:8px; padding:8px 14px; font-weight:800;"
        )
        self.sb.showMessage("Engine running — sticky lock armed")

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
        self.lock_badge.setText("NO TARGET")
        self.lock_badge.setStyleSheet(
            f"background:#121018; color:{TEXT_MUTED}; border:1px solid {BORDER}; "
            f"border-radius:8px; padding:8px 14px; font-weight:800;"
        )
        self.preview.setText("Engine stopped.")
        self.sb.showMessage("Engine stopped.")

    @Slot(object, dict)
    def on_frame(self, frame, stats: dict):
        locked = stats.get("locked")
        target = stats.get("target")
        sticky = stats.get("sticky")
        if locked:
            self.lock_badge.setText("● ON POINT")
            self.lock_badge.setStyleSheet(
                f"background:#0a2a18; color:{GREEN}; border:1px solid {GREEN}; "
                f"border-radius:8px; padding:8px 14px; font-weight:800;"
            )
        elif sticky or target:
            self.lock_badge.setText("● TRACKING")
            self.lock_badge.setStyleSheet(
                f"background:#1a1808; color:{GOLD}; border:1px solid {GOLD}; "
                f"border-radius:8px; padding:8px 14px; font-weight:800;"
            )
        else:
            self.lock_badge.setText("NO TARGET")
            self.lock_badge.setStyleSheet(
                f"background:#121018; color:{TEXT_MUTED}; border:1px solid {BORDER}; "
                f"border-radius:8px; padding:8px 14px; font-weight:800;"
            )

        lock_s = "ON-POINT" if locked else ("TRACK" if target else "—")
        self.stats_lab.setText(
            f"FPS {stats.get('fps', 0)}  ·  "
            f"Dets {stats.get('dets', 0)}  ·  "
            f"Aim {lock_s}  ·  "
            f"Age {stats.get('age', 0)}  ·  "
            f"Act {'HELD' if stats.get('activation') else '—'}  ·  "
            f"Pad {'yes' if stats.get('pad') else 'no'}  ·  "
            f"{stats.get('res', '?')}  ·  "
            f"{stats.get('capture', '')}"
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
    # HiDPI
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
