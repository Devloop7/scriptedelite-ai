"""
ScriptedElite AI - PS5 Optimization (v7 Archetype)
No login. New design using your logo.
Dual YOLO + Color, Chiaki/Capture Card, vgamepad controller emulation,
Linear Prediction, ADS Gating, Calibration, Cronus Recoil.
Uses the exact same model.pt.
"""
import sys
import time
import threading
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QDoubleSpinBox,
    QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QSpinBox,
    QFormLayout, QStatusBar, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot
from PySide6.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QPen, QIcon

import cv2
import numpy as np

from core.config import AppConfig
from core.capture import ScreenCapture
from core.detector import YOLODetector
from core.color_detector import ColorSignatureDetector
from core.predictor import LinearPredictor
from core.controller import AimController
from core.recoil import RecoilMatrix

from pynput import keyboard, mouse

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
LOGO_PATH = ASSETS_DIR / "logo.png"

# Theme
BG_DARK = "#0b111f"
ACCENT_BLUE = "#00c8ff"
TEXT_LIGHT = "#e6e9f0"
TEXT_MUTED = "#8a93a8"
GREEN_ON = "#22c55e"
RED_OFF = "#ef4444"

def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    qss = f"""
    QMainWindow, QWidget {{ background: {BG_DARK}; color: {TEXT_LIGHT}; font-family: "Segoe UI", Arial; }}
    QGroupBox {{ border: 1px solid #1f2a44; border-radius: 8px; margin-top: 8px; padding: 6px; font-weight: 600; }}
    QGroupBox::title {{ color: {ACCENT_BLUE}; }}
    QPushButton {{ background: #1a253f; border: 1px solid #2a3a5c; border-radius: 6px; padding: 6px 14px; }}
    QPushButton:hover {{ border-color: {ACCENT_BLUE}; }}
    QPushButton#aimToggle {{ font-size: 15px; padding: 12px 20px; font-weight: bold; }}
    QPushButton#aimToggle[enabled="true"] {{ background: #0f3a24; border-color: {GREEN_ON}; color: {GREEN_ON}; }}
    QPushButton#aimToggle[enabled="false"] {{ background: #3a1616; border-color: {RED_OFF}; color: {RED_OFF}; }}
    QStatusBar {{ background: #0a0f1c; color: {TEXT_MUTED}; }}
    """
    app.setStyleSheet(qss)


class AimbotWorker(QObject):
    frame_ready = Signal(np.ndarray, dict)
    status_update = Signal(str)

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self.capture = None
        self.yolo = None
        self.color_det = ColorSignatureDetector()
        self.predictor = LinearPredictor(history_len=5, prediction_ms=cfg.prediction_ms)
        self.controller = AimController()
        self.running = False
        self.aim_enabled = False
        self.ads_held = False
        self._last_time = time.time()
        self.stats = {"dets": 0, "fps": 0, "engine": cfg.engine, "gated": False}

    @Slot(bool)
    def set_aim_enabled(self, enabled: bool):
        self.aim_enabled = enabled
        self.controller.set_enabled(enabled)
        if not enabled:
            self.controller.reset()
            self.predictor.reset()

    @Slot(bool)
    def set_ads_held(self, held: bool):
        self.ads_held = held

    def run(self):
        self.running = True
        try:
            self.capture = ScreenCapture(
                mode=self.cfg.capture_mode,
                window_title=self.cfg.chiaki_window,
                capture_device=self.cfg.capture_device
            )
            if self.cfg.engine in ("yolo", "hybrid"):
                self.yolo = YOLODetector(str(MODEL_PATH))
            self.status_update.emit("Engine started")
        except Exception as e:
            self.status_update.emit(f"Init failed: {e}")
            self.running = False
            return

        frame_count = 0
        fps = 0.0

        while self.running:
            t0 = time.time()
            try:
                frame, left, top = self.capture.grab_region()
                if frame is None or frame.size == 0:
                    time.sleep(0.003)
                    continue

                screen_cx = self.capture.get_screen_center()[0] - left + self.cfg.cal_x
                screen_cy = self.capture.get_screen_center()[1] - top + self.cfg.cal_y

                best = None
                engine_used = self.cfg.engine

                if self.cfg.engine in ("yolo", "hybrid") and self.yolo:
                    dets = self.yolo.detect(frame, self.cfg.confidence)
                    best = self.yolo.find_best_target(dets, (screen_cx, screen_cy), self.cfg.max_distance, self.cfg.target_offset)
                    vis = self.yolo.draw_detections(frame, dets, best, self.cfg.target_offset) if best or dets else frame.copy()

                    if best and self.cfg.use_linear_prediction:
                        self.predictor.update(best["aim_x"], best["aim_y"])
                        pred = self.predictor.predict()
                        if pred:
                            best["aim_x"], best["aim_y"] = pred
                            best["dx"] = pred[0] - screen_cx
                            best["dy"] = pred[1] - screen_cy
                else:
                    vis = frame.copy()

                if (self.cfg.engine in ("color", "hybrid")) and (best is None or self.cfg.engine == "color"):
                    ch = self.color_det.detect(frame, self.cfg.color_head_offset)
                    if ch:
                        ax, ay = ch["aim_x"], ch["aim_y"]
                        dx = ax - screen_cx
                        dy = ay - screen_cy
                        dist = (dx*dx + dy*dy) ** 0.5
                        if dist < self.cfg.max_distance:
                            best = {"aim_x": ax, "aim_y": ay, "dx": dx, "dy": dy, "dist": dist}
                            vis = self.color_det.draw(frame, ch)
                            engine_used = "color"

                if 'vis' not in locals():
                    vis = frame.copy()

                should_aim = self.aim_enabled and (not self.cfg.gated_aim or self.ads_held)
                self.stats["gated"] = should_aim

                if should_aim and best:
                    dx = best["dx"]
                    dy = best["dy"]
                    self.controller.move_to_target(
                        dx, dy,
                        smoothing=self.cfg.smoothing,
                        strength=self.cfg.strength * (self.cfg.acquisition_speed / 3.0),
                        humanization=self.cfg.humanization
                    )
                else:
                    self.controller.reset()

                frame_count += 1
                now = time.time()
                if now - self._last_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    self._last_time = now

                self.stats["dets"] = 1 if best else 0
                self.stats["fps"] = int(fps)
                self.stats["engine"] = engine_used

                self.frame_ready.emit(vis, dict(self.stats))

            except Exception as e:
                self.status_update.emit(f"Error: {str(e)[:70]}")
                time.sleep(0.02)

            elapsed = time.time() - t0
            target = 1.0 / max(30, min(self.cfg.fps_limit, 200))
            if elapsed < target:
                time.sleep(target - elapsed)

        if self.capture:
            self.capture.close()
        self.status_update.emit("Engine stopped")

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScriptedElite AI")
        self.resize(1180, 760)

        if LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(LOGO_PATH)))

        self.cfg = AppConfig.load()
        self.worker = None
        self.worker_thread = None
        self.listener = None
        self.recoil = RecoilMatrix()
        self.ads_held = False

        self._setup_ui()
        self._apply_initial_config()
        self._setup_hotkeys_and_gating()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(10, 6, 10, 6)

        # Header
        header = QHBoxLayout()
        if LOGO_PATH.exists():
            header.addWidget(QLabel().setPixmap(QPixmap(str(LOGO_PATH)).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)) or QLabel())
        title = QLabel("SCRIPTED ELITE  •  PS5 AI")
        title.setStyleSheet(f"font-size: 19px; font-weight: 800; color: {ACCENT_BLUE};")
        header.addWidget(title)
        header.addStretch()
        main.addLayout(header)

        # Toggle
        self.aim_btn = QPushButton("AIM DISABLED — HOLD ADS KEY TO ACTIVATE")
        self.aim_btn.setObjectName("aimToggle")
        self.aim_btn.setProperty("enabled", "false")
        self.aim_btn.clicked.connect(self.toggle_aim)
        main.addWidget(self.aim_btn)

        tabs = QTabWidget()
        main.addWidget(tabs, 1)

        # CAPTURE
        cap = QWidget()
        cl = QFormLayout(cap)
        self.cap_mode = QComboBox(); self.cap_mode.addItems(["chiaki", "capture_card", "desktop"])
        cl.addRow("Source", self.cap_mode)
        self.chiaki_win = QLineEdit()
        cl.addRow("Chiaki Window", self.chiaki_win)
        self.cap_dev = QSpinBox(); self.cap_dev.setRange(0, 10)
        cl.addRow("Capture Card #", self.cap_dev)
        tabs.addTab(cap, "Capture")

        # DETECTION
        det = QWidget()
        dl = QFormLayout(det)
        self.engine = QComboBox(); self.engine.addItems(["yolo", "color", "hybrid"])
        dl.addRow("Engine", self.engine)
        self.conf = QSlider(Qt.Horizontal); self.conf.setRange(15, 80)
        dl.addRow("YOLO Confidence", self.conf)
        self.color_off = QDoubleSpinBox(); self.color_off.setRange(15, 90)
        dl.addRow("Color Head Offset", self.color_off)
        self.yolo_offset = QDoubleSpinBox(); self.yolo_offset.setRange(0.0, 0.5); self.yolo_offset.setSingleStep(0.01)
        dl.addRow("YOLO Target Offset (head=0.12)", self.yolo_offset)
        self.pred = QCheckBox("Linear Prediction Targeting")
        dl.addRow(self.pred)
        tabs.addTab(det, "Detection")

        # AIMING
        aim = QWidget()
        al = QFormLayout(aim)
        self.ads = QLineEdit()
        al.addRow("ADS Key (hold to gate)", self.ads)
        self.gate_cb = QCheckBox("Gated (only when ADS held)")
        al.addRow(self.gate_cb)
        self.smooth = QSlider(Qt.Horizontal); self.smooth.setRange(0, 80)
        al.addRow("Smoothing", self.smooth)
        self.strength = QSlider(Qt.Horizontal); self.strength.setRange(40, 220)
        al.addRow("Strength", self.strength)
        self.acq = QDoubleSpinBox(); self.acq.setRange(0.8, 5.8); self.acq.setSingleStep(0.1)
        al.addRow("Acquisition Speed", self.acq)
        self.calx = QSpinBox(); self.calx.setRange(-120, 120)
        self.caly = QSpinBox(); self.caly.setRange(-80, 80)
        al.addRow("Cal X", self.calx)
        al.addRow("Cal Y", self.caly)
        tabs.addTab(aim, "Aiming")

        # CRONUS
        cron = QWidget()
        crl = QVBoxLayout(cron)
        self.wep = QComboBox()
        self.wep.addItems(self.recoil.list_weapons() or ["AR-15","SMG"])
        crl.addWidget(QLabel("Weapon"))
        crl.addWidget(self.wep)
        self.rec_table = QTableWidget(3, 2)
        self.rec_table.setHorizontalHeaderLabels(["Vert", "Horiz"])
        crl.addWidget(self.rec_table)
        saveb = QPushButton("Save Recoil Profile")
        saveb.clicked.connect(self.save_recoil)
        crl.addWidget(saveb)
        tabs.addTab(cron, "Cronus Recoil")

        # Bottom controls
        bot = QHBoxLayout()
        self.status = QLabel("Engine: Stopped")
        bot.addWidget(self.status)
        self.stats = QLabel("Dets:0 FPS:0")
        bot.addWidget(self.stats)
        bot.addStretch()
        self.startb = QPushButton("▶ Start Engine")
        self.startb.clicked.connect(self.start_engine)
        self.stopb = QPushButton("■ Stop")
        self.stopb.clicked.connect(self.stop_engine)
        self.stopb.setEnabled(False)
        bot.addWidget(self.startb)
        bot.addWidget(self.stopb)
        main.addLayout(bot)

        # Preview
        pg = QGroupBox("LIVE PS5 FEED")
        pl = QVBoxLayout(pg)
        self.prev = QLabel("Start engine to see Chiaki / Capture Card feed + detections.")
        self.prev.setMinimumHeight(300)
        self.prev.setAlignment(Qt.AlignCenter)
        pl.addWidget(self.prev)
        self.show_prev = QCheckBox("Enable Preview")
        pl.addWidget(self.show_prev)
        main.addWidget(pg)

        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        self.sb.showMessage("ScriptedElite AI • PS5 • No key")

        # Wire all config widgets
        for w in [self.cap_mode, self.chiaki_win, self.cap_dev,
                  self.engine, self.conf, self.color_off, self.yolo_offset, self.pred,
                  self.ads, self.gate_cb, self.smooth, self.strength, self.acq,
                  self.calx, self.caly, self.show_prev, self.wep]:
            if hasattr(w, 'currentTextChanged'):
                w.currentTextChanged.connect(self.on_config_changed)
            elif hasattr(w, 'textChanged'):
                w.textChanged.connect(self.on_config_changed)
            elif hasattr(w, 'valueChanged'):
                w.valueChanged.connect(self.on_config_changed)
            elif hasattr(w, 'stateChanged'):
                w.stateChanged.connect(self.on_config_changed)
            elif hasattr(w, 'clicked'):
                w.clicked.connect(self.on_config_changed)

    def _apply_initial_config(self):
        self.cap_mode.setCurrentText(self.cfg.capture_mode)
        self.chiaki_win.setText(self.cfg.chiaki_window)
        self.cap_dev.setValue(self.cfg.capture_device)
        self.engine.setCurrentText(self.cfg.engine)
        self.conf.setValue(int(self.cfg.confidence * 100))
        self.color_off.setValue(self.cfg.color_head_offset)
        self.yolo_offset.setValue(self.cfg.target_offset)
        self.pred.setChecked(self.cfg.use_linear_prediction)
        self.ads.setText(self.cfg.ads_key)
        self.gate_cb.setChecked(self.cfg.gated_aim)
        self.smooth.setValue(int(self.cfg.smoothing * 100))
        self.strength.setValue(int(self.cfg.strength * 100))
        self.acq.setValue(self.cfg.acquisition_speed)
        self.calx.setValue(self.cfg.cal_x)
        self.caly.setValue(self.cfg.cal_y)
        self.show_prev.setChecked(self.cfg.show_preview)
        self.wep.setCurrentText(self.cfg.current_weapon)
        self.load_recoil_to_table()
        self.wep.currentTextChanged.connect(self.on_weapon_changed)

    def on_config_changed(self):
        self.cfg.capture_mode = self.cap_mode.currentText()
        self.cfg.chiaki_window = self.chiaki_win.text()
        self.cfg.capture_device = self.cap_dev.value()
        self.cfg.engine = self.engine.currentText()
        self.cfg.confidence = self.conf.value() / 100.0
        self.cfg.color_head_offset = self.color_off.value()
        self.cfg.target_offset = self.yolo_offset.value()
        self.cfg.use_linear_prediction = self.pred.isChecked()
        self.cfg.ads_key = self.ads.text()
        self.cfg.gated_aim = self.gate_cb.isChecked()
        self.cfg.smoothing = self.smooth.value() / 100.0
        self.cfg.strength = self.strength.value() / 100.0
        self.cfg.acquisition_speed = self.acq.value()
        self.cfg.cal_x = self.calx.value()
        self.cfg.cal_y = self.caly.value()
        self.cfg.show_preview = self.show_prev.isChecked()
        self.cfg.current_weapon = self.wep.currentText()

        if self.worker:
            self.worker.cfg = self.cfg

    def set_target_part(self, part, offset):
        self.cfg.target_part = part
        self.cfg.target_offset = offset
        self.conf.setValue(int(self.cfg.confidence*100))  # trigger update
        self.on_config_changed()

    def save_recoil(self):
        w = self.recoil.get(self.wep.currentText())
        v = []
        h = []
        for row in range(self.rec_table.rowCount()):
            try:
                v.append(int(self.rec_table.item(row, 0).text() or 0))
            except:
                v.append(0)
            try:
                h.append(int(self.rec_table.item(row, 1).text() or 0))
            except:
                h.append(0)
        w.vertical = v
        w.horizontal = h
        self.recoil.add_or_update(w)
        self.sb.showMessage(f"Recoil for {w.name} saved", 2000)

    def load_recoil_to_table(self):
        w = self.recoil.get(self.wep.currentText())
        self.rec_table.setRowCount(max(len(w.vertical), len(w.horizontal), 8))
        for i in range(self.rec_table.rowCount()):
            v = w.vertical[i] if i < len(w.vertical) else 0
            h = w.horizontal[i] if i < len(w.horizontal) else 0
            self.rec_table.setItem(i, 0, QTableWidgetItem(str(v)))
            self.rec_table.setItem(i, 1, QTableWidgetItem(str(h)))

    def on_weapon_changed(self):
        self.cfg.current_weapon = self.wep.currentText()
        self.load_recoil_to_table()
        self.on_config_changed()

    def toggle_aim(self):
        if not self.worker:
            self.start_engine()
            return
        new = not self.worker.aim_enabled
        self.worker.set_aim_enabled(new)
        if new:
            self.aim_btn.setText("AIM ENABLED — HOLD ADS")
            self.aim_btn.setProperty("enabled", "true")
        else:
            self.aim_btn.setText("AIM DISABLED — HOLD ADS KEY TO ACTIVATE")
            self.aim_btn.setProperty("enabled", "false")
        self.aim_btn.style().unpolish(self.aim_btn)
        self.aim_btn.style().polish(self.aim_btn)

    def start_engine(self):
        if self.worker and self.worker.running:
            return
        self.worker = AimbotWorker(self.cfg)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_update.connect(self.on_status)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker_thread.start()
        self.status.setText("Engine: Running")
        self.startb.setEnabled(False)
        self.stopb.setEnabled(True)

    def stop_engine(self):
        if self.worker:
            self.worker.stop()
        self.status.setText("Engine: Stopped")
        self.startb.setEnabled(True)
        self.stopb.setEnabled(False)
        self.prev.setText("Engine stopped.")

    @Slot(np.ndarray, dict)
    def on_frame(self, frame, stats):
        self.stats.setText(f"Dets:{stats.get('dets',0)} FPS:{stats.get('fps',0)} Eng:{stats.get('engine','?')} Gated:{stats.get('gated',False)}")
        if not self.cfg.show_preview:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.prev.size() * 0.98, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.prev.setPixmap(pix)

    @Slot(str)
    def on_status(self, msg):
        self.status.setText(msg)

    def _setup_hotkeys_and_gating(self):
        ads_k = (self.cfg.ads_key or "right").lower()
        toggle_k = (self.cfg.toggle_key or "f").lower()

        def on_kb_press(key):
            try:
                if hasattr(key, 'char') and key.char and key.char.lower() == toggle_k:
                    QTimer.singleShot(0, lambda: self.toggle_aim())
                if ads_k not in ['right', 'left', 'middle'] and hasattr(key, 'name') and key.name and key.name.lower() == ads_k:
                    if self.worker: QTimer.singleShot(0, lambda: self.worker.set_ads_held(True))
            except: pass

        def on_kb_release(key):
            try:
                if ads_k not in ['right', 'left', 'middle'] and hasattr(key, 'name') and key.name and key.name.lower() == ads_k:
                    if self.worker: QTimer.singleShot(0, lambda: self.worker.set_ads_held(False))
            except: pass

        self.kb_listener = keyboard.Listener(on_press=on_kb_press, on_release=on_kb_release)
        self.kb_listener.daemon = True
        self.kb_listener.start()

        # Mouse for ADS buttons
        def on_mouse_click(x, y, button, pressed):
            try:
                btn = str(button).split('.')[-1].lower()
                if btn == ads_k:
                    if self.worker:
                        QTimer.singleShot(0, lambda: self.worker.set_ads_held(pressed))
            except: pass

        self.mouse_listener = mouse.Listener(on_click=on_mouse_click)
        self.mouse_listener.daemon = True
        self.mouse_listener.start()

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
        try:
            if hasattr(self, 'kb_listener') and self.kb_listener: self.kb_listener.stop()
            if hasattr(self, 'mouse_listener') and self.mouse_listener: self.mouse_listener.stop()
        except: pass
        self.cfg.save()
        e.accept()


def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()