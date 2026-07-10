"""
Engine worker: capture → detect → sticky lock → virtual stick.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
from PySide6.QtCore import QObject, Signal, Slot

from core.config import AppConfig
from core.capture import ScreenCapture
from core.detector import YOLODetector
from core.color_detector import ColorSignatureDetector
from core.controller import AimController
from core.tracker import StickyTargetTracker, StickyColorTracker

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "assets" / "model.pt"


class AimWorker(QObject):
    frame_ready = Signal(object, dict)
    status_update = Signal(str)

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self.capture: Optional[ScreenCapture] = None
        self.yolo: Optional[YOLODetector] = None
        self.color_det = ColorSignatureDetector()
        self.color_det.set_enemy_color(getattr(cfg, "enemy_color", "#FF00FA"))
        self.color_det.set_tolerance(getattr(cfg, "color_tolerance", 55.0))
        self.controller = AimController()
        self.controller.set_stick_response(getattr(cfg, "stick_response", 55.0))
        self.controller.set_adaptive_response(getattr(cfg, "adaptive_stick", False))
        hold = max(12, int(getattr(cfg, "lock_hold_frames", 28)))
        self.tracker = StickyTargetTracker(
            max_miss_frames=hold,
            match_iou=0.05,
            match_center_px=180.0,
            hold_expand=2.2,
        )
        self.color_tracker = StickyColorTracker(
            max_miss_frames=max(14, hold - 6), stick_radius=200.0,
        )
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
        self.tracker.max_miss_frames = hold
        self.color_tracker.max_miss_frames = max(14, hold - 6)

    def _sync_controller_cfg(self):
        self.controller.set_stick_response(getattr(self.cfg, "stick_response", 55.0))
        self.controller.set_adaptive_response(getattr(self.cfg, "adaptive_stick", False))

    @Slot(bool)
    def set_aim_enabled(self, enabled: bool):
        self.aim_enabled = enabled
        self.controller.set_enabled(enabled)
        if not enabled:
            self.controller.reset()
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
                self._sync_controller_cfg()
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
                    sticky_live = self.tracker.active
                    conf_acq = self.cfg.confidence
                    # Slightly lower conf while holding so weak frames keep identity
                    conf_floor = max(0.18, conf_acq - 0.12) if sticky_live else conf_acq

                    dets = self.yolo.detect(
                        frame,
                        conf_threshold=conf_acq,
                        use_track=True,
                        fov_center=(screen_cx, screen_cy),
                        fov_radius=float(zone_r * (2.2 if sticky_live else 1.15)),
                        max_det=12,
                        conf_floor=conf_floor,
                    )
                    # On acquire, filter by confidence more strictly
                    if not sticky_live:
                        dets = [d for d in dets if d.conf >= conf_acq]

                    pred_lead = 0.0
                    if self.cfg.use_linear_prediction:
                        pred_lead = float(self.cfg.prediction_strength)

                    best = self.tracker.update(
                        dets,
                        (screen_cx, screen_cy),
                        max_dist,
                        offset,
                        priority=self.cfg.target_priority,
                        shake_reduction=self.cfg.shake_reduction,
                        prediction_lead=pred_lead if pred_lead > 0 else 0.0,
                    )
                    if self.cfg.draw_boxes:
                        vis = self.yolo.draw_detections(frame, dets, best, offset)
                    else:
                        vis = frame.copy()
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
                    self.controller.move_to_target(
                        best["dx"], best["dy"],
                        smoothing=self.cfg.smoothing,
                        strength=self.cfg.strength,
                        speed=self.cfg.acquisition_speed / 3.0,
                        humanization=self.cfg.humanization,
                        shake_reduction=self.cfg.shake_reduction,
                    )
                else:
                    self.controller.release_stick()
                    if not activation and not best:
                        self.tracker.reset()
                        self.color_tracker.reset()
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
                    "coasting": bool(best and best.get("coasting")),
                    "res": f"{fw}x{fh}",
                    "age": int(best.get("age", 0)) if best else 0,
                    "track_id": best.get("track_id") if best else None,
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
