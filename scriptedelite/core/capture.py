"""
PS5 Video Capture Subsystem for ScriptedElite AI.
Supports:
- Chiaki Remote Play window capture (preferred for PS5 low-latency)
- Hardware Capture Card (cv2 VideoCapture device)
- Fallback full desktop / primary monitor region
"""
from __future__ import annotations

import time
from typing import Literal, Optional, Tuple

import cv2
import mss
import numpy as np
import pygetwindow as gw


class ScreenCapture:
    """Captures frames from Chiaki, a capture card, or the desktop."""

    def __init__(
        self,
        mode: Literal["chiaki", "capture_card", "desktop"] = "chiaki",
        window_title: str = "Chiaki",
        capture_device: int = 0,
        region_size: int = 0,
        refresh_bounds_sec: float = 0.5,
    ):
        """
        region_size:
          0  -> capture the full Chiaki window (or full primary monitor for desktop)
          >0 -> center-crop a square of this size (legacy / FOV crop mode)
        """
        self.mode = mode
        self.window_title = window_title or "Chiaki"
        self.capture_device = capture_device
        self.region_size = int(region_size) if region_size else 0
        self.refresh_bounds_sec = refresh_bounds_sec

        self.sct = mss.mss()
        self.cap: Optional[cv2.VideoCapture] = None
        self._chiaki_bounds: Optional[Tuple[int, int, int, int]] = None  # left, top, width, height
        self._last_bounds_refresh = 0.0
        self._last_status = ""

        # Logical stream size (updated every grab)
        self.width = 1920
        self.height = 1080
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.found_window = False

        mon = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
        self._desktop_w = mon["width"]
        self._desktop_h = mon["height"]
        self._desktop_l = mon["left"]
        self._desktop_t = mon["top"]

        if mode == "capture_card":
            self._open_capture_card()
        elif mode == "chiaki":
            self._find_chiaki_window(force=True)

    # ------------------------------------------------------------------
    # Window discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _is_noise_window(title: str) -> bool:
        """Skip terminals, browsers, IDEs, and our own UI."""
        t = title.lower()
        noise = (
            "scriptedelite", "scripted elite", "grok", "powershell", "cmd.exe",
            "windows terminal", "visual studio", "pycharm", "vscode", "code -",
            "chrome", "firefox", "edge", "discord", "spotify", "explorer",
            "smoke-test", "python", "notepad",
        )
        return any(n in t for n in noise)

    def _score_window(self, title: str, width: int, height: int, needle: str) -> int:
        """Higher score = better Chiaki candidate."""
        t = title.strip()
        tl = t.lower()
        n = needle.lower()
        if self._is_noise_window(t):
            return -1
        if width < 320 or height < 240:
            return -1

        score = 0
        # Exact / starts-with are best (real Chiaki titles: "Chiaki", "chiaki-ng", …)
        if tl == n or tl.startswith(n):
            score += 100
        elif tl.endswith(n) or f" {n}" in tl or f"{n} " in tl:
            score += 60
        elif n in tl:
            # Weak: title merely contains the needle somewhere (avoid terminal cmdlines)
            if len(t) > 80:
                return -1
            score += 20
        else:
            # Keyword fallbacks
            if "chiaki" in tl:
                score += 50
            elif "ps remote play" in tl or "remote play" in tl:
                score += 40
            else:
                return -1

        # Prefer larger (likely fullscreen game stream)
        score += min(30, (width * height) // 100_000)
        return score

    def _find_chiaki_window(self, force: bool = False) -> bool:
        now = time.time()
        if not force and (now - self._last_bounds_refresh) < self.refresh_bounds_sec:
            return self.found_window

        self._last_bounds_refresh = now
        title = (self.window_title or "Chiaki").strip()
        try:
            scored = []
            for w in gw.getAllWindows():
                wt = (w.title or "").strip()
                if not wt:
                    continue
                try:
                    ww, hh = int(w.width), int(w.height)
                except Exception:
                    continue
                sc = self._score_window(wt, ww, hh, title)
                if sc > 0:
                    scored.append((sc, w))

            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                w = scored[0][1]
                left, top = int(w.left), int(w.top)
                width, height = int(w.width), int(w.height)
                if width > 200 and height > 200:
                    self._chiaki_bounds = (left, top, width, height)
                    self.width = width
                    self.height = height
                    self.center_x = width // 2
                    self.center_y = height // 2
                    self.found_window = True
                    status = f"Chiaki: '{w.title}' {width}x{height} @ ({left},{top}) score={scored[0][0]}"
                    if status != self._last_status:
                        print(f"[Capture] {status}")
                        self._last_status = status
                    return True

            self.found_window = False
            self._chiaki_bounds = None
            if self._last_status != "chiaki-missing":
                print(f"[Capture] Window matching '{title}' not found. Using desktop fallback.")
                self._last_status = "chiaki-missing"
        except Exception as e:
            self.found_window = False
            print(f"[Capture] Error finding Chiaki window: {e}")
        return False

    def _open_capture_card(self):
        self.cap = cv2.VideoCapture(self.capture_device, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.capture_device)
        if self.cap.isOpened():
            # Prefer 1080p if the device supports it
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            print(f"[Capture] Capture card device {self.capture_device} opened.")
        else:
            print(f"[Capture] Warning: Could not open capture device {self.capture_device}")

    # ------------------------------------------------------------------
    # Grab
    # ------------------------------------------------------------------
    def grab_region(self, size: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
        """
        Returns (BGR frame, left, top) where left/top are the top-left of the
        captured region in screen coordinates.
        Frame-local crosshair is always at (frame_w//2, frame_h//2) unless
        calibration offsets are applied by the caller.
        """
        crop = self.region_size if size is None else size

        if self.mode == "capture_card" and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return np.zeros((720, 1280, 3), dtype=np.uint8), 0, 0
            h, w = frame.shape[:2]
            self.width, self.height = w, h
            self.center_x, self.center_y = w // 2, h // 2
            if crop and crop > 0:
                half = crop // 2
                cx, cy = w // 2, h // 2
                left = max(0, cx - half)
                top = max(0, cy - half)
                frame = frame[top : top + crop, left : left + crop]
                return frame, left, top
            return frame, 0, 0

        if self.mode == "chiaki":
            self._find_chiaki_window()
            if self._chiaki_bounds:
                l, t, ww, hh = self._chiaki_bounds
                if crop and crop > 0:
                    cx, cy = l + ww // 2, t + hh // 2
                    half = crop // 2
                    left = max(l, cx - half)
                    top = max(t, cy - half)
                    width = min(crop, l + ww - left)
                    height = min(crop, t + hh - top)
                else:
                    # Full Chiaki window
                    left, top, width, height = l, t, ww, hh

                if width < 2 or height < 2:
                    return np.zeros((480, 640, 3), dtype=np.uint8), left, top

                region = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
                img = np.array(self.sct.grab(region))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                self.width, self.height = frame.shape[1], frame.shape[0]
                self.center_x, self.center_y = self.width // 2, self.height // 2
                return frame, int(left), int(top)

        # Desktop fallback (primary monitor center crop, or full monitor)
        if crop and crop > 0:
            cx = self._desktop_l + self._desktop_w // 2
            cy = self._desktop_t + self._desktop_h // 2
            half = crop // 2
            left = max(self._desktop_l, cx - half)
            top = max(self._desktop_t, cy - half)
            width = height = crop
        else:
            left = self._desktop_l
            top = self._desktop_t
            width = self._desktop_w
            height = self._desktop_h

        region = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
        img = np.array(self.sct.grab(region))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.width, self.height = frame.shape[1], frame.shape[0]
        self.center_x, self.center_y = self.width // 2, self.height // 2
        return frame, int(left), int(top)

    def get_frame_center(self) -> Tuple[int, int]:
        """Crosshair in frame-local coordinates (before calibration)."""
        return self.center_x, self.center_y

    def get_screen_center(self) -> Tuple[int, int]:
        """Legacy alias — returns frame-local center."""
        return self.get_frame_center()

    def get_status(self) -> str:
        if self.mode == "chiaki":
            if self.found_window and self._chiaki_bounds:
                l, t, w, h = self._chiaki_bounds
                return f"Chiaki {w}x{h}"
            return "Chiaki window not found"
        if self.mode == "capture_card":
            ok = self.cap is not None and self.cap.isOpened()
            return f"Capture card #{self.capture_device} {'OK' if ok else 'FAIL'}"
        return f"Desktop {self._desktop_w}x{self._desktop_h}"

    def set_mode(self, mode: str, **kwargs):
        self.mode = mode
        if mode == "chiaki":
            self.window_title = kwargs.get("window_title", self.window_title) or "Chiaki"
            self._find_chiaki_window(force=True)
        elif mode == "capture_card":
            self.capture_device = kwargs.get("device", self.capture_device)
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self._open_capture_card()

    def close(self):
        try:
            if self.cap:
                self.cap.release()
            self.sct.close()
        except Exception:
            pass
