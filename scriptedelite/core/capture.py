"""
PS5 Video Capture Subsystem for ScriptedElite AI.
Supports:
- Chiaki Remote Play window capture (preferred for PS5 low-latency)
- Hardware Capture Card (cv2 VideoCapture device)
- Fallback full desktop region
"""
import mss
import numpy as np
import cv2
import pygetwindow as gw
from typing import Optional, Tuple, Literal

class ScreenCapture:
    def __init__(self, 
                 mode: Literal["chiaki", "capture_card", "desktop"] = "chiaki",
                 window_title: str = "Chiaki",
                 capture_device: int = 0,
                 region_size: int = 720):
        self.mode = mode
        self.window_title = window_title
        self.capture_device = capture_device
        self.region_size = region_size
        self.sct = mss.mss()
        self.cap: Optional[cv2.VideoCapture] = None
        self._chiaki_bounds = None
        self.width = 1920
        self.height = 1080
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        if mode == "capture_card":
            self.cap = cv2.VideoCapture(capture_device)
            if not self.cap.isOpened():
                print(f"[Capture] Warning: Could not open capture device {capture_device}")
        elif mode == "chiaki":
            self._find_chiaki_window()

    def _find_chiaki_window(self):
        try:
            wins = gw.getWindowsWithTitle(self.window_title)
            if wins:
                w = wins[0]
                self._chiaki_bounds = (w.left, w.top, w.width, w.height)
                print(f"[Capture] Found Chiaki window: {self._chiaki_bounds}")
            else:
                print(f"[Capture] Chiaki window '{self.window_title}' not found. Falling back to desktop center.")
        except Exception as e:
            print(f"[Capture] Error finding Chiaki window: {e}")

    def grab_region(self, size: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
        """Returns BGR frame + (left, top) of captured region relative to screen."""
        if size is None:
            size = self.region_size

        if self.mode == "capture_card" and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                # Center crop if needed
                cx, cy = w // 2, h // 2
                half = size // 2
                left = max(0, cx - half)
                top = max(0, cy - half)
                frame = frame[top:top+size, left:left+size]
                return frame, left, top
            else:
                # fallback black frame
                return np.zeros((size, size, 3), dtype=np.uint8), 0, 0

        elif self.mode == "chiaki" and self._chiaki_bounds:
            l, t, ww, hh = self._chiaki_bounds
            cx, cy = l + ww // 2, t + hh // 2
            half = size // 2
            left = max(l, cx - half)
            top = max(t, cy - half)
            region = {"left": left, "top": top, "width": min(size, ww), "height": min(size, hh)}
            img = np.array(self.sct.grab(region))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return frame, left, top

        else:
            # Desktop center fallback
            cx, cy = self.center_x, self.center_y
            half = size // 2
            left = max(0, cx - half)
            top = max(0, cy - half)
            region = {"left": left, "top": top, "width": size, "height": size}
            img = np.array(self.sct.grab(region))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return frame, left, top

    def get_screen_center(self) -> Tuple[int, int]:
        return self.center_x, self.center_y

    def set_mode(self, mode: str, **kwargs):
        self.mode = mode
        if mode == "chiaki":
            self.window_title = kwargs.get("window_title", self.window_title)
            self._find_chiaki_window()
        elif mode == "capture_card":
            self.capture_device = kwargs.get("device", self.capture_device)
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.capture_device)

    def close(self):
        try:
            if self.cap:
                self.cap.release()
            self.sct.close()
        except:
            pass
