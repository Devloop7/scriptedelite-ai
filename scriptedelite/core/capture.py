"""
Video capture subsystem for ScriptedElite AI.
Supports:
- Any application window (user-selected from open windows)
- Chiaki Remote Play window (keyword match)
- Hardware capture card (cv2 VideoCapture)
- Full desktop / primary monitor
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import cv2
import mss
import numpy as np

try:
    import win32gui
    import win32con
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False
    win32gui = None  # type: ignore
    win32con = None  # type: ignore

try:
    import pygetwindow as gw
except Exception:
    gw = None  # type: ignore


CaptureMode = Literal["window", "chiaki", "capture_card", "desktop"]


@dataclass
class WindowInfo:
    """A capturable top-level window."""
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int

    @property
    def label(self) -> str:
        """UI-friendly label with size."""
        t = self.title if len(self.title) <= 60 else self.title[:57] + "..."
        return f"{t}  ({self.width}x{self.height})"


def list_open_windows(
    min_width: int = 160,
    min_height: int = 120,
    exclude_titles: Optional[List[str]] = None,
) -> List[WindowInfo]:
    """
    Enumerate visible top-level windows suitable as capture sources.
    Prefer win32gui (stable hwnd); fall back to pygetwindow.
    """
    exclude = [e.lower() for e in (exclude_titles or [
        "scriptedelite", "scripted elite", "program manager",
        "microsoft text input", "windows input experience",
        "nvidia geforce overlay", "nvidia shadowplay",
        "dwell window", "popuphost", "system tray",
    ])]
    results: List[WindowInfo] = []
    seen_hwnd: set = set()

    if HAS_WIN32 and win32gui is not None:
        def _cb(hwnd, _):
            try:
                if not win32gui.IsWindowVisible(hwnd):
                    return
                if win32gui.GetParent(hwnd) != 0:
                    return
                title = (win32gui.GetWindowText(hwnd) or "").strip()
                if not title:
                    return
                tl = title.lower()
                if any(x in tl for x in exclude):
                    return
                # Skip tool windows / owned popups when possible
                try:
                    ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                    if ex & win32con.WS_EX_TOOLWINDOW:
                        return
                except Exception:
                    pass
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    left, top, right, bottom = rect
                    width = right - left
                    height = bottom - top
                except Exception:
                    return
                if width < min_width or height < min_height:
                    return
                # Minimized / off-screen ghosts
                if width <= 0 or height <= 0:
                    return
                if hwnd in seen_hwnd:
                    return
                seen_hwnd.add(hwnd)
                results.append(WindowInfo(hwnd, title, left, top, width, height))
            except Exception:
                return

        try:
            win32gui.EnumWindows(_cb, None)
        except Exception as e:
            print(f"[Capture] EnumWindows failed: {e}")

    elif gw is not None:
        try:
            for w in gw.getAllWindows():
                title = (w.title or "").strip()
                if not title:
                    continue
                tl = title.lower()
                if any(x in tl for x in exclude):
                    continue
                try:
                    width, height = int(w.width), int(w.height)
                    left, top = int(w.left), int(w.top)
                except Exception:
                    continue
                if width < min_width or height < min_height:
                    continue
                hwnd = int(getattr(w, "_hWnd", 0) or 0)
                if hwnd and hwnd in seen_hwnd:
                    continue
                if hwnd:
                    seen_hwnd.add(hwnd)
                results.append(WindowInfo(hwnd, title, left, top, width, height))
        except Exception as e:
            print(f"[Capture] pygetwindow list failed: {e}")

    # Sort: larger windows first, then title
    results.sort(key=lambda w: (-(w.width * w.height), w.title.lower()))
    return results


class ScreenCapture:
    """Captures frames from a selected window, Chiaki, capture card, or desktop."""

    def __init__(
        self,
        mode: CaptureMode = "window",
        window_title: str = "",
        window_hwnd: int = 0,
        capture_device: int = 0,
        region_size: int = 0,
        refresh_bounds_sec: float = 0.35,
    ):
        """
        region_size:
          0  -> capture the full window (or full primary monitor for desktop)
          >0 -> center-crop a square of this size
        """
        self.mode = mode
        self.window_title = window_title or ""
        self.window_hwnd = int(window_hwnd or 0)
        self.capture_device = capture_device
        self.region_size = int(region_size) if region_size else 0
        self.refresh_bounds_sec = refresh_bounds_sec

        self.sct = mss.mss()
        self.cap: Optional[cv2.VideoCapture] = None
        self._bounds: Optional[Tuple[int, int, int, int]] = None  # left, top, width, height
        self._last_bounds_refresh = 0.0
        self._last_status = ""

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
        elif mode in ("window", "chiaki"):
            self._resolve_window(force=True)

    # ------------------------------------------------------------------
    # Window discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _client_rect_from_hwnd(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """Return (left, top, width, height) of the client area in screen coords."""
        if not HAS_WIN32 or not hwnd:
            return None
        try:
            if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
                return None
            # Client area (excludes title bar / borders) — better for game content
            cl = win32gui.GetClientRect(hwnd)
            cw, ch = cl[2] - cl[0], cl[3] - cl[1]
            if cw < 80 or ch < 80:
                # Fall back to outer rect
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                return left, top, right - left, bottom - top
            pt = win32gui.ClientToScreen(hwnd, (0, 0))
            return int(pt[0]), int(pt[1]), int(cw), int(ch)
        except Exception:
            return None

    @staticmethod
    def _outer_rect_from_hwnd(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        if not HAS_WIN32 or not hwnd:
            return None
        try:
            if not win32gui.IsWindow(hwnd):
                return None
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            return left, top, right - left, bottom - top
        except Exception:
            return None

    def _apply_bounds(self, left: int, top: int, width: int, height: int, label: str) -> bool:
        if width < 80 or height < 80:
            return False
        self._bounds = (int(left), int(top), int(width), int(height))
        self.width = int(width)
        self.height = int(height)
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.found_window = True
        status = f"{label} {width}x{height} @ ({left},{top})"
        if status != self._last_status:
            print(f"[Capture] {status}")
            self._last_status = status
        return True

    def _resolve_window(self, force: bool = False) -> bool:
        """Find bounds for window / chiaki modes."""
        now = time.time()
        if not force and (now - self._last_bounds_refresh) < self.refresh_bounds_sec:
            return self.found_window
        self._last_bounds_refresh = now

        # 1) Exact hwnd if still valid
        if self.window_hwnd:
            rect = self._client_rect_from_hwnd(self.window_hwnd)
            if rect:
                title = ""
                if HAS_WIN32:
                    try:
                        title = win32gui.GetWindowText(self.window_hwnd) or ""
                    except Exception:
                        title = self.window_title or "window"
                if self._apply_bounds(*rect, label=f"Window: '{title}'"):
                    if title:
                        self.window_title = title
                    return True

        # 2) Title match among open windows
        needle = (self.window_title or "").strip()
        if self.mode == "chiaki" and not needle:
            needle = "Chiaki"

        candidates = list_open_windows(min_width=200, min_height=150)
        best: Optional[WindowInfo] = None
        best_score = -1

        for w in candidates:
            sc = self._score_window(w.title, w.width, w.height, needle, chiaki=(self.mode == "chiaki"))
            if sc > best_score:
                best_score = sc
                best = w

        if best and best_score > 0:
            self.window_hwnd = best.hwnd
            rect = self._client_rect_from_hwnd(best.hwnd)
            if rect is None:
                rect = (best.left, best.top, best.width, best.height)
            if self._apply_bounds(*rect, label=f"Window: '{best.title}'"):
                self.window_title = best.title
                return True

        # 3) pygetwindow title search fallback
        if gw is not None and needle:
            try:
                matches = gw.getWindowsWithTitle(needle)
                for w in matches:
                    try:
                        ww, hh = int(w.width), int(w.height)
                        if ww < 200 or hh < 150:
                            continue
                        left, top = int(w.left), int(w.top)
                        hwnd = int(getattr(w, "_hWnd", 0) or 0)
                        if hwnd:
                            self.window_hwnd = hwnd
                            rect = self._client_rect_from_hwnd(hwnd)
                            if rect:
                                if self._apply_bounds(*rect, label=f"Window: '{w.title}'"):
                                    self.window_title = w.title or needle
                                    return True
                        if self._apply_bounds(left, top, ww, hh, label=f"Window: '{w.title}'"):
                            self.window_title = w.title or needle
                            return True
                    except Exception:
                        continue
            except Exception:
                pass

        self.found_window = False
        self._bounds = None
        tag = "window-missing"
        if self._last_status != tag:
            print(f"[Capture] Window matching '{needle or self.window_hwnd}' not found. Using desktop fallback.")
            self._last_status = tag
        return False

    def _score_window(
        self,
        title: str,
        width: int,
        height: int,
        needle: str,
        chiaki: bool = False,
    ) -> int:
        t = title.strip()
        tl = t.lower()
        n = (needle or "").strip().lower()

        if width < 160 or height < 120:
            return -1

        score = 0
        if n:
            if tl == n:
                score += 120
            elif tl.startswith(n):
                score += 100
            elif n in tl:
                if len(t) > 100 and len(n) < 4:
                    return -1
                score += 60
            else:
                # Token overlap
                tokens = [x for x in n.replace("-", " ").split() if x]
                if tokens and all(tok in tl for tok in tokens):
                    score += 40
                else:
                    return -1
        else:
            # No needle: only valid when hwnd already handled; weak size score
            score = 1

        if chiaki:
            if "chiaki" in tl:
                score += 50
            elif "ps remote play" in tl or "remote play" in tl:
                score += 40

        score += min(30, (width * height) // 100_000)
        return score

    def _open_capture_card(self):
        self.cap = cv2.VideoCapture(self.capture_device, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.capture_device)
        if self.cap.isOpened():
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
        Returns (BGR frame, left, top) of the captured region in screen coordinates.
        Frame-local crosshair is (frame_w//2, frame_h//2) before calibration.
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
                frame = frame[top: top + crop, left: left + crop]
                return frame, left, top
            return frame, 0, 0

        if self.mode in ("window", "chiaki"):
            self._resolve_window()
            if self._bounds:
                l, t, ww, hh = self._bounds
                if crop and crop > 0:
                    cx, cy = l + ww // 2, t + hh // 2
                    half = crop // 2
                    left = max(l, cx - half)
                    top = max(t, cy - half)
                    width = min(crop, l + ww - left)
                    height = min(crop, t + hh - top)
                else:
                    left, top, width, height = l, t, ww, hh

                if width < 2 or height < 2:
                    return np.zeros((480, 640, 3), dtype=np.uint8), left, top

                region = {
                    "left": int(left), "top": int(top),
                    "width": int(width), "height": int(height),
                }
                img = np.array(self.sct.grab(region))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                self.width, self.height = frame.shape[1], frame.shape[0]
                self.center_x, self.center_y = self.width // 2, self.height // 2
                return frame, int(left), int(top)

        # Desktop fallback
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

        region = {
            "left": int(left), "top": int(top),
            "width": int(width), "height": int(height),
        }
        img = np.array(self.sct.grab(region))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.width, self.height = frame.shape[1], frame.shape[0]
        self.center_x, self.center_y = self.width // 2, self.height // 2
        return frame, int(left), int(top)

    def get_frame_center(self) -> Tuple[int, int]:
        return self.center_x, self.center_y

    def get_screen_center(self) -> Tuple[int, int]:
        return self.get_frame_center()

    def get_status(self) -> str:
        if self.mode in ("window", "chiaki"):
            if self.found_window and self._bounds:
                l, t, w, h = self._bounds
                name = (self.window_title or "window")[:40]
                return f"{name} {w}x{h}"
            return "Window not found"
        if self.mode == "capture_card":
            ok = self.cap is not None and self.cap.isOpened()
            return f"Capture card #{self.capture_device} {'OK' if ok else 'FAIL'}"
        return f"Desktop {self._desktop_w}x{self._desktop_h}"

    def set_mode(self, mode: str, **kwargs):
        prev = self.mode
        self.mode = mode  # type: ignore
        if "window_title" in kwargs and kwargs["window_title"] is not None:
            self.window_title = kwargs["window_title"] or ""
        if "window_hwnd" in kwargs and kwargs["window_hwnd"] is not None:
            self.window_hwnd = int(kwargs["window_hwnd"] or 0)
        if mode in ("window", "chiaki"):
            self._resolve_window(force=True)
        elif mode == "capture_card":
            self.capture_device = kwargs.get("device", self.capture_device)
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self._open_capture_card()
        elif mode == "desktop":
            self.found_window = False
            self._bounds = None
        if prev != mode:
            print(f"[Capture] Mode → {mode}")

    def set_window(self, title: str = "", hwnd: int = 0):
        """Select a specific window as capture source."""
        self.window_title = title or ""
        self.window_hwnd = int(hwnd or 0)
        if self.mode not in ("window", "chiaki"):
            self.mode = "window"
        self._resolve_window(force=True)

    def close(self):
        try:
            if self.cap:
                self.cap.release()
            self.sct.close()
        except Exception:
            pass
