"""
Controller Input Emulation + Physical Button Polling.
- Output: vgamepad (ViGEm) virtual Xbox 360 gamepad → right stick for aim
- Input:  XInput physical pad triggers/buttons for activation (L2/R2/LT/RT)
"""
from __future__ import annotations

import ctypes
import random
import time
from typing import Optional, Tuple

try:
    import vgamepad as vg
except Exception:
    vg = None  # type: ignore


# ---------------------------------------------------------------------------
# XInput (read physical Xbox / XInput-compatible pads for activation)
# ---------------------------------------------------------------------------
class XInputReader:
    """Lightweight XInput poller for triggers and face buttons."""

    ERROR_SUCCESS = 0
    ERROR_DEVICE_NOT_CONNECTED = 1167

    class XINPUT_GAMEPAD(ctypes.Structure):
        _fields_ = [
            ("wButtons", ctypes.c_ushort),
            ("bLeftTrigger", ctypes.c_ubyte),
            ("bRightTrigger", ctypes.c_ubyte),
            ("sThumbLX", ctypes.c_short),
            ("sThumbLY", ctypes.c_short),
            ("sThumbRX", ctypes.c_short),
            ("sThumbRY", ctypes.c_short),
        ]

    class XINPUT_STATE(ctypes.Structure):
        _fields_ = [
            ("dwPacketNumber", ctypes.c_ulong),
            ("Gamepad", ctypes.c_ubyte * 0),  # placeholder, replaced below
        ]

    # Rebuild with nested gamepad
    class _XINPUT_STATE(ctypes.Structure):
        pass

    _XINPUT_STATE._fields_ = [
        ("dwPacketNumber", ctypes.c_ulong),
        ("Gamepad", XINPUT_GAMEPAD),
    ]

    # Button bitmasks
    BTN = {
        "dpad_up": 0x0001,
        "dpad_down": 0x0002,
        "dpad_left": 0x0004,
        "dpad_right": 0x0008,
        "start": 0x0010,
        "back": 0x0020,
        "ls": 0x0040,
        "rs": 0x0080,
        "lb": 0x0100,
        "rb": 0x0200,
        "a": 0x1000,
        "b": 0x2000,
        "x": 0x4000,
        "y": 0x8000,
    }

    def __init__(self, user_index: int = 0, trigger_threshold: int = 30):
        self.user_index = user_index
        self.trigger_threshold = trigger_threshold
        self._dll = None
        self.connected = False
        for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
            try:
                self._dll = ctypes.WinDLL(name)
                break
            except OSError:
                continue
        if self._dll is None:
            print("[Controller] XInput DLL not found — physical trigger gating unavailable.")

    def read(self) -> Optional[dict]:
        if self._dll is None:
            return None
        state = self._XINPUT_STATE()
        result = self._dll.XInputGetState(self.user_index, ctypes.byref(state))
        if result != self.ERROR_SUCCESS:
            self.connected = False
            return None
        self.connected = True
        g = state.Gamepad
        buttons = int(g.wButtons)
        return {
            "lt": int(g.bLeftTrigger),
            "rt": int(g.bRightTrigger),
            "buttons": buttons,
            "lb": bool(buttons & self.BTN["lb"]),
            "rb": bool(buttons & self.BTN["rb"]),
            "a": bool(buttons & self.BTN["a"]),
            "b": bool(buttons & self.BTN["b"]),
            "x": bool(buttons & self.BTN["x"]),
            "y": bool(buttons & self.BTN["y"]),
            "ls": bool(buttons & self.BTN["ls"]),
            "rs": bool(buttons & self.BTN["rs"]),
        }

    def is_activation_held(self, button: str) -> bool:
        """
        button: one of lt, rt, l2, r2, lb, rb, l1, r1, a, b, x, y, ls, rs
        L2/LT and R2/RT are aliases (PlayStation / Xbox naming).
        """
        state = self.read()
        if state is None:
            return False
        key = (button or "").strip().lower()
        # Trigger aliases
        if key in ("lt", "l2", "left_trigger", "aim_l2"):
            return state["lt"] >= self.trigger_threshold
        if key in ("rt", "r2", "right_trigger", "aim_r2"):
            return state["rt"] >= self.trigger_threshold
        if key in ("lb", "l1"):
            return state["lb"]
        if key in ("rb", "r1"):
            return state["rb"]
        if key in ("a", "cross", "×"):
            return state["a"]
        if key in ("b", "circle"):
            return state["b"]
        if key in ("x", "square"):
            return state["x"]
        if key in ("y", "triangle"):
            return state["y"]
        if key == "ls":
            return state["ls"]
        if key == "rs":
            return state["rs"]
        return False


class AimController:
    """Translates pixel deltas into virtual right-stick aim output."""

    def __init__(self):
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.enabled = False
        self.gamepad = None
        self.xinput = XInputReader()
        self._stick_centered = True

        if vg is not None:
            try:
                self.gamepad = vg.VX360Gamepad()
                self._center_sticks()
                print("[Controller] Virtual Xbox 360 gamepad initialized (ViGEm).")
            except Exception as e:
                print(f"[Controller] Warning: Could not init vgamepad ({e}).")
                self.gamepad = None
        else:
            print("[Controller] vgamepad not installed.")

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled:
            self.reset()

    def _center_sticks(self):
        if not self.gamepad:
            return
        self.gamepad.right_joystick(x_value=0, y_value=0)
        self.gamepad.update()
        self._stick_centered = True

    def move_to_target(
        self,
        dx: float,
        dy: float,
        smoothing: float = 0.35,
        strength: float = 1.0,
        speed: float = 1.0,
        humanization: float = 0.0,
        max_stick: float = 0.92,
        pixel_scale: float = 380.0,
    ):
        """
        dx/dy  : pixel offset from crosshair to aim point (frame-local)
        smoothing : 0..1  higher = more inertia / smoother (less snappy)
        strength  : overall gain multiplier
        speed     : acquisition / response multiplier (tracking speed)
        """
        if not self.enabled:
            return

        # Clamp parameters
        smoothing = max(0.0, min(0.95, float(smoothing)))
        strength = max(0.05, float(strength))
        speed = max(0.1, float(speed))

        if humanization > 0:
            j = random.uniform(-humanization, humanization)
            dx += j
            dy += j * 0.65

        # Strength + speed → target delta
        target_dx = dx * strength * speed
        target_dy = dy * strength * speed

        # Exponential smoothing (EMA). higher smoothing keeps more of last value.
        alpha = 1.0 - smoothing
        smooth_dx = target_dx * alpha + self.last_dx * smoothing
        smooth_dy = target_dy * alpha + self.last_dy * smoothing

        # Adaptive scale: closer targets → finer control (reduces overshoot)
        dist = (dx * dx + dy * dy) ** 0.5
        adaptive = 1.0
        if dist < 40:
            adaptive = 0.45 + 0.55 * (dist / 40.0)
        elif dist < 120:
            adaptive = 0.75 + 0.25 * ((dist - 40) / 80.0)

        scale = max(120.0, float(pixel_scale))
        stick_x = max(-max_stick, min(max_stick, (smooth_dx / scale) * adaptive))
        stick_y = max(-max_stick, min(max_stick, (smooth_dy / scale) * adaptive))

        # Soft deadzone — only zero when truly tiny (avoids micro-jitter spin)
        if abs(stick_x) < 0.02:
            stick_x = 0.0
        if abs(stick_y) < 0.02:
            stick_y = 0.0

        if self.gamepad:
            rx = int(stick_x * 32767)
            ry = int(-stick_y * 32767)  # invert Y for look-up
            self.gamepad.right_joystick(x_value=rx, y_value=ry)
            self.gamepad.update()
            self._stick_centered = abs(rx) < 200 and abs(ry) < 200
        else:
            try:
                import pydirectinput
                ix = int(round(stick_x * 10))
                iy = int(round(stick_y * 10))
                if ix or iy:
                    pydirectinput.moveRel(ix, iy, relative=True)
            except Exception:
                pass

        self.last_dx = smooth_dx
        self.last_dy = smooth_dy

    def is_activation_held(self, button: str) -> bool:
        """Poll physical controller for the configured activation button."""
        return self.xinput.is_activation_held(button)

    def physical_connected(self) -> bool:
        self.xinput.read()
        return self.xinput.connected

    def reset(self):
        self.last_dx = 0.0
        self.last_dy = 0.0
        if self.gamepad and not self._stick_centered:
            self._center_sticks()
        elif self.gamepad:
            # Always force center when explicitly reset
            self._center_sticks()
