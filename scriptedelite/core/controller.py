"""
Controller Input Emulation + Physical Button Polling.
- Output: vgamepad (ViGEm) virtual Xbox 360 gamepad → right stick for aim
- Input:  XInput physical pad triggers/buttons for activation (L2/R2/LT/RT)

Critical control fact
--------------------
Right-stick value is camera *angular rate*, not aim position.
Any non-zero stick keeps moving the crosshair. Stable lock requires:
  • stick = 0 when on target
  • stick magnitude strictly limited by remaining pixel error (no overshoot)
  • filtered error (not raw YOLO jitter)
"""
from __future__ import annotations

import ctypes
import math
import random
from typing import Optional

try:
    import vgamepad as vg
except Exception:
    vg = None  # type: ignore


# ---------------------------------------------------------------------------
# XInput
# ---------------------------------------------------------------------------
class XInputReader:
    ERROR_SUCCESS = 0

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

    class _XINPUT_STATE(ctypes.Structure):
        pass

    _XINPUT_STATE._fields_ = [
        ("dwPacketNumber", ctypes.c_ulong),
        ("Gamepad", XINPUT_GAMEPAD),
    ]

    BTN = {
        "dpad_up": 0x0001, "dpad_down": 0x0002, "dpad_left": 0x0004, "dpad_right": 0x0008,
        "start": 0x0010, "back": 0x0020, "ls": 0x0040, "rs": 0x0080,
        "lb": 0x0100, "rb": 0x0200, "a": 0x1000, "b": 0x2000, "x": 0x4000, "y": 0x8000,
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
            "lt": int(g.bLeftTrigger), "rt": int(g.bRightTrigger), "buttons": buttons,
            "lb": bool(buttons & self.BTN["lb"]), "rb": bool(buttons & self.BTN["rb"]),
            "a": bool(buttons & self.BTN["a"]), "b": bool(buttons & self.BTN["b"]),
            "x": bool(buttons & self.BTN["x"]), "y": bool(buttons & self.BTN["y"]),
            "ls": bool(buttons & self.BTN["ls"]), "rs": bool(buttons & self.BTN["rs"]),
        }

    def is_activation_held(self, button: str) -> bool:
        state = self.read()
        if state is None:
            return False
        key = (button or "").strip().lower()
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
    """
    Stable aim controller.

    Pipeline per frame
    ------------------
    1. Filter error (EMA) — absorbs residual aim-point noise
    2. Deadzone + hysteresis — stick fully released on aim point
    3. Proportional command with *error-fraction cap* — mathematically
       limits how far one frame can move so we cannot fly past the target
    4. Closing-in damping — if error is shrinking, cut gain further
    5. Axis-independent sign brake — kills L/R hunting on overshoot
    """

    # On-target hold (pixels). Wide enough to absorb YOLO wobble after filtering.
    LOCK_IN_PX = 6.0
    LOCK_OUT_PX = 14.0

    # Error filter (separate from user "smoothing" which still affects response)
    ERR_FILTER = 0.55

    # Conservative estimate: full stick (1.0) moves this many px per engine frame
    # at typical Chiaki/game look sens. Used only as a safety cap, not absolute.
    PX_PER_FULL_STICK = 70.0

    # Never close more than this fraction of remaining error in one frame
    MAX_ERROR_FRAC = 0.40

    def __init__(self):
        self._fx = 0.0          # filtered dx
        self._fy = 0.0          # filtered dy
        self._prev_fx = 0.0
        self._prev_fy = 0.0
        self._last_stick_x = 0.0
        self._last_stick_y = 0.0
        self._locked = False
        self._filter_seeded = False
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
        self._last_stick_x = 0.0
        self._last_stick_y = 0.0
        if not self.gamepad:
            self._stick_centered = True
            return
        self.gamepad.right_joystick(x_value=0, y_value=0)
        self.gamepad.update()
        self._stick_centered = True

    def _apply_stick(self, stick_x: float, stick_y: float):
        if abs(stick_x) < 1e-5:
            stick_x = 0.0
        if abs(stick_y) < 1e-5:
            stick_y = 0.0

        if self.gamepad:
            rx = int(max(-1.0, min(1.0, stick_x)) * 32767)
            ry = int(max(-1.0, min(1.0, -stick_y)) * 32767)
            if abs(rx) < 200:
                rx = 0
            if abs(ry) < 200:
                ry = 0
            self.gamepad.right_joystick(x_value=rx, y_value=ry)
            self.gamepad.update()
            self._stick_centered = (rx == 0 and ry == 0)
        else:
            try:
                import pydirectinput
                ix = int(round(stick_x * 10))
                iy = int(round(stick_y * 10))
                if ix or iy:
                    pydirectinput.moveRel(ix, iy, relative=True)
            except Exception:
                pass

        self._last_stick_x = stick_x
        self._last_stick_y = stick_y

    def move_to_target(
        self,
        dx: float,
        dy: float,
        smoothing: float = 0.35,
        strength: float = 1.0,
        speed: float = 1.0,
        humanization: float = 0.0,
        max_stick: float = 0.85,
        pixel_scale: float = 280.0,
    ):
        if not self.enabled:
            return

        smoothing = max(0.0, min(0.85, float(smoothing)))
        strength = max(0.05, min(3.0, float(strength)))
        speed = max(0.1, min(3.0, float(speed)))

        raw_dist = math.hypot(dx, dy)

        # Humanization never near target
        if humanization > 0 and raw_dist > 80:
            j = random.uniform(-humanization, humanization)
            dx += j
            dy += j * 0.5

        # ── 1. Filter error ──────────────────────────────────────────────
        if not self._filter_seeded:
            self._fx, self._fy = dx, dy
            self._filter_seeded = True
        else:
            # More filtering when close (noise is relatively larger)
            a = self.ERR_FILTER
            if raw_dist < 50:
                a = min(0.82, self.ERR_FILTER + 0.20)
            # User smoothing also deepens the error filter slightly
            a = min(0.88, a + smoothing * 0.25)
            self._fx = self._fx * a + dx * (1.0 - a)
            self._fy = self._fy * a + dy * (1.0 - a)

        fx, fy = self._fx, self._fy
        dist = math.hypot(fx, fy)

        # ── 2. Deadzone + hysteresis (true hold on aim point) ────────────
        if self._locked:
            if dist <= self.LOCK_OUT_PX:
                self._prev_fx, self._prev_fy = fx, fy
                self._apply_stick(0.0, 0.0)
                return
            # Target moved enough — resume tracking, keep filtered state
            self._locked = False

        if dist <= self.LOCK_IN_PX:
            self._locked = True
            self._prev_fx, self._prev_fy = fx, fy
            self._apply_stick(0.0, 0.0)
            return

        # ── 3. Closing-in detection (derivative damping) ──────────────────
        prev_dist = math.hypot(self._prev_fx, self._prev_fy)
        closing = dist < prev_dist - 0.15  # error shrinking
        opening = dist > prev_dist + 0.15

        # ── 4. Base proportional command ─────────────────────────────────
        # Map filtered pixels → stick. strength/speed scale far response.
        scale = max(160.0, float(pixel_scale))
        # Near-zone gain floor so strength doesn't dominate when close
        if dist < 80:
            t = (dist - self.LOCK_IN_PX) / max(1.0, 80.0 - self.LOCK_IN_PX)
            t = max(0.0, min(1.0, t))
            t = t * t * (3.0 - 2.0 * t)  # smoothstep
            near = 0.25 + 0.75 * t
        else:
            near = 1.0

        gain = strength * speed * near
        if closing:
            gain *= 0.55          # already heading the right way — ease off
        if opening and self._last_stick_x == 0 and self._last_stick_y == 0:
            gain *= 0.85

        stick_x = (fx / scale) * gain
        stick_y = (fy / scale) * gain

        # ── 5. Error-fraction cap (hard anti-overshoot) ───────────────────
        # Cap so one frame cannot move more than MAX_ERROR_FRAC of remaining
        # error, under the conservative PX_PER_FULL_STICK model.
        # stick * PX_PER_FULL_STICK <= dist * MAX_ERROR_FRAC
        max_by_error = (dist * self.MAX_ERROR_FRAC) / self.PX_PER_FULL_STICK
        max_by_error = max(0.0, min(max_stick, max_by_error))

        # Axis-wise: also cap each axis by its own residual
        max_x = min(max_by_error, abs(fx) * self.MAX_ERROR_FRAC / self.PX_PER_FULL_STICK + 0.005)
        max_y = min(max_by_error, abs(fy) * self.MAX_ERROR_FRAC / self.PX_PER_FULL_STICK + 0.005)

        stick_x = max(-max_x, min(max_x, stick_x))
        stick_y = max(-max_y, min(max_y, stick_y))

        # Global clamp
        stick_x = max(-max_stick, min(max_stick, stick_x))
        stick_y = max(-max_stick, min(max_stick, stick_y))

        # ── 6. Sign brake: if error flipped vs stick, we overshot — stop ─
        if stick_x != 0.0 and fx * stick_x < 0:
            stick_x = 0.0
        if stick_y != 0.0 and fy * stick_y < 0:
            stick_y = 0.0
        if self._last_stick_x != 0.0 and fx * self._last_stick_x < 0:
            stick_x = 0.0
        if self._last_stick_y != 0.0 and fy * self._last_stick_y < 0:
            stick_y = 0.0

        # ── 7. Soft stick deadzone ───────────────────────────────────────
        if abs(stick_x) < 0.015:
            stick_x = 0.0
        if abs(stick_y) < 0.015:
            stick_y = 0.0

        # ── 8. Gentle rate limit (no snap thrash) ────────────────────────
        if dist < 60:
            max_step = 0.10
        elif dist < 120:
            max_step = 0.22
        else:
            max_step = 0.40
        stick_x = self._rate_limit(self._last_stick_x, stick_x, max_step)
        stick_y = self._rate_limit(self._last_stick_y, stick_y, max_step)

        self._apply_stick(stick_x, stick_y)
        self._prev_fx, self._prev_fy = fx, fy

    @staticmethod
    def _rate_limit(prev: float, desired: float, max_step: float) -> float:
        d = desired - prev
        if d > max_step:
            return prev + max_step
        if d < -max_step:
            return prev - max_step
        return desired

    def is_activation_held(self, button: str) -> bool:
        return self.xinput.is_activation_held(button)

    def physical_connected(self) -> bool:
        self.xinput.read()
        return self.xinput.connected

    @property
    def is_locked(self) -> bool:
        return self._locked

    def reset(self):
        self._fx = 0.0
        self._fy = 0.0
        self._prev_fx = 0.0
        self._prev_fy = 0.0
        self._filter_seeded = False
        self._locked = False
        self._center_sticks()
