"""
Controller input emulation + physical button polling.
- Output: vgamepad (ViGEm) virtual Xbox 360 gamepad → right stick for aim
- Input:  XInput physical pad triggers/buttons for activation (L2/R2/LT/RT)

Critical control fact
--------------------
Right-stick value is camera *angular rate*, not aim position.
Any non-zero stick keeps moving the crosshair. Stable lock requires:
  • stick = 0 when on target
  • stick always same sign as remaining error (never reverse-hunt)
  • command strictly limited by remaining pixel error (no overshoot)
  • aim point already Kalman-filtered upstream (tracker owns smoothness)
"""
from __future__ import annotations

import ctypes
import math
import random
import time
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


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class AimController:
    """
    Stable aim controller for hard target lock.

    Model: stick is camera angular *rate*. To settle on a point:
      desired_rate ∝ remaining_error  (exponential approach)
      |command| capped so we never request more motion than remaining error
      stick sign always matches error sign (no reverse hunting)
      stick = 0 inside lock deadzone (hysteresis)

    User knobs:
      strength  → proportional gain
      speed     → far-range close rate
      smoothing → more settle damping near target + lower close fraction
    """

    LOCK_IN_PX = 6.0
    LOCK_OUT_PX = 12.0

    # Max fraction of remaining error to close in one frame
    CLOSE_FRAC = 0.34

    DEFAULT_PX_PER_FULL_STICK = 55.0

    def __init__(self):
        self._prev_ex = 0.0
        self._prev_ey = 0.0
        self._last_stick_x = 0.0
        self._last_stick_y = 0.0
        self._locked = False
        self._seeded = False
        self._frames_on_target = 0
        self._last_t = 0.0
        self._px_per_full = self.DEFAULT_PX_PER_FULL_STICK
        self._adapt_enabled = False
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

    def set_stick_response(self, px_per_full: float):
        """Higher = game more sensitive (less stick needed for same pixel correction)."""
        self._px_per_full = max(20.0, min(140.0, float(px_per_full)))

    def set_adaptive_response(self, enabled: bool):
        self._adapt_enabled = bool(enabled)

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
            if abs(rx) < 160:
                rx = 0
            if abs(ry) < 160:
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

    def release_stick(self):
        """Center stick without wiping lock memory (brief pause)."""
        if self._last_stick_x != 0.0 or self._last_stick_y != 0.0 or not self._stick_centered:
            self._apply_stick(0.0, 0.0)

    def move_to_target(
        self,
        dx: float,
        dy: float,
        smoothing: float = 0.35,
        strength: float = 1.0,
        speed: float = 1.0,
        humanization: float = 0.0,
        max_stick: float = 0.88,
        shake_reduction: bool = True,
    ):
        if not self.enabled:
            return

        smoothing = max(0.0, min(0.85, float(smoothing)))
        strength = max(0.05, min(3.0, float(strength)))
        speed = max(0.1, min(3.0, float(speed)))

        now = time.time()
        if self._last_t > 0:
            _dt = max(0.004, min(0.05, now - self._last_t))
        self._last_t = now

        ex, ey = float(dx), float(dy)
        dist = math.hypot(ex, ey)

        if humanization > 0 and dist > 100:
            j = random.uniform(-humanization, humanization)
            ex += j
            ey += j * 0.5
            dist = math.hypot(ex, ey)

        # ── Hold hysteresis ──────────────────────────────────────────────
        if self._locked:
            if dist <= self.LOCK_OUT_PX:
                self._frames_on_target += 1
                self._prev_ex, self._prev_ey = ex, ey
                self._apply_stick(0.0, 0.0)
                return
            self._locked = False
            self._frames_on_target = 0

        if dist <= self.LOCK_IN_PX:
            self._locked = True
            self._frames_on_target += 1
            self._prev_ex, self._prev_ey = ex, ey
            self._apply_stick(0.0, 0.0)
            return

        self._frames_on_target = 0

        if not self._seeded:
            self._prev_ex, self._prev_ey = ex, ey
            self._seeded = True

        prev_dist = math.hypot(self._prev_ex, self._prev_ey)
        closing = dist < prev_dist - 0.25
        opening = dist > prev_dist + 0.25

        # Adaptive stick response (bounded)
        if self._adapt_enabled and (abs(self._last_stick_x) > 0.05 or abs(self._last_stick_y) > 0.05):
            closed = prev_dist - dist
            stick_mag = math.hypot(self._last_stick_x, self._last_stick_y)
            if closed > 0.5 and stick_mag > 0.05:
                observed = closed / stick_mag
                self._px_per_full = 0.92 * self._px_per_full + 0.08 * observed
                self._px_per_full = max(25.0, min(120.0, self._px_per_full))

        # ── Exponential approach: rate ∝ error ───────────────────────────
        # speed_scale maps user speed (typically ~1 after /3 in worker, or raw 0.5-3)
        speed_scale = 0.65 + 0.45 * min(2.0, speed)
        # Base gain: higher strength = faster close
        gain = 0.62 * strength * speed_scale

        # Near-zone: softstep from lock ring → ~95px (gentle settle, full far)
        if dist < 95.0:
            t = (dist - self.LOCK_IN_PX) / max(1.0, 95.0 - self.LOCK_IN_PX)
            near = 0.30 + 0.70 * _smoothstep(t)
        else:
            near = 1.0

        # Smoothness → stronger settle near target (lower gain, not reverse)
        if dist < 70:
            near *= 1.0 - 0.35 * smoothing
        if shake_reduction and dist < 45:
            near *= 0.78

        # Closing: ease off so we don't blow past; opening: catch up
        if closing and dist < 85:
            ease = 0.48 + 0.52 * (dist / 85.0)
            near *= ease
        elif opening:
            near = min(1.15, near * 1.10)

        # Desired pixel step this frame (proportional, same sign as error)
        cmd_x = ex * gain * near
        cmd_y = ey * gain * near

        # ── Hard never-overshoot ──────────────────────────────────────────
        close_frac = self.CLOSE_FRAC
        if smoothing > 0.25:
            close_frac *= 1.0 - 0.30 * min(1.0, (smoothing - 0.25) / 0.60)
        if shake_reduction and dist < 40:
            close_frac = min(close_frac, 0.22)
        if dist < 25:
            close_frac = min(close_frac, 0.18)

        max_step = max(0.8, dist * close_frac)
        if dist > 110:
            max_step = min(dist * 0.50, max_step * (0.95 + 0.2 * min(2.0, speed)))

        cmag = math.hypot(cmd_x, cmd_y)
        if cmag > max_step and cmag > 1e-6:
            s = max_step / cmag
            cmd_x *= s
            cmd_y *= s

        px = max(20.0, self._px_per_full)
        stick_x = cmd_x / px
        stick_y = cmd_y / px

        stick_x = max(-max_stick, min(max_stick, stick_x))
        stick_y = max(-max_stick, min(max_stick, stick_y))

        # CRITICAL: never reverse vs error — kills L/R hunting
        if stick_x * ex < 0:
            stick_x = 0.0
        if stick_y * ey < 0:
            stick_y = 0.0

        # Soft deadzone — but crawl if still outside lock ring (avoids sticky stall)
        min_crawl = 0.014
        if abs(stick_x) < 0.009:
            stick_x = min_crawl * (1.0 if ex > 0 else -1.0) if abs(ex) > self.LOCK_IN_PX * 0.5 else 0.0
        if abs(stick_y) < 0.009:
            stick_y = min_crawl * (1.0 if ey > 0 else -1.0) if abs(ey) > self.LOCK_IN_PX * 0.5 else 0.0

        # Rate limit: soft near, freer far
        if dist < 30:
            max_step_stick = 0.08
        elif dist < 70:
            max_step_stick = 0.15
        else:
            max_step_stick = 0.36
        if dist > 100:
            max_step_stick = min(0.48, max_step_stick * (0.9 + 0.2 * min(2.0, speed)))

        stick_x = self._rate_limit(self._last_stick_x, stick_x, max_step_stick)
        stick_y = self._rate_limit(self._last_stick_y, stick_y, max_step_stick)

        # Re-enforce sign after rate limit (never reverse-hunt)
        if stick_x * ex < 0:
            stick_x = 0.0
        if stick_y * ey < 0:
            stick_y = 0.0

        self._apply_stick(stick_x, stick_y)
        self._prev_ex, self._prev_ey = ex, ey

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
        self._prev_ex = 0.0
        self._prev_ey = 0.0
        self._seeded = False
        self._locked = False
        self._frames_on_target = 0
        self._last_t = 0.0
        self._center_sticks()
