"""
Controller Input Emulation + Physical Button Polling.
- Output: vgamepad (ViGEm) virtual Xbox 360 gamepad → right stick for aim
- Input:  XInput physical pad triggers/buttons for activation (L2/R2/LT/RT)

Aim model notes
---------------
Right-stick deflection is a *rate* of camera turn, not a position.
A non-zero stick keeps the camera moving forever — so once the crosshair
is on the aim point we MUST fully release the stick. High strength without
a lock deadzone is exactly what causes left/right oscillation.
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
# XInput (read physical Xbox / XInput-compatible pads for activation)
# ---------------------------------------------------------------------------
class XInputReader:
    """Lightweight XInput poller for triggers and face buttons."""

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
    Translates pixel error (crosshair → aim point) into right-stick aim.

    Stable lock design
    ------------------
    Stick deflection is camera *velocity*. To stay locked on the red aim
    point the stick must go fully to zero once error is inside a deadzone.

    1. Pixel lock deadzone + hysteresis → hold on head/body point
    2. Soft near-zone gain reduction (strength cannot punch through lock)
    3. Error-sign reversal brake kills leftover L/R hunting momentum
    4. Stick rate limit prevents snap-flip thrash at high strength
    """

    # Pixel radii for lock (independent of user strength)
    LOCK_IN_PX = 5.0        # enter lock → full stick release
    LOCK_OUT_PX = 11.0      # leave lock only if error grows past this
    SOFT_ZONE_PX = 70.0     # ease gain inside this radius

    def __init__(self):
        self.last_dx = 0.0
        self.last_dy = 0.0
        self._last_stick_x = 0.0
        self._last_stick_y = 0.0
        self._locked = False
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

    def move_to_target(
        self,
        dx: float,
        dy: float,
        smoothing: float = 0.35,
        strength: float = 1.0,
        speed: float = 1.0,
        humanization: float = 0.0,
        max_stick: float = 0.92,
        pixel_scale: float = 320.0,
    ):
        """
        dx/dy   : pixel offset from crosshair to aim point (frame-local)
        smoothing : 0..1  higher = smoother / less snappy acquisition
        strength  : far-range gain (cannot overpower the lock deadzone)
        speed     : acquisition multiplier
        """
        if not self.enabled:
            return

        smoothing = max(0.0, min(0.90, float(smoothing)))
        strength = max(0.05, float(strength))
        speed = max(0.1, float(speed))

        dist = math.hypot(dx, dy)

        # Humanization only far from aim point (never while settling/locked)
        if humanization > 0 and dist > self.SOFT_ZONE_PX:
            j = random.uniform(-humanization, humanization)
            dx += j
            dy += j * 0.65
            dist = math.hypot(dx, dy)

        # ── 1. Lock hysteresis ───────────────────────────────────────────
        # Once on the red aim point: stick = 0 and hold until target moves.
        if self._locked:
            if dist <= self.LOCK_OUT_PX:
                self.last_dx = 0.0
                self.last_dy = 0.0
                self._apply_stick(0.0, 0.0)
                return
            self._locked = False

        if dist <= self.LOCK_IN_PX:
            self._locked = True
            self.last_dx = 0.0
            self.last_dy = 0.0
            self._apply_stick(0.0, 0.0)
            return

        # ── 2. Drive residual error (distance beyond lock ring) ──────────
        # We only correct the excess past LOCK_IN so commands naturally
        # shrink to zero as we approach the lock — no fighting the deadzone.
        residual = dist - self.LOCK_IN_PX
        ux = dx / dist
        uy = dy / dist
        rdx = ux * residual
        rdy = uy * residual

        # ── 3. Near-zone soft gain (caps overshoot at high strength) ─────
        # Linear ease from ~0.35 at lock edge to 1.0 outside SOFT_ZONE.
        # Floor of 0.35 keeps mid-close tracking alive; never full strength
        # right next to the aim point.
        if dist < self.SOFT_ZONE_PX:
            t = (dist - self.LOCK_IN_PX) / max(1.0, self.SOFT_ZONE_PX - self.LOCK_IN_PX)
            t = max(0.0, min(1.0, t))
            # smoothstep
            t = t * t * (3.0 - 2.0 * t)
            near_shape = 0.35 + 0.65 * t
        else:
            near_shape = 1.0

        gain = strength * speed * near_shape

        # ── 4. Sign-flip brake (anti L/R oscillation) ─────────────────────
        # If error flipped vs previous stick direction we already overshot.
        brake = 1.0
        if self._last_stick_x != 0.0 and dx * self._last_stick_x < 0:
            self.last_dx *= 0.1
            brake = min(brake, 0.35)
        if self._last_stick_y != 0.0 and dy * self._last_stick_y < 0:
            self.last_dy *= 0.1
            brake = min(brake, 0.35)
        gain *= brake

        target_dx = rdx * gain
        target_dy = rdy * gain

        # ── 5. EMA smoothing (extra near target) ─────────────────────────
        extra_sm = 0.0
        if dist < self.SOFT_ZONE_PX:
            extra_sm = (1.0 - dist / self.SOFT_ZONE_PX) * 0.25
        sm = min(0.90, smoothing + extra_sm)
        alpha = 1.0 - sm
        smooth_dx = target_dx * alpha + self.last_dx * sm
        smooth_dy = target_dy * alpha + self.last_dy * sm

        # ── 6. Pixels → stick ────────────────────────────────────────────
        scale = max(120.0, float(pixel_scale))
        stick_x = max(-max_stick, min(max_stick, smooth_dx / scale))
        stick_y = max(-max_stick, min(max_stick, smooth_dy / scale))

        # Tiny noise floor only — residual-error model already shrinks output
        if abs(stick_x) < 0.012:
            stick_x = 0.0
        if abs(stick_y) < 0.012:
            stick_y = 0.0

        # ── 7. Rate-limit (stops high-strength snap thrash) ───────────────
        if dist < self.SOFT_ZONE_PX:
            # Allow enough step to close residual, but no wild flips
            max_step = 0.12 + 0.28 * (dist / self.SOFT_ZONE_PX)
        else:
            max_step = 0.50

        stick_x = self._rate_limit(self._last_stick_x, stick_x, max_step)
        stick_y = self._rate_limit(self._last_stick_y, stick_y, max_step)

        # If rate-limit / deadzone zeroed both axes while still outside lock,
        # keep a minimum nudge so we finish into the lock ring instead of
        # stalling a few pixels off the aim point.
        if stick_x == 0.0 and stick_y == 0.0 and residual > 0.05:
            min_nudge = min(0.06, 0.02 + residual * 0.006)
            stick_x = ux * min_nudge
            stick_y = uy * min_nudge

        self._apply_stick(stick_x, stick_y)
        self.last_dx = smooth_dx
        self.last_dy = smooth_dy

    @staticmethod
    def _rate_limit(prev: float, desired: float, max_step: float) -> float:
        delta = desired - prev
        if delta > max_step:
            return prev + max_step
        if delta < -max_step:
            return prev - max_step
        return desired

    def is_activation_held(self, button: str) -> bool:
        return self.xinput.is_activation_held(button)

    def physical_connected(self) -> bool:
        self.xinput.read()
        return self.xinput.connected

    def reset(self):
        self.last_dx = 0.0
        self.last_dy = 0.0
        self._locked = False
        self._center_sticks()
