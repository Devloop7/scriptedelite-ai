"""
Controller Input Emulation (matches original spec).
Uses vgamepad for native thumbstick input profiles (for PS5 via Chiaki/remote play or capture).
Translates spatial deltas into controller stick values.
"""
import time
import random
import vgamepad as vg
from typing import Optional

class AimController:
    def __init__(self):
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.enabled = False
        self.gamepad = None
        try:
            self.gamepad = vg.VX360Gamepad()
            print("[Controller] Virtual Xbox 360 gamepad initialized (ViGEm).")
        except Exception as e:
            print(f"[Controller] Warning: Could not init vgamepad ({e}). Falling back to mouse may be needed.")
            self.gamepad = None

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled and self.gamepad:
            # Center sticks when disabled
            self.gamepad.left_joystick(x_value=0, y_value=0)
            self.gamepad.update()

    def move_to_target(self, dx: float, dy: float, smoothing: float = 0.3, strength: float = 1.0, 
                       humanization: float = 0.0, max_stick: float = 0.85):
        """
        Translate delta to left thumbstick.
        dx/dy are pixel offsets. Converted to normalized stick (-1 to 1).
        strength / smoothing control response.
        For PS5 remote play, this moves the in-game aim stick.
        """
        if not self.enabled:
            return

        if humanization > 0:
            jitter = random.uniform(-humanization, humanization) * 2
            dx += jitter
            dy += jitter * 0.7  # slightly less vertical jitter

        target_dx = dx * strength
        target_dy = dy * strength

        # Exponential smoothing
        smooth_dx = target_dx * (1.0 - smoothing) + self.last_dx * smoothing
        smooth_dy = target_dy * (1.0 - smoothing) + self.last_dy * smoothing

        # Scale pixel delta to stick value. Tune divisor based on resolution/FOV.
        # Typical: 300-600 px full stick for comfortable speed.
        scale = 420.0
        stick_x = max(-max_stick, min(max_stick, smooth_dx / scale))
        stick_y = max(-max_stick, min(max_stick, smooth_dy / scale))

        if self.gamepad:
            # Note: For many games, negative Y is up on left stick for look
            self.gamepad.left_joystick(x_value=int(stick_x * 32767), y_value=int(-stick_y * 32767))
            self.gamepad.update()
        else:
            # Fallback (rare)
            import pydirectinput
            ix = int(round(stick_x * 8))
            iy = int(round(stick_y * 8))
            if abs(ix) > 0 or abs(iy) > 0:
                pydirectinput.moveRel(ix, iy, relative=True)

        self.last_dx = smooth_dx
        self.last_dy = smooth_dy

    def reset(self):
        self.last_dx = 0.0
        self.last_dy = 0.0
        if self.gamepad:
            self.gamepad.left_joystick(x_value=0, y_value=0)
            self.gamepad.update()
