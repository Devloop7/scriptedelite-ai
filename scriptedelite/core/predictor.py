"""
Optional prediction lead — thin helper.

Primary velocity estimation lives in the tracker's Kalman filter.
This module remains for light external lead blending if needed.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple


class LinearPredictor:
    def __init__(self, history_len: int = 6, prediction_ms: float = 60.0):
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=history_len)
        self.prediction_ms = prediction_ms

    def update(self, x: float, y: float):
        t = time.time() * 1000.0
        self.history.append((x, y, t))

    def predict(self, strength: float = 1.0) -> Optional[Tuple[float, float]]:
        if len(self.history) < 2:
            return None
        x0, y0, t0 = self.history[0]
        x1, y1, t1 = self.history[-1]
        dt = t1 - t0
        if dt < 8.0:
            return x1, y1
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        s = max(0.0, min(1.0, float(strength)))
        return x1 + vx * self.prediction_ms * s, y1 + vy * self.prediction_ms * s

    def reset(self):
        self.history.clear()
