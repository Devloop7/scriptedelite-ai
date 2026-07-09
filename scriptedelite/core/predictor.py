"""
Linear Prediction Targeting.
Keeps short history of target positions, computes velocity, predicts future position.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple


class LinearPredictor:
    def __init__(self, history_len: int = 6, prediction_ms: float = 80.0):
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=history_len)
        self.prediction_ms = prediction_ms

    def update(self, x: float, y: float):
        t = time.time() * 1000.0
        self.history.append((x, y, t))

    def predict(self, strength: float = 1.0) -> Optional[Tuple[float, float]]:
        """
        strength 0..1 blends between last known position and full prediction.
        """
        if len(self.history) < 2:
            return None

        # Velocity from oldest→newest sample (need real elapsed time)
        x0, y0, t0 = self.history[0]
        x1, y1, t1 = self.history[-1]
        dt = t1 - t0
        if dt < 8.0:  # need at least ~8ms of history for a stable velocity
            return x1, y1

        vx = (x1 - x0) / dt  # px per ms
        vy = (y1 - y0) / dt

        s = max(0.0, min(1.0, float(strength)))
        pred_x = x1 + vx * self.prediction_ms * s
        pred_y = y1 + vy * self.prediction_ms * s
        return pred_x, pred_y

    def reset(self):
        self.history.clear()
