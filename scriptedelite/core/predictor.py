"""
Linear Prediction Targeting (for Engine A - moving targets).
Keeps short history of target positions, computes velocity, predicts future position.
"""
from collections import deque
from typing import Optional, Tuple, Deque
import time

class LinearPredictor:
    def __init__(self, history_len: int = 5, prediction_ms: float = 80.0):
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=history_len)  # (x, y, t)
        self.prediction_ms = prediction_ms

    def update(self, x: float, y: float):
        t = time.time() * 1000.0
        self.history.append((x, y, t))

    def predict(self) -> Optional[Tuple[float, float]]:
        if len(self.history) < 2:
            return None

        # Use last two points for velocity
        x1, y1, t1 = self.history[-2]
        x2, y2, t2 = self.history[-1]

        dt = max(1.0, t2 - t1)  # ms
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt

        # Predict forward
        pred_x = x2 + vx * self.prediction_ms
        pred_y = y2 + vy * self.prediction_ms

        return pred_x, pred_y

    def reset(self):
        self.history.clear()
