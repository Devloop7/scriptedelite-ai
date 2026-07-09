"""
Sticky target tracker + aim-point filter.

Problems this solves
--------------------
1. Inconsistency: re-picking "closest" every frame switches targets / aim points.
2. Oscillation: raw YOLO boxes jitter several pixels L/R on a static player;
   feeding that noise straight into the stick causes left/right hunting.

Behavior
--------
- First valid detection inside the FOV becomes the locked track.
- Subsequent frames match by IoU / center distance (same player).
- Aim point is EMA-smoothed (heavier when already close to crosshair).
- Brief detection dropouts are tolerated (miss grace) instead of dropping lock.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from core.detector import Detection


def _iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, a.width * a.height)
    area_b = max(1.0, b.width * b.height)
    return inter / (area_a + area_b - inter)


def _center_dist(a: Detection, b: Detection) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


class StickyTargetTracker:
    def __init__(
        self,
        max_miss_frames: int = 12,
        match_iou: float = 0.15,
        match_center_px: float = 120.0,
        aim_smooth_far: float = 0.45,
        aim_smooth_near: float = 0.78,
        near_px: float = 40.0,
    ):
        self.max_miss_frames = max_miss_frames
        self.match_iou = match_iou
        self.match_center_px = match_center_px
        self.aim_smooth_far = aim_smooth_far
        self.aim_smooth_near = aim_smooth_near
        self.near_px = near_px

        self._det: Optional[Detection] = None
        self._aim_x: Optional[float] = None
        self._aim_y: Optional[float] = None
        self._miss = 0
        self._age = 0
        self._last_t = 0.0
        self.active = False

    def reset(self):
        self._det = None
        self._aim_x = None
        self._aim_y = None
        self._miss = 0
        self._age = 0
        self.active = False

    def _match(self, detections: List[Detection]) -> Optional[Detection]:
        if not self._det or not detections:
            return None

        best = None
        best_score = -1.0
        for d in detections:
            iou = _iou(self._det, d)
            cdist = _center_dist(self._det, d)
            # Accept if IoU good OR center stayed close (fast lateral moves)
            if iou < self.match_iou and cdist > self.match_center_px:
                continue
            # Prefer high IoU, then closer center
            score = iou * 2.0 + max(0.0, 1.0 - cdist / max(1.0, self.match_center_px))
            if score > best_score:
                best_score = score
                best = d
        return best

    def _pick_new(
        self,
        detections: List[Detection],
        crosshair: Tuple[float, float],
        max_dist: float,
        offset: float,
        priority: str = "closest",
    ) -> Optional[Detection]:
        cx, cy = crosshair
        best = None
        best_score = float("inf")
        for d in detections:
            ax, ay = d.get_aim_point(offset)
            dist = math.hypot(ax - cx, ay - cy)
            if dist > max_dist:
                continue
            if priority == "highest_conf":
                score = (1.0 - d.conf) * 1000.0 + dist * 0.01
            else:
                score = dist
            if score < best_score:
                best_score = score
                best = d
        return best

    def update(
        self,
        detections: List[Detection],
        crosshair: Tuple[float, float],
        max_dist: float,
        offset: float,
        priority: str = "closest",
    ) -> Optional[Dict]:
        """
        Returns a stable aim dict or None if no valid track.
        Always prefer continuing the current track over switching.
        """
        cx, cy = crosshair
        now = time.time()
        matched: Optional[Detection] = None

        if self.active and self._det is not None:
            matched = self._match(detections)
            if matched is None:
                self._miss += 1
                if self._miss > self.max_miss_frames:
                    self.reset()
                elif self._aim_x is not None:
                    # Coast on last smoothed aim for a few frames (detection blink)
                    dx = self._aim_x - cx
                    dy = self._aim_y - cy
                    dist = math.hypot(dx, dy)
                    if dist <= max_dist:
                        return {
                            "det": self._det,
                            "aim_x": self._aim_x,
                            "aim_y": self._aim_y,
                            "dx": dx,
                            "dy": dy,
                            "dist": dist,
                            "conf": getattr(self._det, "conf", 0.0),
                            "sticky": True,
                            "coasting": True,
                            "age": self._age,
                        }
                    self.reset()
                    return None
            else:
                self._miss = 0
                self._det = matched
        else:
            matched = self._pick_new(detections, crosshair, max_dist, offset, priority)
            if matched is None:
                self.reset()
                return None
            self._det = matched
            self._miss = 0
            self.active = True
            self._age = 0
            # Seed smoother with raw aim (no lag on first lock)
            ax, ay = matched.get_aim_point(offset)
            self._aim_x, self._aim_y = ax, ay

        if matched is None and not self.active:
            return None

        # Fresh aim from current box
        raw_ax, raw_ay = self._det.get_aim_point(offset)

        # Adaptive EMA: heavier smoothing when already near crosshair (kills jitter)
        if self._aim_x is None:
            self._aim_x, self._aim_y = raw_ax, raw_ay
        else:
            cur_dist = math.hypot(self._aim_x - cx, self._aim_y - cy)
            if cur_dist < self.near_px:
                # Near lock: trust history more (filter YOLO wobble)
                beta = self.aim_smooth_near
            else:
                beta = self.aim_smooth_far
            self._aim_x = self._aim_x * beta + raw_ax * (1.0 - beta)
            self._aim_y = self._aim_y * beta + raw_ay * (1.0 - beta)

        self._age += 1
        self._last_t = now

        dx = self._aim_x - cx
        dy = self._aim_y - cy
        dist = math.hypot(dx, dy)

        # If sticky track drifted outside FOV hard limit, drop it
        if dist > max_dist * 1.25:
            self.reset()
            return None

        return {
            "det": self._det,
            "aim_x": self._aim_x,
            "aim_y": self._aim_y,
            "dx": dx,
            "dy": dy,
            "dist": dist,
            "conf": self._det.conf,
            "sticky": True,
            "coasting": False,
            "age": self._age,
            "raw_aim_x": raw_ax,
            "raw_aim_y": raw_ay,
        }
