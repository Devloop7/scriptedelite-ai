"""
Strong sticky target tracker + aim-point filter.

Once a valid target is acquired it is held aggressively until it is truly gone.
Never switches to another player while the current track is still matchable.
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
    """
    Hard sticky lock on a single player box.

    Acquire  → first valid detection in FOV (or closest by priority)
    Hold     → always re-associate that same player; never re-pick another
    Coast    → keep last aim through brief YOLO dropouts
    Release  → only after many consecutive misses OR track leaves hard bounds
    """

    def __init__(
        self,
        max_miss_frames: int = 22,       # strong hold through blinks
        match_iou: float = 0.05,         # very loose IoU still counts as same
        match_center_px: float = 220.0,  # allow fast strafes without losing lock
        aim_smooth_far: float = 0.35,
        aim_smooth_near: float = 0.82,
        near_px: float = 50.0,
        hold_expand: float = 1.85,       # once locked, FOV expands for hold
    ):
        self.max_miss_frames = max_miss_frames
        self.match_iou = match_iou
        self.match_center_px = match_center_px
        self.aim_smooth_far = aim_smooth_far
        self.aim_smooth_near = aim_smooth_near
        self.near_px = near_px
        self.hold_expand = hold_expand

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
        """Pick the detection that continues THIS track only. Never another player."""
        if not self._det or not detections:
            return None

        # Score every det; take best match if above acceptance
        ranked = []
        for d in detections:
            iou = _iou(self._det, d)
            cdist = _center_dist(self._det, d)
            # Size similarity (same person scale shouldn't jump wildly)
            scale = min(self._det.height, d.height) / max(self._det.height, d.height, 1.0)
            ranked.append((iou, cdist, scale, d))

        # 1) Best IoU if any reasonable overlap
        by_iou = [r for r in ranked if r[0] >= self.match_iou]
        if by_iou:
            by_iou.sort(key=lambda r: (-r[0], r[1]))
            return by_iou[0][3]

        # 2) Nearest center within generous radius + similar scale
        by_center = [r for r in ranked if r[1] <= self.match_center_px and r[2] >= 0.35]
        if by_center:
            by_center.sort(key=lambda r: r[1])
            return by_center[0][3]

        # 3) Absolute nearest if extremely close (partial occlusion)
        ranked.sort(key=lambda r: r[1])
        if ranked and ranked[0][1] <= self.match_center_px * 0.55:
            return ranked[0][3]
        return None

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
        cx, cy = crosshair
        now = time.time()
        hold_limit = max_dist * self.hold_expand

        if self.active and self._det is not None:
            # ── HARD STICKY: only re-associate current track ──
            matched = self._match(detections)
            if matched is None:
                self._miss += 1
                if self._miss > self.max_miss_frames:
                    self.reset()
                    return None
                # Coast — stay glued to last aim point
                if self._aim_x is None:
                    return None
                dx = self._aim_x - cx
                dy = self._aim_y - cy
                dist = math.hypot(dx, dy)
                if dist > hold_limit * 1.15:
                    self.reset()
                    return None
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
                    "miss": self._miss,
                }

            # Matched same player — never consider anyone else
            self._miss = 0
            self._det = matched
        else:
            # ── ACQUIRE: first valid target in zone ──
            matched = self._pick_new(detections, crosshair, max_dist, offset, priority)
            if matched is None:
                self.reset()
                return None
            self._det = matched
            self._miss = 0
            self.active = True
            self._age = 0
            ax, ay = matched.get_aim_point(offset)
            self._aim_x, self._aim_y = ax, ay  # instant lock, no lag on acquire

        # Update smoothed aim from current box (same player only)
        raw_ax, raw_ay = self._det.get_aim_point(offset)
        if self._aim_x is None or self._age == 0:
            self._aim_x, self._aim_y = raw_ax, raw_ay
        else:
            cur_dist = math.hypot(self._aim_x - cx, self._aim_y - cy)
            beta = self.aim_smooth_near if cur_dist < self.near_px else self.aim_smooth_far
            # First few frames after acquire: follow raw tightly for "instant stick"
            if self._age < 4:
                beta = min(beta, 0.20)
            self._aim_x = self._aim_x * beta + raw_ax * (1.0 - beta)
            self._aim_y = self._aim_y * beta + raw_ay * (1.0 - beta)

        self._age += 1
        self._last_t = now

        dx = self._aim_x - cx
        dy = self._aim_y - cy
        dist = math.hypot(dx, dy)

        # Once sticky, allow wider FOV so we don't drop on edge of zone
        limit = hold_limit if self._age > 1 else max_dist
        if dist > limit:
            # Don't instantly drop — start miss counter (player briefly off-center)
            self._miss += 1
            if self._miss > self.max_miss_frames:
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
                "coasting": True,
                "age": self._age,
                "miss": self._miss,
            }

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
            "miss": 0,
            "raw_aim_x": raw_ax,
            "raw_aim_y": raw_ay,
        }


class StickyColorTracker:
    """
    Sticky lock for color-mode enemy blobs.
    Once a blob is chosen, stay on it by proximity — never jump to another color mark.
    """

    def __init__(self, max_miss_frames: int = 18, stick_radius: float = 200.0):
        self.max_miss_frames = max_miss_frames
        self.stick_radius = stick_radius
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._aim_x: Optional[float] = None
        self._aim_y: Optional[float] = None
        self._miss = 0
        self._age = 0
        self.active = False

    def reset(self):
        self._cx = self._cy = None
        self._aim_x = self._aim_y = None
        self._miss = 0
        self._age = 0
        self.active = False

    def update(
        self,
        hits: List[Dict],
        crosshair: Tuple[float, float],
        max_dist: float,
        head_offset: float = 0.0,
    ) -> Optional[Dict]:
        cx, cy = crosshair

        if self.active and self._cx is not None:
            # Match closest hit to previous sticky center
            matched = None
            best_d = float("inf")
            for h in hits:
                d = math.hypot(h["cx"] - self._cx, h["cy"] - self._cy)
                if d < best_d:
                    best_d = d
                    matched = h
            if matched is None or best_d > self.stick_radius:
                self._miss += 1
                if self._miss > self.max_miss_frames:
                    self.reset()
                    return None
                if self._aim_x is None:
                    return None
                dx = self._aim_x - cx
                dy = self._aim_y - cy
                return {
                    "aim_x": self._aim_x, "aim_y": self._aim_y,
                    "cx": self._cx, "cy": self._cy,
                    "dx": dx, "dy": dy, "dist": math.hypot(dx, dy),
                    "sticky": True, "coasting": True, "age": self._age,
                    "color": "",
                }
            self._miss = 0
            self._cx, self._cy = matched["cx"], matched["cy"]
            # Smooth aim
            raw_ax, raw_ay = matched["aim_x"], matched["aim_y"]
            if self._age < 3:
                self._aim_x, self._aim_y = raw_ax, raw_ay
            else:
                self._aim_x = self._aim_x * 0.55 + raw_ax * 0.45
                self._aim_y = self._aim_y * 0.55 + raw_ay * 0.45
            self._age += 1
        else:
            # Acquire: closest hit to crosshair within max_dist
            best = None
            best_d = float("inf")
            for h in hits:
                d = math.hypot(h["aim_x"] - cx, h["aim_y"] - cy)
                if d <= max_dist and d < best_d:
                    best_d = d
                    best = h
            if best is None:
                self.reset()
                return None
            self.active = True
            self._miss = 0
            self._age = 1
            self._cx, self._cy = best["cx"], best["cy"]
            self._aim_x, self._aim_y = best["aim_x"], best["aim_y"]

        dx = self._aim_x - cx
        dy = self._aim_y - cy
        dist = math.hypot(dx, dy)
        if dist > max_dist * 1.9 and self._age > 2:
            self._miss += 1
            if self._miss > self.max_miss_frames:
                self.reset()
                return None

        return {
            "aim_x": self._aim_x, "aim_y": self._aim_y,
            "cx": self._cx, "cy": self._cy,
            "dx": dx, "dy": dy, "dist": dist,
            "sticky": True, "coasting": False, "age": self._age,
            "color": "",
        }
