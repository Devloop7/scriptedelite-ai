"""
Hard sticky target tracker + aim-point filter.

Primary objective: once a valid target is acquired, stay locked on that same
target's configured aim point (head / chest / body) as consistently as possible.
Never switch players while the current track is still matchable.
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

    Acquire  → first valid detection in FOV (priority: closest / conf)
    Hold     → re-associate ONLY that same player; never re-pick another
    Follow   → aim point tracks the configured box offset tightly
    Coast    → keep last aim through brief YOLO dropouts
    Release  → only after consecutive misses OR track leaves hard bounds
    """

    def __init__(
        self,
        max_miss_frames: int = 28,
        match_iou: float = 0.04,
        match_center_px: float = 260.0,
        # Lower = stick closer to raw detection (more accurate lock)
        aim_smooth_far: float = 0.18,
        aim_smooth_near: float = 0.42,
        near_px: float = 40.0,
        hold_expand: float = 2.2,
        # Velocity-assisted coast (px/frame estimate)
        use_coast_velocity: bool = True,
    ):
        self.max_miss_frames = max_miss_frames
        self.match_iou = match_iou
        self.match_center_px = match_center_px
        self.aim_smooth_far = aim_smooth_far
        self.aim_smooth_near = aim_smooth_near
        self.near_px = near_px
        self.hold_expand = hold_expand
        self.use_coast_velocity = use_coast_velocity

        self._det: Optional[Detection] = None
        self._aim_x: Optional[float] = None
        self._aim_y: Optional[float] = None
        self._raw_ax: Optional[float] = None
        self._raw_ay: Optional[float] = None
        self._vx: float = 0.0  # aim velocity px/s
        self._vy: float = 0.0
        self._miss = 0
        self._age = 0
        self._last_t = 0.0
        self.active = False

    def reset(self):
        self._det = None
        self._aim_x = None
        self._aim_y = None
        self._raw_ax = None
        self._raw_ay = None
        self._vx = 0.0
        self._vy = 0.0
        self._miss = 0
        self._age = 0
        self._last_t = 0.0
        self.active = False

    def _match(self, detections: List[Detection]) -> Optional[Detection]:
        """Pick the detection that continues THIS track only. Never another player."""
        if not self._det or not detections:
            return None

        ranked = []
        for d in detections:
            iou = _iou(self._det, d)
            cdist = _center_dist(self._det, d)
            scale = min(self._det.height, d.height) / max(self._det.height, d.height, 1.0)
            # Prefer same scale + close center even with weak IoU (partial occlusion)
            ranked.append((iou, cdist, scale, d))

        # 1) Best IoU if any reasonable overlap
        by_iou = [r for r in ranked if r[0] >= self.match_iou]
        if by_iou:
            by_iou.sort(key=lambda r: (-r[0], r[1]))
            return by_iou[0][3]

        # 2) Nearest center within generous radius + similar scale
        by_center = [r for r in ranked if r[1] <= self.match_center_px and r[2] >= 0.30]
        if by_center:
            by_center.sort(key=lambda r: r[1])
            return by_center[0][3]

        # 3) Absolute nearest if still within a tighter radius
        ranked.sort(key=lambda r: r[1])
        if ranked and ranked[0][1] <= self.match_center_px * 0.65:
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

    def _update_velocity(self, ax: float, ay: float, now: float):
        if self._last_t > 0 and self._raw_ax is not None:
            dt = max(1e-3, now - self._last_t)
            ivx = (ax - self._raw_ax) / dt
            ivy = (ay - self._raw_ay) / dt
            # EMA on velocity; clamp extreme jumps
            self._vx = self._vx * 0.55 + ivx * 0.45
            self._vy = self._vy * 0.55 + ivy * 0.45
            speed = math.hypot(self._vx, self._vy)
            if speed > 2500.0:
                s = 2500.0 / speed
                self._vx *= s
                self._vy *= s
        self._raw_ax, self._raw_ay = ax, ay
        self._last_t = now

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
            matched = self._match(detections)
            if matched is None:
                self._miss += 1
                if self._miss > self.max_miss_frames:
                    self.reset()
                    return None
                # Coast — stay glued; optionally advance with last velocity
                if self._aim_x is None:
                    return None
                if self.use_coast_velocity and self._last_t > 0:
                    dt = min(0.05, now - self._last_t)
                    # Decay velocity while coasting so we don't fly off
                    decay = 0.92 ** max(1, self._miss)
                    self._aim_x = self._aim_x + self._vx * dt * decay
                    self._aim_y = self._aim_y + self._vy * dt * decay
                dx = self._aim_x - cx
                dy = self._aim_y - cy
                dist = math.hypot(dx, dy)
                if dist > hold_limit * 1.25:
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
                    "locked_track": True,
                }

            # Matched same player
            self._miss = 0
            self._det = matched
        else:
            # Acquire first valid target in zone
            matched = self._pick_new(detections, crosshair, max_dist, offset, priority)
            if matched is None:
                self.reset()
                return None
            self._det = matched
            self._miss = 0
            self.active = True
            self._age = 0
            self._vx = self._vy = 0.0
            ax, ay = matched.get_aim_point(offset)
            self._aim_x, self._aim_y = ax, ay  # instant lock — no lag on acquire
            self._raw_ax, self._raw_ay = ax, ay
            self._last_t = now

        # Update aim from current box at configured offset (same player only)
        raw_ax, raw_ay = self._det.get_aim_point(offset)
        self._update_velocity(raw_ax, raw_ay, now)

        if self._aim_x is None or self._age == 0:
            self._aim_x, self._aim_y = raw_ax, raw_ay
        else:
            # Adaptive smoothing: track tightly so lock stays on the detection point.
            # Far: follow raw more (snap onto target). Near: light filter for YOLO wobble.
            jump = math.hypot(raw_ax - self._aim_x, raw_ay - self._aim_y)
            cur_dist = math.hypot(self._aim_x - cx, self._aim_y - cy)

            if self._age < 5:
                # First frames after acquire: nearly raw for instant stick
                beta = 0.08
            elif jump > 45:
                # Large box jump (strafe / camera) — catch up fast
                beta = 0.10
            elif cur_dist < self.near_px:
                beta = self.aim_smooth_near
            else:
                beta = self.aim_smooth_far

            self._aim_x = self._aim_x * beta + raw_ax * (1.0 - beta)
            self._aim_y = self._aim_y * beta + raw_ay * (1.0 - beta)

        self._age += 1

        dx = self._aim_x - cx
        dy = self._aim_y - cy
        dist = math.hypot(dx, dy)

        # Once sticky, allow wider FOV so we don't drop on edge of zone
        limit = hold_limit if self._age > 1 else max_dist
        if dist > limit:
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
                "locked_track": True,
                "raw_aim_x": raw_ax,
                "raw_aim_y": raw_ay,
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
            "locked_track": True,
            "raw_aim_x": raw_ax,
            "raw_aim_y": raw_ay,
            "vx": self._vx,
            "vy": self._vy,
        }


class StickyColorTracker:
    """
    Sticky lock for color-mode enemy blobs.
    Once a blob is chosen, stay on it by proximity — never jump to another mark.
    """

    def __init__(self, max_miss_frames: int = 22, stick_radius: float = 240.0):
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
                    "locked_track": True, "color": "",
                }
            self._miss = 0
            self._cx, self._cy = matched["cx"], matched["cy"]
            raw_ax, raw_ay = matched["aim_x"], matched["aim_y"]
            if self._age < 3:
                self._aim_x, self._aim_y = raw_ax, raw_ay
            else:
                # Tight follow for color blobs
                self._aim_x = self._aim_x * 0.30 + raw_ax * 0.70
                self._aim_y = self._aim_y * 0.30 + raw_ay * 0.70
            self._age += 1
        else:
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
        if dist > max_dist * 2.0 and self._age > 2:
            self._miss += 1
            if self._miss > self.max_miss_frames:
                self.reset()
                return None

        return {
            "aim_x": self._aim_x, "aim_y": self._aim_y,
            "cx": self._cx, "cy": self._cy,
            "dx": dx, "dy": dy, "dist": dist,
            "sticky": True, "coasting": False, "age": self._age,
            "locked_track": True, "color": "",
        }
