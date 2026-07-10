"""
Hard sticky target tracker + Kalman aim-point filter.

Primary objective: once a valid target is acquired, stay locked on that same
target's configured aim point (head / chest / body) as consistently as possible.
Never switch players while the current track is still matchable.

Smoothing ownership: Kalman filter owns aim-point stability.
Controller owns stick response — do not double-EMA aim here.
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


class AimKalman:
    """
    Constant-velocity Kalman filter on aim (x, y).
    State: [x, y, vx, vy]
    """

    def __init__(self, process_var: float = 80.0, measure_var: float = 120.0):
        self.q = process_var   # process noise (higher = follow motion more)
        self.r = measure_var   # measurement noise (higher = smoother / less YOLO jitter)
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        # Diagonal covariance approx
        self.p_pos = 400.0
        self.p_vel = 400.0
        self._init = False
        self._last_t = 0.0

    @property
    def initialized(self) -> bool:
        return self._init

    def reset(self):
        self.x = self.y = self.vx = self.vy = 0.0
        self.p_pos = 400.0
        self.p_vel = 400.0
        self._init = False
        self._last_t = 0.0

    def seed(self, ax: float, ay: float, now: float | None = None):
        self.x, self.y = ax, ay
        self.vx = self.vy = 0.0
        self.p_pos = 80.0
        self.p_vel = 200.0
        self._init = True
        self._last_t = now if now is not None else time.time()

    def predict(self, now: float | None = None) -> Tuple[float, float]:
        if not self._init:
            return self.x, self.y
        t = now if now is not None else time.time()
        dt = 0.016
        if self._last_t > 0:
            dt = max(0.004, min(0.05, t - self._last_t))
        self._last_t = t
        self.x += self.vx * dt
        self.y += self.vy * dt
        # Process noise inflates covariance
        self.p_pos += self.q * dt * 4.0
        self.p_vel += self.q * dt
        self.p_pos = min(self.p_pos, 5000.0)
        self.p_vel = min(self.p_vel, 8000.0)
        return self.x, self.y

    def update(self, mx: float, my: float, now: float | None = None,
               measure_scale: float = 1.0) -> Tuple[float, float]:
        """
        measure_scale > 1 trusts measurement less (more smooth, e.g. shake reduction near).
        """
        if not self._init:
            self.seed(mx, my, now)
            return self.x, self.y

        self.predict(now)
        r = self.r * max(0.4, measure_scale)
        # Kalman gain for position
        k = self.p_pos / (self.p_pos + r)
        k = max(0.05, min(0.95, k))
        ix = mx - self.x
        iy = my - self.y
        self.x += k * ix
        self.y += k * iy
        # Velocity update from residual (light)
        kv = min(0.35, k * 0.45)
        t = now if now is not None else time.time()
        dt = 0.016
        if self._last_t > 0:
            # last_t already advanced in predict; use small dt for vel blend
            dt = 0.016
        self.vx = (1.0 - kv) * self.vx + kv * (ix / max(dt, 1e-3)) * 0.35
        self.vy = (1.0 - kv) * self.vy + kv * (iy / max(dt, 1e-3)) * 0.35
        # Clamp velocity
        sp = math.hypot(self.vx, self.vy)
        if sp > 2200.0:
            s = 2200.0 / sp
            self.vx *= s
            self.vy *= s
        self.p_pos = (1.0 - k) * self.p_pos
        self.p_vel = max(50.0, self.p_vel * 0.95)
        return self.x, self.y

    def coast(self, now: float | None = None, decay: float = 0.92) -> Tuple[float, float]:
        """Predict without measurement; decay velocity so we don't fly off."""
        ax, ay = self.predict(now)
        self.vx *= decay
        self.vy *= decay
        return ax, ay


class StickyTargetTracker:
    """
    Hard sticky lock on a single player box.

    Acquire  → first valid detection in FOV (priority: closest / conf)
    Hold     → re-associate ONLY that same player (track_id preferred)
    Follow   → Kalman-filtered aim at configured box offset
    Coast    → predict through brief YOLO dropouts
    Release  → only after consecutive misses OR track leaves hard bounds
    """

    def __init__(
        self,
        max_miss_frames: int = 28,
        match_iou: float = 0.05,
        match_center_px: float = 180.0,
        hold_expand: float = 2.2,
        use_coast_velocity: bool = True,
    ):
        self.max_miss_frames = max_miss_frames
        self.match_iou = match_iou
        self.match_center_px = match_center_px
        self.hold_expand = hold_expand
        self.use_coast_velocity = use_coast_velocity

        self._det: Optional[Detection] = None
        self._track_id: Optional[int] = None
        self._kalman = AimKalman()
        self._miss = 0
        self._age = 0
        self.active = False

    def reset(self):
        self._det = None
        self._track_id = None
        self._kalman.reset()
        self._miss = 0
        self._age = 0
        self.active = False

    def _match(self, detections: List[Detection]) -> Optional[Detection]:
        """Pick the detection that continues THIS track only. Never another player."""
        if not self._det or not detections:
            return None

        # 0) Prefer stable track_id from ByteTrack / BoT-SORT
        if self._track_id is not None:
            for d in detections:
                if d.track_id is not None and int(d.track_id) == int(self._track_id):
                    return d

        ranked = []
        for d in detections:
            iou = _iou(self._det, d)
            cdist = _center_dist(self._det, d)
            scale = min(self._det.height, d.height) / max(self._det.height, d.height, 1.0)
            ranked.append((iou, cdist, scale, d))

        # 1) Best IoU if any reasonable overlap
        by_iou = [r for r in ranked if r[0] >= self.match_iou]
        if by_iou:
            by_iou.sort(key=lambda r: (-r[0], r[1]))
            return by_iou[0][3]

        # 2) Nearest center within radius + similar scale
        by_center = [r for r in ranked if r[1] <= self.match_center_px and r[2] >= 0.35]
        if by_center:
            by_center.sort(key=lambda r: r[1])
            return by_center[0][3]

        # 3) Absolute nearest if still within tighter radius
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
        shake_reduction: bool = True,
        prediction_lead: float = 0.0,
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
                if not self._kalman.initialized:
                    return None
                decay = 0.90 ** max(1, self._miss)
                if self.use_coast_velocity:
                    ax, ay = self._kalman.coast(now, decay=decay)
                else:
                    ax, ay = self._kalman.x, self._kalman.y
                dx = ax - cx
                dy = ay - cy
                dist = math.hypot(dx, dy)
                if dist > hold_limit * 1.25:
                    self.reset()
                    return None
                return {
                    "det": self._det,
                    "aim_x": ax,
                    "aim_y": ay,
                    "dx": dx,
                    "dy": dy,
                    "dist": dist,
                    "conf": getattr(self._det, "conf", 0.0),
                    "sticky": True,
                    "coasting": True,
                    "age": self._age,
                    "miss": self._miss,
                    "locked_track": True,
                    "track_id": self._track_id,
                }

            self._miss = 0
            self._det = matched
            if matched.track_id is not None:
                self._track_id = int(matched.track_id)
        else:
            matched = self._pick_new(detections, crosshair, max_dist, offset, priority)
            if matched is None:
                self.reset()
                return None
            self._det = matched
            self._miss = 0
            self.active = True
            self._age = 0
            self._track_id = int(matched.track_id) if matched.track_id is not None else None
            raw_ax, raw_ay = matched.get_aim_point(offset)
            self._kalman.seed(raw_ax, raw_ay, now)

        raw_ax, raw_ay = self._det.get_aim_point(offset)

        # Measure noise: more smooth near crosshair when shake reduction on
        measure_scale = 1.0
        if self._kalman.initialized:
            cur_dist = math.hypot(self._kalman.x - cx, self._kalman.y - cy)
            if shake_reduction and cur_dist < 50:
                measure_scale = 1.55
            elif self._age < 4:
                measure_scale = 0.55  # trust detection more on acquire

        ax, ay = self._kalman.update(raw_ax, raw_ay, now, measure_scale=measure_scale)

        # Optional lead from Kalman velocity (prediction_strength * small horizon)
        if prediction_lead > 0 and self._age >= 4:
            lead_t = 0.045 * prediction_lead  # seconds
            lx = self._kalman.vx * lead_t
            ly = self._kalman.vy * lead_t
            lmag = math.hypot(lx, ly)
            max_lead = 22.0 * prediction_lead
            if lmag > max_lead and lmag > 1e-6:
                s = max_lead / lmag
                lx *= s
                ly *= s
            ax += lx
            ay += ly

        self._age += 1

        dx = ax - cx
        dy = ay - cy
        dist = math.hypot(dx, dy)

        limit = hold_limit if self._age > 1 else max_dist
        if dist > limit:
            self._miss += 1
            if self._miss > self.max_miss_frames:
                self.reset()
                return None
            return {
                "det": self._det,
                "aim_x": ax,
                "aim_y": ay,
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
                "track_id": self._track_id,
            }

        return {
            "det": self._det,
            "aim_x": ax,
            "aim_y": ay,
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
            "vx": self._kalman.vx,
            "vy": self._kalman.vy,
            "track_id": self._track_id,
        }


class StickyColorTracker:
    """
    Sticky lock for color-mode enemy blobs.
    Once a blob is chosen, stay on it by proximity — never jump to another mark.
    """

    def __init__(self, max_miss_frames: int = 22, stick_radius: float = 200.0):
        self.max_miss_frames = max_miss_frames
        self.stick_radius = stick_radius
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._kalman = AimKalman(process_var=100.0, measure_var=90.0)
        self._miss = 0
        self._age = 0
        self.active = False

    def reset(self):
        self._cx = self._cy = None
        self._kalman.reset()
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
        now = time.time()

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
                if not self._kalman.initialized:
                    return None
                ax, ay = self._kalman.coast(now, decay=0.90)
                dx = ax - cx
                dy = ay - cy
                return {
                    "aim_x": ax, "aim_y": ay,
                    "cx": self._cx, "cy": self._cy,
                    "dx": dx, "dy": dy, "dist": math.hypot(dx, dy),
                    "sticky": True, "coasting": True, "age": self._age,
                    "locked_track": True, "color": "",
                }
            self._miss = 0
            self._cx, self._cy = matched["cx"], matched["cy"]
            raw_ax, raw_ay = matched["aim_x"], matched["aim_y"]
            ax, ay = self._kalman.update(raw_ax, raw_ay, now)
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
            self._kalman.seed(best["aim_x"], best["aim_y"], now)
            ax, ay = best["aim_x"], best["aim_y"]

        dx = ax - cx
        dy = ay - cy
        dist = math.hypot(dx, dy)
        if dist > max_dist * 2.0 and self._age > 2:
            self._miss += 1
            if self._miss > self.max_miss_frames:
                self.reset()
                return None

        return {
            "aim_x": ax, "aim_y": ay,
            "cx": self._cx, "cy": self._cy,
            "dx": dx, "dy": dy, "dist": dist,
            "sticky": True, "coasting": False, "age": self._age,
            "locked_track": True, "color": "",
        }
