"""
Color-signature detection for enemy tags / outlines.
User-configurable enemy color (hex / RGB) so teammates can be ignored.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_hex_color(text: str) -> Tuple[int, int, int]:
    """
    Parse '#RRGGBB', 'RRGGBB', or 'R,G,B' → RGB tuple 0-255.
    Defaults to magenta FF00FA on failure.
    """
    if text is None:
        return (255, 0, 250)
    s = str(text).strip()
    if not s:
        return (255, 0, 250)

    # R,G,B
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 3:
            try:
                r, g, b = [max(0, min(255, int(float(p)))) for p in parts[:3]]
                return (r, g, b)
            except ValueError:
                pass

    s = s.lstrip("#")
    if re.fullmatch(r"[0-9A-Fa-f]{6}", s):
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    if re.fullmatch(r"[0-9A-Fa-f]{3}", s):
        return (int(s[0] * 2, 16), int(s[1] * 2, 16), int(s[2] * 2, 16))
    return (255, 0, 250)


def rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


class ColorSignatureDetector:
    """
    Detects blobs matching a single user-configured enemy color.
    Uses BGR distance + HSV band for robust matching under game lighting.
    """

    def __init__(self):
        self.set_enemy_color("#FF00FA")
        self.tolerance = 55          # 0-100 UI scale → internal thresholds
        self.min_area = 18.0

    def set_enemy_color(self, hex_or_rgb: str):
        self.color_hex = hex_or_rgb if str(hex_or_rgb).startswith("#") else f"#{parse_hex_color(hex_or_rgb)[0]:02X}{parse_hex_color(hex_or_rgb)[1]:02X}{parse_hex_color(hex_or_rgb)[2]:02X}"
        self.rgb = parse_hex_color(hex_or_rgb)
        self.bgr = np.array(rgb_to_bgr(self.rgb), dtype=np.int16)
        # Precompute HSV center from a 1x1 patch
        patch = np.uint8([[list(rgb_to_bgr(self.rgb))]])
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0, 0]
        self.hsv_center = (int(hsv[0]), int(hsv[1]), int(hsv[2]))

    def set_tolerance(self, tol: float):
        """tol: 0-100 user scale."""
        self.tolerance = max(5.0, min(100.0, float(tol)))

    def _thresholds(self) -> Tuple[int, int, int, int]:
        """Return (bgr_dist_max, h_span, s_min, v_min) from tolerance."""
        t = self.tolerance / 100.0
        bgr_max = int(35 + t * 140)          # ~35..175 channel-sum distance
        h_span = int(6 + t * 28)             # hue half-width
        s_min = max(20, int(40 - t * 30))
        v_min = max(20, int(40 - t * 25))
        return bgr_max, h_span, s_min, v_min

    def _mask(self, frame: np.ndarray) -> np.ndarray:
        bgr_max, h_span, s_min, v_min = self._thresholds()

        # BGR Euclidean-ish sum distance
        f = frame.astype(np.int16)
        diff = np.abs(f - self.bgr).sum(axis=2)
        mask_bgr = diff < bgr_max

        # HSV band around target (handles glow/outline colors better)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        hc = self.hsv_center[0]
        # Hue wrap-around (OpenCV H is 0-179)
        dh = np.abs(h.astype(np.int16) - hc)
        dh = np.minimum(dh, 180 - dh)
        mask_hsv = (dh <= h_span) & (s >= s_min) & (v >= v_min)

        # If target is very saturated/neon, prefer intersection; else union soft
        sc = self.hsv_center[1]
        if sc >= 80:
            mask = mask_bgr & mask_hsv
            # If too empty, fall back to union of strong BGR
            if mask.sum() < 30:
                mask = mask_bgr | (mask_hsv & (diff < bgr_max + 40))
        else:
            mask = mask_bgr | mask_hsv

        mask_u8 = (mask.astype(np.uint8) * 255)
        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        return mask_u8

    def detect_all(self, frame: np.ndarray, head_offset: float = 35.0) -> List[Dict]:
        if frame is None or frame.size == 0:
            return []

        mask = self._mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hits: List[Dict] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(c)
            hits.append({
                "aim_x": cx,
                "aim_y": cy + float(head_offset),
                "cx": cx,
                "cy": cy,
                "area": area,
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + w),
                "y2": float(y + h),
                "color": self.color_hex,
                "conf": min(0.99, area / 500.0),
            })
        # Largest first
        hits.sort(key=lambda h: h["area"], reverse=True)
        return hits

    def detect(
        self,
        frame: np.ndarray,
        head_offset: float = 35.0,
        prefer_xy: Optional[Tuple[float, float]] = None,
        max_stick_dist: float = 180.0,
    ) -> Optional[Dict]:
        """
        Single best hit. If prefer_xy is set (sticky previous), pick the blob
        nearest that point within max_stick_dist — never jump to a far teammate blob.
        """
        hits = self.detect_all(frame, head_offset)
        if not hits:
            return None

        if prefer_xy is not None:
            px, py = prefer_xy
            best = None
            best_d = float("inf")
            for h in hits:
                d = (h["cx"] - px) ** 2 + (h["cy"] - py) ** 2
                if d < best_d:
                    best_d = d
                    best = h
            if best is not None and best_d <= max_stick_dist * max_stick_dist:
                return best
            # Prefer sticky: if nothing close enough, keep none (caller coasts)
            return None

        return hits[0]

    def draw(self, frame: np.ndarray, hit: Optional[Dict]) -> np.ndarray:
        vis = frame.copy()
        if not hit:
            return vis
        bgr = tuple(int(x) for x in self.bgr)
        cv2.circle(vis, (int(hit["aim_x"]), int(hit["aim_y"])), 6, (0, 0, 255), -1)
        cv2.circle(vis, (int(hit["cx"]), int(hit["cy"])), 5, bgr, 2)
        if "x1" in hit:
            cv2.rectangle(
                vis,
                (int(hit["x1"]), int(hit["y1"])),
                (int(hit["x2"]), int(hit["y2"])),
                bgr, 1,
            )
        cv2.putText(
            vis, f"ENEMY {hit.get('color', self.color_hex)}",
            (int(hit["cx"]) - 40, int(hit["cy"]) - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1, cv2.LINE_AA,
        )
        return vis
