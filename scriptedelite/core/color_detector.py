"""
Color-Signature Detection Engine (Engine B).
Scans for high-visibility colors (bright magenta / bright green) used for tags.
Applies head offset to estimate aim point.
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


class ColorSignatureDetector:
    def __init__(self):
        self.color_tolerance_mag = 80
        self.color_tolerance_grn = 60

    def detect(self, frame: np.ndarray, head_offset: float = 35.0) -> Optional[Dict]:
        if frame is None or frame.size == 0:
            return None

        target_mag = np.array([250, 0, 255], dtype=np.int16)   # BGR ≈ FF00FA
        target_grn = np.array([0, 255, 0], dtype=np.int16)

        f = frame.astype(np.int16)
        diff_mag = np.abs(f - target_mag).sum(axis=2)
        diff_grn = np.abs(f - target_grn).sum(axis=2)
        mask = (diff_mag < self.color_tolerance_mag) | (diff_grn < self.color_tolerance_grn)

        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area < 20:
            return None

        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        aim_x = cx
        aim_y = cy + float(head_offset)

        iy, ix = int(cy), int(cx)
        iy = max(0, min(frame.shape[0] - 1, iy))
        ix = max(0, min(frame.shape[1] - 1, ix))
        color = "magenta" if diff_mag[iy, ix] < diff_grn[iy, ix] else "green"

        return {
            "aim_x": aim_x,
            "aim_y": aim_y,
            "cx": cx,
            "cy": cy,
            "area": area,
            "color": color,
        }

    def draw(self, frame: np.ndarray, hit: Optional[Dict]) -> np.ndarray:
        vis = frame.copy()
        if not hit:
            return vis
        cv2.circle(vis, (int(hit["aim_x"]), int(hit["aim_y"])), 6, (0, 0, 255), -1)
        cv2.circle(vis, (int(hit["cx"]), int(hit["cy"])), 4, (255, 0, 255), 2)
        cv2.putText(
            vis, f"COLOR {hit.get('color', '?')}",
            (int(hit["cx"]) - 30, int(hit["cy"]) - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
        return vis
