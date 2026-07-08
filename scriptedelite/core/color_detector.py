"""
Light-Weight Color-Signature Detection Engine (Engine B).
Scans for high-visibility colors (bright magenta FF00FA or bright green) typically used for enemy nameplates/tags.
Applies Head Offset Parameter to translate to physical head/chest.
Pure CPU, low overhead fallback.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict

class ColorSignatureDetector:
    def __init__(self):
        # Target colors in BGR (OpenCV)
        self.magenta_bgr = np.array([250, 0, 255])   # FF00FA approx
        self.green_bgr = np.array([0, 255, 0])       # bright green
        self.color_tolerance = 35

    def detect(self, frame: np.ndarray, head_offset: float = 35.0) -> Optional[Dict]:
        """
        Exact hex color signature per spec (FF00FA magenta or bright green).
        Finds largest matching blob, applies head_offset down to head/chest.
        """
        # BGR for the hex
        target_mag = (250, 0, 255)   # FF00FA
        target_grn = (0, 255, 0)

        # Simple per-pixel distance mask (fast enough for small region)
        diff_mag = np.abs(frame.astype(np.int32) - target_mag).sum(axis=2)
        diff_grn = np.abs(frame.astype(np.int32) - target_grn).sum(axis=2)

        mask = (diff_mag < 80) | (diff_grn < 60)   # tolerance

        # Clean
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 20:
            return None

        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        aim_y = cy + head_offset
        aim_x = cx

        return {
            "aim_x": float(aim_x),
            "aim_y": float(aim_y),
            "cx": float(cx),
            "cy": float(cy),
            "area": float(cv2.contourArea(c)),
            "color": "magenta" if diff_mag[cy, cx] < diff_grn[cy, cx] else "green"
        }

    def draw(self, frame: np.ndarray, hit: Optional[Dict]) -> np.ndarray:
        vis = frame.copy()
        if hit:
            cv2.circle(vis, (int(hit["aim_x"]), int(hit["aim_y"])), 6, (0, 0, 255), -1)
            cv2.circle(vis, (int(hit["cx"]), int(hit["cy"])), 4, (255, 0, 255), 2)
            cv2.putText(vis, f"COLOR {hit.get('color','?')}", (int(hit["cx"])-30, int(hit["cy"])-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis
