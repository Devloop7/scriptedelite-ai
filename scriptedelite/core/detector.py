"""
YOLO Detector wrapper for ScriptedElite AI.
Uses the provided model.pt (single 'person' class).
Computes aim points with configurable vertical offset.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
from pathlib import Path

class Detection:
    def __init__(self, x1, y1, x2, y2, conf, cls_id, cls_name):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.width = x2 - x1
        self.height = y2 - y1
        self.cx = (x1 + x2) / 2
        self.cy = (y1 + y2) / 2

    def get_aim_point(self, offset: float = 0.18) -> Tuple[float, float]:
        """
        offset: 0.0 = very top of box (extreme head), 
                0.18 = good head level, 
                0.35-0.45 = upper chest/body
        """
        aim_y = self.y1 + self.height * offset
        return self.cx, aim_y

class YOLODetector:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model: Optional[YOLO] = None
        self.names = {}
        self._load_model()

    def _load_model(self):
        print(f"[Detector] Loading model from {self.model_path} ...")
        self.model = YOLO(str(self.model_path))
        self.names = self.model.names or {0: "person"}
        print(f"[Detector] Model loaded. Classes: {self.names}")

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.4) -> List[Detection]:
        if self.model is None:
            return []

        results = self.model(frame, verbose=False, conf=conf_threshold, iou=0.45)[0]

        detections: List[Detection] = []
        boxes = results.boxes
        if boxes is None:
            return detections

        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls_id = int(box.cls[0])
            cls_name = self.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            det = Detection(x1, y1, x2, y2, conf, cls_id, cls_name)
            detections.append(det)

        return detections

    def find_best_target(self, detections: List[Detection], 
                         crosshair: Tuple[int, int],
                         max_dist: int,
                         offset: float) -> Optional[Dict]:
        """Pick the best target: closest to crosshair within max_dist, using aim point."""
        if not detections:
            return None

        cx, cy = crosshair
        best = None
        best_dist = float("inf")

        for det in detections:
            ax, ay = det.get_aim_point(offset)
            dx = ax - cx
            dy = ay - cy
            dist = (dx*dx + dy*dy) ** 0.5

            if dist < max_dist and dist < best_dist:
                best_dist = dist
                best = {
                    "det": det,
                    "aim_x": ax,
                    "aim_y": ay,
                    "dist": dist,
                    "dx": dx,
                    "dy": dy
                }
        return best

    def draw_detections(self, frame: np.ndarray, detections: List[Detection], 
                        best: Optional[Dict], offset: float) -> np.ndarray:
        """Draw boxes + aim points for preview."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            color = (0, 200, 255) if best and det is best.get("det") else (0, 180, 80)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # aim point
            ax, ay = det.get_aim_point(offset)
            cv2.circle(vis, (int(ax), int(ay)), 5, (0, 0, 255), -1)

            label = f"{det.cls_name} {det.conf:.2f}"
            cv2.putText(vis, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis
