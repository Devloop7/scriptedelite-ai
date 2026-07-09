"""
YOLO Detector wrapper for ScriptedElite AI.
Uses the provided model.pt (single 'person' class).
Computes aim points with configurable vertical offset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore


class Detection:
    def __init__(self, x1, y1, x2, y2, conf, cls_id, cls_name):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.conf = float(conf)
        self.cls_id = int(cls_id)
        self.cls_name = str(cls_name)
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.cx = (self.x1 + self.x2) / 2.0
        self.cy = (self.y1 + self.y2) / 2.0

    def get_aim_point(self, offset: float = 0.18) -> Tuple[float, float]:
        """
        offset: 0.0 = top of box, 0.12 ≈ head, 0.30 ≈ chest, 0.45 ≈ body
        """
        aim_y = self.y1 + self.height * max(0.0, min(1.0, offset))
        return self.cx, aim_y


class YOLODetector:
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = Path(model_path)
        self.model = None
        self.names = {0: "person"}
        self.device = "cpu"
        self._load_model(use_gpu)

    def _load_model(self, use_gpu: bool = True):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        print(f"[Detector] Loading model from {self.model_path} ...")
        self.model = YOLO(str(self.model_path))
        self.names = self.model.names or {0: "person"}

        # Prefer CUDA when available
        self.device = "cpu"
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "0"
                    print("[Detector] Using CUDA GPU")
                else:
                    print("[Detector] CUDA not available — CPU mode")
            except Exception:
                print("[Detector] torch not available for CUDA check — CPU mode")
        print(f"[Detector] Model loaded. Classes: {self.names} | device={self.device}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.4,
        imgsz: int = 640,
    ) -> List[Detection]:
        if self.model is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        # Keep aspect; ultralytics handles letterbox. Cap very large frames for FPS.
        run_imgsz = imgsz
        if max(h, w) > 1600:
            run_imgsz = 640

        results = self.model.predict(
            source=frame,
            verbose=False,
            conf=conf_threshold,
            iou=0.45,
            imgsz=run_imgsz,
            device=self.device,
        )[0]

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
            detections.append(Detection(x1, y1, x2, y2, conf, cls_id, cls_name))
        return detections

    def find_best_target(
        self,
        detections: List[Detection],
        crosshair: Tuple[float, float],
        max_dist: float,
        offset: float,
        priority: str = "closest",
    ) -> Optional[Dict]:
        """Pick best target within max_dist of crosshair using aim point."""
        if not detections:
            return None

        cx, cy = crosshair
        best = None
        best_score = float("inf")

        for det in detections:
            ax, ay = det.get_aim_point(offset)
            dx = ax - cx
            dy = ay - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > max_dist:
                continue

            if priority == "highest_conf":
                # Lower is better: invert conf, slight distance tie-break
                score = (1.0 - det.conf) * 1000.0 + dist * 0.01
            else:
                score = dist

            if score < best_score:
                best_score = score
                best = {
                    "det": det,
                    "aim_x": ax,
                    "aim_y": ay,
                    "dist": dist,
                    "dx": dx,
                    "dy": dy,
                    "conf": det.conf,
                }
        return best

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        best: Optional[Dict],
        offset: float,
    ) -> np.ndarray:
        vis = frame.copy()
        best_det = best.get("det") if best else None

        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            is_best = best_det is not None and det is best_det
            color = (0, 220, 255) if is_best else (0, 180, 80)
            thickness = 2 if is_best else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            ax, ay = det.get_aim_point(offset)
            cv2.circle(vis, (int(ax), int(ay)), 5, (0, 0, 255), -1)
            if is_best:
                # Line from crosshair-ish to aim will be drawn by caller; mark aim clearly
                cv2.circle(vis, (int(ax), int(ay)), 10, (0, 0, 255), 1)

            label = f"{det.cls_name} {det.conf:.2f}"
            cv2.putText(
                vis, label, (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return vis
