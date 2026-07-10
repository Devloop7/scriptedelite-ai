"""
YOLO Detector wrapper for ScriptedElite AI.
Uses the provided model.pt (single 'person' class).
Supports ByteTrack identity tracking, FOV-aware crop inference, and aim points.
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
    def __init__(self, x1, y1, x2, y2, conf, cls_id, cls_name, track_id=None):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.conf = float(conf)
        self.cls_id = int(cls_id)
        self.cls_name = str(cls_name)
        self.track_id = int(track_id) if track_id is not None else None
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
        self.use_half = False
        self._load_model(use_gpu)

    def _load_model(self, use_gpu: bool = True):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        print(f"[Detector] Loading model from {self.model_path} ...")
        self.model = YOLO(str(self.model_path))
        self.names = self.model.names or {0: "person"}

        self.device = "cpu"
        self.use_half = False
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "0"
                    self.use_half = True
                    print("[Detector] Using CUDA GPU (FP16 enabled)")
                else:
                    print("[Detector] CUDA not available — CPU mode")
            except Exception:
                print("[Detector] torch not available for CUDA check — CPU mode")
        print(f"[Detector] Model loaded. Classes: {self.names} | device={self.device}")

    @staticmethod
    def _pick_imgsz(crop_w: int, crop_h: int) -> int:
        m = max(crop_w, crop_h)
        if m <= 420:
            return 416
        if m <= 560:
            return 512
        return 640

    @staticmethod
    def _fov_crop(
        frame: np.ndarray,
        center: Tuple[float, float],
        radius: float,
        min_size: int = 320,
        max_size: int = 960,
    ) -> Tuple[np.ndarray, int, int]:
        """
        Crop a square around the crosshair for denser YOLO inference.
        Returns (crop_bgr, offset_x, offset_y) in full-frame coords.
        """
        h, w = frame.shape[:2]
        cx, cy = center
        # Diameter covers FOV + margin for tall boxes partially in zone
        side = int(max(min_size, min(max_size, radius * 2.4 + 80)))
        half = side // 2
        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = x1 + side
        y2 = y1 + side
        # Clamp to frame
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            shift = x2 - w
            x1 = max(0, x1 - shift)
            x2 = w
        if y2 > h:
            shift = y2 - h
            y1 = max(0, y1 - shift)
            y2 = h
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame, 0, 0
        return crop, x1, y1

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.4,
        imgsz: int = 640,
        use_track: bool = True,
        fov_center: Optional[Tuple[float, float]] = None,
        fov_radius: Optional[float] = None,
        max_det: int = 12,
        conf_floor: Optional[float] = None,
    ) -> List[Detection]:
        """
        Run YOLO with optional ByteTrack IDs and FOV crop.

        conf_floor: if set, use this lower conf for the model (e.g. while sticky)
                    while still returning boxes; caller can filter acquire harder.
        """
        if self.model is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        off_x, off_y = 0, 0
        src = frame

        if fov_center is not None and fov_radius is not None and fov_radius > 20:
            src, off_x, off_y = self._fov_crop(frame, fov_center, float(fov_radius))

        ch, cw = src.shape[:2]
        run_imgsz = imgsz if imgsz else self._pick_imgsz(cw, ch)
        if max(ch, cw) > 1400 and run_imgsz > 512:
            run_imgsz = 512
        else:
            run_imgsz = self._pick_imgsz(cw, ch)

        conf_run = float(conf_floor if conf_floor is not None else conf_threshold)
        conf_run = max(0.12, min(0.9, conf_run))

        kwargs = dict(
            source=src,
            verbose=False,
            conf=conf_run,
            iou=0.50,
            imgsz=run_imgsz,
            device=self.device,
            max_det=max(1, int(max_det)),
        )
        if self.use_half and self.device != "cpu":
            kwargs["half"] = True

        try:
            if use_track:
                results = self.model.track(
                    persist=True,
                    tracker="bytetrack.yaml",
                    **kwargs,
                )[0]
            else:
                results = self.model.predict(**kwargs)[0]
        except Exception as e:
            # Tracking can fail on first frames / missing deps — fall back to predict
            if use_track:
                try:
                    results = self.model.predict(**kwargs)[0]
                except Exception as e2:
                    print(f"[Detector] predict failed: {e2}")
                    return []
            else:
                print(f"[Detector] predict failed: {e}")
                return []

        detections: List[Detection] = []
        boxes = results.boxes
        if boxes is None:
            return detections

        has_id = boxes.id is not None
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            # Soft filter: keep lower conf if tracking (identity continuity)
            if conf < conf_run:
                continue
            cls_id = int(box.cls[0])
            cls_name = self.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            # Map crop → full frame
            x1 += off_x
            x2 += off_x
            y1 += off_y
            y2 += off_y
            tid = None
            if has_id:
                try:
                    tid = int(boxes.id[i].item())
                except Exception:
                    tid = None
            detections.append(
                Detection(x1, y1, x2, y2, conf, cls_id, cls_name, track_id=tid)
            )
        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        best: Optional[Dict],
        offset: float,
    ) -> np.ndarray:
        vis = frame.copy()
        best_det = best.get("det") if best else None
        best_id = best.get("track_id") if best else None

        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            is_best = False
            if best_det is not None and det is best_det:
                is_best = True
            elif best_id is not None and det.track_id is not None and det.track_id == best_id:
                is_best = True
            color = (0, 220, 255) if is_best else (0, 180, 80)
            thickness = 2 if is_best else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            ax, ay = det.get_aim_point(offset)
            cv2.circle(vis, (int(ax), int(ay)), 5, (0, 0, 255), -1)
            if is_best:
                cv2.circle(vis, (int(ax), int(ay)), 10, (0, 0, 255), 1)

            tid = f"#{det.track_id} " if det.track_id is not None else ""
            label = f"{tid}{det.cls_name} {det.conf:.2f}"
            cv2.putText(
                vis, label, (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return vis
