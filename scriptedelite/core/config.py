"""
ScriptedElite AI - Configuration (no auth, simple JSON persistence)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

CONFIG_PATH = Path(__file__).parent.parent / "settings.json"


@dataclass
class AppConfig:
    # Detection Engine
    engine: str = "yolo"                 # "yolo" | "color" | "hybrid"
    confidence: float = 0.42
    color_head_offset: float = 38.0      # pixels down from color tag to head
    target_offset: float = 0.18          # YOLO box vertical offset (0=top, 1=bottom)
    max_distance: int = 520              # max px from crosshair for target pick
    use_gpu: bool = True

    # Targeting
    use_linear_prediction: bool = True
    prediction_ms: float = 75.0
    prediction_strength: float = 0.5     # 0..1 blend of prediction
    target_part: str = "head"            # head | chest | body
    target_priority: str = "closest"     # closest | highest_conf

    # Tracking (user-facing names)
    # smoothing = tracking smoothness (0..1, higher = smoother / less snappy)
    # strength  = tracking strength (gain)
    # acquisition_speed = tracking speed
    smoothing: float = 0.35
    strength: float = 1.0
    humanization: float = 0.0
    acquisition_speed: float = 3.2
    fps_limit: int = 120
    curve: str = "linear"                # linear | ease (reserved)

    # Features
    sticky_aim: bool = True              # always keep same player once acquired
    shake_reduction: bool = True

    # Activation / Gating
    # activation_button: physical controller — "l2","r2","lt","rt","lb","rb",...
    # ads_key: keyboard/mouse fallback — "right","left","ctrl","shift", letter keys
    activation_button: str = "l2"
    ads_key: str = "right"
    gated_aim: bool = True
    toggle_key: str = "f"

    # Calibration
    cal_x: int = 0
    cal_y: int = 0

    # Capture
    capture_mode: str = "chiaki"         # chiaki | capture_card | desktop
    chiaki_window: str = "Chiaki"
    capture_device: int = 0
    region_size: int = 0                 # 0 = full window; >0 = center square crop

    # UI / Misc
    show_preview: bool = True
    preview_scale: float = 0.55
    draw_boxes: bool = True

    # Detection Zone (circle around crosshair)
    zone_radius: int = 200
    draw_zone: bool = True

    # Controller platform (for UI labels)
    controller_platform: str = "playstation"  # playstation | xbox
    controller_mode: str = "controller"       # controller | mouse

    # Cronus / Recoil
    current_weapon: str = "AR-15"

    def save(self):
        try:
            CONFIG_PATH.parent.mkdir(exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as e:
            print("Config save error:", e)

    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                defaults = {f: getattr(cls(), f) for f in cls.__dataclass_fields__}
                merged = {**defaults, **data}
                return cls(**{k: v for k, v in merged.items() if k in cls.__dataclass_fields__})
            except Exception as e:
                print("Config load error:", e)
        cfg = cls()
        cfg.save()
        return cfg

    def offset_for_part(self, part: str | None = None) -> float:
        p = (part or self.target_part or "head").lower()
        return {"head": 0.12, "chest": 0.30, "body": 0.45}.get(p, self.target_offset)
