"""
ScriptedElite AI - Configuration (JSON persistence)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "settings.json"


@dataclass
class AppConfig:
    # Detection Engine
    engine: str = "yolo"                 # "yolo" | "color" | "hybrid"
    confidence: float = 0.42
    color_head_offset: float = 38.0      # pixels down from color tag to aim
    target_offset: float = 0.12          # YOLO box vertical offset (synced from target_part)
    max_distance: int = 520              # hard max px from crosshair for target pick
    use_gpu: bool = True

    # Enemy color filter (Color / Hybrid color path)
    enemy_color: str = "#FF00FA"
    color_tolerance: float = 55.0

    # Targeting
    use_linear_prediction: bool = False  # off by default — pure lock on detection point
    prediction_ms: float = 60.0
    prediction_strength: float = 0.35
    target_part: str = "head"            # head | chest | body
    target_priority: str = "closest"     # closest | highest_conf
    aim_fine_tune: float = 0.0

    # Tracking (user-facing)
    # smoothing = tracking smoothness (0..1, higher = smoother / more damping)
    # strength  = tracking strength (gain)
    # acquisition_speed = tracking speed
    smoothing: float = 0.35
    strength: float = 1.0
    humanization: float = 0.0
    acquisition_speed: float = 3.0
    fps_limit: int = 120

    # Lock behavior
    shake_reduction: bool = True
    lock_hold_frames: int = 28

    # Stick response: estimated px of camera motion per full stick deflection / frame
    # Higher = game is more sensitive (use less stick for same error)
    stick_response: float = 55.0
    adaptive_stick: bool = False

    # Activation / Gating
    activation_button: str = "l2"
    ads_key: str = "right"
    gated_aim: bool = True
    toggle_key: str = "f"

    # Calibration
    cal_x: int = 0
    cal_y: int = 0

    # Capture
    capture_mode: str = "window"
    chiaki_window: str = "Chiaki"
    window_title: str = ""
    window_hwnd: int = 0
    capture_device: int = 0
    region_size: int = 0

    # UI / Misc
    show_preview: bool = True
    draw_boxes: bool = True

    # Detection Zone (circle around crosshair)
    zone_radius: int = 200
    draw_zone: bool = True

    # Controller platform (UI labels)
    controller_platform: str = "playstation"

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
                if "window_title" not in data and data.get("chiaki_window"):
                    data["window_title"] = data["chiaki_window"]
                if data.get("capture_mode") == "chiaki" and not data.get("window_title"):
                    data["window_title"] = data.get("chiaki_window") or "Chiaki"
                # Drop legacy unknown keys
                merged = {**defaults, **{k: v for k, v in data.items() if k in defaults}}
                return cls(**merged)
            except Exception as e:
                print("Config load error:", e)
        cfg = cls()
        cfg.save()
        return cfg

    def offset_for_part(self, part: str | None = None) -> float:
        p = (part or self.target_part or "head").lower()
        base = {"head": 0.12, "chest": 0.30, "body": 0.45}.get(p, self.target_offset)
        fine = max(-0.10, min(0.10, float(self.aim_fine_tune or 0.0)))
        return max(0.0, min(1.0, base + fine))
