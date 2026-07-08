"""
ScriptedElite AI - Configuration (no auth, simple JSON persistence)
"""
import json
from pathlib import Path
from dataclasses import dataclass, asdict

CONFIG_PATH = Path(__file__).parent.parent / "settings.json"

@dataclass
class AppConfig:
    # Detection Engine
    engine: str = "yolo"                 # "yolo" | "color" | "hybrid"
    confidence: float = 0.42
    color_head_offset: float = 38.0      # pixels down from color tag to head (Engine B)
    target_offset: float = 0.18          # YOLO box offset
    max_distance: int = 520              # pixels / FOV limit

    # Targeting
    use_linear_prediction: bool = True
    prediction_ms: float = 75.0
    target_part: str = "head"

    # Aiming / Acquisition
    smoothing: float = 0.28
    strength: float = 1.15
    humanization: float = 0.22
    acquisition_speed: float = 3.2       # 1.0-5.8 range per spec archetype
    fps_limit: int = 140

    # Gating (ADS / Aim Key)
    ads_key: str = "right"               # pynput style: 'right', 'ctrl', 'f', etc. Must be held.
    gated_aim: bool = True

    # Calibration
    cal_x: int = 0                       # Crosshair calibration offsets
    cal_y: int = 0

    # Capture
    capture_mode: str = "chiaki"         # chiaki | capture_card | desktop
    chiaki_window: str = "Chiaki"
    capture_device: int = 0

    # Hotkeys
    toggle_key: str = "f"                # master enable toggle

    # UI / Misc
    show_preview: bool = True
    preview_scale: float = 0.55

    # Cronus / Recoil
    current_weapon: str = "AR-15"

    def save(self):
        try:
            CONFIG_PATH.parent.mkdir(exist_ok=True)
            with open(CONFIG_PATH, "w") as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as e:
            print("Config save error:", e)

    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text())
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except Exception:
                pass
        cfg = cls()
        cfg.save()
        return cfg
