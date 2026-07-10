# ScriptedElite AI

**Precision sticky target lock** — capture any window, detect players (YOLO + ByteTrack), commit to one target, and hold the configured aim point without overshoot thrash.

- No login / license key
- Scripted Elite branding (electric blue)
- `assets/model.pt` person detection with track IDs
- FOV-aware crop inference + Kalman aim filter
- Critically damped stick control (strength / smoothness / speed)
- Dual engine: YOLO + color-signature fallback
- Virtual controller aim (vgamepad / ViGEm → right stick)

## Quick Start (Windows)

1. Python 3.10+ and (recommended) NVIDIA GPU + CUDA.
2. `pip install -r requirements.txt`
3. Install **ViGEmBus**.
4. `python main.py` or `run.bat`

## How to use

1. **Setup** — pick capture window → Start Engine.
2. Align green zone + X on your crosshair (Calibrate X/Y if needed).
3. **Lock** — Head / Chest / Body, FOV, Smoothness / Strength / Speed.
4. Enable **AIM** (button or **F**) → hold activation (default L2 / right mouse).

## Lock pipeline

| Stage | Behavior |
|-------|----------|
| **Acquire** | First valid detection in FOV |
| **Commit** | Sticky pin by track ID (ByteTrack) |
| **Filter** | Kalman aim at configured body point |
| **Drive** | Damped PD stick — no overshoot fraction |
| **Hold** | Stick zeroed on point |
| **Coast** | Predict through brief miss frames |
| **Release** | Miss budget or left expanded hold zone |

## Project layout

| Path | Role |
|------|------|
| `core/capture.py` | Window / card / desktop frames |
| `core/detector.py` | YOLO track + FOV crop |
| `core/tracker.py` | Sticky identity + Kalman aim |
| `core/controller.py` | ViGEm stick + XInput activation |
| `core/worker.py` | Engine loop |
| `ui/` | Themed PySide6 UI |
| `main.py` | Entry point |

## Key settings

| Setting | Meaning |
|---------|---------|
| Smoothness | Damping / settle softness |
| Strength | Aim gain |
| Speed | Close rate on large gaps |
| Stick Response | Game sensitivity (px per full stick) |
| Zone Radius | Acquire FOV |
| Hold Frames | Coast tolerance |

## Building .exe

```powershell
pyinstaller --noconfirm --onedir --windowed --name "ScriptedElite AI" --icon assets/logo.png --add-data "assets;assets" --add-data "core;core" --add-data "ui;ui" main.py
```

Or `build_exe.bat`.
