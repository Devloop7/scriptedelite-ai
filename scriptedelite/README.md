# ScriptedElite AI

**PS5-optimized capture + YOLO tracking** for Chiaki Remote Play.

- No login / license key
- Scripted Elite branding (electric blue)
- `assets/model.pt` person detection (YOLO)
- Dual engine: YOLO + color-signature fallback
- Full Chiaki window capture (or capture card / desktop)
- Detection zone circle + center X overlay for calibration
- Virtual controller aim (vgamepad / ViGEm → right stick)
- Activation via L2 / R2 / LT / RT (XInput) or mouse / keyboard
- Tracking smoothness, strength, and speed controls

## Quick Start (Windows)

1. Python 3.10+ and (recommended) NVIDIA GPU + CUDA.

2. Install deps:
   ```powershell
   cd C:\scriptedelite
   pip install -r requirements.txt
   # Optional CUDA torch:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install **ViGEmBus** (required for virtual controller output).

4. Run:
   ```powershell
   python main.py
   ```
   Or double-click `run.bat`.

## How to use

1. Open **Chiaki**, connect to your PS5 (fullscreen is best).
2. In ScriptedElite → **Capture** tab:
   - Source = `chiaki`
   - Window title = `Chiaki` (or your exact title)
   - Region size = `Full window` (0)
3. Click **Start Engine**.
4. Confirm the **green circle + center X** sits on your in-game crosshair.
   - If not, use **Calibrate X/Y**.
5. **Aim** tab: set zone radius, smoothness, strength, speed, head/chest/body.
6. **Activation** tab: pick controller button (default **L2**) and/or mouse (default **right**).
7. Enable **AIM** (button or **F** key).
8. Hold the activation button → tracking follows the target inside the zone.

## Architecture

| Piece | Role |
|-------|------|
| `core/capture.py` | Chiaki / capture card / desktop frames |
| `core/detector.py` | YOLO person boxes + aim point |
| `core/color_detector.py` | Magenta/green tag fallback |
| `core/predictor.py` | Linear lead on moving targets |
| `core/controller.py` | ViGEm stick out + XInput button in |
| `core/config.py` | `settings.json` persistence |
| `main.py` | UI + engine loop |

## Key settings

| Setting | Meaning |
|---------|---------|
| Tracking Smoothness | Higher = smoother, less snappy |
| Tracking Strength | Overall aim gain |
| Tracking Speed | How fast the stick closes the gap |
| Zone Radius | FOV circle around crosshair |
| Activation Button | L2/R2/LT/RT/… held to track |
| Calibrate X/Y | Nudge center X onto crosshair |

## Controls

- Master toggle: **F** (configurable)
- Activation: hold **L2** (or configured pad / mouse button)

## Building .exe

```powershell
pyinstaller --noconfirm --onedir --windowed --name "ScriptedElite AI" --icon assets/logo.png --add-data "assets;assets" --add-data "core;core" main.py
```

Or use `build_exe.bat`.
