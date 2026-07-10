# ScriptedElite AI

**Precision sticky target lock** — capture any window, detect players (YOLO), commit to one target, and hold the configured aim point as consistently as possible.

- No login / license key
- Scripted Elite branding (electric blue)
- `assets/model.pt` person detection (YOLO)
- Dual engine: YOLO + color-signature fallback
- **Any application window** as capture source (game, Chrome, Chiaki, …)
- Capture card / desktop options
- Detection zone + center X for crosshair calibration
- Virtual controller aim (vgamepad / ViGEm → right stick)
- Activation via L2 / R2 / LT / RT (XInput) or mouse / keyboard
- Hard sticky lock: same player until truly gone

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

1. Open the game / Chiaki / app you want to capture.
2. **Capture** tab:
   - Source = `Application Window`
   - Click **Refresh List** → select the target window
   - Region size = `Full window` (0)
3. Click **Start Engine**.
4. Confirm the **green circle + center X** sits on your in-game crosshair.
   - If not, use **Calibrate X/Y**.
5. **Aim Lock** tab: set zone radius, head/chest/body, smoothness, strength, speed.
6. **Activation** tab: pick controller button (default **L2**) and/or mouse (default **right**).
7. Enable **AIM** (button or **F** key).
8. Hold the activation button → first valid target in the zone is locked and held.

## How lock works

| Stage | Behavior |
|-------|----------|
| **Acquire** | First valid detection in FOV (closest or highest conf) |
| **Commit** | That player is sticky — others are ignored |
| **Aim point** | Configured offset on the box (head / chest / body + fine tune) |
| **Hold** | Stick is zeroed when on point; target motion re-engages tracking |
| **Coast** | Brief YOLO dropouts keep the last aim so lock does not blink off |
| **Release** | Only after many missed frames or track leaves the expanded hold zone |

## Architecture

| Piece | Role |
|-------|------|
| `core/capture.py` | Window / Chiaki / capture card / desktop frames |
| `core/detector.py` | YOLO person boxes + aim point |
| `core/color_detector.py` | Enemy-color blob fallback |
| `core/tracker.py` | Hard sticky identity + aim-point follow |
| `core/predictor.py` | Optional capped lead on moving targets |
| `core/controller.py` | ViGEm stick out + XInput activation in |
| `core/config.py` | `settings.json` persistence |
| `main.py` | UI + engine loop |

## Key settings

| Setting | Meaning |
|---------|---------|
| Tracking Smoothness | Higher = softer motion |
| Tracking Strength | Overall aim gain |
| Tracking Speed | How fast the stick closes the gap |
| Zone Radius | FOV circle for *acquiring* a new target |
| Lock Hold Frames | Miss tolerance while coasting a sticky lock |
| Aim Fine Tune | Small vertical nudge on head/chest/body |
| Activation Button | L2/R2/LT/RT/… held to drive stick |
| Calibrate X/Y | Nudge center X onto crosshair |

## Controls

- Master toggle: **F** (configurable)
- Activation: hold **L2** (or configured pad / mouse button)

## Building .exe

```powershell
pyinstaller --noconfirm --onedir --windowed --name "ScriptedElite AI" --icon assets/logo.png --add-data "assets;assets" --add-data "core;core" main.py
```

Or use `build_exe.bat`.
