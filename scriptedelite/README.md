# ScriptedElite AI

**PS5-Optimized AI Aim Tool** (rebranded from Ghost AI Tool v7 archetype).

- No login / license key — direct access for friends
- Completely new modern design (Scripted Elite branding + your logo)
- Uses the **exact same** `model.pt` (YOLO person detection)
- Dual-engine: YOLO + Color-signature fallback (magenta FF00FA / green)
- Chiaki Remote Play + Capture Card support
- Controller thumbstick emulation (vgamepad / ViGEm)
- Linear Prediction Targeting
- ADS-gated aim (only when aim key held)
- Crosshair calibration + acquisition speed
- Cronus Zen recoil matrix + ZPU profile support

## Quick Start (Windows)

1. Ensure Python 3.10+ and (recommended) NVIDIA GPU with CUDA.

2. Activate venv:
   ```powershell
   cd C:\scriptedelite
   .\.venv\Scripts\Activate.ps1
   ```

3. Install deps:
   ```powershell
   pip install -r requirements.txt
   # For CUDA performance:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   **Important for controller emulation:**
   - Install ViGEmBus driver (required by vgamepad, same as original).

4. Run:
   ```powershell
   python main.py
   ```

## Core Architecture (per spec)
- **Engine A (YOLO)**: Full neural detection + Linear Prediction.
- **Engine B (Color)**: Fast CPU fallback on bright magenta/green nameplates + Head Offset.
- **Capture**: Chiaki window (preferred) or capture card device.
- **Gating**: Only applies stick input while your ADS/Aim key is held.
- **Output**: Virtual controller thumbsticks for PS5 remote play / console.
- **Cronus**: Built-in recoil tables per weapon. Export notes/profiles for ZPU.

## Key Settings
- Capture Source (Chiaki / Capture Card / Desktop)
- Detection Engine (YOLO / Color / Hybrid)
- ADS Key (must be held to activate aim)
- Crosshair Calibration (X/Y offsets)
- Acquisition Speed (up to ~5.8 range)
- Head Offset (for color engine)
- Recoil Profiles (per weapon vertical/horizontal)

## Controls
- Master Toggle: **F** (or configured)
- ADS Gate: Hold your configured aim key (e.g. right mouse or controller button mapped in Chiaki)

## Building .exe
Use `build_exe.bat` or:
```powershell
pyinstaller --noconfirm --onedir --windowed --name "ScriptedElite AI" --icon assets/logo.png --add-data "assets;assets" --add-data "core;core" main.py
```

## Credits
Core detection model preserved. Full rebrand + architecture aligned to the v7 PS5 optimization spec. For private/friends use only.
