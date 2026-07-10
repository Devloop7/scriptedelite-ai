@echo off
echo Building ScriptedElite AI executable...
cd /d "%~dp0"

python -m pip install pyinstaller --quiet

pyinstaller ^
  --noconfirm ^
  --onedir ^
  --windowed ^
  --name "ScriptedElite AI" ^
  --icon "assets/logo.png" ^
  --add-data "assets;assets" ^
  --add-data "core;core" ^
  --add-data "ui;ui" ^
  main.py

echo.
echo Build complete. Check the dist\ folder.
pause