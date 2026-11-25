@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run INSTALL_CPU.bat first.
  pause
  exit /b 2
)
.venv\Scripts\python.exe -m phone_vlm gui --run runs/experiment_001
if errorlevel 1 pause
