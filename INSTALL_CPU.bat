@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Device cpu
if errorlevel 1 echo Installation failed. Read the message above.
pause
