@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run install.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" main.py %*
if errorlevel 1 (
  echo AeroSentry exited with an error. Copy the details above for troubleshooting.
  pause
  exit /b 1
)
