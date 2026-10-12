@echo off
setlocal
cd /d "%~dp0"
echo AeroSentry - Python 3.12 CPU installation
py -3.12 -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if errorlevel 1 (
  echo Install Python 3.12 x64 from python.org with the Python Launcher enabled.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] == (3,12), 'Recreate .venv using Python 3.12'"
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto fail
echo.
echo Installation complete. Run run.bat. First launch uses labeled SIMULATION.
echo For real detection press F2 and select a local drone-trained .pt model and source.
pause
exit /b 0
:fail
echo.
echo Installation failed. Read the error above and README.md troubleshooting.
pause
exit /b 1
