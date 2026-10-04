@echo off
rem Veil launcher: sets itself up on first run, then starts the tray app with no console window.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo First run: setting up Veil. This takes a few minutes and only happens once.
    where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
    if not exist ".venv\Scripts\python.exe" (
        echo Could not create a Python environment. Install Python 3.10+ from python.org and try again.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :failed

    rem The OCR package pulls in CPU-only onnxruntime; swap in the fastest GPU build for this PC.
    ".venv\Scripts\python.exe" -m pip uninstall -y onnxruntime >nul 2>nul
    where nvidia-smi >nul 2>nul && (
        echo NVIDIA GPU found: installing CUDA OCR runtime...
        ".venv\Scripts\python.exe" -m pip install "onnxruntime-gpu[cuda,cudnn]" || goto :failed
    ) || (
        echo Installing DirectML OCR runtime...
        ".venv\Scripts\python.exe" -m pip install onnxruntime-directml || goto :failed
    )
)

start "" ".venv\Scripts\pythonw.exe" -m veil
exit /b 0

:failed
echo Installing dependencies failed. See the messages above.
pause
exit /b 1
